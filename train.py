import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.dataset import CustomImageDataset
import os
import cv2

from utils.util import *
from model.model import SegFusion
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.data import decollate_batch
import torch.nn.functional as F

import shutil
import random
import time

import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def train(opt_path, img_size, load_model = False):

    torch.manual_seed(1234)
    random.seed(1234)

    opt = load_yaml(opt_path)
    model = SegFusion(opt)
    

    log_dir = opt['log_dir']
    batch_size = opt['dataset']['batch_size']

    epochs = opt['epochs']
    dilation = opt['dilation']
    phase = opt['phase']
    grayscale = opt['grayscale']


    if not os.path.exists(log_dir):
        os.mkdir(log_dir)    

    if load_model:
        model.load_network()
        model.load_opt()
    else:
        if phase == 'train':
            try:
                shutil.rmtree(log_dir + '/')
                os.rmdir(log_dir)
            except:
                pass
    modality = opt['dataset']['modality']
    input_modality = list(modality.values())

    train_transform = A.Compose([
        A.geometric.Resize(height= img_size, width = img_size, interpolation = cv2.INTER_LINEAR, p=1),
        A.geometric.transforms.Affine(scale = (0.9, 1.1),
                                    rotate=(-15, 15),
                                    shear=(-18,18),
                                    interpolation=cv2.INTER_LINEAR,
                                    mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value = 1),
        ToTensorV2(transpose_mask=True),
        ],
        additional_targets=create_dict(input_modality)
        )

    valid_transform = A.Compose([
        A.geometric.Resize(height= img_size, width = img_size, interpolation = cv2.INTER_LINEAR, p=1),
        A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value = 1),
        ToTensorV2(transpose_mask=True),
        ],
        additional_targets=create_dict(input_modality)
        )


    train_dataset = CustomImageDataset(data_dir = opt['dataset']['train']['root'], 
                                       modality = input_modality, 
                                       transforms = train_transform,
                                       grayscale = grayscale)
    valid_dataset = CustomImageDataset(data_dir = opt['dataset']['val']['root'], 
                                       modality = input_modality, 
                                       transforms = valid_transform,
                                       grayscale = grayscale)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 32)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 32)


    
    step = model.begin_step

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)    
    
    writer = SummaryWriter(log_dir)
    epoch_loss_values = list()
    metric_values = list()

    dataformat = "HWC"


    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    best_metric = -1
    if not load_model:
        for epoch in range(model.begin_epoch+1, epochs, 1):
            start_time = time.time()
            print('{:-^30}'.format(f'{epoch}/{epochs}'))

            epoch_loss = 0
            cnt = 0
            for (train_data, _) in train_loader:
                cnt += 1
                step +=1

                model.feed_data(train_data, epoch, dilation)
                model.optimize_parameters()
                l_fusion_ssim = model.log_dict['l_fusion_ssim']
                l_fusion_mse = model.log_dict['l_fusion_mse']
                l_fusion_texture = model.log_dict['l_fusion_texture']
                l_fusion = model.log_dict['l_fusion']
                l_seg = model.log_dict['l_seg']
                l_tot = model.log_dict['l_tot']


                epoch_loss += l_tot
                epoch_len = len(train_loader)
                # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("l_fusion_ssim", l_fusion_ssim, epoch_len * epoch + cnt)
                writer.add_scalar("l_fusion_mse", l_fusion_mse, epoch_len * epoch + cnt)
                writer.add_scalar("l_fusion_texture", l_fusion_texture, epoch_len * epoch + cnt)

                writer.add_scalar("l_fusion", l_fusion, epoch_len * epoch + cnt)
                writer.add_scalar("l_seg", l_seg, epoch_len * epoch + cnt)
                writer.add_scalar("l_tot", l_tot, epoch_len * epoch + cnt)

            with torch.no_grad():
                for i, (valid_data, _) in enumerate(valid_loader):
                    model.feed_data(valid_data, epoch, dilation)
                    model.test(model.src1, model.src2, model.mask)
                    val_outputs = [post_trans(i) for i in decollate_batch(model.predict)]
                    dice_metric(y_pred=val_outputs, y=model.label)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                writer.add_scalar("val_mean_dice", metric, epoch)        


                fused_img = model.sample
                mask = model.mask[0].transpose(2,0).detach().to('cpu').numpy()
                src1 = min_max_scaling(model.src1[0].transpose(2,0).detach().to('cpu').numpy())
                src1 = cv2.cvtColor(src1, cv2.COLOR_GRAY2RGB)
                src2 = min_max_scaling(model.src2[0].transpose(2,0).detach().to('cpu').numpy())
                src2 = cv2.cvtColor(src2, cv2.COLOR_GRAY2RGB)
                fused_img = min_max_scaling(fused_img[0].transpose(2,0).detach().to('cpu').numpy())
                label = model.label[0].transpose(2,0).detach().to('cpu').numpy().astype(np.float32)
                pred =  val_outputs[0].transpose(2,0).detach().to('cpu').numpy()

                '''
                IPD Map 출력을 위한거
                '''
                if opt['loss_type']['Fusion_loss']['which_loss'] == 'new_loss': 
                    IPD = model.loss['Fusion_loss'].IPD
                    weight1, _ = IPD(model.src1)
                    weight2, _ = IPD(model.src2)
                    IPD_map = torch.cat((weight1, weight2), dim = 1)
                    IPD_map = F.softmax(IPD_map, dim=1)[0].transpose(2,0).to('cpu').numpy()
                    colormap = plt.get_cmap('PuOr')
                    IPD_map = colormap(IPD_map[:,:,0])
                    IPD_map = (IPD_map[:, :, :3] * 255).astype(np.uint8)
                    writer.add_image("IPD_map", IPD_map, epoch, dataformats = dataformat)


                writer.add_image("T1", src1, epoch, dataformats = dataformat)
                writer.add_image("FLAIR", src2, epoch, dataformats = dataformat)
                writer.add_image("Fusion", fused_img, epoch, dataformats = dataformat)
                writer.add_image("output", overlay_plot(fused_img * mask, pred, color = [101,255,101]), epoch, dataformats = dataformat)
                writer.add_image("label", overlay_plot(src2 * mask, label, color = [255,0,0]), epoch, dataformats = dataformat)
                
            # for dilation
            if model.i > 0:
                metric = 0

            if metric > best_metric :
                best_metric = metric
                best_metric_epoch = epoch
                model.save_network(epoch, step, log_dir)
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch, metric, best_metric, best_metric_epoch
                )
            )
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            end_time = time.time()

            model.update_learning_rate(epoch, model.optimizers['netFusion'], opt['train']['optimizers']['netFusion']['lr'], 0.995)
            # model.update_learning_rate(epoch, model.optimizers['netFusion'], opt['train']['optimizers']['netSeg']['lr'], 1.01)

            execution_time = end_time - start_time
            print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
            print(f"Execution time: {execution_time} seconds")
        print("training Done!")
    else:
        print('phase is val')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegFusion')
    parser.add_argument('--opt_path', '-p', type=str, default='')
    parser.add_argument('--img_size', '-i', type=int, default=256)
    parser.add_argument('--load_model', '-l', type=bool, default=False)

    args = parser.parse_args()
    train(opt_path = args.opt_path,
          img_size = args.img_size,
          load_model = args.load_model
          )  
