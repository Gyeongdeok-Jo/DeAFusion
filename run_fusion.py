from utils.util import *
from model.model import SegFusion
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.dataset import CustomImageDataset
from torch.utils.data import DataLoader
import os 
import argparse
import torch
import cv2

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def run_fusion(opt_path, img_size):
    opt = load_yaml(opt_path)
    model = SegFusion(opt)
    model.load_network()
    model.load_opt()
    save_dir = opt['path']['save_dir']
    grayscale = opt['grayscale']
    log_dir = opt['log_dir']

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)    
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)    

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    


    modality = opt['dataset']['modality']
    input_modality = list(modality.values())

    test_transform = A.Compose([
        # A.geometric.Resize(height= img_size, width = img_size, interpolation = cv2.INTER_LINEAR, p=1),
        A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value = 1),
        ToTensorV2(transpose_mask=True),
        ],
        additional_targets=create_dict(input_modality)
        )
    dataset = CustomImageDataset(data_dir = opt['dataset']['test']['root'], 
                                    modality = input_modality, 
                                    transforms = test_transform,
                                    grayscale = grayscale)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers = 2)

    with torch.no_grad():
        model.netFusion.eval()
        for (test_data, src_path) in dataloader:

            name = src_path[0].split('/')[-1]
            name = name.replace(input_modality[-1], 'fusion')
            model.feed_data(test_data)
            model.test(model.src1, model.src2, run_fusion = True)
            fused_image = model.sample
            save_path = os.path.join(save_dir, name)
            
            image = fused_image[0, :, :, :]
            image = image.cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = min_max_scaling(image)*255.
            image = image.astype(np.uint8)
            cv2.imwrite(save_path, image)
            print('Fusion {0} Sucessfully!'.format(save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegFusion')
    parser.add_argument('--opt_path', '-p', type=str, default='')
    parser.add_argument('--img_size', '-i', type=int, default=256)

    args = parser.parse_args()
    run_fusion(opt_path = args.opt_path,
                img_size = args.img_size,

    )
    print("Inference Done!")