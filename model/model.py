import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from collections import OrderedDict
from utils.registry import MODEL_REGISTRY
import os
import logging
from loss.loss import *
import numpy as np
logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class SegFusion(BaseModel):
    def __init__(self, opt):
        super(SegFusion, self).__init__(opt)
        self.network_names = ["netFusion", "netSeg"]  
        self.begin_step = 0
        self.begin_epoch = 0
        self.pretrained = opt['pretrained']
        self.networks = {}
        self.optimizers = {}
        self.loss = {}
        self.define_network(opt)
        self.define_optimizers(opt['train']['optimizers'])
        self.define_loss(opt['loss_type'])
        self.set_network_state("train")
        self.alpha = opt['alpha']
        self.modality = opt['dataset']['modality']

        if self.opt['phase'] == 'train':
            self.log_dict = OrderedDict()
        if self.opt['distributed'] == True:
            print(f"Device : {self.opt['gpu_ids']}")
        else:
            print(f'Device : {self.device}')
            
    def feed_data(self, data, epoch = 0, iter = 0):
        self.data = self.set_device(data)
        if self.opt['phase'] == 'train':
            self.src1 = self.data[self.modality['src1']]
            self.src2 = self.data[self.modality['src2']]
            self.label = self.data[self.modality['label']].to(torch.float)
            self.mask = self.data[self.modality['mask']].to(torch.long)
            
            '''
            Dilation Label
            '''
            # label 맨처음에 크게 나중 epochs에 작게
            kernel_size = 3
            final_scale = 1e-2  # 목표 스케일 값으로 매우 작은 값을 설정합니다.

            kernel = torch.ones((1, 1, kernel_size, kernel_size)).to(self.device).to(torch.float)
            decay_rate = -np.log(final_scale / (iter + 1e-10)) / (self.opt['epochs'] - 0)
            i = int(iter * np.exp(-decay_rate * epoch))
            self.i = i
            for _ in range(i):
                self.label = F.conv2d(self.label, kernel, padding=kernel_size // 2)
            self.label[self.label>0] = 1
            self.label = self.label.to(torch.long)
            self.label *= self.mask
        else:
            self.src1 = self.data[self.modality['src1']]
            self.src2 = self.data[self.modality['src2']]


    def optimize_parameters(self):

        self.set_requires_grad(['netFusion'], True)
        self.set_requires_grad(['netSeg'], False)

        self.fusion = self.netFusion(self.src1, self.src2)

        l_fusion, l_fusion_1, l_fusion_2, l_fusion_3 = self.loss['Fusion_loss']((self.fusion+1)/2, (self.src1+1)/2, (self.src2+1)/2)
        
        #BisNet 
        # if self.fusion.shape[1] != 3:
        #     self.fusion = self.fusion.repeat(1, 3, 1, 1)
        inputs = torch.concat([self.fusion,self.mask], axis = 1)
        out, out16, _ = self.netSeg(inputs)

        l_seg_p =self.loss['Seg_loss'](out, self.label)
        l_seg_16 =self.loss['Seg_loss'](out16, self.label)
        l_seg = l_seg_p + 0.1 * l_seg_16

        l_tot = l_fusion + self.alpha * l_seg

        self.optimizers['netFusion'].zero_grad()
        l_tot.backward()       
        self.optimizers['netFusion'].step()

        self.log_dict['l_fusion_ssim'] = l_fusion_1
        self.log_dict['l_fusion_mse'] = l_fusion_2
        self.log_dict['l_fusion_texture'] = l_fusion_3

        self.log_dict['l_fusion'] = l_fusion.item()
        self.log_dict['l_seg'] = l_seg.item()
        self.log_dict['l_tot'] = l_tot.item()




        self.set_requires_grad(['netSeg'], True)
        self.set_requires_grad(['netFusion'], False)

        fusion_copy = self.fusion.detach().clone().requires_grad_(True)
        inputs = torch.concat([fusion_copy,self.mask], axis = 1)
        out, out16, _ = self.netSeg(inputs)

        l_seg_p =self.loss['Seg_loss'](out, self.label)
        l_seg_16 =self.loss['Seg_loss'](out16, self.label)
        l_seg = l_seg_p + 0.75*l_seg_16

        
        self.optimizers['netSeg'].zero_grad()
        l_seg.backward()
        self.optimizers['netSeg'].step()


    def test(self, src1, src2, mask = None, run_fusion = False):
        self.netFusion.eval()
        self.sample = self.netFusion(src1, src2)
                    
        self.netFusion.train()
        if not run_fusion:
            inputs = torch.concat([self.sample, self.mask], axis = 1)
            self.netSeg.eval()
            self.predict, _, _ = self.netSeg(inputs)
            self.netSeg.train()

        if self.sample.shape[1] != 3:
            self.sample = self.sample.repeat(1, 3, 1, 1)


    def save_network(self, epoch, iter_step, log_dir):
        check_path = os.path.join(log_dir, 'checkpoints')
        if not os.path.isdir(check_path):
            os.mkdir(check_path)
        netFusion_path = os.path.join(check_path, 'I{}_E{}_netFusion.pth'.format(iter_step, epoch))
        netSeg_path = os.path.join(check_path, 'I{}_E{}_netSeg.pth'.format(iter_step, epoch))
        optFusion_path = os.path.join(check_path, 'I{}_E{}_optFusion.pth'.format(iter_step, epoch))
        optSeg_path = os.path.join(check_path, 'I{}_E{}_optSeg.pth'.format(iter_step, epoch))

        # netFusion
        network = self.netFusion
        if isinstance(self.netFusion, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, netFusion_path, _use_new_zipfile_serialization=False)
        # netSeg
        network = self.netSeg
        if isinstance(self.netSeg, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, netSeg_path, _use_new_zipfile_serialization=False)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optimizers['netFusion'].state_dict()
        torch.save(opt_state, optFusion_path)
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optimizers['netSeg'].state_dict()
        torch.save(opt_state, optSeg_path)

        logger.info('Saved model in [{:s}] ...'.format(netFusion_path))



    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info('Loading pretrained model for netFusion [{:s}] ...'.format(load_path))
            netFusion_path = '{}_netFusion.pth'.format(load_path)

            network = self.netFusion
            if isinstance(self.netFusion, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(netFusion_path))#, strict=(not self.opt['model']['finetune_norm']))


            if self.opt['phase'] == 'train':
                netSeg_path = '{}_netSeg.pth'.format(load_path)
                network = self.netSeg
                if isinstance(self.netSeg, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(netSeg_path))#, strict=(not self.opt['model']['finetune_norm']))

    def load_opt(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for Fusion [{:s}] ...'.format(load_path))
            optFusion_path = '{}_optFusion.pth'.format(load_path)
            optSeg_path = '{}_optSeg.pth'.format(load_path)

            # optimizer
            optFusion = torch.load(optFusion_path)
            self.optimizers['netFusion'].load_state_dict(optFusion['optimizer'])
            self.begin_step = optFusion['iter']
            self.begin_epoch = optFusion['epoch']
            optSeg = torch.load(optSeg_path)
            self.optimizers['netSeg'].load_state_dict(optSeg['optimizer'])
            self.begin_step = optSeg['iter']
            self.begin_epoch = optSeg['epoch']

    
    def _set_lr(self, optimizer, lr):
        """set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def update_learning_rate(self, epoch, optimizer, init_lr, rate):
        print(f'lr : {init_lr} -> {init_lr*(rate **epoch)}')
        self._set_lr(optimizer, init_lr*(rate **epoch))

