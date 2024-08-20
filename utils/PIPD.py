# Modified from: https://github.com/ytZhang99/U2Fusion-pytorch/blob/master/train.py

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg

class PIPD(nn.Module):

    def __init__(self, c = 0.1, device = 'cpu', layer = [3, 8, 15, 22, 29]):
        super(PIPD, self).__init__()
        vgg_model = torchvision.models.vgg16(weights=vgg.VGG16_Weights.DEFAULT).to(device)
        blocks = [vgg_model.features[i].eval() for i in range(len(vgg_model.features))]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.c = c
        self.layer = layer
        self.device = device

    def forward(self, input):
        size = input.shape[2:]
        gaussian_blur = torchvision.transforms.GaussianBlur((3,3), sigma=(0.1, 5.0))

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        x = input
        blured_x = gaussian_blur(input)

        w_list = []
        feature_maps = []
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in self.layer:
                    feature_maps.append(x)
                    m = torch.mean(self.features_grad(x).pow(2), dim=[1]).unsqueeze(1)  #(b,1,h,w)

                    m = self.transform(m, mode='bilinear', size=size, align_corners=False)  #(b,1, img_size, img_size)
                    w = torch.unsqueeze(m, dim=-1)  #(b,1,img_size,img_size, 1)
                    w_list.append(w)
    
            w = torch.cat(w_list, dim=-1)   #(b,1,img_size, img_size, 5)
            weight = torch.mean(w, dim=-1)  #(b,1,img_size, img_size)
        return weight / self.c, feature_maps
    
    def features_grad(self, features):
        kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(self.device)
        _, c, _, _ = features.shape
        c = int(c)
        for i in range(c):
            feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
            if i == 0:
                feat_grads = feat_grad
            else:
                feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
        return feat_grads
