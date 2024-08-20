# Modified from: https://github.com/ytZhang99/U2Fusion-pytorch/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY
import sys
import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np

class DenseLayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(DenseLayer, self).__init__()
        self.conv = ConvBlock(num_channels, growth, kernel_size=3, act_type='lrelu', norm_type=None)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out


@ARCH_REGISTRY.register()
class DenseNet(nn.Module):
    def __init__(self, in_channels = 2, out_channels = 1,num_features = 64, growth = 64, num_layers = 5):
        super(DenseNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.growth = growth
        modules = []
        self.conv_1 = ConvBlock(self.in_channels, self.num_features, kernel_size=3, act_type='lrelu', norm_type=None)
        for i in range(num_layers):
            modules.append(DenseLayer(self.num_features, self.growth))
            self.num_features += self.growth
        self.dense_layers = nn.Sequential(*modules)
        self.sub = nn.Sequential(ConvBlock(self.num_features, 128, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(128, 64, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(64, 32, kernel_size=3, act_type='lrelu', norm_type=None),
                                 nn.Conv2d(32, self.out_channels, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())

    def forward(self, x_over, x_under):
        x = torch.cat((x_over, x_under), dim=1)
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.sub(x)
        return x

# network functions
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type='prelu', slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(negative_slope=slope, inplace=True)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % norm_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,
              act_type='prelu', norm_type='bn', pad_type='zero'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)

    act = activation(act_type) if act_type else None
    n = norm(out_channels, norm_type) if norm_type else None
    return sequential(p, conv, n, act)







class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)
    

@ARCH_REGISTRY.register()
class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(torch.cat((x_vis_p2,x_inf_p2),dim=1))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x