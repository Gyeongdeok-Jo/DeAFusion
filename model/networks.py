import torch
import torch.nn as nn
from utils.registry import ARCH_REGISTRY

def build_network(net_opt, opt, pretrained, name):

    which_network = net_opt['which_network']
    net = ARCH_REGISTRY.get(which_network)(**net_opt['setting'])
    if pretrained and (name == 'netFusion'):
        net.load_state_dict(torch.load('./model/densenet.pth')['model'])

    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net, device_ids=opt['gpu_ids'])

    return net
