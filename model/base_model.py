import torch
import torch.nn as nn
from model.networks import build_network
from utils.registry import LOSS_REGISTRY


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        if opt['distributed']:
            self.device = torch.device(f'cuda:{opt["gpu_ids"][0]}' if opt['gpu_ids'] is not None else 'cpu')
        else:
            self.device = torch.device(f'cuda:{opt["gpu_ids"][0]}' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0



    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None and not isinstance(item, list):
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n


    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return list(self.optimizers.values())[0].param_groups[0]["lr"]
    
    def set_requires_grad(self, names, requires_grad):
        for name in names:
            if isinstance(self.networks[name], nn.Module):
                for v in self.networks[name].parameters():
                    v.requires_grad = requires_grad


    def define_network(self, opt):
        nets_opt = opt["networks"]
        defined_network_names = list(nets_opt.keys())

        assert set(defined_network_names).issubset(set(self.network_names))

        for name in defined_network_names:
            setattr(self, name, build_network(nets_opt[name], opt, self.pretrained, name))
            self.networks[name] = self.set_device(getattr(self, name))


    def define_optimizers(self, optim_opts):

        if "default" in optim_opts.keys():
            default_optim = optim_opts.pop("default")

        defined_optimizer_names = list(optim_opts.keys())
        assert set(defined_optimizer_names).issubset(self.networks.keys())

        for name in defined_optimizer_names:
            optim_opt = optim_opts[name].copy()
            if optim_opt is None:
                optim_opt = default_optim.copy()

            params = []
            for v in self.networks[name].parameters():
                if v.requires_grad:
                    params.append(v)

            optim_type = optim_opt.pop("type")
            optimizer = getattr(torch.optim, optim_type)(params=params, **optim_opt)
            self.optimizers[name] = optimizer

    def define_loss(self, loss_opts):
        defined_loss_names = list(loss_opts.keys())
        for name in defined_loss_names:
            loss_opt = loss_opts[name]
            which_loss = loss_opt['which_loss']
            self.loss[name] = LOSS_REGISTRY.get(which_loss)(device = self.device, **loss_opt['setting'])
            
    def set_network_state(self, state):
        for name in self.networks.keys():
            if isinstance(self.networks[name], nn.Module):
                getattr(self.networks[name], state)()


    def set_requires_grad(self, names, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        for name in names:
            if isinstance(self.networks[name], nn.Module):
                for v in self.networks[name].parameters():
                    v.requires_grad = requires_grad
