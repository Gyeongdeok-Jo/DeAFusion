import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.dice import DiceLoss

from .msssi import ssim
from utils.registry import LOSS_REGISTRY
from utils.PIPD import PIPD


@LOSS_REGISTRY.register()
class Diceloss(DiceLoss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False, 
        *args, 
        **kwargs
    ) -> None:
        # 부모 클래스의 초기화 메서드 호출
        super(Diceloss, self).__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        # 부모 클래스의 forward 메서드 호출
        return super(Diceloss, self).forward(input, target)



class Sobelxy(nn.Module):
    def __init__(self, device, grayscale):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.grayscale = grayscale
        if self.grayscale:
            self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
            self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
        else:
            self.weightx = nn.Parameter(data=kernelx.expand(3, 1, 3, 3), requires_grad=False).to(device)
            self.weighty = nn.Parameter(data=kernely.expand(3, 1, 3, 3), requires_grad=False).to(device)

    def forward(self, x):
        # 각 채널에 대해 필터를 적용합니다.
        if self.grayscale:
            sobelx = F.conv2d(x, self.weightx, padding=1)
            sobely = F.conv2d(x, self.weighty, padding=1)
        else:
            sobelx = F.conv2d(x, self.weightx, padding=1, groups=3)
            sobely = F.conv2d(x, self.weighty, padding=1, groups=3)
        return torch.abs(sobelx) + torch.abs(sobely)
    

@LOSS_REGISTRY.register()
class Fusion_loss(nn.Module):
    def __init__(self, device, lamb1=1, lamb2=1, lamb3=1, c = 0.1, grayscale = False, *args, **kwargs):
        super(Fusion_loss, self).__init__()
        self.loss_mse = nn.MSELoss(reduction = 'mean').to(device)
        self.sobelconv=Sobelxy(device = device, grayscale = grayscale)
        self.PIPD = PIPD(c = c, device = device)
        self.device = device
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lamb3 = lamb3

    def forward(self, fused_img, src1, src2, **kwargs):
        weight1, _ = self.PIPD(src1)
        weight2, _ = self.PIPD(src2)
        PIPD_map = torch.cat((weight1, weight2), dim = 1)
        PIPD_map = F.softmax(PIPD_map, dim=1)
        
        w1 = PIPD_map[:, :1,:,:]
        w2 = PIPD_map[:, 1:,:,:]


        s1_grad = self.sobelconv(src1)
        s2_grad = self.sobelconv(src2)
        # x_grad_joint = torch.max(s1_grad, s2_grad)
        fused_img_grad = self.sobelconv(fused_img)

        loss_1 = torch.mean(w1 * (1 - ssim(fused_img, src1, map_return = True))) \
                    + torch.mean(w2 * (1 - ssim(fused_img, src2, map_return = True)))
        loss_1 = self.lamb1*torch.mean(loss_1)

        loss_2 = self.loss_mse(fused_img*torch.sqrt(w1), src1*torch.sqrt(w1)) \
                    + self.loss_mse(fused_img*torch.sqrt(w2), src2*torch.sqrt(w2))
        loss_2 = self.lamb2*torch.mean(loss_2)
        loss_3 = self.loss_mse(fused_img_grad*torch.sqrt(w1), s1_grad*torch.sqrt(w1)) \
                   + self.loss_mse(fused_img_grad*torch.sqrt(w2), s2_grad*torch.sqrt(w2))
        loss_3 = self.lamb3*torch.mean(loss_3)

        loss_total = loss_1 + loss_2 + loss_3
        return loss_total, loss_1.item(), loss_2.item(), loss_3.item()

