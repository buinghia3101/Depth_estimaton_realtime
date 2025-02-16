import torch
import torch.nn as nn
from torchmetrics.functional.image import image_gradients
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from critera import MaskedL1Loss,MaskedMSELoss
class SSIM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ssim_loss_weight = 0.3
        self.l1_loss_weight = 0.4
        self.edge_loss_weight = 0.3
        self.l2=MaskedMSELoss()
        self.l1=MaskedL1Loss()
        self.huber=nn.HuberLoss()


    def forward(self,pred,target):
        #Edges
        dy_true,dx_true=image_gradients(target)
        dy_pred,dx_pred=image_gradients(pred)
        weights_x=torch.exp(torch.mean(torch.abs(dx_true)))
        weights_y=torch.exp(torch.mean(torch.abs(dy_true)))
        
        
        #Depth smoothness
        
        smoothness_x=dx_pred*weights_x
        smoothness_y=dy_pred*weights_y
        
        depth_smoothness_loss=torch.mean(abs(smoothness_x)+torch.mean(abs(smoothness_y)))
        ssim_loss=torch.mean((1-ssim(target.float(),pred.float(),data_range=1.0,win_size=7,win_sigma=0.03)))
        #Point-wise depth
        diff=torch.abs(target.float()-pred.float())
        l1_loss=self.l1(target,pred)
        l2_loss=self.l2(target.float(),pred.float())      
        total_loss1=(((self.edge_loss_weight*depth_smoothness_loss))+(self.ssim_loss_weight*ssim_loss)+(self.l1_loss_weight*l1_loss))
        
        total_loss=l1_loss
        return total_loss1