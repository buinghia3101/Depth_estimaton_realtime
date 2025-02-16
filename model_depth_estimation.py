import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
class Model_depth_estimation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model=mobilenet_v3_large()
        self.feat=model.features[:-1]
        self.conv3=nn.Conv2d(160,256,kernel_size=1)
        # GAP=nn.AvgPool2d(2)
        self.transpose1=nn.ConvTranspose2d(256,3,kernel_size=256,padding=7,stride=2)
        self.norm=nn.BatchNorm2d(3*256*256)
        self.transpose2=nn.ConvTranspose2d(3,1,kernel_size=3,padding=1,stride=1)
    def forward(self,x):
        x=self.feat(x) 
        x=self.transpose1(x)
        x=self.norm(x)
        x=self.transpose2(x)
        return x
        

