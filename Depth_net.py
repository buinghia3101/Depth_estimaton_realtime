from Resnet_depth import *

import timm
class Depth(nn.Module):
    def __init__(self, decoder, output_size, in_channels=3, pretrained=True) -> None:
        super(Depth,self).__init__()
        self.model_s=timm.create_model('lcnet_050.ra2_in1k', pretrained=pretrained)
        selected_layers = list(self.model_s.children())[:2]

        self.feat = torch.nn.Sequential(*selected_layers)

        
        num_channels=512
        self.output_size = output_size

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)
    def forward(self,x):
        x=self.feat(x)
        
        x=self.conv2(x)
        x=self.bn2(x)
        
        # decoder
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)
        return x

