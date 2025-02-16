import  os
import pandas as pd
import torch
from torchvision.io import read_image
import torchvision
from scipy.interpolate import NearestNDInterpolator
from matplotlib import pyplot as plt
import numpy as np
import os
train_labels = []
val_labels = []
for dirname, _, filenames in os.walk('/kaggle/input/cityscapes-depth-and-segmentation/data/train'):
    for filename in filenames:
        train_labels.append(filename)
for dirname, _, filenames in os.walk('/kaggle/input/cityscapes-depth-and-segmentation/data/val'):
    for filename in filenames:
        val_labels.append(filename)
class DataSet(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, all_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.all_transform = all_transform
        
        self.kernel=torch.ones(9)
        self.kernel=self.kernel.view(3,3)
        self.kernel[1][1] = 3
        self.kernel /= self.kernel.sum()
        self.kernel = self.kernel.view(1,1,3,3)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        


        transform = torchvision.transforms.CenterCrop((228,304))
        pad = torchvision.transforms.Pad((23,14,23,14),padding_mode="edge")
        image_x =  torch.Tensor(np.load(os.path.join(self.img_dir,"image", self.img_labels[idx])) )
        image_y_depth =  torch.Tensor(np.load(os.path.join(self.img_dir,"depth", self.img_labels[idx])))
#         image_y_depth =  transform(image_y_depth.view(1,128,256))
#         image_y_depth= torch.stack([pad(torch.nn.functional.conv2d( image_y_depth.view(1,100,210), torch.nn.Parameter(self.kernel), padding='same').detach().view(100,210))]).permute(1,2,0)
            
#         image_y_depth = (np.load(os.path.join(self.img_dir,"depth", self.img_labels[idx]))*255).astype("uint8")
        image_y_depth = (image_y_depth.numpy()*255).astype("uint8")
        data = image_y_depth
        mask = np.where(~(data <=0.4))
        interp = NearestNDInterpolator(np.transpose(mask), data[mask])
        image_result = interp(*np.indices(data.shape))
        image_y_depth = (torch.Tensor(image_result)/255)
        image_y_depth = pad(transform((image_y_depth.view(1,128,256)))).permute(1,2,0)
        image_y_depth = image_y_depth / image_y_depth.max()
        
        image_y_label =  torch.Tensor(np.load(os.path.join(self.img_dir,"label", self.img_labels[idx])) )
        
        if self.transform:
            tr_x = self.transform(image=image_x.numpy())
            image_x = torch.from_numpy(tr_x["image"])
        if self.all_transform:
            tr_y = self.all_transform(image=image_x.numpy(),masks=[image_y_depth.numpy(),image_y_label.numpy()])
            image_x,image_y_depth,image_y_label = torch.from_numpy(tr_y["image"]),torch.from_numpy(tr_y["masks"][0]),torch.from_numpy(tr_y["masks"][1])
        return image_x.permute(2,0,1),image_y_depth.permute(2,0,1)