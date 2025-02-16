import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from sklearn.model_selection import train_test_split
import numpy as np
sample_tfms = [
    A.HorizontalFlip(),
    A.GaussNoise(p=0.2),
    A.OneOf([
        A.MotionBlur(p=.3),
        A.MedianBlur(blur_limit=3, p=0.3),
        A.Blur(blur_limit=3, p=0.5),
    ], p=0.3),
    A.RGBShift(),
    A.RandomBrightnessContrast(),
    A.RandomResizedCrop(128,156),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
]
train_tfms = A.Compose([
    *sample_tfms,
    A.Resize(128,256),
    A.Normalize(always_apply=True),
    ToTensorV2()
])
valid_tfms = A.Compose([
    A.Resize(128,256),
    A.Normalize(always_apply=True),
    ToTensorV2()
])
class DepthEstimationDataset(Dataset):
    def __init__(self, root_dir,tfms):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'image')
        self.depth_folder = os.path.join(root_dir, 'depth')
        self.image_filenames = os.listdir(self.image_folder)
        self.tfms=tfms
        
    def open_im(self,p):
        im = np.load(p)
        return im

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        depth_name = os.path.join(self.depth_folder, self.image_filenames[idx])

        im, dp = self.open_im(img_name).astype(np.uint8), (np.squeeze(self.open_im(depth_name),2)*255).astype(np.uint8)
        augs = self.tfms(image=im,mask=dp)
        sample={"rgb":None,"depth":None}
        im, dp = augs['image'], augs['mask']/255
        sample["rgb"]=im
        sample["depth"]=dp.unsqueeze(0)

        return sample

# Example usage:
train_root_directory = 'CitySpaces/data/train'
val_root_directory = 'CitySpaces/data/val'

train_dataset = DepthEstimationDataset(train_root_directory,tfms=train_tfms)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


val_dataset = DepthEstimationDataset(val_root_directory,tfms=valid_tfms)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
