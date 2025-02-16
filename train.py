
"""
Created on Fri Feb  2 19:16:42 2018
@author: norbot
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse

from torch.autograd import Variable

import utils
import lr_scheduler as lrs
from tqdm import tqdm
from SSIM_loss import SSIM
from diode_loader import DIODE
from Unet_pytorch import UNet
from segmentation.data_loader.segmentation_dataset import SegmentationDataset
from segmentation.data_loader.transform import Rescale, ToTensor
from segmentation.trainer import Trainer
from segmentation.predict import *
from segmentation.models import all_models
from util.logger import Logger
from dataloader_v2 import *
from model_dense import Model
from Resnet_depth import ResNet
from critera import MaskedMSELoss,berHuLoss
from CitySpaces_dataloader import *
from Depth_net import Depth
parser = argparse.ArgumentParser(description='PyTorch Sparse To Dense Training')

# net parameters
parser.add_argument('--n_sample', default=200, type=int, help='sampled sparse point number')
parser.add_argument('--data_set', default='nyudepth', type=str, help='train dataset')

# optimizer parameters
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (L2 penalty)')
parser.add_argument('--dampening', default=0.0, type=float, help='dampening for momentum')
parser.add_argument('--nesterov', '-n', action='store_true', help='enables Nesterov momentum')
parser.add_argument('--num_epoch', default=60, type=int, help='number of epoch for training')

# network parameters
parser.add_argument('--cspn_step', default=24, type=int, help='steps of propagation')
parser.add_argument('--cspn_norm_type', default='8sum', type=str, help='norm type of cspn')

# batch size
parser.add_argument('--batch_size_train', default=16, type=int, help='batch size for training')
parser.add_argument('--batch_size_eval', default=1, type=int, help='batch size for eval')

#data directory
parser.add_argument('--save_dir', default='result/base_line', type=str, help='result save directory')
parser.add_argument('--best_model_dir', default='result/base_line', type=str, help='best model load directory')
parser.add_argument('--train_list', default='nyu_data/data/nyu2_train.csv', type=str, help='train data lists')
parser.add_argument('--eval_list', default='nyu_data/data/nyu2_test.csv', type=str, help='eval data list')
parser.add_argument('--model', default='base_model', type=str, help='model for net')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--pretrain', '-p', action='store_true', help='load pretrained resnet model')

args = parser.parse_args()

sys.path.append("./models")


use_cuda = torch.cuda.is_available()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:",device)

# global variable
best_rmse = sys.maxsize  # best test rmse
cspn_config = {'step': args.cspn_step, 'norm_type': args.cspn_norm_type}
start_epoch = 0 # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
# import nyu_dataset_loader as dataset_loader
# trainset = dataset_loader.NyuDepthDataset(csv_file=args.train_list,
#                                             root_dir='nyu_data',
#                                             split = 'train',
#                                             n_sample = args.n_sample,
#                                             input_format='img')
# valset = dataset_loader.NyuDepthDataset(csv_file=args.eval_list,
#                                         root_dir='nyu_data',
#                                         split = 'val',
#                                         n_sample = args.n_sample,
#                                         input_format='img')

# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=args.batch_size_train,
#                                           shuffle=True,
#                                           num_workers=2,
#                                           pin_memory=True,
#                                           drop_last=True)
# valloader = torch.utils.data.DataLoader(valset,
#                                         batch_size=args.batch_size_eval,
#                                         shuffle=False,
#                                         num_workers=2,
#                                         pin_memory=True,
#                                         drop_last=True)
# DEFINE MODEL 
model_name = "fcn32_resnet34"
device = 'cuda'
batch_size = 16
n_classes = 1
num_epochs = 10
image_axis_minimum_size = 200
pretrained = True
fixed_feature = False
### Model
# net = all_models.model_from_name[model_name](n_classes, batch_size,
#                                                 pretrained=pretrained,
#                                                 fixed_feature=fixed_feature)

net=Depth(decoder="upconv",output_size=(128,256))
if use_cuda:
    net.to(device=device)
    assert torch.cuda.device_count() == 1, 'only support single gpu'
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


criterion =berHuLoss().to(device)
# optimizer = optim.SGD(net.parameters(),
#                     #   lr=args.lr,
#                     #   momentum=args.momentum,
#                     #   weight_decay=args.weight_decay,
#                     #   nesterov=args.nesterov,
#                     #   dampening=args.dampening)
optimizer=torch.optim.Adam(net.parameters(),lr=args.lr,amsgrad=True)

scheduler = lrs.ReduceLROnPlateau(optimizer, 'min') # set up scheduler


# Training
def train(epoch):
    net.train()
    total_step_train = 0
    train_loss = 0.0
    error_sum_train = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                       'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                       'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,}

    tbar = tqdm(train_dataloader)
    for batch_idx, sample in enumerate(tbar):
        [inputs, targets] = [sample['rgb'] , sample['depth']]
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        # print(targets.shape)
        # print(outputs.shape)
        # break
        loss= criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        error_str = 'Epoch: %d, loss=%.4f' % (epoch, train_loss / (batch_idx + 1))
        tbar.set_description(error_str)

        targets = targets.data.cpu()
        outputs = outputs.data.cpu()
        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        total_step_train += args.batch_size_train
        error_avg = utils.avg_error(error_sum_train,
                                    error_result,
                                    total_step_train,
                                    args.batch_size_train)
        if batch_idx % 500 == 0:
            utils.print_error('training_result: step(average)',
                              epoch,
                              batch_idx,
                              loss,
                              error_result,
                              error_avg,
                              print_out=True)

    error_avg = utils.avg_error(error_sum_train,
                                error_result,
                                total_step_train,
                                args.batch_size_train)
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
    utils.log_result_lr(args.save_dir, error_avg, epoch, old_lr, False, 'train')

    tmp_name = "epoch_%02d.pth" % (epoch)
    save_name = os.path.join(args.save_dir, tmp_name)
    # torch.save(net.state_dict(), save_name)


def val(epoch):
    global best_rmse
    is_best_model = False
    net.eval()
    total_step_val = 0
    eval_loss = 0.0
    error_sum_val = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                     'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                     'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,}

    tbar = tqdm(val_dataloader)
    for batch_idx, sample in enumerate(tbar):
        [inputs, targets] = [sample['rgb'] , sample['depth']]
        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
        loss= criterion(outputs, targets)

        targets = targets.data.cpu()
        outputs = outputs.data.cpu()
        
        loss = loss.data.cpu()
        eval_loss += loss.item()
        error_str = 'Epoch: %d, loss=%.4f' % (epoch, eval_loss / (batch_idx + 1))
        tbar.set_description(error_str)

        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        total_step_val += args.batch_size_eval
        error_avg = utils.avg_error(error_sum_val, error_result, total_step_val, args.batch_size_eval)

    utils.print_error('eval_result: step(average)',
                      epoch, batch_idx, loss,
                      error_result, error_avg, print_out=True)

    #log best_model
    if utils.updata_best_model(error_avg, best_rmse):
        is_best_model = True
        best_rmse = error_avg['RMSE']
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
    utils.log_result_lr(args.save_dir, error_avg, epoch, old_lr, is_best_model, 'eval')

    # saving best_model
    if is_best_model:
        print('==> saving best model at epoch %d' % epoch)
        best_model_pytorch = os.path.join(args.save_dir, 'best_model.pth')
        torch.save(net.state_dict(), best_model_pytorch)

    #updata lr
    scheduler.step(error_avg['MAE'], epoch)


def train_val():
    for epoch in range(0, args.num_epoch):
        train(epoch)
        val(epoch)

if __name__ == '__main__':
    train_val()