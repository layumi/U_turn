# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import torch.utils.data as data
import models.cifar as models
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/zzd/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='resnet20', type=str, help='save model path')
parser.add_argument('--batchsize', default=100, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--depth', default=20, type=int, help='model depth')

###########################################
# python test.py --name resnet20 --depth 20
##########################################

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataloader = datasets.CIFAR10
testset = dataloader(root='./data', train=False, download=False, transform=data_transforms)
testloader = data.DataLoader(testset, batch_size=opt.batchsize, shuffle=False, num_workers=8)

class_names =  ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./checkpoint',name,'model_best.pth.tar')
    checkpoint = torch.load(save_path)
    network.load_state_dict(checkpoint['state_dict'])
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#

def evaluate(model,dataloaders):
    count = 0
    score = 0.0
    score5 = 0.0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        input_img = Variable(img.cuda(), volatile=True)
        outputs = model(input_img) 
        outputs = outputs.data.cpu()
        _, preds = outputs.topk(5, dim=1)
        correct = preds.eq(label.view(n,1).expand_as(preds))
        score += torch.sum(correct[:,0])
        score5 += torch.sum(correct)
    print("top1: %.4f top5:%.4f"% (score/count, score5/count))
    return

######################################################################
# Load Collected data Trained model
#print('-------test-----------')

model_structure = models.__dict__['resnet'](
                    num_classes=10,
                    depth=opt.depth,
                )

model = torch.nn.DataParallel(model_structure).cuda()
model = load_network(model)

# Change to test mode
model = model.eval()

# test
evaluate(model,testloader)

