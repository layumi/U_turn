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
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, 10))
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/zzd/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='.', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--depth', default=20, type=int, help='model depth')
parser.add_argument('--method_id', default=5, type=int, help='1.fast || 2.least likely || 3.label smooth')
parser.add_argument('--rate', default=2, type=int, help='attack rate')
###########################################
# python test.py --name resnet20 --depth 20
##########################################

opt = parser.parse_args()

#gpu_ids = opt.gpu_ids.split(',')
#torch.cuda.set_device(0)
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



def attack(model,dataloaders, method_id):
    count = 0
    score = 0
    score5 = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        inputs = Variable(img.cuda(), requires_grad = True)
        outputs = model(inputs) 
        #-----------------attack-------------------
        # The input has been whiten.
        # So when we recover, we need to use a alpha
        alpha = 1.0 / (0.2 * 255.0)
        criterion = nn.CrossEntropyLoss()
        inputs_copy = Variable(inputs.data, requires_grad = False)
        diff = torch.FloatTensor(inputs.shape).zero_()
        diff = Variable(diff.cuda(), requires_grad = False)
        #1. FGSM, GradientSignAttack
        if method_id == 1:
            _, preds = torch.max(outputs.data, 1)
            labels = Variable(preds.cuda())
            loss = criterion(outputs, labels)
            loss.backward()
            inputs = inputs + torch.sign(inputs.grad) * opt.rate * alpha
            inputs = clip(inputs,n)
        #2. IterativeGradientSignAttack
        elif method_id == 2:
            _, preds = torch.max(outputs.data, 1)
            labels = Variable(preds.cuda())
            for iter in range( round(min(1.25 * opt.rate, opt.rate+4))):
                loss = criterion(outputs, labels)
                loss.backward()
                diff += torch.sign(inputs.grad)
                mask_diff = diff.abs() > opt.rate
                diff[mask_diff] = opt.rate * torch.sign(diff[mask_diff])
                inputs = inputs_copy + diff * 1.0  * alpha # we use 1 instead of opt.rate
                inputs = clip(inputs,n)
                inputs = Variable(inputs.data, requires_grad=True)
                outputs = model(inputs)
        #3. Iterative Least-likely method
        elif method_id == 3:
            # least likely label is fixed
            _, ll_preds = torch.min(outputs.data, 1)
            ll_label = Variable(ll_preds, requires_grad=False)
            for iter in range( round(min(1.25 * opt.rate, opt.rate+4))):
                loss = criterion(outputs, ll_label)
                loss.backward()
                diff += torch.sign(inputs.grad)
                mask_diff = diff.abs() > opt.rate
                diff[mask_diff] = opt.rate * torch.sign(diff[mask_diff])
                inputs = inputs_copy - diff * 1.0 * alpha # we use 1 instead of opt.rate
                inputs = clip(inputs,n)
                inputs = Variable(inputs.data, requires_grad=True)
                outputs = model(inputs)
        #4. Label-smooth method
        elif method_id == 4:
            batch_size = inputs.shape[0]
            smooth_label = torch.ones(batch_size, 10) /10.0
            target = Variable(smooth_label.cuda(), requires_grad=False)
            criterion2 = nn.MSELoss()
            sm = nn.Softmax(dim = 1) #softmax work on the second dim (sum of the 751 elements = 1)
            for iter in range( round(min(1.25 * opt.rate, opt.rate+4))):
                sm_outputs = sm(outputs)
                loss2 = criterion2(sm_outputs, target)
                loss2.backward()
                prob,_ = torch.max(sm_outputs,1)
                #print('iter:%d smooth-loss:%4f max-pre:%4f'%(iter, loss2.data[0],torch.mean(prob)))
                diff += torch.sign(inputs.grad)
                mask_diff = diff.abs() > opt.rate
                diff[mask_diff] = opt.rate * torch.sign(diff[mask_diff])
                inputs = inputs_copy - diff * 1.0 * alpha 
                inputs = clip(inputs,n)
                inputs = Variable(inputs.data, requires_grad=True)
                outputs = model(inputs)
        #5. MSE on feature
        elif method_id == 5:
            #remove classifier
            outputs = get_feature(model,inputs)
            #model.module.fc = nn.Sequential()
            #outputs = model(inputs)
            #print(outputs.shape)
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            outputs = outputs.div(fnorm.expand_as(outputs))
            feature_dim = outputs.shape[1]
            batch_size = inputs.shape[0]
            #zero_feature = torch.zeros(batch_size,feature_dim)
            target = Variable(-outputs.data, requires_grad=False)
            criterion2 = nn.MSELoss()
            #s = target*target
            #print(torch.sum(s))
            for iter in range( round(min(1.25 * opt.rate, opt.rate+4))):
                loss2 = criterion2(outputs, target)
                loss2.backward()
                diff += torch.sign(inputs.grad)
                mask_diff = diff.abs() > opt.rate
                diff[mask_diff] = opt.rate * torch.sign(diff[mask_diff])
                inputs = inputs_copy - diff * 1.0 * alpha
                inputs = clip(inputs,n)
                inputs = Variable(inputs.data, requires_grad=True)
                #outputs = model(inputs)
                outputs = get_feature(model,inputs)
                fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
                outputs = outputs.div(fnorm.expand_as(outputs))
            #print( torch.sum(outputs*target))
        else:
            print('unknow method id')

        #reload model
        
        outputs = model(inputs)
        location = get_feature(model, inputs)
        draw(location, label)
        outputs = outputs.data.cpu()
        #print(outputs.shape)
        _, preds = outputs.topk(5, dim=1)
        correct = preds.eq(label.view(n,1).expand_as(preds))
        score += float(torch.sum(correct[:,0]))
        score5 += float(torch.sum(correct))
    print( '%.4f | %.4f'%(score/count, score5/count))
    return score, score5

######################################################################
# Load Collected data Trained model
#print('-------test-----------')

# wrn or resnet
model_structure = models.__dict__['resnet'](
                    num_classes=10,
                    depth=opt.depth,
                    #widen_factor=10,
                )


model_wrn = torch.nn.DataParallel(model_structure).cuda(device=0)
model = load_network(model_wrn)
# Change to test mode
model = model.eval()
#print(model)
#######################################################################
# Creat Up bound and low bound
# Clip 
zeros = np.zeros((32,32,3),dtype=np.uint8)
zeros = Image.fromarray(zeros) 
zeros = data_transforms(zeros)

ones = 255*np.ones((32,32,3), dtype=np.uint8)
ones = Image.fromarray(ones)
ones = data_transforms(ones)

zeros,ones = zeros.cuda(),ones.cuda()

is_appear = np.zeros(10)
def draw(location, label):  #location and label
    location = location.data.cpu()
    label = label.data.cpu().numpy()
    for i in range(location.size(0)):
        l =  label[i]
        if is_appear[l]==0:
            is_appear[l] = 1
            ax.scatter( location[i, 0], location[i, 1], c=colors[l], s=10, label = l,
             alpha=0.7, edgecolors='none')
        else:
            ax.scatter( location[i, 0], location[i, 1], c=colors[l], s=10, 
            alpha=0.7, edgecolors='none')
    return

def get_feature(model,x):
        x = model.module.conv1(x)
        x = model.module.bn1(x)
        x = model.module.relu(x)    # 32x32

        x = model.module.layer1(x)  # 32x32
        x = model.module.layer2(x)  # 16x16
        x = model.module.layer3(x)  # 8x8

        x = model.module.avgpool(x)
        x = x.view(x.size(0), -1)
        x = model.module.fc(x)

        return x

def clip(inputs, batch_size):
    inputs = inputs.data
    for i in range(batch_size):
        inputs[i] = clip_single(inputs[i])
    inputs = Variable(inputs.cuda())
    return inputs

def clip_single(input):       
    low_mask = input<zeros
    up_mask = input>ones
    input[low_mask] = zeros[low_mask]
    input[up_mask] = ones[up_mask]
    return input

#################################################################
# Attack
fig, ax = plt.subplots()
score,score5 = attack(model,testloader,opt.method_id)

#######################################################################
# Draw
#test(model, test_loader)
ax.grid(True)
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.legend(loc='best')
ax.set_title('epsilon=%d, Top-1=%.2f%%, Top-5=%.2f%%'%(opt.rate, score/100, score5/100))
fig.savefig('train%d-%.4f-%.4f.jpg'%(opt.rate, score, score5))
