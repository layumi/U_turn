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
from model import ft_net, ft_net_dense
from PIL import Image

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/zzd/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--method_id', default=5, type=int, help='1.fast || 2.least likely || 3.label smooth')
parser.add_argument('--rate', default=16, type=int, help='attack rate')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','train']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=4) for x in ['gallery','query','train']}

class_names = image_datasets['train'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# recover image
# -----------------
def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp

######################################################################
# Generate attack
# ----------------------
#
# Generate a attack from  a trained model.
#
def generate_attack(model,dataloaders, method_id):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data # Note that this is the label in the testing set (different with the training set)
        n, c, h, w = img.size()
        inputs = Variable(img.cuda(), requires_grad=True)
        if method_id != 5:
            outputs = model(inputs)
        # ---------------------attack------------------
        # The input has been whiten.
        # So when we recover, we need to use a alpha
        alpha = 1.0 / (0.226 * 255.0)
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
            smooth_label = torch.ones(batch_size, 751) /751.0
            target = Variable(smooth_label.cuda(), requires_grad=False)
            criterion2 = nn.MSELoss()
            sm = nn.Softmax(dim = 1) #softmax work on the second dim (sum of the 751 elements = 1)
            for iter in range( round(min(1.25 * opt.rate, opt.rate+4))):
                sm_outputs = sm(outputs)
                loss2 = criterion2(sm_outputs, target)
                loss2.backward()
                prob,_ = torch.max(sm_outputs,1)
                print('iter:%d smooth-loss:%4f max-pre:%4f'%(iter, loss2.data[0],torch.mean(prob)))
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
            #L2norm = nn.InstanceNorm1d(2048, affine=False)
            model.model.fc = nn.Sequential() #nn.Sequential(*L2norm)
            model.classifier = nn.Sequential()
            outputs = model(inputs)
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
                outputs = model(inputs)
                fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
                outputs = outputs.div(fnorm.expand_as(outputs))
            #print( torch.sum(outputs*target))
        else:
            print('unknow method id')
 
        #print(torch.mean(diff.abs()))
        #Save attack images
        attack = inputs.data.cpu()
        for j in range(inputs.shape[0]):
            im = recover(attack[j,:,:,:])
            im_path = train_path[count+j][0].split('/')[-1]
            subdir_path = train_path[count+j][0].split('/')[-2]
            subdir_path = os.path.join( opt.test_dir, 'train_adv', subdir_path)
            if not os.path.isdir(subdir_path):
                os.mkdir(subdir_path)
            im = Image.fromarray(im.astype('uint8'))
            im.save( subdir_path+ '/'+ im_path + '%d-'%opt.method_id + '%d'%opt.rate +'.jpg'  )
        count += n
        #print(count)


train_path = image_datasets['train'].imgs
query_path = image_datasets['query'].imgs

######################################################################
# Load Collected data Trained model
#print('-------generate-----------')
if opt.use_dense:
    model_structure = ft_net_dense(751)
else:
    model_structure = ft_net(751)
model = load_network(model_structure)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

train_name = os.path.join(opt.test_dir, 'train')
dir_name = os.path.join( opt.test_dir, 'train_adv')
if not os.path.isdir(dir_name):
    os.system('cp -r %s %s'%(train_name, dir_name) )

#######################################################################
# Creat Up bound and low bound
# Clip 
zeros = np.ones((256,128,3),dtype=np.uint8)
zeros = Image.fromarray(zeros) 
zeros = data_transforms(zeros)

ones = 255*np.ones((256,128,3), dtype=np.uint8)
ones = Image.fromarray(ones)
ones = data_transforms(ones)

zeros,ones = zeros.cuda(),ones.cuda()

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

##########################################################################
# Generate Attack Samples
generate_attack(model,dataloaders['train'],opt.method_id)

