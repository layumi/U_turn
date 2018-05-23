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
from shutil import copyfile

opt=2

if opt==1:
    data_dir = '/home/zzd/Market/pytorch'
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x)) for x in ['gallery','query']}

    query_path = image_datasets['query'].imgs
    gallery_path = image_datasets['gallery'].imgs

    save_path = './original'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    with open("../original_list.txt","r") as rank:
        for i in range(3368):
            result = rank.readline()
            rank15 = result.split(',')
            query = query_path[i]
            os.mkdir(save_path + '/%d'%i)
            copyfile(query[0], save_path + '/%d/'%i + 'query.jpg')
            for j in range(15):
                img_name = gallery_path[int(rank15[j])]
                copyfile(img_name[0], save_path + '/%d/'%i + '%d.jpg'%j)


#############################
# adv
if opt==2:
    query_path = datasets.ImageFolder('../attack_query/pytorch/query').imgs
    gallery_path = datasets.ImageFolder('/home/zzd/Market/pytorch/gallery').imgs

    save_path = './adv'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    with open("../adv_list.txt","r") as rank:
        for i in range(3368):
            result = rank.readline()
            rank15 = result.split(',')
            query = query_path[i]
            os.mkdir(save_path + '/%d'%i)
            copyfile(query[0], save_path + '/%d/'%i + 'query.jpg')
            for j in range(15):
                img_name = gallery_path[int(rank15[j])]
                img_name = img_name[0]
                #print(img_name[38:])
                copyfile(img_name, save_path + '/%d/'%i + '%d.jpg'%j)
                #copyfile(img_name, save_path + '/%d/'%i + img_name[38:])



