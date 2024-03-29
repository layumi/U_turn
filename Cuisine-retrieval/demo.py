import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from shutil import copyfile
from PIL import Image
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=300, type=int, help='test_image_index')
parser.add_argument('--method', default=5, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='./Food-cropped/pytorch',type=str, help='./test_data')
parser.add_argument('--adv',action='store_true', help='./test_data')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')

if opts.adv:
    result = scipy.io.loadmat('attack_query/ft_ResNet50_all-%d/16/query.mat'%opts.method)

query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]

result = scipy.io.loadmat('pytorch_result.mat')
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, gf, gl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.argwhere(gl==-1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

i = opts.query_index
adv = ''
if opts.adv:
    adv = 'a'

if not os.path.isdir(str(opts.query_index)+adv+str(opts.method)):
    os.mkdir(str(opts.query_index)+adv+str(opts.method))
index = sort_img(query_feature[i],query_label[i],gallery_feature,gallery_label)

########################################################################
# Visualize the rank result

query_path, _ = image_datasets['query'].imgs[i]
query_label = query_label[i]
if opts.adv:
    query_path = query_path.replace('./Food-cropped/pytorch/query','attack_query/ft_ResNet50_all-%d/16/'%opts.method)
print(query_path)
print('Top 10 images are as follow:')
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(12,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path)
    query256 = Image.open(query_path)
    query256 = query256.resize((256,256))
    query256.save('./%d%s%d/query.jpg'%(opts.query_index,adv,opts.method) )
    #copyfile(query_path, './%d%s/query.jpg'%(opts.query_index,adv) )
    #imshow(query_path,'query')
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i+1]]
        label = gallery_label[index[i+1]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        print(img_path)
        img256 = Image.open(img_path)
        img256 = img256.resize((256,256))
        #copyfile(img_path, './%d%s/%d.jpg'%(opts.query_index,adv,i) )
        img256.save('./%d%s%d/%d.jpg'%(opts.query_index,adv,opts.method,i) )
    fig.savefig('result.jpg')
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i+1]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

