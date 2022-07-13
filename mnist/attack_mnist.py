import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from model import Net
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from model import Net
import matplotlib.cm as cm
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--method_id', default=5, type=int, help='1.fast || 2.least likely || 3.label smooth')
parser.add_argument('--rate', default=2, type=int, help='attack rate')

opt = parser.parse_args()

colors = cm.rainbow(np.linspace(0, 1, 10))

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model/best.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def get_feature(model, x):
    x = F.max_pool2d(model.conv1(x), 2)
    x = F.max_pool2d(model.conv2(x), 2)
    x = F.relu(model.conv3(x))
    x = x.view(-1, 500)
    x = model.fc1(x)
    return x

def attack(model,dataloaders, method_id=5):
    count = 0.0
    score = 0.0
    score5 = 0.0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        inputs = Variable(img.cuda(), requires_grad = True)
        outputs = model(inputs) 
        #-----------------attack-------------------
        # The input has been whiten.
        # So when we recover, we need to use a alpha
        alpha = 1.0 / (0.3081 * 255.0)
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
            #fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            #outputs = outputs.div(fnorm.expand_as(outputs))
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
                outputs = get_feature(model, inputs)
                #fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
                #outputs = outputs.div(fnorm.expand_as(outputs))
            #print( torch.sum(outputs*target))
        else:
            print('unknow method id')

        outputs = model(inputs)
        location = get_feature(model, inputs)
        test(location, label)
        outputs = outputs.data.cpu()
        _, preds = outputs.topk(5, dim=1)
        correct = preds.eq(label.view(n,1).expand_as(preds))
        score += torch.sum(correct[:,0])
        score5 += torch.sum(correct)
    print( '%f | %f'%(score, score5))
    return score


is_appear = np.zeros(10)
def test(location, label):  #location and label
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

data_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download = True, transform=data_transform),
        batch_size=100, shuffle=False)

model = Net()
model = load_network(model)
model = model.eval()
model = model.cuda()

#######################################################################
# Creat Up bound and low bound
# Clip 
zeros = np.zeros((28,28),dtype=np.uint8)
zeros = Image.fromarray(zeros) 
zeros = data_transform(zeros)

ones = 255*np.ones((28,28), dtype=np.uint8)
ones = Image.fromarray(ones)
ones = data_transform(ones)

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

#######################################################################
# Main
#test(model, test_loader)
fig, ax = plt.subplots()
score = attack(model, test_loader)
ax.grid(True)
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.legend(loc='best')
fig.savefig('train%d-%d.jpg'%(opt.rate, score))
