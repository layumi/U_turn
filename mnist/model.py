import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        #init.constant(m.bias.data, 0.0)

# LeNet, I follow the practice in 
# https://github.com/vlfeat/matconvnet/blob/34742d978151d8809f7ab2d18bb48db23fb9bb47/examples/mnist/cnn_mnist_init.m
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv1.apply(weights_init_kaiming)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.conv2.apply(weights_init_kaiming)
        self.conv3 = nn.Conv2d(50, 500, kernel_size=4)
        self.conv3.apply(weights_init_kaiming)

        self.fc1 = nn.Linear(500, 2)
        self.fc2 = nn.Linear(2, 10, bias=False)
        self.fc2.apply(weights_init_classifier)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 500)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
