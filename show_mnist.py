import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from model import Net
import matplotlib.cm as cm
import numpy as np

colors = cm.rainbow(np.linspace(0, 1, 10))
print(colors)
######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model/best.pth')
    network.load_state_dict(torch.load(save_path))
    return network


def test(model, test_loader):
    test_loss = 0
    correct = 0
    is_appear = np.zeros(10)
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            output = model(data)
            location = output.data.cpu()
            for i in range(data.size(0)):
                l = target[i].data.numpy()
                if is_appear[l]==0:
                    is_appear[l] = 1
                    ax.scatter( location[i, 0], location[i, 1], c=colors[l], s=10, label = l,
                alpha=0.7, edgecolors='none')
                else:
                    ax.scatter( location[i, 0], location[i, 1], c=colors[l], s=10, 
                alpha=0.7, edgecolors='none')

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=100, shuffle=False)

model = Net()
model = load_network(model)
model.fc2 = nn.Sequential()
model = model.eval()
model = model.cuda()

fig, ax = plt.subplots()
test(model, test_loader)
ax.grid(True)
ax.legend(loc='best')
fig.savefig('train.jpg')

