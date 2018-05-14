import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

######################################################################
# Draw Curve
#---------------------------
x_epoch = [2,4,8,12,16]

top1 = {}
top5 = {}

for i in range(6):
    top1[i] = []
    top5[i] = []

top1[0] = [93.14, 93.14, 93.14, 93.14, 93.14]
top5[0] = [99.76, 99.76, 99.76, 99.76, 99.76]

with open("./Output.txt", "r") as f:
    for line in f:
        score = line.split('|')
        method_id = int(score[2])
        top1_acc, top5_acc = float(score[3]), float(score[4])
        top1[method_id].append(top1_acc*100)
        top5[method_id].append(top5_acc*100)


fig = plt.figure(figsize=(10,4),dpi=180)
ax0 = fig.add_subplot(121, ylabel="Top-1(%)", xlabel='epsilon')
ax0.plot(x_epoch, top1[0], 'k-', label='Clean')
ax0.plot(x_epoch, top1[1], 'b^-', label='Fast')
ax0.plot(x_epoch, top1[2], 'rs-', label='Basic')
ax0.plot(x_epoch, top1[3], 'gv-', label='Least-likely')
#ax0.plot(x_epoch, top1[4], 'mo-', label='Label-smooth')
ax0.plot(x_epoch, top1[5], 'yo-', label='Our')
ax0.grid(True)
ax0.legend()
plt.ylim(0,100)
plt.xlim(1,17)

ax0 = fig.add_subplot(122, ylabel="Top-5(%)", xlabel='epsilon')
ax0.plot(x_epoch, top5[0], 'k-', label='Clean')
ax0.plot(x_epoch, top5[1], 'b^-', label='Fast')
ax0.plot(x_epoch, top5[2], 'rs-', label='Basic')
ax0.plot(x_epoch, top5[3], 'gv-', label='Least-likely')
#ax0.plot(x_epoch, top5[4], 'mo-', label='Label-smooth')
ax0.plot(x_epoch, top5[5], 'yo-', label='Our')
ax0.grid(True)
ax0.legend()
plt.ylim(0,100)
plt.xlim(1,17)

fig.savefig( 'Cifar.jpg')
