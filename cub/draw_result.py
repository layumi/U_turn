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
top10 = {}
mAP = {}

for i in range(6):
    top1[i] = []
    top10[i] = []
    mAP[i] = []

top1[0] = [44.53] *5
top10[0] = [82.24] *5
mAP[0] = [19.51] *5

with open("./Output.txt", "r") as f:
    for line in f:
        score = line.split('|')
        method_id = int(score[2])
        top1_acc, top10_acc, mAP_acc = float(score[3]), float(score[5]), float(score[6])
        top1[method_id].append(top1_acc)
        top10[method_id].append(top10_acc)
        mAP[method_id].append(mAP_acc)

fig = plt.figure(figsize=(15,4), dpi=90)
ax0 = fig.add_subplot(131, ylabel="Rank-1(%)",xlabel='epsilon')
ax0.plot(x_epoch, top1[0], 'k-', label='Clean')
ax0.plot(x_epoch, top1[1], 'b^-', label='Fast')
ax0.plot(x_epoch, top1[2], 'rs-', label='Basic')
ax0.plot(x_epoch, top1[3], 'gv-', label='Least-likely')
ax0.plot(x_epoch, top1[5], 'yo-', label='Our')
ax0.grid(True)
ax0.legend()
plt.ylim(0.0,100.0)
plt.xlim(1,17)

ax0 = fig.add_subplot(132, ylabel="Rank-10(%)",xlabel='epsilon')
ax0.plot(x_epoch, top10[0], 'k-', label='Clean')
ax0.plot(x_epoch, top10[1], 'b^-', label='Fast')
ax0.plot(x_epoch, top10[2], 'rs-', label='Basic')
ax0.plot(x_epoch, top10[3], 'gv-', label='Least-likely')
ax0.plot(x_epoch, top10[5], 'yo-', label='Our')
ax0.grid(True)
ax0.legend()
plt.ylim(0,100)
plt.xlim(1,17)

ax0 = fig.add_subplot(133, ylabel="mAP(%)", xlabel='epsilon')
ax0.plot(x_epoch, mAP[0], 'k-', label='Clean')
ax0.plot(x_epoch, mAP[1], 'b^-', label='Fast')
ax0.plot(x_epoch, mAP[2], 'rs-', label='Basic')
ax0.plot(x_epoch, mAP[3], 'gv-', label='Least-likely')
ax0.plot(x_epoch, mAP[5], 'yo-', label='Our')
ax0.grid(True)
ax0.legend()
plt.ylim(0,100)
plt.xlim(1,17)

fig.savefig('CUB.jpg')
