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

for i in range(6):
    top1[i] = []
    top10[i] = []

top1[0] = [0.88, 0.88, 0.88, 0.88, 0.88]
top10[0] = [0.97, 0.97, 0.97, 0.97, 0.97]

with open("Output.txt", "r") as f:
    for line in f:
        score = line.split('|')
        method_id = int(score[2])
        top1_acc, top10_acc = float(score[3]), float(score[5])
        top1[method_id].append(top1_acc)
        top10[method_id].append(top10_acc)

fig = plt.figure()
ax0 = fig.add_subplot(121, title="top1")
ax0.plot(x_epoch, top1[0], 'ko-', label='GT')
ax0.plot(x_epoch, top1[1], 'bo-', label='Fast')
ax0.plot(x_epoch, top1[2], 'ro-', label='Basic')
ax0.plot(x_epoch, top1[3], 'go-', label='Least-likely')
ax0.plot(x_epoch, top1[5], 'yo-', label='Our')
ax0.legend()


ax0 = fig.add_subplot(122, title="top10")
ax0.plot(x_epoch, top10[0], 'ko-', label='GT')
ax0.plot(x_epoch, top10[1], 'bo-', label='Fast')
ax0.plot(x_epoch, top10[2], 'ro-', label='Basic')
ax0.plot(x_epoch, top10[3], 'go-', label='Least-likely')
ax0.plot(x_epoch, top10[5], 'yo-', label='Our')
ax0.legend()

fig.savefig( 'accuracy.jpg')
