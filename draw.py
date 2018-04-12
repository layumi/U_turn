import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

######################################################################
# Draw Curve
#---------------------------
x_epoch = [2,4,8,12,16]

y0 = [0.88, 0.88, 0.88, 0.88, 0.88]

y1 = [0.818587,
0.668943,
0.38747,
0.194181,
0.085808
]
y2 = [0.813539,
0.512767,
0.237827,
0.139252,
0.094418
]
y3 = [0.83848,
0.611639,
0.233967,
0.088777,
0.037411
]

y5 = [0.824525,
0.438836,
0.078682,
0.020487,
0.005344
]
fig = plt.figure()
ax0 = fig.add_subplot(111, title="accuracy")
ax0.plot(x_epoch, y0, 'ko-', label='GT')
ax0.plot(x_epoch, y1, 'bo-', label='Fast')
ax0.plot(x_epoch, y2, 'ro-', label='Basic')
ax0.plot(x_epoch, y3, 'go-', label='Least-likely')
ax0.plot(x_epoch, y5, 'yo-', label='Our')
ax0.legend()
fig.savefig( 'accuracy.jpg')
