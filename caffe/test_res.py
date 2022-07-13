import caffe
import os
import numpy as np
from PIL import Image

def clip(input, zeros, ones):
    low_mask = input<zeros 
    up_mask = input>ones
    input[low_mask] = zeros[low_mask]
    input[up_mask] = ones[up_mask]
    return input


def attack(img_pth, transformer):
    # load data
    data = caffe.io.load_image(img_pth)
    data = data * 255
    data = caffe.io.resize_image( data,(256,128))
    data = transformer.preprocess('data', data)
    net.blobs['data'].reshape(1,3,256,128)
    net.blobs['data'].data[...] = data

    # calculate the up bound and low bound of the input
    zeros = np.zeros((256,128,3),dtype=np.float32)
    ones = np.ones((256,128,3),dtype=np.float32)*255
    zeros = transformer.preprocess('data', zeros)
    ones = transformer.preprocess('data', ones)
    
    # As in my paper, I set the rate = 16
    rate = 16
    for i in range(int(min(1.25*rate, rate+4))):
        net.forward()
        net.backward()
        loss = net.blobs['loss'].data
        grad = net.blobs['data'].diff
        # In the first round, I set the -f (more detail in the paper)
        if i==0:
            fc7_adv = -net.blobs['fc7'].data
            fc7_adv = fc7_adv.reshape(1,512,1,1)
            print 'make adv'
            #test = fc7.flatten()
            #print sum(test*test)
            net.blobs['fc7-adv'].data[...] = fc7_adv

        # use the acculmulate grad (just small trick)
        if i==0:
            acc_grad = np.sign(grad)
        else:
            acc_grad = acc_grad + np.sign(grad)
           
        acc_grad = acc_grad.reshape(3,256,128)
        mask_diff = np.abs(acc_grad) > rate
        acc_grad[mask_diff] = rate * np.sign(acc_grad[mask_diff])

        net.blobs['data'].data[...] = clip(data - acc_grad, zeros, ones)

        print i, loss

    data = net.blobs['data'].data
    data = transformer.deprocess('data',data)
    return data


##########################################
# Set data path
##########################################
query_pth = './query/'
query_save_pth = './query-adv-res-mvn/'
# Set model path
# Remember to remvoe the ImageData layer in the deploy prototxt !!
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('resnet.prototxt', 'resnet_340000.caffemodel', caffe.TEST)

##########################################
# Prepare
##########################################
# define transformer
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # 192*96*3 -> 3*192*96
transformer.set_mean('data', np.asarray([107.72, 103.77, 109.23]) ) # mean pixel
transformer.set_channel_swap('data', (2,1,0))  # RGB -> BGR

#########################################
#Attack->save data
#########################################
for root,dirs,files in os.walk(query_pth):
    for name in files:
        src_pth = query_pth + name #load path
        print src_pth
        dst_pth = query_save_pth + name #save path
        im = attack(src_pth, transformer)  # attack
        im = Image.fromarray(im.astype('uint8'))
        im.save(dst_pth) # save image
