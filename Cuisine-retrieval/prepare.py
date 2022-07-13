import os
from shutil import copyfile
from PIL import Image

# You only need to change this line to your dataset download path

download_path = '../Food'
dst_path = '../Food-cropped/'
train_path = dst_path+'/pytorch/train_all/'
gallery_path = dst_path+'/pytorch/gallery/'
query_path = dst_path + '/pytorch/query'
if not os.path.isdir(download_path):
    print('please change the download_path')

if not os.path.isdir(dst_path):
    os.mkdir(dst_path)

for root, dirs, files in os.walk(download_path, topdown=True):
    for name in files:
        if name == 'bb_info.txt':
            with open(root+'/'+name) as fp:
                for i, line in enumerate(fp):
                    if i==0:
                        continue
                    img_name, left, upper, right, lower = line.split(' ')
                    left, upper, right, lower = int(left), int(upper), int(right), int(lower)
                    if right-left <10 or lower-upper<10:
                        continue
                    im = Image.open(root + '/' + img_name + '.jpg')
                    im = im.crop(box= (left-1, upper-1, right-1, lower-1) )
                    root_dst = root.replace('Food', 'Food-cropped')
                    if not os.path.isdir(root_dst):
                        os.mkdir(root_dst)
                    im.save(root_dst + '/' + img_name + '.jpg')

if not os.path.isdir(dst_path+'/pytorch'):
    os.mkdir(dst_path+'/pytorch')
    os.mkdir(dst_path+'/pytorch/train_all')
    os.mkdir(dst_path+'/pytorch/gallery')
    os.mkdir(query_path)


# Split
for i in range(224):
    print(dst_path+str(i+1))
    os.system('mv %s %s'% ( dst_path+str(i+1), train_path) )

for i in range(32):
    os.system('mv %s %s'%( dst_path+str(i+225), gallery_path) )

# Query
for root, dirs, files in os.walk(gallery_path, topdown=True):
    count = 0
    for name in files:
        root_dst = root.replace('gallery', 'query')
        if not os.path.isdir(root_dst):
            os.mkdir(root_dst)
        if name[-3:] == 'jpg':
            os.system('mv %s %s' %(root+'/'+name, root_dst+'/'+name))
            count +=1
        if count == 16: 
            break       
