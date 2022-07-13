import os
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description='Prepare')
parser.add_argument('--method_id', default=3, type=int, help='1.fast || 2.least likely || 3.label smooth')
parser.add_argument('--rate', default=2, type=int, help='attack rate')
parser.add_argument('--name', default='baseline', type=str, help='save model path')

opt = parser.parse_args()

# You only need to change this line to your dataset download path
download_path = './attack_query'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#query
query_path = download_path + '/%s-'%opt.name + str(opt.method_id) + '/' + str(opt.rate)
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0] 
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

