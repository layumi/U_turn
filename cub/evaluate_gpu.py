import scipy.io
import torch
import numpy as np
import time

#######################################################################
# Evaluate
def evaluate(qf,ql,gf,gl,junk_index):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    good_index = np.setdiff1d(query_index, junk_index, assume_unique=True)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def save_index(index):
# save the top 15 id
    with open('adv_list.txt','a') as fp:
        for i in range(15):
            fp.write("%d," %index[i])
        fp.write(" \n")

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # save index 
    #save_index(index)    

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
# main function
def main( query_path ):
    result_q = scipy.io.loadmat(query_path+'/query.mat')
    query_feature = torch.FloatTensor(result_q['img_f'])
    query_label = result_q['label'][0]

    result_g = scipy.io.loadmat('pytorch_result.mat')
    gallery_feature = torch.FloatTensor(result_g['img_f'])
    gallery_label = result_g['label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    junk_index1 = np.argwhere(gallery_label==-1) #not well-detected
    #print(query_label)
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label, i)
        if CMC_tmp[0]==-1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
    save_result = (CMC[0],CMC[4],CMC[9],ap/len(query_label))
    return save_result

if __name__=='__main__':
    #since = time.time()
    query_path = './attack_query/ft_ResNet50_all-1/2/'
    main(query_path)
    #print(time.time()-since)
