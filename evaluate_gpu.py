import scipy.io
import torch
import numpy as np
import time

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc,junk_index1):
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
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    #junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

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
def main( query_path = './' ):
    #result_q = scipy.io.loadmat(query_path+'/query_result_normal.mat')
    result_q = scipy.io.loadmat(query_path+'/query_result.mat')
    query_feature = torch.FloatTensor(result_q['query_f'])
    query_cam = result_q['query_cam'][0]
    query_label = result_q['query_label'][0]

    result_g = scipy.io.loadmat('gallery_result.mat')
    gallery_feature = torch.FloatTensor(result_g['gallery_f'])
    gallery_cam = result_g['gallery_cam'][0]
    gallery_label = result_g['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    fail_index = []
    junk_index1 = np.argwhere(gallery_label==-1) #not well-detected
    #print(query_label)
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam, junk_index1)
        if CMC_tmp[0]==-1:
            continue
        if CMC_tmp[0]==1: fail_index.append(i)
        CMC += CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])
    print(len(fail_index), fail_index)
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
    save_result = (CMC[0],CMC[4],CMC[9],ap/len(query_label))
    return save_result

if __name__=='__main__':
    #since = time.time()
    #query_path = './attack_query/baseline-9/16'
    query_path = './'
    main(query_path)
    #print(time.time()-since)
