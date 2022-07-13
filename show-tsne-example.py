from sklearn.datasets import load_digits
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import scipy
import torch
import numpy as np
#digits = load_digits()

query_path = '.'
result_n = scipy.io.loadmat(query_path+'/query_result_normal.mat')
query_n = torch.FloatTensor(result_n['query_f'])
label_n = result_n['query_label'][0]

result_q = scipy.io.loadmat(query_path+'/query_result.mat')
query_q = torch.FloatTensor(result_q['query_f'])
label_q = result_q['query_label'][0]

data = torch.cat( (query_n, query_q), 0)

flag = -1
label_t1 = torch.zeros(label_n.shape)
for index, xx in enumerate(label_n):
    if index == 0:
        flag = xx
        continue
    if xx !=flag:
        flag = xx
        label_t1[index] = label_t1[index-1] +1 
    else:
        label_t1[index] = label_t1[index-1]

flag = -1
label_t2 = torch.zeros(label_q.shape)
for index, xx in enumerate(label_q):
    if index == 0:
        flag = xx
        continue
    if xx !=flag:
        flag = xx
        label_t2[index] = label_t2[index-1] +1
    else:
        label_t2[index] = label_t2[index-1]

label = np.concatenate( (label_t1, label_t2), 0) 
print(label)
#label = torch.cat( (torch.zeros(label_n.shape), torch.ones(label_q.shape)), 0)

print(data.shape, label.shape)
embeddings = TSNE(n_jobs=16).fit_transform(data)

fig = plt.figure(dpi=1200)


top = 10
vis_x = [] #embeddings[0:first20, 0]
vis_y = [] #embeddings[0:first20, 1]
label_t = []
for i in range(500): 
    if label_t1[i] == top:
        break
    if i==0 or label_t1[i] != label_t1[i-1]:
        vis_x.append(embeddings[i, 0])
        vis_y.append(embeddings[i, 1])
        label_t.append(label_t1[i])
print(label_t)
plt.scatter(vis_x, vis_y, c=label_t, cmap=plt.cm.get_cmap("jet", top), marker='.')

start = len(label_t1)
vis_x = [] #embeddings[0:first20, 0]
vis_y = [] #embeddings[0:first20, 1]
label_t = []
for i in range(500):
    if label_t2[i] == top:
        break
    if i==0 or label_t2[i] != label_t2[i-1]:
        vis_x.append(embeddings[start+i, 0])
        vis_y.append(embeddings[start+i, 1])
        label_t.append(label_t2[i])
print(label_t)
plt.scatter(vis_x, vis_y, c=label_t, cmap=plt.cm.get_cmap("jet", top), marker='*')
plt.grid(True)
plt.colorbar(ticks=range(top))
plt.clim(-0.5, top-0.5)
plt.show()
fig.savefig( 'tsne.jpg')
