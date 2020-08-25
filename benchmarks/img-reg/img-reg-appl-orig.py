import torch

from time import time

# vals is a vector of pairs of values, i.e., (zip vdataR vval) in C code.
def Histogram2D(vals,device):
    #preconpute the range

    #rangeh=torch.ceil(vals.max()-vals.min()).int()+6
    lbs = vals.min()
    rangeh=torch.floor(vals.max()-lbs).int()+4

    #compute indices

    t_idx=vals.floor()

    #index array

    p=torch.arange(vals.size(0))

    #setup varibles

    t=torch.tensor(vals-t_idx,dtype=torch.float32,requires_grad=True,device=device)

    ones4=torch.ones([4,4],dtype=torch.int32,device=device)

    onesp=torch.ones(t.size(0),dtype=torch.int32,device=device)

    stride_x, stride_y=torch.meshgrid([torch.arange(0,4,dtype=torch.int32,device=device), torch.arange(0,4,dtype=torch.int32,device=device)])

    #t_idx=t_idx.flatten().int()
    t_idx=(vals-lbs).abs().flatten().int()

    indices=torch.einsum('a,bc->abc',t_idx[2*p],ones4)*(rangeh)

    indices+=torch.einsum('a,bc->abc',onesp,stride_x)*rangeh

    indices+=torch.einsum('a,bc->abc',t_idx[2*p+1],ones4)

    indices+=torch.einsum('a,bc->abc',onesp,stride_y)

    a=torch.stack([t.flatten()*0+1, t.flatten(), t.flatten()**2, t.flatten()**3],dim=1)

    b=torch.tensor(([1, 4, 1, 0],[-3, 0, 3, 0],[3, -6, 3, 0],[-1, 3, -3, 1]),dtype=torch.float32,device=device)/6

    y=torch.mm(a,b)

    #print(y)

    res=(torch.einsum('ab,ac->abc',y[2*p,:],y[2*p+1,:])).flatten()

    #sort_res,nid=torch.sort(indices.flatten())

    v,ids=indices.flatten().unique(return_counts=True)

    val=torch.split(res.flatten(),ids.tolist());

    hist=torch.zeros(v.size(),device=device,dtype=torch.float32)

    va=(v%rangeh)

    vb=((v//rangeh).long())

    for index, value in enumerate(val):

        hist[index]=value.sum()

    v_a,ids=va.unique(return_counts=True)

    hist_a=torch.zeros(v_a.size(),device=device,dtype=torch.float32)

    vala=torch.split(hist,ids.tolist());

    for index, value in enumerate(vala):

        hist_a[index]=value.sum()

    v_b,ids=vb.unique(return_counts=True)

    hist_b=torch.zeros(v_b.size(),device=device,dtype=torch.float32)

    valb=torch.split(hist,ids.tolist());

    for index, value in enumerate(vala):

        hist_b[index]=value.sum()

    return hist, hist_a, hist_b

from time import time

device='cuda'

torch.manual_seed(123)

import sys
dataset = sys.argv[1]
if dataset == 'small':
    x=torch.rand([2000000,2], dtype=torch.float32,device=device)*50-30
elif dataset == 'large':
    x=torch.rand([8000000,2], dtype=torch.float32,device=device)*200
else:
    raise Exception('No such dataset: ' + dataset)

runs=100
st = time()
for i in range(runs):
    hist_c, hist_a, hist_b = Histogram2D(x,device)
en = time()
print("Pytorch GPU Histogram Runtime:",(en-st)/runs)
open(dataset + '-orig.runtime', 'w').write('%.2f' % ((en-st)/runs*1000))
