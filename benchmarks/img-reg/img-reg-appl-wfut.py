import torch

import wrapFuthark

from time import time

# vals is a vector of pairs of values, i.e., (zip vdataR vval) in C code.
def Histogram2D(dataset,vals,device):

    dataR=(vals[:,0]).cpu().detach().numpy()
    valsO=(vals[:,1]).cpu().detach().numpy()

    # creating futhark object
    wrapobj = wrapFuthark.HISTOFUTH(dataR, valsO)

    # warmup call to Futhark wrapper
    h1, h2, hist10_cl, hist20_cl, histc0_cl = wrapobj.computeHistos()

    GPU_RUNS = 100
    st = time()
    # count call to Futhark wrapper
    for idx in range(GPU_RUNS):
        h1, h2, hist1_cl, hist2_cl, histc_cl = wrapobj.computeHistos()


    hist_a = hist1_cl.get()
    hist_b = hist2_cl.get()
    hist_c = histc_cl.get()
    et = time()

    print('Futhark Histogram Runtime:', (et-st)/GPU_RUNS )
#    print('Hist_a: ', hist_a, sum(hist_a))
#    print('Hist_b: ', hist_b)
#    print('Hist_c: ', hist_c[:111])
#    print('h1/2: ', h1, h2)

    open(dataset + '-fut.runtime', 'w').write('%.2f' % ((et-st)/GPU_RUNS*1000))

    return hist_c, hist_a, hist_b

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

histo = Histogram2D(dataset,x,device)
