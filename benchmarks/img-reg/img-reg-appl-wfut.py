import torch

import wrapFuthark

from time import time

def CubicBSpline3D(pts,x,device='cpu',return_jac=False):

    t_idx=pts.floor()

    #t_idx.detach()

    p=torch.arange(pts.size(0))

    t=pts-t_idx

    #t_idx=torch.tensor(_idx.clone().flatten(),dtype=torch.int32,device=device)

    ones4=torch.ones([4,4,4],dtype=torch.int32,device=device)

    onesp=torch.ones(t.size(0),dtype=torch.int32,device=device)

    t_idx=t_idx.flatten().int()

    stride_x, stride_y, stride_z=torch.meshgrid([torch.arange(0,4,device=device)-2, torch.arange(0,4,device=device)-2, torch.arange(0,4,device=device)-2])

    indices=torch.einsum('a,bcd->abcd',t_idx[3*p],ones4)*(x.size(2)*x.size(1))

    indices+=torch.einsum('a,bcd->abcd',onesp,stride_x.int())*(x.size(2)*x.size(1))

    indices+=torch.einsum('a,bcd->abcd',t_idx[3*p+1],ones4)*(x.size(2))

    indices+=torch.einsum('a,bcd->abcd',onesp,stride_y.int())*(x.size(2))

    indices+=torch.einsum('a,bcd->abcd',t_idx[3*p+2],ones4)+torch.einsum('a,bcd->abcd',onesp,stride_z.int())

    indices=indices.long()

    jac=None

    a=torch.stack([t.flatten()*0+1, t.flatten(), t.flatten()**2, t.flatten()**3],dim=1) 

    b=torch.tensor(([1, 4, 1, 0],[-3, 0, 3, 0],[3, -6, 3, 0],[-1, 3, -3, 1]),dtype=torch.float32,device=device)/6 

    y=torch.mm(a,b)

    w=torch.sum(torch.einsum('ab,ac,ad->abcd',y[3*p,:],y[3*p+1,:],y[3*p+2,:])*x.flatten(start_dim=1)[:,indices],dim=[2,3,4])

    if(return_jac):

        da=torch.stack([t.flatten()*0, t.flatten()*0+1, t.flatten()*2, 3*t.flatten()**2],dim=1) 

        dy=torch.mm(da,b)

        wx=torch.sum(torch.einsum('ab,ac,ad->abcd',dy[3*p,:],y[3*p+1,:],y[3*p+2,:])*x.flatten(start_dim=1)[:,indices],dim=[2,3,4])

        wy=torch.sum(torch.einsum('ab,ac,ad->abcd',y[3*p,:],dy[3*p+1,:],y[3*p+2,:])*x.flatten(start_dim=1)[:,indices],dim=[2,3,4])

        wz=torch.sum(torch.einsum('ab,ac,ad->abcd',y[3*p,:],y[3*p+1,:],dy[3*p+2,:])*x.flatten(start_dim=1)[:,indices],dim=[2,3,4])

        jac=torch.stack([wx.t(),wy.t(),wz.t()],dim=2)

    return w,jac

def LinearSpline3D(pts,x,device='cpu',return_jac=False):

    t_idx=pts.floor()

    # create a enumerated list for the points

    p=torch.arange(pts.size(0))

    # create the tri-liniar parametrization for the evaluation points 

    t=torch.tensor(pts-t_idx,dtype=torch.float32,device=device)

    #make a cube of 2x2x2 of ones as a helper for converting indices

    ones4=torch.ones([2,2,2],dtype=torch.int32,device=device)

    #make a array of ones as a helper for converting indices

    onesp=torch.ones(t.size(0),dtype=torch.int32,device=device)

    #make a cube of 2x2x2 of indices (0-origin) as a helper for converting indices

    stride_x, stride_y, stride_z=torch.meshgrid([torch.arange(0,2,device=device), torch.arange(0,2,device=device), torch.arange(0,2,device=device)])

    #compute/transform the indices to the flatten domain

    t_idx=t_idx.flatten().int()

    indices=torch.einsum('a,bcd->abcd',t_idx[3*p],ones4)*(x.size(2)*x.size(1))

    indices+=torch.einsum('a,bcd->abcd',onesp,stride_x.int())*(x.size(2)*x.size(1))

    indices+=torch.einsum('a,bcd->abcd',t_idx[3*p+1],ones4)*(x.size(2))

    indices+=torch.einsum('a,bcd->abcd',onesp,stride_y.int())*(x.size(2))

    indices+=torch.einsum('a,bcd->abcd',t_idx[3*p+2],ones4)+torch.einsum('a,bcd->abcd',onesp,stride_z.int())

    indices=indices.long()

    #the parameter vector for the spline (1D)

    y=torch.stack([1-t.flatten(), t.flatten()],dim=1) 

    #compute the element-wise kroenecker products 

    w=torch.sum(torch.einsum('ab,ac,ad->abcd',y[3*p,:],y[3*p+1,:],y[3*p+2,:])*x.flatten(start_dim=1)[:,indices.long()],dim=[2,3,4])

    # derivatives

    #the parameter vector for the derivative spline (1D)

    jac=None

    if(return_jac):

        dy=torch.stack([t.flatten().clone()*0-1, t.flatten().clone()*0+1],dim=1) 


        wx=torch.sum(torch.einsum('ab,ac,ad->abcd',dy[3*p,:],y[3*p+1,:],y[3*p+2,:])*x.flatten(start_dim=1)[:,indices],dim=[2,3,4])

        wy=torch.sum(torch.einsum('ab,ac,ad->abcd',y[3*p,:],dy[3*p+1,:],y[3*p+2,:])*x.flatten(start_dim=1)[:,indices],dim=[2,3,4])

        wz=torch.sum(torch.einsum('ab,ac,ad->abcd',y[3*p,:],y[3*p+1,:],dy[3*p+2,:])*x.flatten(start_dim=1)[:,indices],dim=[2,3,4])

        #construct the array of jacobians

        jac=torch.stack([wx.t(),wy.t(),wz.t()],dim=2)

    return w,jac 

# vals is a vector of pairs of values, i.e., (zip vdataR vval) in C code.
def Histogram2D(vals,device):
    #print('In Histo2D, shape input: ', vals.shape, vals.size(0))
    #rangeh=torch.ceil(vals.max()-vals.min()).int()+6
    #print('Range, vals: ',rangeh)


    dataR=(vals[:,0]).cpu().detach().numpy()
    valsO=(vals[:,1]).cpu().detach().numpy()

    #print("inp vals1: ", dataR[:32])
    #print("inp vals2: ", valsO[:32])

    #print('In Histo2D, vals: ',valsO[:32], dataR[:32])

    # creating futhark object
    wrapobj = wrapFuthark.HISTOFUTH(dataR, valsO)
    
    # warmup call to Futhark wrapper
    h1, h2, hist10_cl, hist20_cl, histc0_cl = wrapobj.computeHistos()

    GPU_RUNS = 1
    st = time()
    # count call to Futhark wrapper
    for idx in range(GPU_RUNS):
        h1, h2, hist1_cl, hist2_cl, histc_cl = wrapobj.computeHistos()


    hist_a = hist1_cl.get()
    hist_b = hist2_cl.get()
    hist_c = histc_cl.get()
    et = time()

    print('Futhark Histogram Runtime:', (et-st)/GPU_RUNS )
    print('Hist_a: ', hist_a)
    print('Hist_b: ', hist_b)
    #print('Hist_c: ', hist_c[:111])
    #print('Hist_c: ', hist_c[53*50:])
    print('h1/2: ', h1, h2)

    return hist_c, hist_a, hist_b

#print(y,x.grad)

def NMI(x,device):

    h1,h2,h3=Histogram2D(x,device)

    h1=h1/h1.sum()

    h2=h2/h2.sum()

    h3=h3/h3.sum()

    return(-torch.sum(h2*torch.log(h2))+(-torch.sum(h3*torch.log(h3))))/(-torch.sum(h1*torch.log(h1)))

def MI(x,device):

    h1,h2,h3=Histogram2D(x,device)

    h1=h1/h1.sum()

    h2=h2/h2.sum()

    h3=h3/h3.sum()

    return+(-torch.sum(h1*torch.log(h1)))-(-torch.sum(h2*torch.log(h2))+(-torch.sum(h3*torch.log(h3))))

def NCC(x,device):

    mx=torch.mean(x,dim=0)

    return torch.mean((x[:,0]-mx[0])*(x[:,1]-mx[1]))/torch.prod(torch.var(x,dim=0))

def SSD(x,device):

    return torch.sum((x[:,0]-x[:,1])**2)

#######################################

#torch.cuda.empty_cache()

#device='cpu'
device='cuda'

x=torch.zeros([1, 250, 250, 250], dtype=torch.float32, requires_grad=True,device=device)

pts=torch.rand([1000000,3],dtype=torch.float32,device=device)*200+10

st = time()

test_eval,_=CubicBSpline3D(pts,x,device,True)

print('diff',time()-st)

#torch.cuda.empty_cache()

st = time()

test_eval,_=LinearSpline3D(pts,x,device,True)

print('diff',time()-st)

torch.manual_seed(123)
x=torch.rand([2000000,2], dtype=torch.float32,device=device)*50-30
#x=torch.rand([8000000,2], dtype=torch.float32,device=device)*50-30

x.requires_grad_()

histo = Histogram2D(x,device)

st = time()

res=SSD(x,device)

#print('diff',time()-st,res,x.grad.data)

st = time()

resNcc=NCC(x,device)

resNcc.backward()

print('diff',time()-st,resNcc)
