import torch
import torch.nn as nn
import numpy as np 
from torchdiffeq import odeint
import math
from TorchDiffEqPack.odesolver import odesolve
import os

def my_kmeans(Y,k):
    from scipy.cluster.vq import kmeans
    n   = Y.shape[-1]
    Y_  = Y.detach().cpu().reshape([-1,n]).numpy()
    std = Y_.std(0)
    Yw  = Y_ / std
    rnd_idx = torch.randint(0,Yw.shape[0],[k]).numpy()
    Zw = kmeans(Yw,Yw[rnd_idx])[0]
    Z = torch.tensor(Zw*std, dtype=Y.dtype, device=Y.device) 
    if Z.shape[0] != k:
        Z = torch.cat([Z,torch.randn(k-Z.shape[0],Z.shape[1],device=Z.device)],0)
    return Z
    

def aca_odesolve(f, z0, ts, method, rtol=1e-5, atol=1e-6, step_size=None):
    options = {}
    method = 'rk2' if method=='midpoint' else method
    step_size = (ts[1]-ts[0]).item() if step_size is None else step_size
    options.update({'method': method})
    options.update({'h': step_size})
    options.update({'t0': ts[0].item()})
    options.update({'t1': ts[-1].item()})
    options.update({'rtol': rtol})
    options.update({'atol': atol})
    options.update({'t_eval': ts.tolist()})
    options.update({'interpolation_method':'cubic'})
    return odesolve(f, z0, options=options)

def save_policy_wo_data(fname):
    import envs
    import ctrl.ctrl as base
    env = envs.MyAcrobot(device='cpu', obs_trans=True, solver='dopri5')
    ctrl,D = base.CTRL.load(env, f'{fname}')
    ctrl.save(fname=fname+'-nodata')

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, torch.tensor):
            return m + torch.log(sum_exp)
        else:
            return m + math.log(sum_exp)

def flatten_(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat)>0 else torch.tensor([])

def dense_integrate(f, z0, ts, dt, method, ret_time_grid=False):
    input_tuple = isinstance(z0,tuple)
    T = torch.max(ts) # T
    td = torch.arange(0, T, dt, dtype=z0.dtype, device=ts.device)
    td = torch.cat((td,ts)) 
    td = torch.unique(td,True)
    # ts_idx = torch.cat([(td==t_).nonzero() for t_ in ts]).squeeze()
    ts_idx = torch.cat([torch.nonzero(td==t_,as_tuple=False) for t_ in ts]).squeeze()
    zd = odeint(f, z0, td, method=method) # T,N,n
    if not input_tuple:
        z  = zd[ts_idx,:,:] # len(ts),N,n
    else:
        z  = [zd_[ts_idx] for zd_ in zd] # len(ts),N,n
    if ret_time_grid: 
        return z,zd,td
    return z,zd

def smooth(x,w=7):
    x = np.array(x)
    y = np.zeros_like(x)
    for i in range(len(y)):
        y[i] = x[max(0,i-w):min(i+w,len(y))].mean()
    return y


def get_minibatch_jacobian(y, x, create_graph=False):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                      create_graph=create_graph)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac
    
def sq_dist(X1, X2, ell=1.0):
    X1  = X1 / ell
    X1s = torch.sum(X1**2, dim=-1, keepdim=True) # N,1 or n,N,1
    X2  = X2 / ell
    X2s = torch.sum(X2**2, dim=-1, keepdim=True).transpose(-1,-2)  # 1,N or n,1,N
    sq_dist = -2*X1@X2.transpose(-1,-2) + X1s + X2s # N,N or n,N,N
    return sq_dist 

def draw_from_gp(inputs, sf, ell, L=1, N=1, n_out=1, eps=1e-5):
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(1) 
    T = inputs.shape[0]
    cov  = K(inputs,inputs,ell,sf,eps=eps) # T,T
    L_ = torch.cholesky(cov)
    # L,N,T,n_out or N,T,n_out or T,n_out
    return L_ @ torch.randn([L,N,T,n_out],device=inputs.device).squeeze(0).squeeze(0)


def K(X1, X2, ell=1.0, sf=1.0, eps=1e-5):
    dnorm2 = sq_dist(X1,X2,ell)
    K_ = sf**2 * torch.exp(-0.5*dnorm2)
    if X1.shape[-2]==X2.shape[-2]:
        return K_ + torch.eye(X1.shape[-2],device=X1.device)*eps
    return K_    
    
    
    
    
    
    
    
    
    
    
    
    
    
    