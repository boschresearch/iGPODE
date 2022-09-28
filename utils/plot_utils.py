# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Cagatay Yildiz, cagatay.yildiz1@gmail.com

import os, sys, pickle
sys.path.append("..") # Adds higher directory to python modules path.

import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np

from ode_model import ODEModel
from utils.train_utils import compute_empirical_likelihood, compute_results


def plot_2D_data(Y, fname):
    from matplotlib.gridspec import GridSpec
    plt.close()
    [N,nb,T,n] = Y.shape
    for n_ in range(N):
        Y_   = Y[n_].permute(1,0,2).detach().cpu() # T,b,ndim
        cols = ['tab:blue','tab:orange','tab:green','tab:red','tab:pink']
        fig = plt.figure(1,(5*(nb+1),max(6,1.5*n)))
        gs  = GridSpec(max(5,n+1),nb+1)
        ax  = fig.add_subplot(gs[:4,0])
        for b in range(nb):
            ax.scatter(Y_[0,b,0], Y_[0,b,1], color=cols[b], s=50)
            h1 = ax.scatter(Y_[:,b,0], Y_[:,b,1], color=cols[b], s=25, label='Data')
        ax.legend(handles=[h1], fontsize=15)
        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        for b in range(nb):
            for d in range(n):
                ax = fig.add_subplot(gs[d,b+1])
                ax.plot(Y_[:,b,d], '.-', color=cols[b], ms=1)
            plt.tight_layout()
        plt.savefig(f'{fname}-{n_}.png',dpi=200)
        plt.close()


def plot_2D_system(Y, Ypred, dens, ts, fname, mins=None, maxs=None):
    ''' 
        Y     - [B,T,n]
        Ypred - [L,B,T,n]
        dens  - [B,T]
        ts    - [T]
    '''
    plt.close()
    assert Ypred.shape[0]>1, 'multiple sequences needed to plot the predictive uncertainty!'
    from matplotlib.gridspec import GridSpec
    [nb,T,n] = Y.shape
    ts    = ts.detach().cpu()
    Y     = Y.permute(1,0,2).detach().cpu() # T,b,ndim
    Ypred = Ypred.permute(0,2,1,3).detach().cpu() # L,T,b,ndim
    Ypred_mean = Ypred.mean(0)
    cols = ['tab:blue','tab:orange','tab:green','tab:red','tab:pink']
    fig = plt.figure(1,(5*max(2,nb),max(12,3*n)))
    gs  = GridSpec(n+5,max(2,nb))
    # first 2D plot
    ax  = fig.add_subplot(gs[:4,0])
    for b in range(nb):
        ax.scatter(Y[0,b,0], Y[0,b,1], color=cols[b], s=50)
        for t in range(T):
            ax.scatter(Y[t,b,0], Y[t,b,1], color=cols[b], alpha=1-.8*t/T, s=25)
    ax.set_title('True trajectory',fontsize=15)
    if maxs is not None:
        ax.set_xlim([mins[0],maxs[0]])
        ax.set_ylim([mins[1],maxs[1]])
    # second 2D plot
    ax  = fig.add_subplot(gs[:4,1])
    for b in range(nb):
        ax.scatter(Ypred_mean[0,b,0], Ypred_mean[0,b,1], color=cols[b], s=50)
        for t in range(T):
            ax.scatter(Ypred_mean[t,b,0], Ypred_mean[t,b,1], color=cols[b], alpha=1-.8*t/T, s=25)
    ax.set_title('Mean predicted trajectory',fontsize=15)
    if maxs is not None:
        ax.set_xlim([mins[0],maxs[0]])
        ax.set_ylim([mins[1],maxs[1]])
    # objects over time
    for b in range(nb):
        for d in range(n):
            ax = fig.add_subplot(gs[4+d,b])
            ax.fill_between(ts, *np.quantile(Ypred[:,:,b,d], q=(0.025, 0.975), axis=0), color=cols[b], alpha=0.15)
            ax.plot(ts, Y[:,b,d], '.-', color=cols[b], ms=1)
            if maxs is not None:
                ax.set_ylim([mins[d],maxs[d]])
        ax   = fig.add_subplot(gs[4+n,b])
        ax.plot(ts, dens[b], '-', color=cols[b])
    plt.tight_layout()
    plt.savefig(f'{fname}',dpi=200)


def plot_predictions(exp, ode_model, L=40, idx=[0,1,2], fname=None, use_side_inf=False):
    if isinstance(ode_model,str):
        ode_model = exp.load(fname)
    fname = ode_model.name if fname is None else fname
    Ytr, Ytest, ztr, ztest, ts = exp.Ytr, exp.Ytest, exp.ztr, exp.ztest, exp.ts # N,B,T,n
    idx = [i if idx_>Ytr.shape[0] or idx_>Ytest.shape[0] else idx_ for i,idx_ in enumerate(idx)]
    Ypreds = []
    B = Ytr.shape[1]
    for z,Y,prefix in zip([ztr,ztest],[Ytr,Ytest],['tr','test']):
        Y_ = Y.reshape(-1,Y.shape[-1])
        maxs,mins = Y_.max(0)[0].cpu().numpy()+0.25,Y_.min(0)[0].cpu().numpy()-0.25
        z,Y = z[idx],Y[idx]
        with torch.no_grad():
            inputs = [Y,ts,z] if use_side_inf else [Y,ts]
            Ypred = torch.stack([ode_model.integrate(*inputs) for _ in range(L)]) # L,N,b,T,n
            Ypreds.append(Ypred) 
        dens = torch.stack([compute_empirical_likelihood(Y[:,b:b+1],Ypred[:,:,b:b+1]) for b in range(B)]) # B,N,T
        for i in range(len(idx)):
            fname_whole = os.path.join(exp.log_folder, f'{fname}-{prefix}-{idx[i]}')
            exp.plot_fnc(Y[i], Ypred[:,i], dens[:,i], ts, fname=fname_whole+'.png', mins=mins, maxs=maxs)
    return torch.stack([Ytr[idx],Ytest[idx]],0), torch.stack(Ypreds).permute(1,0,2,3,4,5) # [2,N,b,T,n], [L,2,N,b,T,n]
     
           
def plot_all_models(exp, L=20, idx=[0,6,8]):
    model_names = exp.get_all_trained_models()
    for fname in model_names:
        try:
            ode_model = exp.load(fname)
        except:
            print(f'{fname} not loaded!')
            break
        _ = plot_predictions(exp, ode_model, idx=idx, L=L, fname=fname)
        