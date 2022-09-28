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

import os, sys
sys.path.append("..") # Adds higher directory to python modules path.

import torch
from utils.logger import Logger


def compute_empirical_likelihood(Y, Ypred):
    ''' 
        Y     - [N,b,T,n]     
        Ypred - [L,N,b,T,n]   
        returns - [N,T]
    '''
    Y = Y.permute(0,2,1,3) # [N,T,B,n]
    Ypred = Ypred.permute(1,3,0,2,4) # [N,T,L,B,n]
    [N,T,L,b,n] = Ypred.shape
    Y     = Y.detach().reshape(N*T,b*n).cpu() # NT,bn
    Ypred = Ypred.detach().reshape(N*T,L,b*n).cpu() # NT,L,bn
    mean,var = Ypred.mean(1), Ypred.var(1)
    mvn = torch.distributions.MultivariateNormal(mean,torch.stack([(1e-2+v).diag() for v in var]))
    return mvn.log_prob(Y).reshape(N,T).detach()


def compute_results(exp, ode_model, Nmax=1000, L=20, use_side_inf=False):
    Ytr, Ytest, ztr, ztest, ts = exp.Ytr, exp.Ytest, exp.ztr, exp.ztest, exp.ts
    Nmini = min(100, Ytr.shape[0], Ytest.shape[0])
    def compute_minibatch_results(Y,ts,z):
        N = min(Nmax, Y.shape[0])
        Ypreds = []
        for i in range(N//Nmini):
            Y_,X_ = Y[i*Nmini:(i+1)*Nmini],z[i*Nmini:(i+1)*Nmini]
            inputs = [Y_,ts,X_] if use_side_inf else [Y_,ts]
            Ypred = torch.stack([ode_model.integrate(*inputs) for _ in range(L)]) # L,Nmini,b,T,n
            Ypreds.append(Ypred)
        Ypreds = torch.cat(Ypreds,1) # L,N,b,T,n
        mse  = ((Ypreds - Y[:N])**2).mean(0).mean(0).sum(0).sum(-1) # T
        dens = compute_empirical_likelihood(Y[:N], Ypreds).mean(0)
        return mse,dens
    with torch.no_grad():
        tr_mse,   tr_dens   = compute_minibatch_results(Ytr,   ts, ztr)
        test_mse, test_dens = compute_minibatch_results(Ytest, ts, ztest)
        return [tr_mse, test_mse, tr_dens, test_dens]


def crop_data(Y, ts, Tsub):
    # extract subsequences of length Tsub
    if Tsub > 1:
        N,b,T,n = Y.shape
        t0s = torch.randint(T-Tsub,[N],device=Y.device)
        Y   = torch.stack([Y[i,:,t0:t0+Tsub] for i,t0 in enumerate(t0s)])
        ts  = ts[:Tsub]
    return Y,ts


def compute_elbo_terms(ode_model, Y, ts, zs=None, Tsub=-1, Nsub=-1):
    # extract dataset if minibatching
    N = Y.shape[0]
    if Nsub > 1:
        idx = torch.randint(N, [Nsub], device=Y.device)
        Y  = Y[idx]
        zs = None if zs is None else zs[idx]
    else:
        Nsub = N
    # compute the likelihood
    lhood_, kl_z_, kl_f = ode_model.minibatch_elbo(Y, ts, zs=zs, Tsub=Tsub)
    lhood = lhood_ / Nsub * N
    kl_z  = kl_z_  / Nsub * N
    return  lhood, kl_z+kl_f


def optimization_loop(exp, ode_model, Niter, logger, plot_fnc=None, eta=1e-3, beta=1.0, Tsub=-1, Nsub=-1, \
                      save_every=100, fname=None, kl_anneal=False, use_side_inf=False):
    fname = ode_model.name if fname is None else fname
    Ytr, ts  = exp.Ytr, exp.ts
    ztr      = exp.ztr if use_side_inf else None
    opt_pars = ode_model.parameters() 
    opt = torch.optim.Adam(opt_pars, eta)
    for it in range(Niter):
        beta = min(1,2*(it+1)/Niter) if kl_anneal else beta 
        opt.zero_grad()
        ode_model.train()
        lhood,kl = compute_elbo_terms(ode_model, Ytr, ts, ztr, Tsub=Tsub, Nsub=Nsub)
        loss     = -lhood + beta*kl
        loss.backward()
        logger.log(lhood,kl)
        opt.step()
        ode_model.fix_gpytorch_cache(it)
        if (it+1) % save_every == 0:
            ode_model.save(os.path.join(exp.log_folder,fname))
        # except Exception as e:
    # plot/log results at the end of optimization
    with torch.no_grad():
        try:
            if plot_fnc is not None:
                plot_fnc(exp, ode_model, fname=fname, use_side_inf=use_side_inf)
            tr_err, test_err, tr_dens, test_dens = compute_results(exp, ode_model, L=20, use_side_inf=use_side_inf)
            logger.print_message('\ntr_MSE={:.3f} \ttest_MSE={:.3f}\n'.format(tr_err.mean(),test_err.mean()))
            logger.print_message('\ntr_dens={:.3f} \ttest_dens={:.3f}\n'.format(tr_dens.mean(),test_dens.mean()))
            exp.save(ode_model, fname, results=[tr_err, test_err, tr_dens, test_dens])
        except Exception as e:
            print(e)


def optimize_model(exp, ode_model, Niter, plt_fnc, N=50, fname=None, use_side_inf=False, log_every=100):
    '''
        Groundtruth charge information is used if use_side_inf
        Also, the first round of optimization is executed 2*Niter iterations
    '''
    fname = ode_model.name if fname is None else fname
    if ode_model.is_gp:
        eta = 3e-3
    elif ode_model.func=='nn': 
        eta = 5e-4
    elif ode_model.func=='bnn': 
        eta = 1e-4
    # logging
    logger = Logger(['lhood','kl'], print_every=log_every, log_folder=exp.log_folder)
    logger.print_message(f'Experiment folder: {exp.log_folder}.')
    logger.print_message(f'{fname} training started.')
    logger.print_message(ode_model.__repr__())
    # stage-1
    optimization_loop(exp, ode_model, 2*Niter, logger, plt_fnc, eta=eta, Tsub=5,  Nsub=N, \
                      fname=fname, kl_anneal=True, use_side_inf=use_side_inf, save_every=log_every)
    # stage-2
    optimization_loop(exp, ode_model, Niter,   logger, plt_fnc, eta=eta, Tsub=16, Nsub=N, \
                  fname=fname, kl_anneal=True, use_side_inf=use_side_inf, save_every=log_every)
    # stage-3
    optimization_loop(exp, ode_model, Niter,   logger, plt_fnc, eta=eta, Tsub=33, Nsub=N, \
                  fname=fname, kl_anneal=False, use_side_inf=use_side_inf, save_every=log_every)
    # stage-4
    optimization_loop(exp, ode_model, Niter,   logger, plt_fnc, eta=eta, Tsub=50, Nsub=N, \
                  fname=fname, kl_anneal=False, use_side_inf=use_side_inf, save_every=log_every)









