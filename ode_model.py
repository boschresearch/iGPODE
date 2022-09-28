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

import numpy as np
import io, pickle, warnings
import torch, torch.nn as nn
from torch.distributions.kl import kl_divergence as kl

from TorchDiffEqPack.odesolver import odesolve

from core import GPModel, MLP, BNN, GRUEncoder

    
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
    
"""
The following three functions are taken from NRI V 1.0
(https://github.com/ethanfetaya/NRI
Copyright 2018 Ethan Fetaya, Thomas Kipf, MIT license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree)
to facilitate modeling interaction signals.
"""
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def edge2node(x, rel_rec, rel_send):
    # NOTE: Assumes that we have the same graph across all samples.
    incoming = torch.matmul(rel_rec.t(), x)
    return incoming / incoming.size(1)

def node2edge(x, rel_rec, rel_send):
    # NOTE: Assumes that we have the same graph across all samples.
    receivers = torch.matmul(rel_rec, x)
    senders = torch.matmul(rel_send, x)
    edges = torch.cat([receivers,senders], dim=-1)
    return edges


class ODEModel(nn.Module):
    def __init__(self, func, B, n, obs_vel, order=1, n_latent=-1, solver='rk4',
                      n_aug_dyn=0, n_aug_st=0, encoder=True, decoder=False,
                      M=750,      H=256,     L=2,     act='softplus',     diag_covar=True,
                      M_int=2250, H_int=512, L_int=2, act_int='softplus', diag_covar_int=True,
                      T_iv=5,   rnn_output_iv=20,  H_iv=100, act_iv='relu',
                      T_glv=49, rnn_output_glv=25, H_glv=50, act_glv='elu'):
        '''
        Parameters
        ----------
        func           - str  - the function approximator, should be in ['nn','bnn','rbf','0.5','1.5','2.5']
        B              - int  - number of objects
        n              - int  - position/velocity dimensionality
        obs_vel        - bool - denotes whether velocities are observed or not, needed for encoding
        order          - int  - the order of the dynamics
        n_latent       - int  - latent position/velocity dimensionality (replaces n)
        solver         - str  - ODE solver
        n_aug_dyn      - int  - the number of augmented dynamic dimensions
        n_aug_st       - int  - the number of augmented static dimensions
        encoder        - bool  - denotes whether an encoder is used
        decoder        - bool  - denotes whether an decoder is used
        
        M              - int  - the number of inducing points    of the independent kinematics function
        H              - int  - number of hidden neurons         of the independent kinematics function
        L              - int  - number of hidden layers          of the independent kinematics function
        act            - str  - activation function              of the independent kinematics function
        diag_covar     - bool - variational posterior covariance of the independent kinematics function is diagonal or not
        
        M_int          - int  - the number of inducing points    of the interaction function
        H_int          - int  - number of hidden neurons         of the interaction function
        L_int          - int  - number of hidden layers          of the interaction function
        act_int        - str  - activation function              of the interaction function
        diag_covar_int - bool - variational posterior covariance of the interaction function is diagonal or not
        
        T_iv           - int  - the number of initial data points given as input to the initial value RNN
        rnn_output_iv  - int  - initial value rnn encoder output dimension
        H_iv           - int  - number of hidden neurons      of the MLP following the initial value RNN output
        act_iv         - str  - activation function           of the MLP following the initial value RNN output
        
        T_glv          - int  - the number of initial data points given as input to the global latent variable RNN
        rnn_output_glv - int  - global latent variable RNN encoder output dimension
        H_glv          - int  - number of hidden neurons      of the MLP following global latent variable RNN output
        act_glv        - str  - activation function           of the MLP following global latent variable RNN output
        '''
        super().__init__()
        assert n_latent<0 or encoder, 'must use encoder if learning a latent ODE system'
        assert    obs_vel or encoder, 'must use encoder velocities are not observed'
        assert func  in ['nn','bnn','rbf','0.5','1.5','2.5'], 'wrong function approximator'
        # model parameters
        self.func  = func
        self.B     = B
        self.n     = n
        self.order = order
        self.n_latent    = n_latent
        self.solver      = solver
        self.n_aug_dyn   = n_aug_dyn
        self.n_aug_st    = n_aug_st
        self.decoder     = decoder
        self.encoder     = encoder
        self.obs_vel     = obs_vel
        self.solver      = 'rk4'
        # independent kinematics function parameters
        self.M   = M
        self.H   = H
        self.L   = L
        self.act = act
        self.diag_covar = diag_covar
        # interaction function parameters
        self.M_int   = M_int
        self.H_int   = H_int
        self.L_int   = L_int
        self.act_int = act_int
        self.diag_covar_int = diag_covar_int
        # initial value encoder parameters 
        self.T_iv   = T_iv
        self.H_iv   = H_iv
        self.act_iv = act_iv
        self.rnn_output_iv  = rnn_output_iv
        # global latent variable parameters
        self.T_glv   = T_glv
        self.H_glv   = H_glv
        self.act_glv = act_glv
        self.rnn_output_glv  = rnn_output_glv
        
        # differential functions
        if n_latent>0:
            print(f'Replacing the position/velocity dimensionality with {n_latent}')
            n_ode = n_latent
        else:
            n_ode = n
        if B==1:
            self.f = BaseODEModel(func, n_ode, order=order, n_aug_dyn=n_aug_dyn, n_aug_st=n_aug_st, \
                            solver=solver, M=M, H=H, L=L, act=act, diag_covar=diag_covar)
        elif B>1:
            self.f = BaseIODEModel(func, B, n_ode, order=order, n_aug_dyn=n_aug_dyn, n_aug_st=n_aug_st, \
                            solver=solver, M=M, H=H, L=L, act=act, diag_covar=diag_covar,\
                            M_int=M_int, H_int=H_int, L_int=L_int, act_int=act_int, diag_covar_int=diag_covar_int)
                
        # initial value encoder
        self.nobs = (1+self.obs_vel) * self.n
        if self.encoder:
            print('Using an encoder for initial values')
            enc_outs = [it for it in [n_ode,n_ode,n_aug_dyn] if it>0] # position, velocity, aug_dyn, aug_st
            self.iv_encoder = GRUEncoder(enc_outs, self.nobs, rnn_output_size=rnn_output_iv, H=H_iv, act=act_iv)
        else:
            print('Skipping initial value encoder, using the initial values in input sequences')
            
        # global latent variable encoder
        if n_aug_st>0:
            print('Using an encoder for global latent variables')
            self.glv_encoder = GRUEncoder([B*n_aug_st], B*self.nobs, rnn_output_size=rnn_output_glv, H=H_glv, act=act_glv)
        else:
            print('No global latent variable is used')
        
        # decoder
        self.sp = torch.nn.Softplus()
        if decoder:
            print('Using an decoder for map latent states into observated space')
            nin = 2*n_ode + n_aug_dyn + n_aug_st
            self._decoder = MLP(nin, 2*self.nobs, n_hid_layers=0, act='linear')
        else:
            self.__sn = torch.nn.Parameter(-2 * torch.ones(self.nobs), requires_grad=True)
            
    @property
    def sn(self):
        return self.sp(self.__sn)  
    
    @property
    def device(self):
        return self.f.device
        
    @property
    def is_gp(self):
        return self.f.is_gp
    
    def encode(self, Y):
        ''' Y - [N,b,T,n]
            returns - array of MultivariateNormals of shape Nb,n
        '''
        [N,b,T,n] = Y.shape
        Y_ = Y.reshape(N*b,T,n) # Nb,T,n
        z0 = self.iv_encoder(Y_[:,:self.T_iv]) # Nb,n
        return z0
    
    def get_latent_variable(self, Y, zs=None):
        [N,b,_,n] = Y.shape
        # if no latent dim, skip
        if self.n_aug_st==0:
            return None,None
        # if zs is already given, no computation
        if zs is not None:
            return zs.reshape(N*b,self.n_aug_st), None
        # else, encode and sample
        qzs = self.glv_encoder(Y[:,:,:self.T_glv].permute(0,2,1,3).reshape(N, self.T_glv, b*n), False)[0] # Nb,n
        zs  = qzs.rsample() # [N,B,n_aug_st]
        return zs.reshape(N*b,self.n_aug_st),qzs # [NB,n_aug_st]
            
    def _latent_integrate(self, Y, ts, zs=None):
        ''' 
            This function should not be called outside of this class!
            Inputs:
                Y  - a sequence [N,b,T,n] or [N,T,n]
                ts - integration time points
                zs - optional static information - None or [N,B,n_aug_st]
            Returns:
                yhat - if use decoder  - latent trajectory [N,b,T,n] or [N,T,n]
                     - if not decoder  - Normal of shape   [N,b,T,n] or [N,T,n]
                qz0  - List of MultivariateNormals of shape [Nb,_]
        '''
        # add the object dimension if the input is 3D
        y3d = Y.ndim==3
        if y3d:
            Y = Y.unsqueeze(1)
        [N,b,_,n] = Y.shape
        T = len(ts)
        # static encoding
        zs = self.get_latent_variable(Y,zs)[0]
        # dynamic encoding
        if self.encoder:
            qz0 = self.encode(Y)
            qz0_d = qz0[:2] if self.n_aug_dyn==0 else qz0[:3]
            zd_0  = torch.cat([qz0_.rsample() for qz0_ in qz0_d],-1) # [Nb,2n] or [Nb,2n+n_aug_dyn]
        else:
            qz0 = []
            zd_0 = Y[:,:,0].reshape(N*b,n)
        # integrate
        zt = self.f.integrate(zd_0, ts, zs).reshape(N,b,T,-1) # N,b,T,n
        if y3d:
            zt = zt.squeeze(1) # discard the object dimension if one object
        # decode
        if self.decoder:
            dec_out = self._decoder(zt)
            mean,var = dec_out.split(2*[dec_out.shape[-1]//2],-1)
            yhat = torch.distributions.Normal(mean,self.sp(var))
        else:
            yhat = zt[...,:len(self.sn)]
        return yhat,qz0

    def integrate(self, Y, ts, zs=None):
        ''' 
            Returns the states in the integrated trajectory that correspond to observations
            Inputs:
                Y  - a sequence [N,b,T,n] or [N,T,n]
                ts - integration time points
                zs - optional static information - None or [N,B,n_aug_st]
            Returns:
                yhat - predictions [N,b,T,n] or [N,T,n]
        '''
        yhat = self._latent_integrate(Y, ts, zs)[0]
        return yhat.mean if self.decoder else yhat
    
    def minibatch_elbo(self, Y, ts, zs=None, Tsub=-1):
        '''
            Inputs:
                Y  - input sequence [N,b,T,n] or [N,T,n]
                ts - integration time points
                zs - optional static information - None or [N,B,n_aug_st]
                Tsub - optional subsequence length used for likelihood 
            Returns:
                lhood - torch.tensor (scaled according to minibatch length Tsub)
                kl_z  - torch.tensor
                kl_f  - torch.tensor
        '''
        # static encoding on whole sequence
        [N,B,T,n] = Y.shape
        zs,qzs = self.get_latent_variable(Y,zs)
        # minibatch over time
        if Tsub > 1:
            t0s = torch.randint(T-Tsub,[N],device=Y.device)
            Y   = torch.stack([Y[i,...,t0:t0+Tsub,:] for i,t0 in enumerate(t0s)])
            ts  = ts[:Tsub]
        else:
            Tsub = T
        # integrate
        Ypred,qz0 = self._latent_integrate(Y, ts, zs)
        # lhood
        if self.decoder:
            lhood = Ypred.log_prob(Y).sum()
        else:
            # Y,Ypred = Y/Y.max(0)[0].abs(), Ypred/Y.max(0)[0].abs()
            mvn   = torch.distributions.MultivariateNormal(Y,self.sn.diag())
            lhood = mvn.log_prob(Ypred).sum()
        # kl_z
        qz = qz0 + [qzs] if qzs is not None else qz0
        kl_z = []
        for qz_ in qz:
            d     = qz_.mean.shape[-1]
            zeros = torch.zeros(d, dtype=Y.dtype, device=self.device)
            ones  = torch.ones (d, dtype=Y.dtype, device=self.device)
            N0I   = torch.distributions.Normal(zeros,ones)
            kl_z.append(kl(qz_,N0I).reshape(-1))
        kl_z = torch.cat(kl_z,-1).sum() if len(kl_z)>0 else torch.zeros(1,device=self.device)
        # kl_f
        kl_f = self.f.kl()
        if self.func=='bnn':
            kl_f = kl_f * self.n * self.B * 2
        return lhood/Tsub*T, kl_z, kl_f 
        
    def save(self, fname, results=None):
        fname = fname if fname.endswith('.pkl') else fname+'.pkl'
        results = None if results is None else [r.detach().cpu() for r in results]
        with open(fname, 'wb') as f: 
            pickle.dump([self.func, self.B, self.n, self.obs_vel, self.order, self.n_latent, 
                      self.solver, self.n_aug_dyn, self.n_aug_st, self.encoder, self.decoder,
                      self.M,     self.H,     self.L,     self.act,     self.diag_covar,
                      self.M_int, self.H_int, self.L_int, self.act_int, self.diag_covar_int,
                      self.T_iv,  self.rnn_output_iv,  self.H_iv,  self.act_iv,
                      self.T_glv, self.rnn_output_glv, self.H_glv, self.act_glv, self.state_dict(), results], f)
                
    @staticmethod
    def load(fname):
        ''' Loads a saved model into CPU. 
            Below CPU_Unpickler class was needed to load a GPU-trained model into CPU
        '''
        fname = fname if fname.endswith('.pkl') else fname+'.pkl'
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else: 
                    return super().find_class(module, name)
        with open(fname, 'rb') as f:
            outputs = CPU_Unpickler(f).load()
            func, B, n, obs_vel, order, n_latent, solver, n_aug_dyn, n_aug_st, encoder, decoder,
            M, H, L, act, diag_covar, M_int, H_int, L_int, act_int, diag_covar_int,
            T_iv, rnn_output_iv, H_iv, act_iv, T_glv, rnn_output_glv, H_glv, act_glv, 
            state_dict, results = outputs   
            
        ode_model = ODEModel(func, B, n, obs_vel, orderorder, n_latent=n_latent, solver=solver,
                      n_aug_dyn=n_aug_dyn, n_aug_st=n_aug_st, encoder=encoder, decoder=decoder,
                      M=M,      H=H,     L=L,     act=act,     diag_covar=diag_covar,
                      M_int=M_int, H_int=H_int, L_int=L_int, act_int=act_int, diag_covar_int=diag_covar_int,
                      T_iv=T_iv,   rnn_output_iv=rnn_output_iv,   H_iv=H_iv  , act_iv=act_iv,
                      T_glv=T_glv, rnn_output_glv=rnn_output_glv, H_glv=H_glv, act_glv=act_glv)
        ode_model.load_state_dict(state_dict)
        ode_model.eval()
        return ode_model

    @property
    def name(self):
        t1 = self.f.name
        t2 = '-ivenc'  if self.iv_encoder else ''
        t3 = '-glvenc' if self.n_aug_st>0 else ''
        t4 = '-dec'    if self.decoder    else ''
        return t1+t2+t3+t4 

    def fix_gpytorch_cache(self, it):
        self.f.fix_gpytorch_cache(it)
    
    
class BaseODEModel(nn.Module):
    def __init__(self, func, n, order=1, solver='rk4', n_aug_dyn=0, n_aug_st=0, \
                      M=750, H=256, L=2, act='softplus', diag_covar=True):
        super().__init__()
        self.n = n # position/velocity dimensionality
        self.M = M
        self.H = H
        self.L = L
        self.func  = func
        self.order = order
        self.act   = act
        self.diag_covar  = diag_covar
        self.n_aug_dyn   = n_aug_dyn
        self.n_aug_st    = n_aug_st
        self.solver      = solver
            
        # set the input and output dimensions
        nin     = 2*self.n + n_aug_dyn + n_aug_st
        n_f_out = 2*self.n + n_aug_dyn if order==1 else self.n + n_aug_dyn
            
        # build the differential function
        self.build_differential_function(nin, n_f_out)
            
    def build_differential_function(self, nin, n_f_out):
        if self.func=='nn':
            self.f = MLP(nin, n_f_out, n_hid_layers=self.L, H=self.H, act=self.act)
        elif self.func=='bnn':
            self.f = BNN(nin, n_f_out, n_hid_layers=self.L, H=self.H, act=self.act)
        else:
            inducing_points = torch.rand(self.M, nin) * 4 - 2
            self.f = GPModel(inducing_points=inducing_points, nout=n_f_out, \
                              kernel=self.func, diag_covar=self.diag_covar)
    
    def kl(self):
        if self.func!='bnn':
            return self.f.kl()
        else:
            return self.f.kl().mean()
    
    @property
    def device(self):
        return self.f.device
    
    def draw_f(self):
        if self.is_gp:
            f    = self.f.post_draw(S=100,P=1)
            odef = lambda x: f(x).squeeze(0) # f_s
        elif self.func=='bnn':
            odef = self.f.draw_f()
        else:
            odef = self.f
        return odef
            
    def build_ode_func(self, z_static=None):
        ''' z_static is either None or [N,n_aug_st] '''
        odef = self.draw_f()
        def odef_order_wrapper(t,z):
            ''' z - [N,n]
                returns - [N,n]
            '''
            vel     = z[:,self.n:2*self.n]
            z       = torch.cat([z,z_static],-1) if z_static is not None else z
            dz_self = odef(z)
            return dz_self if self.order==1 else torch.cat([vel,dz_self],-1)
        return odef_order_wrapper
            
    def integrate(self, zd_0, ts, zs_0=None):
        ''' 
            Inputs:
                zd_0 - dynamics initial values, [N, 2*self.n + n_aug_dyn] (position, velocity and augmented dynamics)
                zd_0 - static   initial values, [N, n_aug_st] or None
                ts   - integration time points
            Returns:
                yhat - [N,T,2*self.n+n_aug_dyn]
        '''
        # build differential function
        odef = self.build_ode_func(zs_0)
        # integrate
        yhat = aca_odesolve(odef, zd_0, ts, method=self.solver)
        return yhat.permute(1,0,2) # Nb,T_,nout
    
    @property
    def is_gp(self):
        return isinstance(self.f, GPModel)
    
    def __repr__(self):
        return self.f.__repr__()
    
    def train(self,train=False):
        if self.is_gp:
            _ = self.f.train(train)
    
    def fix_gpytorch_cache(self, it):
        if self.is_gp:
            self.f.fix_gpytorch_cache(it)
                
    @property
    def name(self):
        return self.func + f'-ord{self.order}'



class BaseIODEModel(nn.Module):  
    def __init__(self, func, B, n, order=1, solver='rk4', n_aug_dyn=0, n_aug_st=0, 
                      M=750,      H=256,     L=2,     act='softplus',     diag_covar=True,
                      M_int=2250, H_int=512, L_int=2, act_int='softplus', diag_covar_int=True):
        super().__init__()
        assert B>1, 'number of objects should be bigger than 1'
        self.func  = func
        self.B     = B
        self.n     = n
        self.order = order
        self.n_aug_dyn = n_aug_dyn
        self.n_aug_st  = n_aug_st
        self.solver    = solver
        # independent dynamics parameters
        self.M   = M
        self.H   = H
        self.L   = L
        self.act = act
        self.diag_covar = diag_covar
        # interaction function parameters
        self.M_int   = M_int
        self.H_int   = H_int
        self.L_int   = L_int
        self.act_int = act_int
        self.diag_covar_int = diag_covar_int
            
        # set the input and output dimensions
        nin     = 2*self.n + n_aug_dyn + n_aug_st
        n_f_out = 2*self.n + n_aug_dyn if order==1 else self.n + n_aug_dyn
        
        # build the independent dynamics
        self.build_independent_function(nin, n_f_out)
            
        # build the interaction function
        self.build_interaction_units(nin, n_f_out)
            
        off_diag = np.ones([B, B]) - np.eye(B)
        rel_rec  = np.array(encode_onehot(np.where(off_diag)[0])).astype(float)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1])).astype(float)
        self.register_buffer("rel_rec", torch.tensor(rel_rec))
        self.register_buffer("rel_send", torch.tensor(rel_send))
    
    def build_independent_function(self, nin, n_f_out):
        if self.func=='nn':
            self.f = MLP(nin, n_f_out, n_hid_layers=self.L, H=self.H, act=self.act)
        elif self.func=='bnn':
            self.f = BNN(nin, n_f_out, n_hid_layers=self.L, H=self.H, act=self.act)
        else:
            inducing_points = torch.rand(self.M, nin) * 4 - 2
            self.f = GPModel(inducing_points=inducing_points, nout=n_f_out, \
                              kernel=self.func, diag_covar=self.diag_covar)
        
    
    def build_interaction_units(self, nin, n_f_out):
        # build input range
        nin = 2*nin - self.n # due to interaction parameterization
        if self.func=='nn':
            self.fint = MLP(nin, n_f_out, n_hid_layers=self.L_int, H=self.H_int, act=self.act_int).to(self.device)
        elif self.func=='bnn':
            self.fint = BNN(nin, n_f_out, n_hid_layers=self.L_int, H=self.H_int, act=self.act_int).to(self.device)
        else:
            inducing_points = torch.rand(self.M_int, nin) * 4 - 2
            self.fint = GPModel(inducing_points=inducing_points, nout=n_f_out, \
                              kernel=self.func, diag_covar=self.diag_covar_int) #.to(self.device)
            
    
    def kl(self):
        if self.func!='bnn':
            return self.f.kl() + self.fint.kl()
        else:
            return self.f.kl().mean() + self.fint.kl().mean()
    
    @property
    def device(self):
        return self.f.device
    
    def draw_f(self):
        if self.is_gp:
            fdraw = self.f.post_draw(S=100,P=1)
            f = lambda x: fdraw(x).squeeze(0)
            fdrawint = self.fint.post_draw(S=100,P=1)
            fint = lambda x: fdrawint(x).squeeze(0)
        elif self.func=='bnn':
            f = self.f.draw_f()
            fint = self.fint.draw_f()
        else:
            f = self.f
            fint = self.fint
        return f,fint
    
    def build_ode_func(self, zs=None):
        ''' zs is either None or [Nb,n_aug_st] '''
        # set up the interaction kernel function
        f,fint = self.draw_f()
        def odefint(x): # x is [_,2n]
            p1,v1,d1,c1,p2,d2,v2,c2 = x.split(2*[self.n,self.n,self.n_aug_dyn,self.n_aug_st],-1)
            inp = torch.cat([p1-p2,v1,d1,c1,v2,d2,c2],-1) 
            return fint(inp) # f_b
        def odef_order_wrapper(t,z):
            ''' z - [NB,n]
                returns - [NB,n]
            '''
            # auto-regressive dynamics
            vel     = z[:,self.n:2*self.n]
            z_conc  = torch.cat([z,zs],-1) if zs is not None else z
            dz_self = f(z_conc) # Nb,_
            # interacting dynamics
            z_conc  = z_conc.reshape(-1, self.B, z_conc.shape[-1]) # N,b,n
            [N,b,n] = z_conc.shape
            other_pairs = node2edge(z_conc, self.rel_rec, self.rel_send).permute(1,0,2) # b*(b-1),N,2n
            dz_int  = odefint(other_pairs.reshape(b*(b-1)*N,2*n)) # b*(b-1)*N,n
            dz_int  = dz_int.reshape(b,b-1,N,-1) # b,(b-1),N,n
            dz_self = dz_self.reshape(N,b,-1).permute(1,0,2) # b,N,n
            dz_int = dz_int.sum(1) # sum neighboring messages
            dz = (dz_self + dz_int).permute(1,0,2).reshape(N*b,-1)
            return dz if self.order==1 else torch.cat([vel,dz],-1) # _,n
        return odef_order_wrapper

    def integrate(self, zd_0, ts, zs_0=None):
        ''' 
            Inputs:
                zd_0 - dynamics initial values, [NB, 2*self.n + n_aug_dyn] (position, velocity and augmented dynamics)
                zd_0 - static   initial values, [NB, n_aug_st] or None
                ts   - integration time points
            Returns:
                yhat - [NB,T,2*self.n + n_aug_dyn]
        '''
        # build differential function
        odef = self.build_ode_func(zs_0)
        # integrate
        yhat = aca_odesolve(odef, zd_0, ts, method=self.solver)
        return yhat.permute(1,0,2) # NB,T_,nout

    @property
    def is_gp(self):
        return isinstance(self.f, GPModel)
    
    def __repr__(self):
        return self.f.__repr__() + '\n' + self.fint.__repr__()
    
    def train(self, train=False):
        if self.is_gp:
            _ = self.f.train(train)
            _ = self.fint.train(train)
    
    def fix_gpytorch_cache(self, it):
        if self.is_gp:
            self.f.fix_gpytorch_cache(it)
            self.fint.fix_gpytorch_cache(it)
                
    @property
    def name(self):
        t1 = f'{self.B}'
        t2 = f'-{self.func}'
        t3 = f'-ord{self.order}'
        return t1+t2+t3