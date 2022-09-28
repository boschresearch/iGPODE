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

"""
The following class is adapted from ODE-RL V 1.0
(https://github.com/cagatayyildiz/oderl
Copyright 2022 Cagatay Yildiz, MIT license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree)
to test Bayesian neural network based baseline method.
"""
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import MultivariateNormal, Normal, Bernoulli, kl_divergence as kl
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


def get_act(act="relu"):
    if act=="relu":         return nn.ReLU()
    elif act=="elu":        return nn.ELU()
    elif act=="celu":       return nn.CELU()
    elif act=="leaky_relu": return nn.LeakyReLU()
    elif act=="sigmoid":    return nn.Sigmoid()
    elif act=="tanh":       return nn.Tanh()
    elif act=="sin":        return torch.sin
    elif act=="linear":     return nn.Identity()
    elif act=='softplus':   return nn.modules.activation.Softplus()
    elif act=='swish':      return lambda x: x*torch.sigmoid(x)
    elif act=='lipswish':   return lambda x: 0.909 * torch.nn.functional.silu(x)
    else:                   return None


class BNN(nn.Module):
    def __init__(self, n_in, n_out, n_hid_layers=2, H=100, act='relu', logsig0=-3, bnn=True,  var_apr='mf'):
        super().__init__()
        layers_dim = [n_in] + n_hid_layers*[H] + [n_out]
        self.weight_mus  = nn.ParameterList([])
        self.bias_mus    = nn.ParameterList([])
        self.sp   = torch.nn.Softplus()
        self.acts = []
        self.act  = act 
        self.bnn  = bnn
        self.var_apr = var_apr
        for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
            self.weight_mus.append(Parameter(torch.Tensor(n_in, n_out)))
            self.bias_mus.append(Parameter(torch.Tensor(1,n_out)))
            self.acts.append(get_act(act) if i<n_hid_layers else get_act('linear')) # no act. in final layer
        if bnn:
            self.weight_logsigs = nn.ParameterList([])
            self.bias_logsigs   = nn.ParameterList([])
            self.logsig0 = logsig0
            for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
                self.weight_logsigs.append(Parameter(torch.Tensor(n_in, n_out)))
                self.bias_logsigs.append(Parameter(torch.Tensor(1,n_out)))
        self.reset_parameters()

    @property
    def device(self):
        return self.weight_mus[0].device

    @property
    def dtype(self):
        return self.weight_mus[0].dtype

    def __transform_sig(self,sig):
        return self.sp(sig) + 1e-3

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weight_mus,self.bias_mus)):
            nn.init.xavier_uniform_(weight,gain)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
        if self.bnn:
            for w,b in zip(self.weight_logsigs,self.bias_logsigs):
                nn.init.uniform_(w,self.logsig0-1,self.logsig0+1)
                nn.init.uniform_(b,self.logsig0-1,self.logsig0+1)

    def draw_noise(self, L):
        P = parameters_to_vector(self.parameters()).numel() // 2 # single noise term needed per (mean,var) pair
        noise = torch.randn([L,P], device=self.device, dtype=self.dtype)
        if self.var_apr == 'mf':
            return noise
        elif self.var_apr == 'radial':
            noise /= noise.norm(dim=1,keepdim=True)
            r = torch.randn([L,1], device=self.device, dtype=self.dtype)
            return noise * r

    def __sample_weights(self, L):
        if self.bnn:
            noise_vec = self.draw_noise(L) # L,P
            weights = []
            i = 0
            for weight_mu,weight_sig in zip(self.weight_mus,self.weight_logsigs):
                p = weight_mu.numel()
                weights.append( weight_mu + noise_vec[:,i:i+p].view(L,*weight_mu.shape)*self.__transform_sig(weight_sig) )
                i += p
            biases = []
            for bias_mu,bias_sig in zip(self.bias_mus,self.bias_logsigs):
                p = bias_mu.numel()
                biases.append( bias_mu + noise_vec[:,i:i+p].view(L,*bias_mu.shape)*self.__transform_sig(bias_sig) )
                i += p
            else:
                biases = [torch.zeros([L,1,weight_mu.shape[1]], device=self.device, dtype=self.dtype)*1.0 \
                    for weight_mu,bias_mu in zip(self.weight_mus,self.bias_mus)] # list of zeros
        else:
            raise ValueError('This is a NN, not a BNN!')
        return weights,biases

    def draw_f(self, L=1, mean=False):
        """ 
            x=[N,n] & mean=True ---> out=[N,n]
            x=[N,n] & bnn=False ---> out=[N,n]
            x=[N,n] & L=1 ---> out=[N,n]
            x=[N,n] & L>1 ---> out=[L,N,n]
            x=[L,N,n] -------> out=[L,N,n]
        """
        if mean or not self.bnn:
            def f(x):
                for (weight,bias,act) in zip(self.weight_mus,self.bias_mus,self.acts):
                    x = act(F.linear(x,weight.T,bias))
                return x
            return f
        else:
            weights,biases = self.__sample_weights(L)
            def f(x):
                x2d = x.ndim==2
                if x2d:
                    x = x.unsqueeze(0) if L==1 else torch.stack([x]*L) # [L,N,n]
                for (weight,bias,act) in zip(weights,biases,self.acts):
                    x = act(torch.baddbmm(bias, x, weight))
                return x.squeeze(0) if x2d and L==1 else x
            return f

    def forward(self, x, L=1, mean=False):
        return self.draw_f(L,mean)(x)

    def kl(self, L=100):
        if not self.bnn:
            return torch.zeros([1],device=self.device)*1.0
        if self.var_apr == 'mf':
            mus      = [weight_mu.view([-1]) for weight_mu in self.weight_mus]
            logsigs  = [weight_logsig.view([-1]) for weight_logsig in self.weight_logsigs]
            mus     += [bias_mu.view([-1]) for bias_mu in self.bias_mus]
            logsigs += [bias_logsigs.view([-1]) for bias_logsigs in self.bias_logsigs]
            mus  = torch.cat(mus)
            sigs = self.__transform_sig(torch.cat(logsigs))
            q = Normal(mus,sigs)
            N = Normal(torch.zeros_like(mus, device=self.device, dtype=self.dtype), \
                       torch.ones_like(mus, device=self.device, dtype=self.dtype)
                )
            return kl(q,N)
        elif self.var_apr == 'radial':
            weights,biases = self.__sample_weights(L)
            weights = torch.cat([w.view(L,-1) for w in weights],1)
            sigs = torch.cat([weight_sig.view([-1]) for weight_sig in self.weight_logsigs])
            biases = torch.cat([b.view(L,-1) for b in biases],1)
            weights = torch.cat([weights,biases],1)
            bias_sigs = torch.cat([bias_sig.view([-1]) for bias_sig in self.bias_logsigs])
            sigs = torch.cat([sigs,bias_sigs])
            cross_entr = -(weights**2).mean(0)/2 - np.log(2*np.pi)
            entr = -self.__transform_sig(sigs).log()
            return entr - cross_entr
    
    def __repr__(self):
        str_ = 'BNN\n' if self.bnn else 'NN\n'
        for i,(weight,act) in enumerate(zip(self.weight_mus,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_


