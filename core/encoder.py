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
The following class is adapted from latent ODE
(https://github.com/YuliaRubanova/latent_ode/blob/master/lib/encoder_decoder.py
Copyright unknown, unknown license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree)
to encode observed sequences.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules.rnn import GRU
import torch.nn.functional as F

# import os, sys
# sys.path.append("..") # Adds higher directory to python modules path.
from core.mlp import MLP

class GRUEncoder(nn.Module):
    def __init__(self, output_dims, input_dim, rnn_output_size=20, H=50, act='relu'):
        super(GRUEncoder, self).__init__()
        assert type(output_dims) == list
        self.input_dim       = input_dim
        self.output_dims     = output_dims
        self.rnn_output_size = rnn_output_size # number of outputs per output_dim

        rnn_hidden_to_latent_nets = [nn.Sequential(nn.Linear(self.rnn_output_size, H), 
                                                    nn.ReLU(True) if act=='relu' else nn.ELU(True),
                                                    nn.Linear(H, 2*d))
                                    for d in self.output_dims]
        
        self.rnn_hiddens_to_latents = nn.ModuleList(rnn_hidden_to_latent_nets)
        self.gru = GRU(self.input_dim, self.rnn_output_size*len(output_dims))

    def forward(self, data, run_backwards=True):
        assert (not torch.isnan(data).any())

        data = data.permute(1,0,2)  # (N, T, D) -> (T, N, D)
        if run_backwards:
            data = torch.flip(data, [0])  # (T, N, D)

        outputs, _ = self.gru(data)  # (T, N, K)

        idx = 0
        q_outputs = []
        for (dim, net) in zip(self.output_dims, self.rnn_hiddens_to_latents):
            output = outputs[-1][:, idx:idx+self.rnn_output_size]
            idx += self.rnn_output_size
            net_out   = net(output)
            mu, sigma = net_out[...,:dim],net_out[...,dim:]
            sigma = F.softplus(sigma) + 1e-3
            assert (not torch.isnan(mu).any())
            assert (not torch.isnan(sigma).any())
            q_outputs.append(Normal(mu, sigma))

        return q_outputs











































