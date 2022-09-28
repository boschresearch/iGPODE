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
import torch

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from data.bouncing_balls import BouncingBallsSim
from data.charged_particles import ChargedParticlesSim

def generate_data(dataset, A=1, dt=0.5, T=100, N=(1000,200,200), SIZE=5.0, noise_level='zero', \
                  obs_vel=True, probs=[0.5, 0, 0.5], device='cpu', dtype=torch.float32):
    '''
        dataset - ['balls','charges']
        A       - number of objects
        dt      - time difference between observations
        N       - triplet denoting the number of train/valid/test sequences
        SIZE    - box size
        noise_level - 'zero' for charges, ['zero','low','high'] for balls
        obs_vel - True if velocities are to be returned
        probs   - charge probabilities (minus,null,plus)
    '''
    assert dataset in ['balls', 'charges']
    
    Ntr, Nval, Ntest = N
    N = Ntr + Nval + Ntest
    s = np.zeros((N,T,A,4),dtype=np.float32) # ball locations
    sig = 0.0 # observation noise parameters
    
    Z   = np.zeros((N,A,1), dtype=np.float32) # charge information for the charges dataset
    if dataset=='balls':
        sim = BouncingBallsSim(box_size=SIZE)
        if noise_level=='zero':
            sig = 0.0
        elif noise_level=='low':
            sig = 0.02
        elif noise_level=='high':
            sig = 0.04
    elif dataset=='charges':
        sim = ChargedParticlesSim(box_size=SIZE, loc_std=1., vel_norm=.5, interaction_strength=1.)
        
    for i in range(N):
        if i%100==0:
            print(f'iteration {i}...')
        s_,z_ = sim.sample_trajectory(A=A, T=T, dt=dt, probs=probs)
        s[i], Z[i,:,0] = s_, z_
        
    Z,s = torch.tensor(Z).to(device).to(dtype),torch.tensor(s).permute([0,2,1,3]).to(device).to(dtype) # N,A,T,4
    Ztr,Zval,Ztest = Z[:Ntr], Z[Ntr:Ntr+Nval], Z[-Ntest:]
    s_tr_clean,s_val_clean,s_test_clean = s[:Ntr], s[Ntr:Ntr+Nval], s[-Ntest:]
    data_range = s.reshape(-1,4).max(0)[0]-s.reshape(-1,4).min(0)[0] # 4
    s_tr   = s_tr_clean   + sig * data_range * torch.randn_like(s_tr_clean)
    s_val  = s_val_clean  + sig * data_range * torch.randn_like(s_val_clean)
    s_test = s_test_clean + sig * data_range * torch.randn_like(s_test_clean)
    ts = torch.arange(T) * dt
    
    nin = 2 + 2*obs_vel
    return Ztr, Zval, Ztest, s_tr[...,:nin], s_val[...,:nin], s_test[...,:nin], ts