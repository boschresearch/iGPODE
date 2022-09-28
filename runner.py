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

import os, numpy as np
import argparse
import torch

import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ode_model           import ODEModel
from utils.experiment    import Experiment
from utils.plot_utils    import plot_predictions, plot_2D_system
from utils.data_utils    import generate_data
from utils.train_utils   import optimize_model

parser = argparse.ArgumentParser()
# computer settings
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--use-float', action='store_true', default=False,
                    help='Disables the use of doubles')
# train and log settings
parser.add_argument('--exp-name', type=str, default='dummy-exp',
                    help='Experiment name')  
parser.add_argument('--num-iter', type=int, default=10000,
                    help='Number of iterations per training round')
parser.add_argument('--batch-size', type=int, default=50,
                    help='Number of samples per batch.')
parser.add_argument('--log-every', type=int, default=50,
                    help='How many iterations to wait before printing the train loss')
# experiment settings
parser.add_argument('--dataset', type=str, default='balls',
                    help='Dataset name')  
parser.add_argument('--N', type=int, default=100,
                    help='Number of training sequences')  
parser.add_argument('--noise-level', type=str, default='zero',
                    help='Dataset noise level')      
parser.add_argument('--no-vel', action='store_true', default=False,
                    help='Velocities are observed or not')       
# model settings       
parser.add_argument('--ftype', type=str, default='rbf',
                    help='Function approximator')
parser.add_argument('--order', type=int, default=1,
                    help='Order or the dynamics')
parser.add_argument('--num-latent', type=int, default=-1,
                    help='Number of latent dimensionality (-1 is not used)')
parser.add_argument('--solver', type=str, default='rk4',
                    help='ODE solver')  
parser.add_argument('--n-aug-dyn', type=int, default=0,
                    help='The dimensionality of additional latent dynamics states')
parser.add_argument('--n-aug-st', type=int, default=0,
                    help='The dimensionality of global latent variables')
parser.add_argument('--no-encoder', action='store_true', default=False,
                    help='Initial value encoder is used or not')     
parser.add_argument('--decoder', action='store_true', default=False,
                    help='Decoder used or not')   
# independent kinematics function parameters  
parser.add_argument('--M', type=int, default=750,
                    help='Number of inducing points used in the independent kinematics function')
parser.add_argument('--H', type=int, default=256,
                    help='Number of hidden neurons used in the independent kinematics function')
parser.add_argument('--L', type=int, default=2,
                    help='Number of hidden layers used in the independent kinematics function')
parser.add_argument('--act', type=str, default='softplus',
                    help='Activation function used in the independent kinematics function')
parser.add_argument('--full-covar', action='store_true', default=False,
                    help='Diagonal covariance approximation used in the independent kinematics function')
# interaction function parameters
parser.add_argument('--M-int', type=int, default=2250,
                    help='Number of inducing points used in the interaction function')
parser.add_argument('--H-int', type=int, default=512,
                    help='Number of hidden neurons used in the interaction function')
parser.add_argument('--L-int', type=int, default=2,
                    help='Number of hidden layers used in the interaction function')
parser.add_argument('--act-int', type=str, default='softplus',
                    help='Activation function used in the interaction function')
parser.add_argument('--full-covar-int', action='store_true', default=False,
                    help='Diagonal covariance approximation used in the interaction function')     
# initial value encoder parameters
parser.add_argument('--T-iv', type=int, default=5,
                    help='Number of time points initial value encoder takes as input')
parser.add_argument('--rnn-output-iv', type=int, default=20,
                    help='Output dimensionality of the RNN in the initial value encoder')
parser.add_argument('--H-iv', type=int, default=100,
                    help='Number of hidden neurons of the MLP in the initial value encoder')
parser.add_argument('--act-iv', type=str, default='relu',
                    help='Activation function of the MLP in the initial value encoder')
# global latent variable encoder parameters
parser.add_argument('--T-glv', type=int, default=5,
                    help='Number of time points global latent variable encoder takes as input')
parser.add_argument('--rnn-output-glv', type=int, default=25,
                    help='Output dimensionality of the RNN in the global latent variable encoder')
parser.add_argument('--H-glv', type=int, default=50,
                    help='Number of hidden neurons of the MLP in the global latent variable encoder')
parser.add_argument('--act-glv', type=str, default='elu',
                    help='Activation function of the MLP in the global latent variable encoder')
                    
                    
# parse the parameters
args = parser.parse_args()
device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
dtype  = torch.float32 if args.use_float else torch.float64
dataset     = args.dataset
noise_level = args.noise_level
exp_name    = args.exp_name

obs_vel        = not args.no_vel
encoder        = not args.no_encoder
diag_covar     = not args.full_covar
diag_covar_int = not args.full_covar_int
num_obj     = 3   if dataset=='balls' else 5
dt          = 0.5 if dataset=='balls' else 0.05

# generate a dataset
ztr, zval, ztest, Ytr, Yval, Ytest, ts = generate_data(dataset, A=num_obj, dt=dt, T=100, N=(args.N,args.N//10,args.N//10), 
                    noise_level=noise_level, obs_vel=obs_vel, device=device, dtype=dtype)
exp = Experiment(exp_name, Ytr, Yval, Ytest, ztr, zval, ztest, ts, plot_2D_system)

# build the model
ode_model = ODEModel(args.ftype, num_obj, 2, obs_vel, order=args.order, n_latent=args.num_latent, solver=args.solver,
                        n_aug_dyn=args.n_aug_dyn, n_aug_st=args.n_aug_st, encoder=encoder, decoder=args.decoder,
                        M=args.M, H=args.H, L=args.L, act=args.act, diag_covar=diag_covar,
                        M_int=args.M_int, H_int=args.H_int, L_int=args.L_int, act_int=args.act_int, diag_covar_int=diag_covar_int,
                        T_iv=args.T_iv,   rnn_output_iv=args.rnn_output_iv,   H_iv=args.H_iv,   act_iv=args.act_iv,
                        T_glv=args.T_glv, rnn_output_glv=args.rnn_output_glv, H_glv=args.H_glv, act_glv=args.act_glv).to(device).to(dtype)

optimize_model(exp, ode_model, args.num_iter, plot_predictions, N=args.batch_size, log_every=args.log_every)
























