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
import torch, gpytorch

from gpytorch.models      import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution
from gpytorch.variational import VariationalStrategy, IndependentMultitaskVariationalStrategy, UnwhitenedVariationalStrategy
from gpytorch.kernels     import RBFKernel, MaternKernel

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, nout, kernel='rbf', whitened=True, diag_covar=True):
        assert kernel in ['rbf','0.5','1.5','2.5'], 'Wrong kernel type!'
        U_VAR_DIST = MeanFieldVariationalDistribution if diag_covar else CholeskyVariationalDistribution
        variational_distribution = U_VAR_DIST(inducing_points.size(-2), batch_shape=torch.Size([nout]))
        VAR_CLS = VariationalStrategy if whitened else UnwhitenedVariationalStrategy
        base_variational_strategy = VAR_CLS(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        variational_strategy = IndependentMultitaskVariationalStrategy(base_variational_strategy, num_tasks=nout)
        super().__init__(variational_strategy)
        self.kernel   = kernel
        self.whitened = whitened
        self.diag_covar = diag_covar
        self.nout = nout
        self.nin  = inducing_points.shape[-1]
        self.M    = inducing_points.shape[-2]
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([nout]))
        if self.kernel=='rbf':
            self.base_kernel = RBFKernel(self.nin) 
        else:
            nu = float(kernel)
            self.rnd_gam = torch.distributions.gamma.Gamma(nu,nu)
            self.base_kernel = MaternKernel(nu, ard_num_dims=self.nin)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel, batch_shape=torch.Size([nout]))
    
    @property
    def device(self):
        return self.Z.device
    
    @property
    def dtype(self):
        return self.Z.dtype
    
    @property
    def sf(self):
        return self.covar_module.outputscale.to(self.device) # num_out
    
    @property
    def ell(self):
        return self.covar_module.base_kernel.lengthscale.squeeze().to(self.device) # num_out
    
    @property
    def Z(self):
        return self.variational_strategy.base_variational_strategy.inducing_points
    
    @property 
    def Kzz(self):
        ''' return LazyTensor [nout,M,M] '''
        return self.covar_module(self.Z)
    
    @property
    def Lzz(self):
        ''' return LazyTensor [nout,M,M] '''
        return self.Kzz.cholesky()
        
    @property
    def U_mean(self):
        ''' return [nout,M]'''
        U = self.variational_strategy.variational_distribution.mean
        return (self.Lzz @ U.unsqueeze(-1)).squeeze(-1) if self.whitened else U
    
    @property
    def U_cov(self):
        ''' return [n,M,M]'''
        S = self.variational_strategy.variational_distribution.covariance_matrix # nout,M,M
        L = self.Lzz.evaluate()
        return L @ S @ L.transpose(1,2) if self.whitened else S
    
    def draw_U(self,P):
        if self.whitened:
            L = self.Lzz.evaluate() @ self.variational_strategy.variational_distribution.scale_tril
            return self.U_mean + (L @ torch.randn([self.nout,self.M,P],device=self.device, dtype=L.dtype)).permute(2,0,1)
        else:
            L = self.variational_strategy.variational_distribution.scale_tril
            return torch.distributions.MultivariateNormal(self.U_mean,scale_tril=L).rsample(torch.Size([P]))
        
    def draw_omega(self,P,S,nout,nin):
        # due to https://github.com/j-wilson/GPflowSampling/blob/develop/gpflow_sampling/bases/fourier_initializers.py
        normal_rvs = torch.randn(P, S, nout, nin, device=self.device, dtype=self.dtype)
        if self.kernel == 'rbf':
            return normal_rvs
        else:
            gamma_rvs = self.rnd_gam.sample([P,S,nout,nin]).to(self.device)
            return torch.rsqrt(gamma_rvs) * normal_rvs
            
    def __cache(self, S=100, P=1):
        '''
            S - number of bases
            P - number of function draws
        '''
        nin, nout = self.nin, self.nout
        omega_= self.draw_omega(P,S,nout,nin)
        omega = omega_ / self.ell.reshape([1,1,1,nin]).sqrt() # P,S,nout,nin
        bias  = torch.rand(P, S, nout, 1, device=self.device, dtype=self.dtype) * 2 * np.pi # P,S,nout
        w     = torch.randn(P, S, nout, 1, device=self.device, dtype=self.dtype) # P,S,nout
        # draw nu
        phi_w_z = self.rff(self.Z, omega, bias, w) # P,M,nout
        u_draws = self.draw_U(P) # nout,M,P
        f_m     = u_draws.permute(0,2,1) - phi_w_z # P,M,nout
        nu      = self.Kzz.inv_matmul(f_m.permute(2,1,0)).permute(2,1,0) # P,M,nout
        return omega,bias,w,nu
    
    def rff(self, x, omega, bias, w): # x [N,nin], output [P,N,nout]
        nout = self.nout
        # version-1
        proj      = omega@x.T
        features  = torch.cos(proj+bias) # P,S,nout,N
        out_scale = np.sqrt(2/omega.shape[1]) * self.sf.sqrt().reshape(1,1,nout,1)
        phi       = features * out_scale # P,S,nout,N
        return (phi*w).sum(1).permute(0,2,1) # P,N,nout
        # version-2
        # proj = omega@x.T
        # phi_ = torch.cat( [torch.cos(proj),torch.sin(proj)], 1) # P,2S,nout,N
        # phi  = self.sf.sqrt().reshape(1,1,nout,1) * phi_ / np.sqrt(omega.shape[1]) # P,S,nout,N
        # return (phi*w).sum(1).permute(0,2,1) # P,N,nout
    
    def function_space_prior(self, xs, P):
        ''' xs - evaluation points [N] 
            P  - number of samples
            returns - samples [P,n]
        '''
        rnd = torch.randn(len(xs), P, device=self.device, dtype=self.dtype)
        return self.covar_module(xs)._cholesky().matmul(rnd).detach().T
    
    def prior_draw(self, S=100, P=1):
        omega,tau,w,nu = self.__cache(S,P)
        return lambda x: self.rff(x,omega,tau,w)
    
    def function_space_posterior(self, x ,P):
        ''' xs - evaluation points [N,n] 
            P  - number of samples
            returns - samples [P,N,n]
        '''
        return self(x).rsample(torch.Size([P]))
    
    def post_draw(self, S=100, P=1):
        ''' S - number of bases
            P - number of function draws
        '''
        nout = self.nout
        omega,tau,w,nu = self.__cache(S,P) # nu [P,M,nout]
        def f(x):
            # x [N,nin], outputs [P,N,nout]
            N = x.shape[0]
            prior = self.rff(x, omega, tau, w) # P,N,nout
            Kxz = self.covar_module(x,self.Z).evaluate() # nout,N,M
            Kxz = Kxz.unsqueeze(0).repeat([P,1,1,1]).permute(0,3,2,1) # P,M,N,nout
            nu_ = nu.unsqueeze(2).repeat([1,1,N,1])  # P,M,N,nout
            update = (nu_*Kxz).sum(1).reshape(P,N,nout) # P,N,out
            return prior + update # P,N,nout
        return f
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def kl(self):
        return self.variational_strategy.kl_divergence()
    
    def fix_gpytorch_cache(self, it):
        del self.variational_strategy.base_variational_strategy._memoize_cache
        if it==0:
            _ = self.train(False)
            _ = self(self.Z)
            
    def __repr__(self):
        type_ = 'Whitened' if self.whitened else 'Unwhitened'
        str_ = f'{type_} GP model with {self.kernel} kernel and {self.M} inducing points, from R^{self.nin} to R^{self.nout}.'
        return str_
 