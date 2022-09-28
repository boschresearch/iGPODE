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
The following class is adapted from NRI V 1.0
(https://github.com/ethanfetaya/NRI
Copyright 2018 Ethan Fetaya, Thomas Kipf, MIT license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree)
to generate charged particles data sequences
"""

import torch
import numpy as np

class ChargedParticlesSim(object):
    def __init__(self, box_size=5., loc_std=2., vel_norm=0.5, interaction_strength=1.0):
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self._charge_types = np.array([-1., 0., 1.])

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        # assert (np.all(loc < self.box_size * 3))
        # assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, dt=0.1, A=5, T=100, probs=[0.5, 0, 0.5]):
        ''' A  - number of objects
            T  - sequence length
            dt - time between two observations
        '''
        diag_mask = np.ones((A,A), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        # Sample edges
        charges = np.random.choice(self._charge_types, size=(A,1), p=probs)
        edges = charges.dot(charges.transpose())
        # Initialize location and velocity
        loc_next = np.random.randn(2, A) * self.loc_std
        vel_next = np.random.randn(2, A)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc_next,vel_next = self._clamp(loc_next, vel_next)
        _max_F = 10
        
        def odef(t,x):
            # x is [2,2,A] = [loc,vel]
            x = np.reshape(x, [2,2,A])
            loc,vel = x[0],x[1]
            loc,vel = self._clamp(loc, vel)

            l2_dist_power3 = np.power(
                self._l2(loc.transpose(), loc.transpose()),
                3. / 2.)
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size, 0)

            F = (forces_size.reshape(1, A, A) *
                 np.concatenate((
                     np.subtract.outer(loc[0, :],
                                       loc[0, :]).reshape(1, A, A),
                     np.subtract.outer(loc[1, :],
                                       loc[1, :]).reshape(1, A, A)
                     )
                )).sum(
            axis=-1)
            F[F > _max_F]  =  _max_F
            F[F < -_max_F] = -_max_F
            return np.reshape(np.stack([vel,F]), [2*2*A])
        
        def integrate(f, x0, ts, d=1000, euler=True):
            dt = ts[1]-ts[0]
            T = len(ts)
            dense_ts = np.arange(T*d)*dt/d 
            h = dt/d
            def rk4_step(t,x):
                if euler:
                    return h*f(t,x)
                else:
                    k1 = h * (f(t, x))
                    k2 = h * (f((t+h/2), (x+k1/2)))
                    k3 = h * (f((t+h/2), (x+k2/2)))
                    k4 = h * (f((t+h), (x+k3)))
                    return (k1+2*k2+2*k3+k4) / 6
            xs = [x0]
            for t in dense_ts:
                dx    = rk4_step(t,xs[-1])
                xnext = xs[-1] + dx
                xs.append(xnext)
            xs = xs[:-1:d]
            return np.stack(xs).T
            
        with np.errstate(divide='ignore'):
            x0 = np.reshape(np.stack([loc_next,vel_next]), [2*2*A])
            ts = np.arange(T)*dt
            sol = integrate(odef, x0, ts)
            sol = np.reshape(sol.T, [T,2,2,A])
            loc,vel = sol[:,0],sol[:,1]
            sol = np.concatenate([loc,vel],1) # T,4,A
            return sol.transpose(0,2,1), charges[:,0] # [T,A,4], [A]
        
        
        
        
        
        
        
        
        
        
        
        
        
        