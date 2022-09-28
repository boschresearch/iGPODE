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
The following script is adapted from RTRBM V 1.0
(http://www.cs.utoronto.ca/~ilya/pubs/
Copyright 2009 Ilya Sutskever, MIT license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree)
to generate bouncing balls sequences.
"""

import numpy as np, os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

def norm(x): 
    return np.sqrt((x**2).sum())

def sigmoid(x):        
    return 1./(1.+np.exp(-x))

def bounce_n(dt, SIZE, T=100, n=2, r=None, m=None):
    if m==None: m=np.array([1]*n)
    if r==None: r=np.array([1.2]*n)
    X = np.zeros((T, n, 2)) # position
    V = np.zeros((T, n, 2)) # velocity
    v = np.random.randn(n,2)
    v = v / norm(v)*.5
    good_config=False
    while not good_config:
        x = -3+np.random.rand(n,2)*6
        good_config=True
        for i in range(n):
            for z in range(2):
                if x[i][z]-r[i]<-SIZE or x[i][z]+r[i]>SIZE:      
                    good_config=False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i]-x[j])<r[i]+r[j]:
                    good_config=False


    eps = dt/10 # 10 intermediate steps between two observations
    for t in range(T):

        for i in range(n): # for each ball
            X[t,i] = x[i]
            V[t,i] = v[i]

        for mu in range(int(dt/eps)): # intermediate steps
            for i in range(n): 
                x[i] += eps*v[i]
            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<-SIZE:  
                        v[i][z] =  abs(v[i][z]) # want positive
                    if x[i][z]+r[i]>SIZE: 
                        v[i][z] = -abs(v[i][z]) # want negative
            for i in range(n):
                for j in range(i):
                    if norm(x[i]-x[j])<r[i]+r[j]:
                        # the bouncing off part:
                        w   = x[i]-x[j]
                        w   = w / norm(w)
                        v_i = np.dot(w.transpose(),v[i])
                        v_j = np.dot(w.transpose(),v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w*(new_v_i - v_i)
                        v[j] += w*(new_v_j - v_j)

    return np.concatenate([X,V],-1)

def ar(x,y,z):
    return z/2+np.arange(x,y,z)

def show_image_seq(V):
    V = V.cpu().numpy()
    T   = len(V)
    res = int(np.sqrt(V.shape[1]))
    for t in range(T):
        plt.imshow(V[t].reshape(res,res),cmap=matplotlib.cm.Greys_r)
        fname = os.path.join('data',str(t)+'.png')
        plt.savefig(fname)
        plt.close()


class BouncingBallsSim(object):
    def __init__(self, box_size=5.0):
        self.box_size = box_size

    def sample_trajectory(self, dt=1.0, A=2, T=128, r=None, m=None, **kwargs):
        x = bounce_n(dt, self.box_size, T, A, r, m)
        V = np.zeros([A])
        return x,V