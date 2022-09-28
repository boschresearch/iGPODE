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

import torch
import os, sys
sys.path.append("..") # Adds higher directory to python modules path.

from ode_model import ODEModel

class Experiment:
    def __init__(self, log_folder, Ytr, Yval, Ytest, ztr, zval, ztest, ts, plot_fnc=None):
        self.Ytr   = Ytr
        self.Yval  = Yval
        self.Ytest = Ytest
        self.ztr   = ztr
        self.zval  = zval
        self.ztest = ztest
        self.ts    = ts
        self.plot_fnc     = plot_fnc
        self.__log_folder = log_folder
        self.__exp_root   = 'exps' # os.path.join('..','exps') 
        self.init_exp_folder()
        
    def maybe_create_folder(self,fname):
        if not os.path.exists(fname):
            try:
                os.makedirs(fname)
            except Exception as e:
                print('Exception: ' + str(e))
        
    def init_exp_folder(self):
        self.maybe_create_folder(os.path.join(self.__exp_root, self.__log_folder))
            
    def rm(self,fname):
        file = os.path.join(self.log_folder, fname)
        if os.path.isfile(file):
            os.remove(file) 

    def get_all_trained_models(self):
        return [it[:-4] for it in os.listdir(self.log_folder) if '.pkl' in it]
        
    @property
    def log_folder(self):
        return os.path.join(self.__exp_root,self.__log_folder)

    def __repr__(self):
        return self.__log_folder
    
    def save(self, ode_model, fname=None, results=None):
        fname = ode_model.name if fname is None else fname 
        ode_model.save(os.path.join(self.log_folder,fname),results)
    
    def load(self,fname):
        return ODEModel.load(os.path.join(self.log_folder,fname)).to(self.Ytr.device).to(self.Ytr.dtype)