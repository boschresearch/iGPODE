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


import os

import numpy as np
import torch
import logging

class Logger:
    def __init__(self, metrics, log_folder=None, print_every=100, interval_mean=True):
        ''' Inputs
                metrics         - list of str denoting the metrics to be printed
                log_folder      - which folder to dump the logs.
                                    if None, print only to console. else, console + file
                print_every     - logging interval
                interval_mean   - if True, print the mean since the last log. 
        '''
        self.metrics = metrics
        self.print_every = print_every
        self.interval_mean = interval_mean
        self.N = len(metrics)
        self.logs = np.zeros([0,self.N],dtype=np.float32)
        self.iter = 0
        # create logger
        self.logger = logging.getLogger('IODE')
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        # create file handler if requested
        if log_folder is not None:
            log_fname = os.path.join(log_folder, 'logs.log')
            file = logging.FileHandler(log_fname, mode='a')
            file.setLevel(logging.INFO)
            file.setFormatter(logging.Formatter("%(asctime)s  %(message)s",datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(file)
        # create stream handler 
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        stream.setFormatter(logging.Formatter("%(asctime)s  %(message)s",datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(stream)
        
    def print_message(self,msg):
         self.logger.info(msg)
        
    def print_traced_log(self):
        msg = 'iter={:<4d}'.format(self.iter)
        if self.interval_mean:
            means = np.mean(self.logs[-self.print_every:],0) # N
        else:
            means = np.mean(self.logs,0) # N
        for name,mean in zip(self.metrics,means):
            msg += '\t{:s}={:.3f}'.format(name,mean)
        self.logger.info(msg)
        
    def log(self, *args):
        assert len(args)==self.N, 'number of arguments is wrong'
        self.iter += 1
        if self.iter%self.print_every == 0:
            self.print_traced_log()
        args = [arg.item() if isinstance(arg,torch.Tensor) else arg for arg in args]
        self.logs = np.concatenate([self.logs,np.array([args])],0)


















