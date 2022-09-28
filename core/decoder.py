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

import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim, identity=True):
        super(SimpleDecoder, self).__init__()
        if identity:
            self.decoder = nn.Identity()
        else:
            self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim))
            
    def forward(self, z):
        return self.decoder(z)