"""
<PetPals>
Copyright (C) 2024 Alvin Kollçaku

Author: Alvin Kollçaku
Contact: kollcakualvin@gmail.com
Year: 2024
Original repository of the project: https://github.com/AlvinKollcaku/PetPals.git

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class PetMatchingModel(nn.Module):
    def __init__(self,dropoutRate,batchNorm,actFun):
        super().__init__()

        self.inputLayer =  nn.Linear(6, 64)
        self.bnorm1 = nn.BatchNorm1d(64)  # the number of units into this layer
        self.hiddenLayer1 =  nn.Linear(64, 32)
        self.bnorm2 = nn.BatchNorm1d(32)
        self.hiddenLayer2 =  nn.Linear(32, 16)
        self.outputLayer =  nn.Linear(16, 7)

        self.batchNorm = batchNorm
        self.dr = dropoutRate
        self.actFun = actFun

        if not hasattr(torch.nn.functional, self.actFun):
            raise ValueError(f"Activation function '{self.actFun}' not found in torch.nn.functional")

    # forward pass
    def forward(self, x):
        # get activation function type
        # this code replaces torch.relu with torch.<self.actfun>
        actfun = getattr(torch.nn.functional, self.actFun)
        # input
        x = actfun(self.inputLayer(x))
        x = F.dropout(x, p=self.dr, training=self.training)  # switches dropout off during .eval()
        if self.batchNorm:
            # hidden1
            x = self.bnorm1(x)
            x = actfun(self.hiddenLayer1(x))
            # hidden2
            x = self.bnorm2(x)
            x = actfun(self.hiddenLayer2(x))
        else:
            # hidden1
            x = actfun(self.hiddenLayer1(x))
            x = F.dropout(x, p=self.dr, training=self.training)
            # hidden2
            x = actfun(self.hiddenLayer2(x))
            x = F.dropout(x, p=self.dr, training=self.training)

        x = self.outputLayer(x)

        return x

#The forward function is called when data is passed through the model,
#such as in this line: model = PetMatchingModel(DROPOUT_RATE)
#Behind the scenes, this calls the __call__ method of the nn.Module class,
#which in turn calls the custom forward method.