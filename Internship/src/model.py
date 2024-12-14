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

class PetMatchingModel(nn.Module):
    def __init__(self,dropoutRate):
        super().__init__()

        self.inputLayer =  nn.Linear(6, 64)
        self.hiddenLayer1 =  nn.Linear(64, 32)
        self.hiddenLayer2 =  nn.Linear(32, 16)
        self.outputLayer =  nn.Linear(16, 7)

        self.dr = dropoutRate

    # forward pass
    def forward(self, x):
        # input
        x = F.relu(self.inputLayer(x))
        x = F.dropout(x, p=self.dr, training=self.training)  # switches dropout off during .eval()

        # hidden1
        x = F.relu(self.hiddenLayer1(x))
        x = F.dropout(x, p=self.dr, training=self.training)

        # hidden2
        x = F.relu(self.hiddenLayer2(x))
        x = F.dropout(x, p=self.dr, training=self.training)

        x = self.outputLayer(x)

        return x

#The forward function is called when data is passed through the model,
#such as in this line: model = PetMatchingModel(DROPOUT_RATE)
#Behind the scenes, this calls the __call__ method of the nn.Module class,
#which in turn calls the custom forward method.