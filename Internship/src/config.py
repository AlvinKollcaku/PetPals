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

# Configuration for training
import torch

MODEL_SAVE_PATH = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\ANN_models\\pet_matching_model4.pth"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
NUM_EPOCHS = 200

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

