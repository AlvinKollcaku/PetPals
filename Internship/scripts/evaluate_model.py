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

from src.dataloader import DataLoaderManager
from src.model import PetMatchingModel
from src.utils import load_model, calculate_mae
from src.config import *
import torch

def main():
    data_manager = DataLoaderManager(
        csv_path="C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Preprocessing\\preprocessed_matchedUserWithPet.csv",
        batch_size=BATCH_SIZE
    )

    _, _, test_loader = data_manager.get_loaders()

    # Loading the trained model
    model = PetMatchingModel(DROPOUT_RATE,True).to(DEVICE)
    model = load_model(model, MODEL_SAVE_PATH, DEVICE)

    # Evaluation
    model.eval()
    mae = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Moving inputs and targets to the appropriate device
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Getting model predictions
            predictions = model(inputs)

            # Computing MAE for the batch
            mae += calculate_mae(predictions, targets)  # MAE is per-sample

            total_samples += 1

    # Computing the average MAE over all test samples
    print(f"Test MAE: {mae / total_samples:.4f}")

if __name__ == "__main__":
    main()
