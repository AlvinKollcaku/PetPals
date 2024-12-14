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

from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

class DataLoaderManager:
    def __init__(self, csv_path, batch_size=16):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self._prepare_data()

    def _prepare_data(self):
        data = pd.read_csv(self.csv_path)

        inputs, labels = data.iloc[:, 1:7].values, data.iloc[:, 8:].values

        # Splitting into train and temp (dev+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            inputs, labels, test_size=0.3, random_state=42
        )

        # Splitting temp into dev and test
        X_dev, X_test, y_dev, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        X_train, X_dev, X_test = map(lambda x: torch.tensor(x).float(), [X_train, X_dev, X_test])
        y_train, y_dev, y_test = map(lambda y: torch.tensor(y).float(), [y_train, y_dev, y_test])

        train_dataset = TensorDataset(X_train, y_train)
        dev_dataset = TensorDataset(X_dev, y_dev)
        test_dataset = TensorDataset(X_test, y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.dev_loader = DataLoader(dev_dataset, batch_size=1)
        self.test_loader = DataLoader(test_dataset, batch_size=1)

    def get_loaders(self):
        return self.train_loader, self.dev_loader, self.test_loader


csv_path = 'C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Preprocessing\\preprocessed_matchedUserWithPet.csv'  # Replace with your actual CSV file path

data_loader_manager = DataLoaderManager(csv_path, batch_size=10)

train_loader, dev_loader, test_loader = data_loader_manager.get_loaders()

def debug_loader(loader, loader_name):
    total_samples = 0
    for i, (inputs, labels) in enumerate(loader):
        total_samples += len(inputs)
        if i == 0:  # Print the first batch for verification
            print(f"--- {loader_name} First Batch ---")
            print(f"Inputs: {inputs}")
            print(f"Labels: {labels}")
            print(f"Inputs Shape: {inputs.shape}")
            print(f"Labels Shape: {labels.shape}")
    print(f"Total {loader_name} Samples: {total_samples}\n")

print("Debugging DataLoaderManager Splits:")
debug_loader(train_loader, "Train Loader")
debug_loader(dev_loader, "Dev Loader")
debug_loader(test_loader, "Test Loader")

# Additional check -> checking that the number of samples in inputs tensor matches those in the labels tensor
def check_consistency(loader, loader_name):
    for i, (inputs, labels) in enumerate(loader):
        assert inputs.size(0) == labels.size(0), f"Mismatch in {loader_name} at batch {i}"
    print(f"{loader_name} data consistency verified.")

check_consistency(train_loader, "Train Loader")
check_consistency(dev_loader, "Dev Loader")
check_consistency(test_loader, "Test Loader")