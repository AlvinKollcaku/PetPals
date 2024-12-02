from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import torch

class DataLoaderManager:
    def __init__(self, csv_path, batch_size=32):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.dataset = None
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self._prepare_data()

    def _prepare_data(self):
        # Load the dataset
        data = pd.read_csv(self.csv_path)

        # Split into inputs and labels
        inputs = torch.tensor(data.iloc[:, 0:4].values).float()
        labels = torch.tensor(data.iloc[:, 4:9].values).float()

        # Create a TensorDataset
        full_dataset = TensorDataset(inputs, labels)

        # Split into train, dev, and test sets
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        dev_size = int(0.15 * total_size)
        test_size = total_size - train_size - dev_size

        train_dataset, dev_dataset, test_dataset = random_split(
            full_dataset, [train_size, dev_size, test_size]
        )

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.dev_loader = DataLoader(dev_dataset, batch_size=1)  # Batch size 1 for precise evaluation
        self.test_loader = DataLoader(test_dataset, batch_size=1)  # Batch size 1 for precise evaluation

    def get_loaders(self):
        return self.train_loader, self.dev_loader, self.test_loader
