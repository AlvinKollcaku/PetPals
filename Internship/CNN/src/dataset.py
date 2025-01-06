# src/dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DogBreedDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with 'file_path' and 'label'.
            transform (callable, optional): Optional transform to be applied to an image sample.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = sorted(self.dataframe['label'].unique())

        # Mapping from label to an integer (for classification)
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.loc[idx, 'file_path']
        label_str = self.dataframe.loc[idx, 'label']

        # Convert label to an integer class
        label = self.label2idx[label_str]

        # Open the image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

