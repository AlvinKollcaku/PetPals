import torch.nn as nn

class PetMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),  # Input size: 4 user attributes
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # Output size: 5 pet attributes
            nn.Sigmoid()       # To scale output to [0, 1]
        )

    def forward(self, x):
        return self.layers(x)
