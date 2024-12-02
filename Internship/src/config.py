# Configuration for training
import torch

# Paths
TRAIN_DATA_PATH = "data/train.csv"
VAL_DATA_PATH = "data/val.csv"
MODEL_SAVE_PATH = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\pet_matching_model.pth"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
