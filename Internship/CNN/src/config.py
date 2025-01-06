import torch

# Paths
CNN_TRAIN_DATA_PATH = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\CNN\\Dataset_prep\\initial_model_test.csv"
CNN_MODEL_SAVE_PATH = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\CNN\\CNN_models\\CNN_model.pth"

# Hyperparameters
CNN_BATCH_SIZE = 2
CNN_LEARNING_RATE = 0.01
CNN_DROPOUT_RATE = 0.2
CNN_NUM_EPOCHS = 3

# Device
CNN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"