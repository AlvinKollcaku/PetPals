from src.dataloader import DataLoaderManager
from src.model import PetMatchingModel
from src.utils import load_model, calculate_mae
from src.config import *
import torch

def main():
    # Initialize DataLoaderManager
    data_manager = DataLoaderManager(
        csv_path="C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Preprocessing\\preprocessed_matchedUserWithPet.csv",
        batch_size=BATCH_SIZE
    )

    # Access the test loader
    _, _, test_loader = data_manager.get_loaders()

    # Load the trained model
    model = PetMatchingModel().to(DEVICE)
    model = load_model(model, MODEL_SAVE_PATH, DEVICE)

    # Evaluate the model
    model.eval()
    mae = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move inputs and targets to the appropriate device
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Get model predictions
            predictions = model(inputs)

            # Compute MAE for the batch
            mae += calculate_mae(predictions, targets)  # MAE is per-sample

            total_samples += 1

    # Compute the average MAE over all test samples
    print(f"Test MAE: {mae / total_samples:.4f}")

if __name__ == "__main__":
    main()
