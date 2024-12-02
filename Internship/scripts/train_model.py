from src.dataloader import DataLoaderManager
from src.model import PetMatchingModel
from src.train import train, validate
from src.utils import save_model
from src.config import *
import torch

def main():
    # Initialize the DataLoaderManager
    data_manager = DataLoaderManager(
        csv_path="C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Preprocessing\\preprocessed_matchedUserWithPet.csv",
        batch_size=BATCH_SIZE
    )

    # Retrieve train, dev, and test loaders
    train_loader, dev_loader, test_loader = data_manager.get_loaders()

    # Initialize the model
    model = PetMatchingModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Train the model
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)

        # Validate on the dev set
        dev_loss = validate(model, dev_loader, criterion, DEVICE)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Train Loss = {train_loss:.4f}, Dev Loss = {dev_loss:.4f}")

    # Save the trained model
    save_model(model, MODEL_SAVE_PATH)

    # Optional: Evaluate on the test set
    print("\nEvaluating on the Test Set...")
    test_loss = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss = {test_loss:.4f}")

if __name__ == "__main__":
    main()
