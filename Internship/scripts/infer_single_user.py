from src.model import PetMatchingModel
from src.utils import load_model
from src.config import *
import torch

def infer(user_attributes):
    # Normalize user attributes
    user_attributes = torch.tensor(user_attributes, dtype=torch.float32).to(DEVICE) / 5.0

    # Load model
    model = PetMatchingModel().to(DEVICE)
    model = load_model(model, MODEL_SAVE_PATH, DEVICE)

    # Perform inference
    model.eval()
    with torch.no_grad():
        pet_attributes = model(user_attributes.unsqueeze(0))  # Add batch dimension
    return pet_attributes.squeeze(0).cpu().numpy()


import pandas as pd
import numpy as np


def find_closest_pet(predicted_attributes, csv_path):
    # Load the dataset
    data = pd.read_csv(csv_path)

    data = data.dropna()

    # Extract the relevant columns and PetName
    pet_attributes = data[["grooming_frequency_value", "shedding_value", "energy_level_value",
                           "trainability_value", "demeanor_value"]].values
    pet_names = data["PetName"].values

    # Calculate the Euclidean distance between predicted attributes and dataset rows
    distances = np.linalg.norm(pet_attributes - predicted_attributes, axis=1)

    # Find the index of the closest match
    closest_index = np.argmin(distances)

    # Get the corresponding pet name
    closest_pet_name = pet_names[closest_index]

    return closest_pet_name


if __name__ == "__main__":
    # Example user attributes
    user_attributes = [4, 4, 3, 3]  # Replace with actual input values
    # moderate activity level
    # high grooming preference
    # full time available
    # experienced

    # Perform inference
    predicted_pet_attributes = infer(user_attributes)

    # Find the closest matching pet
    csv_path = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\akc-data-latest.csv"
    closest_pet = find_closest_pet(predicted_pet_attributes, csv_path)

    print(f"The closest matching pet is: {closest_pet}")



