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

from src.model import PetMatchingModel
from src.utils import load_model
from src.config import *
import torch

def infer(user_attributes):
    # Normalizing user attributes
    user_attributes = torch.tensor(user_attributes, dtype=torch.float32).to(DEVICE)

    # Loading model
    model = PetMatchingModel(DROPOUT_RATE,True,'relu').to(DEVICE)
    model = load_model(model, MODEL_SAVE_PATH, DEVICE)

    # Performing inference
    model.eval()
    with torch.no_grad():
        pet_attributes = model(user_attributes.unsqueeze(0))  # Add batch dimension
    return pet_attributes.squeeze(0).cpu().numpy()


import pandas as pd
import numpy as np


def find_closest_pet(predicted_attributes, csv_path):
    data = pd.read_csv(csv_path)

    data = data.dropna()

    #extracting relevant pet attributes
    pet_attributes = data[["grooming_frequency_value", "shedding_value", "energy_level_value",
                           "trainability_value", "demeanor_value","avg_weight_z","avg_height_z"]].values
    pet_names = data["PetName"].values

    #TODO -> utilizing calculate_match_score() in matching_user_pet.py
    # Calculating the Euclidean distance between predicted attributes and dataset rows
    distances = np.linalg.norm(pet_attributes - predicted_attributes, axis=1)

    closest_index = np.argmin(distances) #confidence level meauser

    closest_pet_name = pet_names[closest_index]

    return closest_pet_name

#TODO take top 3 dogs
if __name__ == "__main__":
    # Example user attributes
    #user_attributes = [0.5, 0, 0.25, 0.666667,1,0] # Label = Bolognese
    user_attributes = [0.75, 0.66667, 0.75, 0,0.5,1] # Great Dane

    predicted_pet_attributes = infer(user_attributes)
    print(predicted_pet_attributes)

    csv_path = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Dataset_prep\\dogs_normalized.csv"
    closest_pet = find_closest_pet(predicted_pet_attributes, csv_path)

    print(f"The closest matching pet is: {closest_pet}")



