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

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Dataset_prep\\user_dog_matches.csv")

inputs = data.iloc[:, 1:7] #Label encoding and normalization will be applied here

print(f"Duplicates found: {data.duplicated().sum()}")
data = data.drop_duplicates()

# Dropping unnecessary columns (e.g., PetName and PetDescription that are not needed for training)
data = data.drop(columns=['min_height','max_height','min_weight','max_weight','avg_weight','avg_height','match_score'])

#Dropping rows with NaN values->There should be no NaN values since I ran this commend at the Dataset_prep
#but it does not hurt to run the command again
data = data.dropna()

#Label encoding string values and then applying min-max normalization
#Using dictionary comprehension
mappings = {
    'activity_level': {v: i for i, v in enumerate(["Very Low", "Low", "Moderate", "High", "Very High"])},
    'home_type': {v: i for i, v in enumerate(["Small Apartment", "Large Apartment", "House", "Rural"])},
    'grooming_preference': {v: i for i, v in enumerate(["Very Low", "Low", "Medium", "High", "Very High"])},
    'time_available': {v: i for i, v in enumerate(["Full-time", "Part-time", "Limited", "Minimal"])},
    'experience_level': {v: i for i, v in enumerate(["First-time owner", "Experienced", "Expert"])},
    'preferred_dog_size': {v: i for i, v in enumerate(["Small", "Medium", "Large"])}
}

#Applying the encoding
for col, mapping in mappings.items():
    data[col] = data[col].map(mapping)

#Normalization
columns_to_normalize = ['activity_level', 'home_type', 'grooming_preference',
                        'time_available', 'experience_level', 'preferred_dog_size']
scaler = MinMaxScaler() # (x-min)/(max-min)
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

data.to_csv("preprocessed_matchedUserWithPet.csv", index=False)
print("Preprocessing completed. Data saved to 'preprocessed_matchedUserWithPet.csv'.")
