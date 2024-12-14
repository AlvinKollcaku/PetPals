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

input_csv = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Dataset_prep\\akc-data-latest.csv"
output_csv_normalized = "dogs_normalized.csv"

df = pd.read_csv(input_csv)

# Creating new columns for average weight and average height
df["avg_weight"] = (df["min_weight"] + df["max_weight"]) / 2
df["avg_height"] = (df["min_height"] + df["max_height"]) / 2

# Z-score normalizing the `avg_weight` and `avg_height` columns
df["avg_weight_z"] = (df["avg_weight"] - df["avg_weight"].mean()) / df["avg_weight"].std()
df["avg_height_z"] = (df["avg_height"] - df["avg_height"].mean()) / df["avg_height"].std()

# Dropping unnecessary columns (e.g., PetName and PetDescription that are not needed for training)
df = df.drop(columns=['description', 'temperament','popularity','min_expectancy', 'max_expectancy', 'group',
                      'grooming_frequency_category','shedding_category', 'energy_level_category','energy_level_category',
                      'trainability_category','demeanor_category'])

# Dropping rows with null values
df = df.dropna()

df.to_csv(output_csv_normalized, index=False)

print(f"Final CSV with z-score normalization saved as: {output_csv_normalized}")
