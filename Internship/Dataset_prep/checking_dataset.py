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

df = pd.read_csv("C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Dataset_prep\\user_dog_matches.csv")

column_name = 'PetName'

# Checking for uniqueness
unique_values = df[column_name].unique()
is_unique = df[column_name].is_unique

print(f"Unique values: {unique_values}")
print(f"Is the column unique? {is_unique}")
print(f"Number of unique values: {len(unique_values)}")

#There are 41 dog breeds in the dataset that will be used for training -> Good amount of breeds

