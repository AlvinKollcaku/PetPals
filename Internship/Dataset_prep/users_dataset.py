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


import random
import pandas as pd

activity_levels = ["Very Low", "Low", "Moderate", "High", "Very High"]
home_types = ["Small Apartment", "Large Apartment", "House", "Rural"]
grooming_preferences = ["Very Low", "Low", "Medium", "High", "Very High"]
time_available = ["Full-time", "Part-time", "Limited", "Minimal"]
experience_levels = ["First-time owner", "Experienced", "Expert"]
preferred_dog_sizes = ["Small", "Medium", "Large"]

time_activity_mapping = {
    "Full-time": activity_levels,            # All levels allowed
    "Part-time": activity_levels,           # All levels allowed
    "Limited": ["Very Low", "Low", "Moderate"], # Limited to lower levels
    "Minimal": ["Very Low", "Low"],         # Only very low and low
}

size_home_mapping = {
    "Small Apartment": ["Small"],  # Only small dogs are suitable
    "Large Apartment": ["Small", "Medium"],  # Small and medium dogs are suitable
    "House": preferred_dog_sizes,  # All sizes are suitable
    "Rural": ["Medium", "Large"],  # Larger dogs are more suited to rural areas
}

users = []
for user_id in range(1, 1001):
    time_avail = random.choice(time_available)
    activity_level = random.choice(time_activity_mapping[time_avail])
    home_type = random.choice(home_types)
    grooming_pref = random.choice(grooming_preferences)
    experience_level = random.choice(experience_levels)
    preferred_size = random.choice(size_home_mapping[home_type])

    user = {
        "user_id": user_id,
        "activity_level": activity_level,
        "home_type": home_type,
        "grooming_preference": grooming_pref,
        "time_available": time_avail,
        "experience_level": experience_level,
        "preferred_dog_size": preferred_size,
    }
    users.append(user)

# Converting to a DataFrame to save it as csv
user_df = pd.DataFrame(users)

user_df.to_csv("users_dataset.csv", index=False) #to not include the row indices 0,1,2...
print("User dataset created and saved to 'users_dataset.csv'")
