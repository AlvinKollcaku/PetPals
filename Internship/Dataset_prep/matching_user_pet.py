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

users = pd.read_csv("users_dataset.csv")
dogs = pd.read_csv("dogs_normalized.csv")

# Matching function
def calculate_match_score(user, dog):
    score = 0

    # Activity Level ↔ Energy Level matching
    activity_to_energy = {
        "Very Low":0.4, #dogs table has 0.4 as lowest values so it is a good match for both
        "Low": 0.4,     #very low and low
        "Moderate": 0.6,
        "High": 0.8,
        "Very High": 1.0,
    }
    target_energy = activity_to_energy[user["activity_level"]]
    score += 1 - abs(dog["energy_level_value"] - target_energy)  # Closer energy = higher score

    # Grooming Preference ↔ Grooming Frequency
    grooming_to_frequency = {
        "Very Low": 0.2,
        "Low": 0.4,
        "Medium": 0.6,
        "High": 0.8,
        "Very High": 1.0,
    }
    target_grooming = grooming_to_frequency[user["grooming_preference"]]
    score += 1 - abs(dog["grooming_frequency_value"] - target_grooming)

    # Time Available ↔ Energy Level & Grooming Frequency
    #In this case we cant really substract user value with dog values since there is no direct correlation
    #A Full-time available user can very well have a low energy dog. That is why it is best
    #to simply compare if the dogs energy level and grooming_frequency are not above the
    #users time_available
    time_restriction = {
        "Full-time": 1.0,
        "Part-time": 0.8,
        "Limited": 0.6,
        "Minimal": 0.4,
    }
    score += (0.5 if dog["energy_level_value"] <= time_restriction[user["time_available"]] else 0)
    #If dogs energy level is <= or equal the user time available value than it is a good match
    score += (0.5 if dog["grooming_frequency_value"] <= time_restriction[user["time_available"]] else 0)

    # Experience Level ↔ Trainability & Demeanor
    #The values here correspond to the minimum value expected for the dogs trainability and demeanor
    experience_to_trainability = {
        "First-time owner": 0.8, #First time owner expects high trainability and demeanor
        "Experienced": 0.6,
        "Expert": 0.2,
    }
    exp_min = experience_to_trainability[user["experience_level"]]
    if exp_min <= dog["trainability_value"]:
        score += 1
    if exp_min <= dog["demeanor_value"]:
        score += 1

    # Home Type ↔ Energy, Shedding
    home_restrictions = { #Since we also compare here I am putting the maximum values
        "Small Apartment": {"energy": 0.4, "shedding": 0.2},
        "Large Apartment": {"energy": 0.6, "shedding": 0.4},
        "House": {"energy": 1.0, "shedding": 0.6},
        "Rural": {"energy": 1.0, "shedding": 1},
    }
    home_restriction = home_restrictions[user["home_type"]]
    score += (1 if dog["energy_level_value"] <= home_restriction["energy"] else 0)
    score += (1 if dog["shedding_value"] <= home_restriction["shedding"] else 0)

    # Preferred Size ↔ Avg Height & Avg Weight
    #Since the user specifies the size I have used ranges here
    size_to_dimensions = {
        "Small": (0, 30), #These values were specified by looking at min max of the average sizes
        "Medium": (30, 56), #max = 80 for both , mins were 3 and 16.
        "Large": (56, 80),
    } # Dog max weight = 110
    min_size, max_size = size_to_dimensions[user["preferred_dog_size"]]
    if min_size <= dog["avg_height"] <= max_size:
        score += 1
    if min_size <= dog["avg_weight"] <= max_size:
        score += 1

    return score

# Finding the best match for each user
matches = []
for _, user in users.iterrows():
    best_score = -1
    best_dog = None

    for _, dog in dogs.iterrows():
        score = calculate_match_score(user, dog)
        if score > best_score:
            best_score = score
            best_dog = dog

    if best_dog is not None:
        matches.append({
            "user_id": user["user_id"],
            **user.to_dict(),
            **best_dog.to_dict(),
            "match_score": best_score,
        })

# Converting matches to DataFrame and saving to CSV
matches_df = pd.DataFrame(matches)
matches_df.to_csv("user_dog_matches.csv", index=False)
