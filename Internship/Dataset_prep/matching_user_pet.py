import pandas as pd

users = pd.read_csv("users_dataset.csv")
dogs = pd.read_csv("dogs_dataset.csv")

# Normalize function for dogs attributes -> min-max normalization
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Matching function
def calculate_match_score(user, dog):
    score = 0

    # Activity Level ↔ Energy Level matching
    activity_to_energy = {
        "Very Low": 0.4,
        "Low": 0.6,
        "Moderate": 0.8,
        "High": 1.0,
        "Very High": 1.0,
    }
    target_energy = activity_to_energy[user["activity_level"]]
    score += 1 - abs(dog["energy_level"] - target_energy)  # Closer energy = higher score

    # Grooming Preference ↔ Grooming Frequency
    grooming_to_frequency = {
        "Very Low": 0.2,
        "Low": 0.4,
        "Medium": 0.6,
        "High": 0.8,
        "Very High": 1.0,
    }
    target_grooming = grooming_to_frequency[user["grooming_preference"]]
    score += 1 - abs(dog["grooming_frequency"] - target_grooming)

    # Time Available ↔ Energy Level & Grooming Frequency
    time_restriction = {
        "Full-time": 1.0,
        "Part-time": 0.8,
        "Limited": 0.6,
        "Minimal": 0.4,
    }
    score += (1 if dog["energy_level"] <= time_restriction[user["time_available"]] else 0)
    #If dogs energy level is <= or equal the user time available value than it is a good match
    score += (1 if dog["grooming_frequency"] <= time_restriction[user["time_available"]] else 0)

    # Experience Level ↔ Trainability & Demeanor
    experience_to_trainability = {
        "First-time owner": (0.8, 1.0),
        "Experienced": (0.6, 1.0),
        "Expert": (0.2, 1.0),
    }
    trainability_min, trainability_max = experience_to_trainability[user["experience_level"]]
    if trainability_min <= dog["trainability_value"] <= trainability_max:
        score += 1
    demeanor_min, demeanor_max = experience_to_trainability[user["experience_level"]]
    if demeanor_min <= dog["demeanor_value"] <= demeanor_max:
        score += 1

    # Home Type ↔ Energy, Shedding
    home_restrictions = {
        "Small Apartment": {"energy": 0.6, "shedding": 0.4},
        "Large Apartment": {"energy": 0.8, "shedding": 0.6},
        "House": {"energy": 1.0, "shedding": 1.0},
        "Rural": {"energy": 1.0, "shedding": 1.0},
    }
    home_restriction = home_restrictions[user["home_type"]]
    score += (1 if dog["energy_level"] <= home_restriction["energy"] else 0)
    score += (1 if dog["shedding_value"] <= home_restriction["shedding"] else 0)

    # Preferred Size ↔ Avg Height & Avg Weight
    size_to_dimensions = {
        "Small": (0, 30),
        "Medium": (30, 60),
        "Large": (60, 100),
    }
    min_size, max_size = size_to_dimensions[user["preferred_size"]]
    if min_size <= dog["avg_height"] <= max_size:
        score += 1
    if min_size <= dog["avg_weight"] <= max_size:
        score += 1

    return score

# Find the best match for each user
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
            "dog_id": best_dog["dog_id"],
            **best_dog.to_dict(),
            "match_score": best_score,
        })

# Convert matches to DataFrame and save to CSV
matches_df = pd.DataFrame(matches)
matches_df.to_csv("user_dog_matches.csv", index=False)
