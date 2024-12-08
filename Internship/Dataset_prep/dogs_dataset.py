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
