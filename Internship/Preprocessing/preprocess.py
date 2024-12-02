import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data = pd.read_csv("C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\matchedPets.csv")

print(f"Duplicates found: {data.duplicated().sum()}")
data = data.drop_duplicates()

"""
User Attributes:
Activity Levels: Very Low, Low, Moderate, High, Very High.
Home Types: Apartment, House, Rural, Urban.
Grooming Preferences: Very Low, Low, Medium, High, Very High.
Time Available: Full-time, Part-time, Limited, Minimal.
Experience Levels: First-time owner, Experienced, Expert.
"""
# Dropping unnecessary columns (e.g., PetName and PetDescription that are not needed for training)
data = data.drop(columns=['User ID','Pet ID','Pet Description', 'Grooming Frequency Category','Shedding Category',
                      'Energy Level Category', 'Trainability Category', 'Demeanor Category','Pet Temperament',
                    'Pet Popularity', 'Pet Min Height', 'Pet Max Height', 'Pet Min Weight', 'Pet Max Weight',
                      'Pet Min Expectancy', 'Pet Max Expectancy','Home Type', 'Pet Group'])

#Dropping rows with NaN values
data = data.dropna()

# Manually Label Encoding user attributes (ordinal columns)
ordinal_columns = ['Activity Level', 'Grooming Preference', 'Time Available', 'Experience Level']
category_mappings = [{'Very High': 5, 'High': 4, 'Moderate': 3, 'Low': 2, 'Very Low': 1},
                     {'Very High': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Very Low': 1},
                     {'Full-time': 4, 'Part-time': 3, 'Limited': 2, 'Minimal': 1},
                     {'Expert':3 ,'Experienced':2 ,'First-time owner':1}]

for col, mapping in zip(ordinal_columns, category_mappings):
    data[col] = data[col].map(mapping)

# Saving preprocessed data for training
data.to_csv("preprocessed_matchedUserWithPet.csv", index=False)
print("Preprocessing completed. Data saved to 'preprocessed_matchedUserWithPet.csv'.")
