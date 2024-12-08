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

