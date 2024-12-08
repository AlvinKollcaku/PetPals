import pandas as pd
#Most of the preprocessing was done at Dataset_prep-> Here I simply kept the columns that were going to be used for training
#with the user_id and PetName as additional

data = pd.read_csv("C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\Dataset_prep\\user_dog_matches.csv")

print(f"Duplicates found: {data.duplicated().sum()}")
data = data.drop_duplicates()

# Dropping unnecessary columns (e.g., PetName and PetDescription that are not needed for training)
data = data.drop(columns=['min_height','max_height','min_weight','max_weight','avg_weight','avg_height','match_score'])

#Dropping rows with NaN values->There should be no NaN values, but it does not hurt to run the command again
data = data.dropna()

# Saving preprocessed data for training
data.to_csv("preprocessed_matchedUserWithPet.csv", index=False)
print("Preprocessing completed. Data saved to 'preprocessed_matchedUserWithPet.csv'.")
