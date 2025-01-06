import os
import shutil

def flatten_folder_structure(base_folder):
    organized_folder = os.path.join(base_folder, "Organized_Images")

    for breed in os.listdir(organized_folder):
        breed_folder = os.path.join(organized_folder, breed)
        if os.path.isdir(breed_folder):
            for root, _, files in os.walk(breed_folder):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        source = os.path.join(root, file)
                        destination = os.path.join(breed_folder, file)
                        # Move the file to the breed folder
                        shutil.move(source, destination)
                        print(f"Moved {source} to {destination}")

            # Removing any empty subfolders
            for subfolder in os.listdir(breed_folder):
                subfolder_path = os.path.join(breed_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    shutil.rmtree(subfolder_path)
                    print(f"Removed empty folder {subfolder_path}")

BASE_FOLDER = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\CNN\\Dataset_prep"
flatten_folder_structure(BASE_FOLDER)
