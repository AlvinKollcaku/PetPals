import os
import csv

def create_csv(base_folder, csv_file):
    organized_folder = os.path.join(base_folder, "Organized_Images")
    data = []

    for breed in os.listdir(organized_folder):
        breed_folder = os.path.join(organized_folder, breed)
        if os.path.isdir(breed_folder):
            for file_name in os.listdir(breed_folder):
                file_path = os.path.join(breed_folder, file_name)
                data.append([file_path, breed])

    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'label'])  # CSV headers
        writer.writerows(data)
    print(f"CSV saved at {csv_file}")

BASE_FOLDER = "C:\\Users\\alvin\\OneDrive\\Desktop\\PetPals\\Internship\\CNN\\Dataset_prep"  # Update this to your project directory
CSV_FILE = os.path.join(BASE_FOLDER, "image_labels.csv")
create_csv(BASE_FOLDER, CSV_FILE)
