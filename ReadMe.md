# Pet Match and Recognition System

## Project Overview  

This project consists of two neural networks:

1. **User-to-Pet Matching Neural Network:**  
   - Matches users with pet dogs based on their characteristics.

2. **Pet Image Recognition Neural Network:**  
   - Uses Convolutional Neural Networks (CNNs) to recognize and classify dog breeds from images.

Both functionalities are accessible via a **Flask REST API**.

---

## Main Technologies Used  

- **PyTorch, NumPy, Pandas, Scikit-learn, Optuna:** Model development and optimization.
- **Flask:** RESTful API interface.
- **Google Cloud Platform (GCP):** Image storage for training.

---

## Datasets Used

### 1. User-to-Dog Matching Dataset
- **Dog Characteristics Dataset:** [AKC Dog Data](https://tmfilho.github.io/akcdata/)
- **User Data:** Custom dataset created for matching users with suitable dog breeds.

### 2. Dog Breed Image Classification Dataset
- **Dataset Used:** [Stanford Dog Breeds Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
- **Selected Breeds:** The model was trained to recognize only 12 breeds due to training time constraints:
  
  ```
  0: Afghan Hound
  1: African Hunting Dog
  2: Airedale
  3: American Staffordshire Terrier
  4: Appenzeller
  5: Australian Terrier
  6: Bedlington Terrier
  7: Bernese Mountain Dog
  8: Blenheim Spaniel
  9: Border Collie
  10: Border Terrier
  11: Boston Bull
  ```

---

## User-to-Dog Pet Matching Key Points

### 1. User-Dog Matching
- A script was created to generate a user-dog match dataset by merging the user dataset with the dog characteristics dataset.
- After preprocessing, the data was used to train the model, achieving the following results:
  
  ![User-Dog Matching Results](https://github.com/user-attachments/assets/cb93781e-0a8d-4e8e-af09-47ef3766b815)
  
  ![User-Dog Matching Results](https://github.com/user-attachments/assets/3d354b1b-7ac6-4924-8ac7-eed41fdb5a94)

### 2. Dog Breed Recognition from Image
- The CNN model was trained to classify the 12 selected breeds.
- The model performed exceptionally well, achieving the following accuracy:
  
  ![Dog Breed Recognition Accuracy](https://github.com/user-attachments/assets/7286924f-c6c8-4a7d-a24a-d82429c183b0)

---

## REST API

### API Endpoints:

![API Endpoints](https://github.com/user-attachments/assets/42881233-1c7c-4f8b-a56e-da7a1dbe6bb9)

### Authentication
- Uses JWT for authentication, where the user receives an `access_token` upon login.

### **1. User-Dog Match Endpoint**
- **Endpoint:** `POST http://127.0.0.1:5000/dogmatch/`
- **Request Format:**
  
  ```json
  {
     "activity_level": 0.75,
     "home_type": 0.66667,
     "grooming_preference": 0.75,
     "time_available": 0.0,
     "experience_level": 0.5,
     "preferred_dog_size":1
  }
  ```
- **Response:**
  
  ![Dog Match Response](https://github.com/user-attachments/assets/13ba997b-0fe0-4f50-bb40-018a92060b3a)

### **2. Dog Breed Recognition Endpoint**
- **Endpoint:** `POST http://127.0.0.1:5000/dogbreed/classify`
- **Request:** Accepts an image as input (currently via file upload, but might be updated to accept URLs).
  
  ![Input Image](https://github.com/user-attachments/assets/a515b3d3-e903-47eb-a095-58194e84e98f)
  
- **Example Input Image: African Hunting Dog**
  
  ![Example Dog Image](https://github.com/user-attachments/assets/4cf97fdd-46d0-49a6-9468-9514c330840e)

- **Response:**
  
  ![Classification Response](https://github.com/user-attachments/assets/c00a6658-e5ff-4fa2-af41-c4c7769a2e4a)

  - The model accurately predicted the breed as **African Hunting Dog** with almost 100% (99.6%) certainty.

---

## Google Cloud Platform (GCP)

- **Storage:**
  - All 120 dog breed images were stored in a GCP bucket.
  - For training, a CSV file was created containing image paths for the selected 12 breeds and their corresponding labels.
  
  ![GCP Dataset](https://github.com/user-attachments/assets/ef8abdeb-e066-4e3c-b0e9-00905072dc8b)

---

## Future Improvements

- Expanding the breed recognition model to all 120 breeds.
- Optimizing API response times and deploying the system for public use.
  
---

## License

This project is licensed under the **GNU General Public License (GPL)**.

---

## Author

Developed by **Alvin Kollcaku**.

