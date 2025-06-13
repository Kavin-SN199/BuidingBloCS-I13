# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip ASL dataset
import zipfile
zipfile.ZipFile("/content/drive/MyDrive/archive.zip", 'r').extractall("/content/asl_data")
print("Dataset unzipped.")

# Convert images to CSV
import cv2
import os
import csv
from tqdm.notebook import tqdm

IMG_SIZE = 64
csv_path = "asl_alphabet.csv"
data_dir = "/content/asl_data/asl_alphabet_train/asl_alphabet_train"

with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'pixel_{i}' for i in range(IMG_SIZE * IMG_SIZE)])

    for label in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue

        for file in tqdm(os.listdir(folder), desc=label):
            if not file.endswith(".jpg"):
                continue

            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            pixels = (img / 255.0).flatten()
            writer.writerow([label.lower()] + pixels.tolist())

print("CSV creation complete.")

# Copy CSV to Google Drive
!cp asl_alphabet.csv /content/drive/MyDrive/
print("CSV saved to Google Drive.")

