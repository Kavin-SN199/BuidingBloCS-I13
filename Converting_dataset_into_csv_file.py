# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#  2. Unzip your ASL dataset
import zipfile
import os

zip_path = "/content/drive/MyDrive/archive.zip"  
extract_dir = "/content/asl_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(" Unzipped dataset")

#  3. Process images into CSV
import cv2
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import csv

IMG_SIZE = 64
csv_file = "asl_alphabet_full.csv"
header = ['label'] + [f'pixel_{i}' for i in range(IMG_SIZE * IMG_SIZE)]

with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    root = os.path.join(extract_dir, "asl_alphabet_train/asl_alphabet_train")  

    for label in sorted(os.listdir(root)):
        label_path = os.path.join(root, label)
        if not os.path.isdir(label_path):
            continue

        for file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            flat = img.flatten()
            writer.writerow([label.lower()] + flat.tolist())

print(f" Done. CSV saved to: {csv_file}")
!cp asl_alphabet_full.csv /content/drive/MyDrive/asl_alphabet_full.csv
print(" CSV saved to your Google Drive.")
