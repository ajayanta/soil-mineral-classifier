import os
import cv2
import numpy as np
from tqdm import tqdm

def resize_normalize(input_dir, output_dir, size=(150, 150)):
    os.makedirs(output_dir, exist_ok=True)

    for soil_type in os.listdir(input_dir):
        soil_folder = os.path.join(input_dir, soil_type)
        save_folder = os.path.join(output_dir, soil_type)
        os.makedirs(save_folder, exist_ok=True)

        for img_name in tqdm(os.listdir(soil_folder), desc=f"Processing {soil_type}"):
            img_path = os.path.join(soil_folder, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, size)
            img = img / 255.0

            save_path = os.path.join(save_folder, img_name)
            cv2.imwrite(save_path, (img * 255).astype(np.uint8))

    print("Resize + Normalization complete!")
