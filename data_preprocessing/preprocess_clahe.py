import os
import cv2
import numpy as np

def apply_clahe(input_dir, output_dir, size=(150,150)):
    os.makedirs(output_dir, exist_ok=True)

    def preprocess_image(path):
        image = cv2.imread(path)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = cv2.resize(img, size)
        return img / 255.0

    for soil_type in os.listdir(input_dir):
        soil_folder = os.path.join(input_dir, soil_type)
        output_folder = os.path.join(output_dir, soil_type)
        os.makedirs(output_folder, exist_ok=True)

        for img_name in os.listdir(soil_folder):
            path = os.path.join(soil_folder, img_name)
            img = preprocess_image(path)
            save_path = os.path.join(output_folder, img_name)
            cv2.imwrite(save_path, (img * 255).astype(np.uint8))

        print(f" Processed {soil_type}")

    print("CLAHE Preprocessing Complete!")
