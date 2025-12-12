import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, split_ratio=0.8):
    train_dir = os.path.join(base_dir, "Train")
    test_dir = os.path.join(base_dir, "Test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    soil_types = os.listdir(base_dir)

    for soil in soil_types:
        soil_path = os.path.join(base_dir, soil)
        if not os.path.isdir(soil_path):
            continue

        train_path = os.path.join(train_dir, soil)
        test_path = os.path.join(test_dir, soil)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        images = os.listdir(soil_path)
        train_imgs, test_imgs = train_test_split(images, train_size=split_ratio, random_state=42)

        for img in train_imgs:
            shutil.move(os.path.join(soil_path, img), os.path.join(train_path, img))

        for img in test_imgs:
            shutil.move(os.path.join(soil_path, img), os.path.join(test_path, img))

    print("Dataset split into Train & Test")
