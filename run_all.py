import os

# --- Import modules from your project ---
from data_preprocessing.augment import augment_dataset
from data_preprocessing.preprocess_resize import resize_normalize
from data_preprocessing.preprocess_clahe import apply_clahe
from data_preprocessing.split_dataset import split_dataset

from model.train_model import train_model
from model.evaluate_model import evaluate_model


# -------------------------------
# 📁 SET PATHS (EDIT THESE)
# -------------------------------
BASE_DATASET = r"C:\Users\KIIT\OneDrive\Desktop\Soil types\Original"
AUGMENTED = "datasets/augmented"
RESIZED = "datasets/resized"
CLAHE = "datasets/clahe"
TRAIN_TEST_BASE = "datasets/processed"

MODEL_SAVE_PATH = "saved_models/soil_vgg16.h5"
LABELS = ["Black Soil", "Laterite Soil", "Peat Soil", "Yellow Soil"]


def main():
    print("\n========== STEP 1: AUGMENTATION ==========")
    augment_dataset(BASE_DATASET, AUGMENTED, augment_per_image=50)

    print("\n========== STEP 2: RESIZE + NORMALIZE ==========")
    resize_normalize(AUGMENTED, RESIZED, size=(150,150))

    print("\n========== STEP 3: CLAHE PREPROCESSING ==========")
    apply_clahe(RESIZED, CLAHE, size=(150,150))

    print("\n========== STEP 4: TRAIN / TEST SPLIT ==========")
    split_dataset(CLAHE, split_ratio=0.8)

    print("\n========== STEP 5: TRAINING MODEL ==========")
    model, history = train_model(TRAIN_TEST_BASE)

    os.makedirs("saved_models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"🎉 Model saved at: {MODEL_SAVE_PATH}")

    print("\n========== STEP 6: EVALUATION ==========")
    evaluate_model(MODEL_SAVE_PATH, f"{TRAIN_TEST_BASE}/Test", LABELS)

    print("\n🎯 ALL STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
