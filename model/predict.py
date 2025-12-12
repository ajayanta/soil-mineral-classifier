import cv2
import numpy as np
from tensorflow.keras.models import load_model

soil_nutrients = {
    "Black Soil": "Calcium Carbonate, Magnesium, Potash, Humus",
    "Laterite Soil": "Iron, Aluminium, Manganese, Titanium",
    "Peat Soil": "High Organic Matter, Nitrogen, Sulfur",
    "Yellow Soil": "Iron Oxide, Aluminium Oxide, Silica"
}

def predict_soil(model_path, image_path):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150,150))
    img = img / 255.0
    img = np.expand_dims(img, 0)

    labels = ["Black Soil", "Laterite Soil", "Peat Soil", "Yellow Soil"]
    preds = model.predict(img)
    soil_type = labels[np.argmax(preds)]

    print("Predicted Soil:", soil_type)
    print("Nutrients:", soil_nutrients[soil_type])
