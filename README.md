# Soil Mineral Classification using Deep Learning

This project focuses on classifying different types of soil using images. The goal is to take an image of soil and predict which category it belongs to, along with some basic information about the nutrients typically found in that soil.

---

## What this project does

The project follows a full workflow starting from raw images all the way to prediction:

* Expands the dataset using image augmentation
* Cleans and preprocesses images (resizing, normalization, CLAHE)
* Splits the dataset into training and testing sets
* Trains a deep learning model (VGG16)
* Evaluates model performance
* Predicts soil type for new images

---

## Soil categories used

The model is trained on the following types:

* Black Soil
* Laterite Soil
* Peat Soil
* Yellow Soil

---

## Project structure

```
soil-mineral-classifier/
│
├── data_preprocessing/
├── model/
├── utils/
├── notebooks/
│   └── soilmineral.ipynb
│
├── run_all.py
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset is not included in this repository because of its size.

To use this project, you need to organize your dataset like this:

```
dataset/
├── Black Soil/
├── Laterite Soil/
├── Peat Soil/
└── Yellow Soil/
```

Once you have the dataset ready, update the paths inside `run_all.py` before running the code.

---

## Setup

This project works best with **Python 3.10** because TensorFlow does not support newer versions like 3.13.

Create a virtual environment and install dependencies:

```
py -3.10 -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the project

To run the entire pipeline (from preprocessing to training and evaluation):

```
python run_all.py
```

---

## Model details

* Model: VGG16 (pre-trained)
* Input size: 150 × 150
* Number of classes: 4

Transfer learning is used to improve performance with a smaller dataset.

---

## Prediction

You can test the model on a new image using:

```python
from model.predict import predict_soil

predict_soil("saved_models/soil_vgg16.h5", "image.jpg")
```

---

## Nutrient information

Based on the predicted soil type, the project also gives a rough idea of nutrients:

* Black Soil → rich in calcium, magnesium, potash
* Laterite Soil → high in iron and aluminium
* Peat Soil → high organic matter and nitrogen
* Yellow Soil → contains iron oxides and silica

---

## Notes

* Make sure to update dataset paths before running
* The dataset is not included in this repo
* Python 3.10 is required for compatibility

---

## About

This project was built as part of learning and experimenting with deep learning and image classification.
