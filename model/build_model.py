from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def build_vgg16_model(input_shape=(150,150,3), num_classes=4):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    conv_base.trainable = False

    model = Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()
    return model
