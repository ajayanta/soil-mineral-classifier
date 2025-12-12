from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from build_model import build_vgg16_model

def train_model(train_dir, epochs=20, batch_size=64):
    model = build_vgg16_model()

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    return model, history
