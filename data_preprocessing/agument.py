import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def augment_dataset(input_dir, output_dir, augment_per_image=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    for soil_type in os.listdir(input_dir):
        soil_folder = os.path.join(input_dir, soil_type)
        save_folder = os.path.join(output_dir, soil_type)
        os.makedirs(save_folder, exist_ok=True)

        for img_name in os.listdir(soil_folder):
            img_path = os.path.join(soil_folder, img_name)

            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            count = 0
            for batch in datagen.flow(img_array, batch_size=1,
                                      save_to_dir=save_folder,
                                      save_prefix="aug",
                                      save_format="jpg"):
                count += 1
                if count >= augment_per_image:
                    break

    print("Augmentation complete!")
