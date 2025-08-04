from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATASET_PATH = "dataset/"

num_classes = len(os.listdir(DATASET_PATH))  # 5 классов
class_mode = "binary" if num_classes == 2 else "categorical"

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),  # используем 224x224 для предварительно обученной модели
    batch_size=32,
    class_mode=class_mode,
    subset="training",
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),  # используем 224x224 для предварительно обученной модели
    batch_size=32,
    class_mode=class_mode,
    subset="validation",
    shuffle=False
)
