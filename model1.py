import os
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "dataset/"

num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл не найден: {image_path}")
        return

    try:
        img = Image.open(image_path)
        img.verify()  # Проверка на повреждение
        img = Image.open(image_path)  # Открываем снова
    except (OSError, IOError):
        print(f"Ошибка: Поврежденное изображение - {image_path}")
        return

    model = tf.keras.models.load_model("image_classifier.h5")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось прочитать изображение - {image_path}")
        return

    img = cv2.resize(img, (224, 224))  # Важно: подгоняем изображение под размер модели
    img = img / 255.0  # Нормализуем
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_names = sorted(os.listdir(DATASET_PATH))  # Отсортировать для сопоставимости

    if class_mode == "binary":
        confidence = prediction[0][0]
        predicted_class = class_names[0] if confidence < 0.5 else class_names[1]
    else:
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_class = class_names[predicted_idx]

    print(f"Модель определила: {predicted_class} (уверенность: {confidence:.2f})")

    # Отображение
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Модель определила: {predicted_class}")
    plt.axis("off")
    plt.show()

# Пример вызова
predict_image("dataset/humans/human9.jpg")
