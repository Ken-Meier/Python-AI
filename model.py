from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np

# Загружаем данные
from pets import train_data  # Импортируем train_data из pets.py

DATASET_PATH = "dataset/"

num_classes = len(os.listdir(DATASET_PATH))  # 5 классов
class_mode = "binary" if num_classes == 2 else "categorical"

# Подключаем MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Замораживаем предобученные слои

# Добавляем собственные слои для классификации
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # Добавление Dropout для предотвращения переобучения
x = Dense(256, activation="relu")(x)
x = Dense(num_classes, activation="softmax")(x)  # Слой для многоклассовой классификации

# Создаем финальную модель
model = Model(inputs=base_model.input, outputs=x)

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",  # Используем категориальный кросс-энтропию для многоклассовой классификации
              metrics=["accuracy"])

# Получаем имена классов из генератора
class_names = list(train_data.class_indices.keys())  # Получаем список имен классов

# Вычисление весов для каждого класса (если необходимо)
class_labels = train_data.classes
weights = compute_class_weight(class_weight="balanced", classes=np.unique(class_labels), y=class_labels)
class_weights = dict(enumerate(weights))

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

# Обучение модели
model.fit(train_data, validation_data=train_data, epochs=30, class_weight=class_weights, callbacks=[early_stop, checkpoint])

# Оценка модели
test_loss, test_accuracy = model.evaluate(train_data)
print(f"Точность модели на валидационных данных: {test_accuracy:.2f}")

# Сохранение модели
model.save("image_classifier.h5")
