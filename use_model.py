import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os


# Загрузка сохраненной модели
model = tf.keras.models.load_model('fine_tuned_model_digits.keras')

# Параметры изображения
img_height, img_width = 224, 224

# Функция для предсказания цифры на изображении
def predict_digit(img_path):
    # Загружаем и предобрабатываем изображение
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # добавляем размер для батча
    img_array /= 255.0  # нормализация

    # Используем модель для предсказания
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # получаем индекс класса с наибольшей вероятностью

    return predicted_class


directory_path = 'tests'
for root, dirs, files in os.walk('tests'):
    for file in files:
        img_path = f'{directory_path}/{file}'  # Замените на путь к вашему изображению
        predicted_digit = predict_digit(img_path)
        print(f'Предсказанная цифра для файла {file}: {predicted_digit}')

