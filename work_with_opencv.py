import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Загрузка предобученной модели
model = tf.keras.models.load_model('fine_tuned_model_digits.keras')

# Параметры изображения
img_height, img_width = 224, 224


# Функция для предсказания цифры на изображении
def predict_digit(img):
    # Приведение изображения к нужному размеру
    img_resized = cv2.resize(img, (img_height, img_width))
    img_array = np.expand_dims(img_resized, axis=0)  # добавляем размер для батча
    img_array = img_array / 255.0  # нормализация

    # Предсказание цифры
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # получаем индекс класса с наибольшей вероятностью

    return predicted_class


# Захват видео с камеры
cap = cv2.VideoCapture(0)

# Проверяем, удалось ли открыть камеру
if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    # Чтение кадра с камеры
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр")
        break

    # Прогоняем кадр через нейросеть
    predicted_digit = predict_digit(frame)

    # Вывод предсказанной цифры в консоль
    print(f'Предсказанная цифра: {predicted_digit}')

    # Отображаем кадр в окне
    cv2.imshow('Camera Feed', frame)

    # Прерывание по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение камеры и закрытие всех окон
cap.release()
cv2.destroyAllWindows()
