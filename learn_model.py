import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import os

# Параметры
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

# Функция для загрузки и предобработки изображений
def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0  # нормализация
    return img, label

# Загрузка CSV и создание списка файлов и меток
def load_data(csv_file, base_path):
    df = pd.read_csv(csv_file)
    df['file'] = df['file'].apply(lambda x: os.path.join(base_path, x))
    paths = df['file'].values
    labels = df['label'].values
    return paths, labels

# Создание датасета TensorFlow
def create_dataset(csv_file, base_path):
    paths, labels = load_data(csv_file, base_path)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Пути к CSV-файлам
train_csv = './labelTrain.csv'
validation_csv = './labelTest.csv'

# Базовые пути к изображениям
train_base_path = './dataset/train'
validation_base_path = './dataset/validation'

# Создание датасетов
train_dataset = create_dataset(train_csv, train_base_path)
validation_dataset = create_dataset(validation_csv, validation_base_path)

# Загрузка предварительно обученной модели VGG16 без верхних слоев
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Добавление новых слоев
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)  # 10 классов для цифр от 0 до 9

# Создание модели
model = Model(inputs=base_model.input, outputs=predictions)

# Замораживание слоев базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Компиляция модели
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Вывод структуры модели
model.summary()

# Обучение модели
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

# Сохранение модели после начального обучения
model.save('initial_model_digits.h5')

# Разморозка последних слоев и дополнительное обучение
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Компиляция модели с меньшим learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Дополнительное обучение модели
history_fine_tune = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10  # дополнительные эпохи
)

# Сохранение дообученной модели
model.save('fine_tuned_model_digits.h5')

# Оценка модели
final_loss, final_accuracy = model.evaluate(validation_dataset)
print(f'Final validation accuracy: {final_accuracy * 100:.2f}%')
