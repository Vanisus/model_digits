import tensorflow as tf

# Загрузка модели из .h5 файла
model = tf.keras.models.load_model('fine_tuned_model_digits.h5')

# Компиляция модели с метриками
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile_metrics(metrics=['accuracy'])
# Сохранение модели в формате .keras
model.save('fine_tuned_model_digits.keras')
