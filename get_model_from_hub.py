from huggingface_hub import hf_hub_download
import tensorflow as tf
# Загрузка модели с Hugging Face
model_file = hf_hub_download(repo_id="vanisus/definition_digits", filename="fine_tuned_model_digits.keras")

# Использование загруженной модели
model = tf.keras.models.load_model(model_file)

# Сохранение модели локально
model.save("fine_tuned_model_digits.keras")
