Install all dependencies
```bash
pip3 install tensorflow pandas huggingface_hub
```
Use your huggingface_token for auth 
```bash
huggingface-cli login
```
After that use file get_model_from_hub.py for downloading model from hub
```bash
python3 get_model_from_hub.py
```
Use file use_model.py for definition your images (images for tests included in folder 'tests')


Установите все зависимости

```bash
pip3 install tensorflow pandas huggingface_hub
```
Используйте ваш huggingface_token для аутентификации 
```bash
huggingface-cli login
```
После этого используйте файл get_model_from_hub.py для загрузки модели из хаба
```bash
python3 get_model_from_hub.py
```
Используйте файл use_model.py для определения ваших изображений (изображения для тестов включены в папку 'tests')
