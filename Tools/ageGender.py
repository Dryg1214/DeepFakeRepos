import sys
from age_and_gender import AgeAndGender
from PIL import Image
import os

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Проверяем целостность файла
        return True
    except (IOError, SyntaxError):
        return False


def resize_and_crop_image(img, target_size, face_coordinates):
    # Открываем изображение
    #img = Image.open(image_path)
    
    # Рассчитываем соотношение сторон исходного изображения
    width, height = img.size
    aspect_ratio = width / height
    # Определяем меньшую сторону и масштабируем её до 2048, а другую пропорционально
    if aspect_ratio > 1:  # ширина больше высоты
        new_height = target_size
        new_width = int(new_height * aspect_ratio)
    else:  # высота больше ширины
        new_width = target_size
        new_height = int(new_width / aspect_ratio)
    # Масштабируем изображение
    img = img.resize((new_width, new_height))

    left_top_x, left_top_y, right_bottom_x, right_bottom_y = face_coordinates
    difference_first = left_top_y
    difference_second = height - right_bottom_y

    #Обрезаем изображение до квадрата
    left = (new_width - target_size) / 2 #Не трогать
    top = (new_height - target_size) / 2
    right = (new_width + target_size) / 2 #Не трогать
    bottom = (new_height + target_size) / 2

    if int(left) == 0 and int(right) == target_size and difference_second > difference_first:
      bottom = bottom - top
      top = top - top
      
    img = img.crop((left, top, right, bottom))
    return img




dataSetsPath = "/media/pavel/5B99-804C/Datasets/aboba.png"
DATA_SAVE_PATH = "sfdfssdfs/"
target_size = 1024
directory_path = "dsdf"


#Загружаем модель AgeAndGender
data = AgeAndGender()
modelsPath = "UbuntuParts/AgeGenderDetector/example/models/"
data.load_shape_predictor(f'{modelsPath}shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier(f'{modelsPath}dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor(f'{modelsPath}dnn_age_predictor_v1.dat')

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if not is_valid_image(file_path):
        os.remove(file_path)
        print(f"Deleted invalid image file: {file_path}")
    try:
        image = Image.open(file_path)
        result = data.predict(image)
        object_count = len(result)
        # Проверяем, если объект только один, то получаем возраст и центральные координаты лица
        if object_count == 1:
            age = result[0]['age']['value']
            face = result[0]['face']['value']
            # Если возраст человека меньше 18 лет, удаляем изображение
            if age < 18:
                os.remove(file_path)
                print(f"Deleted img with age < 18: {file_path}")
                continue
            # Изменение размера и обрезка изображения до размера target_size
            save_image = resize_and_crop_image(image, target_size, face)
            save_image.save(f'{DATA_SAVE_PATH}/f{filename}')
        # Если объектов несколько (группы лиц), то удаляем изображение.
        else:
            os.remove(file_path)
            print(f"Deleted img with any Faces: {file_path}")
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        os.remove(file_path)
        
        
         