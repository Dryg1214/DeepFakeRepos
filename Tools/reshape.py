import imutils
import cv2
import numpy as np
from PIL import Image


#reshape с сохранением пропорций картинки
def reshape_save_proporshions(file_path, file_path_save, width = 256, height = 256):
    img = cv2.imread(file_path)
    img = imutils.resize(img, width, height)
    cv2.imwrite(file_path_save, img)


def resize_image_save_proporshions(input_path, output_path, target_size):
    # Открываем изображение
    original_image = Image.open(input_path)

    # Изменяем размер с сохранением пропорций
    original_image.thumbnail(target_size)

    # Сохраняем измененное изображение
    original_image.save(output_path)


def resize_hard_image(file_path, file_path_save, size=(256,256)):
    img = cv2.imread(file_path)
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    img = cv2.resize(mask, size, interpolation)
    cv2.imwrite(file_path_save, img)


# Разделение изображения на n частей
def cut_n_image_from_image(file_path, filepath_save, name):
    img = cv2.imread(file_path)
    dimensions = img.shape
    # харкодом под 1024, для e4e можно будет потом в параметр
    N = dimensions[1] // 1024
    width_cutoff = dimensions[1] // N
    
    list_img = list()
    for i in range(N):
        counter = i * width_cutoff
        img_save = img[:, counter:counter + width_cutoff]
        list_img.append(img_save)
        cv2.imwrite(f"{filepath_save}\\{name}{i}.png", img_save)



file_path = "ReshapeImage\\input_image\\tones.png"
file_path_save = "ReshapeImage\\output_image"

# Пример использования
input_image_path = 'F:\\DataBaseImage\\ffhq\\00868.png'
output_image_path = '00868_process_3IMU.png'
target_size = (2048, 2048)

#resize_image_save_proporshions(input_image_path, output_image_path, target_size)
#resize_hard_image(input_image_path, output_image_path, target_size)
reshape_save_proporshions(input_image_path, output_image_path, 2048, 2048)


#cut_n_image_from_image(file_path, file_path_save, "tone")


#cv2.imshow('image' , img)
#cv2.waitKey()

# Работает, но качество очень сильно умирает
#resize_hard_image(file_path, file_path_save)
