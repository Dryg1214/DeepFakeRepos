from PIL import Image
import cv2
import imutils
import numpy as np



def resize_and_crop_image(image_path, target_size):
    # Открываем изображение
    img = Image.open(image_path)

    # Рассчитываем соотношение сторон исходного изображения
    width, height = img.size
    aspect_ratio = width / height
    print("Базовое разрешение")
    print(width)
    print(height)
    # Определяем меньшую сторону и масштабируем её до 2048, а другую пропорционально
    if aspect_ratio > 1:  # ширина больше высоты
        new_height = target_size
        new_width = int(new_height * aspect_ratio)
    else:  # высота больше ширины
        new_width = target_size
        new_height = int(new_width / aspect_ratio)

    print("Обработанное разрешение")
    print(new_width)
    print(new_height)
    # Масштабируем изображение
    img = img.resize((new_width, new_height))

    # Обрезаем изображение до квадрата
    left = (new_width - target_size) / 2
    top = (new_height - target_size) / 2
    right = (new_width + target_size) / 2
    bottom = (new_height + target_size) / 2

    img = img.crop((left, top, right, bottom))

    return img


# Пример использования
image_path = "F:\\DeepFakeRepos\\Tools\\TestImageResize\\01896.png"
target_width = 1024
target_height = 1024

new_image = resize_and_crop_image(image_path, 1024)
new_image.save('01896_res.png')