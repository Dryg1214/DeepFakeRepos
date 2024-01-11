from PIL import Image
import os

def is_valid_jpg(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Проверяем целостность файла
        return True
    except (IOError, SyntaxError):
        return False


# Удаление изображений, которые не являются изображениями
def delete_invalid_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not is_valid_jpg(file_path):
            # Если файл не является валидным, удаляем его
            os.remove(file_path)
            print(f"Deleted invalid JPG file: {file_path}")


# Для папки папок DataBaseImage с реальными изображениями
def delete_invalid_images_recursive(root_directory):
    for root, root_folders, _ in os.walk(root_directory):
        # Используем функцию delete_invalid_images для каждой подпапки
        for root_folder in root_folders:
            folder_path = os.path.join(root, root_folder)
            delete_invalid_images(folder_path)


# Удаление изображений, ниже определенного разрешения
def delete_low_resolution_images(folder_path, min_resolution):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            flag = False
            with Image.open(filepath) as img:
                width, height = img.size
                if width < min_resolution or height < min_resolution:
                    flag = True
            if flag == True:
                os.remove(filepath)
                print(f"Deleted image: {filepath}")
        except Exception as e:
            print(f"Error processing image {filepath}: {e}")


def delete_low_resolution_images_recursive(root_directory, min_resolution):
     for root, root_folders, _ in os.walk(root_directory):
        for root_folder in root_folders:
            folder_path = os.path.join(root, root_folder)
            delete_low_resolution_images(folder_path, min_resolution)

min_resolution = 400
directory_to_clean = "F:\\DataBaseImage\\dir_003"
#delete_invalid_images_recursive(directory_to_clean)
delete_low_resolution_images_recursive(directory_to_clean, min_resolution)