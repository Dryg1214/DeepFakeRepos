import re
import os
from PIL import Image


def check_images_equal(image_first, image_second):
    img1 = Image.open(image_first)
    img2 = Image.open(image_second)

    if img1.size != img2.size:
        return False
    
    pixels1 = list(img1.getdata())
    pixels2 = list(img2.getdata())
    
    # Такая штука не работает. Показатели на пару значений яркости отличаются
    return pixels1 == pixels2
    

# Проверка является ли заменонное фото, точной копией destination - объектом куда встраивают изображение
# Сделана по причине не совсем корректной работы roop
def check_images_dest_equal(folder_destinashion, folder_fakes):
    for filename in os.listdir(folder_fakes):
        match = re.search(r'_dest(.+?)(?=\.\w+$)', filename)
        if match:
            result = match.group(1)
            dest_filepath = os.path.join(folder_destinashion, result)
            fakepath = os.path.join(folder_fakes, filename)
            if check_images_equal(dest_filepath, fakepath) == True:
                os.remove(fakepath)
                print(f"Deleted image: {fakepath}")
        else:
            print("Текст после '_dest' не найден.")


folder_dest_name = "F:\\DeepFakeRepos\\Datasets\\destination"
folder_fakes = "F:\\DeepFakeRepos\\Datasets\\RoopData\\fake"
check_images_dest_equal(folder_dest_name, folder_fakes)