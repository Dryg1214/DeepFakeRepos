import cv2
import numpy as np
import os
import dlib

# conda install conda-forge::dlib


# Функция для определения лиц на изображении
# Нужна чтобы подготовить датасеты, на наличие лиц людей (1)
# И лиц впринципе/ Xaar плохо работает
def count_images_without_faces_hhar(image_path, face_cascade):
    image = cv2.imread(image_path)
    # преобразуем изображение к оттенкам серого
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray)
    # печатать количество найденных лиц
    print(f"{len(faces)} лиц обнаружено на изображении.")
    # для всех обнаруженных лиц рисуем синий квадрат
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
    # сохраним изображение с обнаруженными лицами
    cv2.imwrite("image_faces1.jpg", image)


# Работает но не всегда. Мультяшные изображения трактует как человеческие.
def count_face_in_images(prototxt_path, model_path, image_path): 
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    image = cv2.imread(image_path)
    # получаем ширину и высоту изображения
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # устанавливаем на вход нейронной сети изображение
    model.setInput(blob)
    # выполняем логический вывод и получаем результат
    output = np.squeeze(model.forward())
    counter = 0
    for i in range(0, output.shape[0]):
        # получить уверенность
        confidence = output[i, 2]
        # если достоверность выше 50%, то нарисуйте окружающий прямоугольник
        if confidence > 0.5:
            counter += 1
            """
            # получить координаты окружающего блока и масштабировать их до исходного изображения
            box = output[i, 3:7] * np.array([w, h, w, h])
            # преобразовать в целые числа
            start_x, start_y, end_x, end_y = box.astype(int)
            # рисуем прямоугольник вокруг лица
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
            # также нарисуем текст
            cv2.putText(image, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
            # show the image
            cv2.imshow("image", image)
            cv2.waitKey(0)
            # save the image with rectangles
            #cv2.imwrite("kids_detected_dnn.jpg", image)
            """
    # if counter != 1:
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0) 
    #     cv2.destroyAllWindows() 
    return counter


# Функция для удаления изображений из одной папки, в которых нет лиц или больше 1
def delete_images_without_faces_folder(folder_path, prototxt_path, model_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            count_faces = count_face_in_images(prototxt_path, model_path, filepath)
            if count_faces != 1:
                os.remove(filepath)
                print(f"Deleted image: {filepath}")
        except Exception as e:
            print(f"Error processing image {filepath}: {e}")
            os.remove(filepath)


# Функция для удаления изображений из папки с папками, в которых нет лиц или больше 1
def delete_images_without_faces_many_folders(root_directory, prototxt_path, model_path):
    for root, root_folders, _ in os.walk(root_directory):
        for root_folder in root_folders:
            folder_path = os.path.join(root, root_folder)
            delete_images_without_faces_folder(folder_path, prototxt_path, model_path)



face_cascade = cv2.CascadeClassifier("F:\\DeepFakeRepos\\Tools\\materials\\haarcascade_frontalface_default.xml")
# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "F:\\DeepFakeRepos\\Tools\\materials\\deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
model_path = "F:\\DeepFakeRepos\\Tools\\materials\\res10_300x300_ssd_iter_140000_fp16.caffemodel"

#image = "F:\\DeepFakeRepos\\Datasets\\destination\\14.jpg"
image2 = "F:\\DeepFakeRepos\\Datasets\\destination\\785.jpg"
image = "F:\\DeepFakeRepos\\Tools\\TestImageForFind\\14.jpg"
image3 = "F:\\DeepFakeRepos\\Tools\\TestImageForFind\\322.jpg"
image4 = "F:\\DeepFakeRepos\\Tools\\TestImageForFind\\243.jpg"
image5 = "F:\\DeepFakeRepos\\Tools\\TestImageForFind\\540.jpg"
image6 = "F:\\DataBaseImage\\dir_003\\Conor Mcgregor\\400.jpg"
#delete_images_without_faces(image, face_cascade)
#count_face_in_images(prototxt_path, model_path, image)
folder_path = "F:\\DataBaseImage\\dir_003"
delete_images_without_faces_many_folders(folder_path, prototxt_path, model_path)