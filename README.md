**ЭТАПЫ ВЫПОЛНЕНИЯ НАУЧНОЙ РАБОТЫ**  

**1. Скачиваем БД FaceForensics++**  
  
Скачиваем фейковые изображения с официального репозитория FF 
https://github.com/ondyari/FaceForensics

После отправки заявки к авторам, на почту пришло письмо с ссылкой на скачивание бенчмарка с изображениями:
[FaceForensics_benchmark](http://kaldir.vc.in.tum.de/faceforensics_benchmark_images.zip)

Данный бенчмарк, создан для тестирования детектирования при сжатии изображений.

В качестве реальных изображений использовались изображения знаменитостей высокого качества
[Celebrity-Face-Recognition-Dataset](https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset)

**2. Берем 2-3 готовых (не из списка FaceForensics++) решения по замене/изменению лиц:**  
Для замены лиц были выбраны сети **GHOST**, **ROOP**
- Сеть GHOST располагается в [AllModels/ModelGenerate/GHOST](AllModels/ModelGenerate/GHOST)
- Сеть ROOP располагается в [AllModels/ModelGenerate/roop](AllModels/ModelGenerate/roop)

Для преобразования использовалась сеть **Encoder4editing**
- Сеть Encoder4editing располагается в [AllModels/ModelGenerate/encoder4editing](AllModels/ModelGenerate/encoder4editing)
     
**3. Дополняем FaceForensics++ новыми генерациями, причем БД разбиваем на 2 блока - замена лиц и изменение лица**  

Для генерации данных с помощью модели ROOP был написан скрипт для автоматической генерации:  
[scriptROOP.py](GenerateDeepFakeRoopScripts/scriptROOP.py)

В дочерней папке скрипта располагается текстовый файл, с примером запуска этого файла. 
 
Для генерации данных с помощью модели GHOST используется Jupyter Notebook, изменный под массовую генерацию изображений:  
[GHOST Jupyter Notebook](AllModels/ModelGenerate/GHOST/GHOST_upd.ipynb)  

Для генерации данных с помощью моделей ROOP и GHOST были необходимы две папки. Одна нужна для источника контекста [Source](Datasets/source), а другая для его замены [Destination](Datasets/destination). Разрешение выходного изображения равно разрешению входного изображения, меняется лишь "лицо" объекта.

Для генерации данных с помощью модели Encoder4Editing используется Jupyter Notebook, изменный под массовую генерацию изображений:  
[Encoder4Editing Jupyter Notebook](https://github.com/Dryg1214/DeepFakeRepos/blob/main/AllModels/ModelGenerate/encoder4editing/E4emyUpdate.ipynb)  

Для генерации изменнных атрибутов необходимо загрузить одну папку с целевыми изображениями. Выходное изображение модели Encoder4Editing = 1024x1024

**4. Находим готовые решения (модель+веса) по детекции подделок из числа, указаных в таблице 2**  

Для детектирования были выбраны три модели: **Xception, EfficientNetB4, MesoNet.**  

Для предсказаний модели **XceptionNet** использовался Jupyter Notebook [XceptionNet](AllModels/DeepFakeDetectionModels/XceptionNet/XceptionNet.ipynb)  

Данные для обучающей выборки и код для обучения модели **XceptionNet** представлен в Jupyter Notebook [XceptionTraining](AllModels/DeepFakeDetectionModels/XceptionNet/Team_Dark_HAIYA_XceptionNet_Deepfake_Detector_Training.ipynb) 

Исследование для **XceptionNet** проводилось для данных: [ROOP](Datasets/RoopData/fake), [GHOST](Datasets/GHOSTdata/fake), [FaceForensics](Datasets/FaceForensics/fake), [RealImage](Datasets/CelebaHQReal/Data/real), [SmileAttribute](Datasets/Encoder4EditingData/Smile/fake), [OldAgeAttribute](Datasets/Encoder4EditingData/OldAge/fake) и [Базовой тестовой выборкой XceptionNet](AllModels/DeepFakeDetectionModels/XceptionNet/testing_images)  


Для предсказаний модели **MesoNet** использовался Jupyter Notebook [MesoNet](AllModels/DeepFakeDetectionModels/MesoNet-DeepFakeDetection/notebook/Meso_4.ipynb). 
Также в коде данного ноутбука есть методы для обучения модели. Датасет для обучения и тестирования модели [MesoNetData](AllModels/DeepFakeDetectionModels/MesoNet-DeepFakeDetection/data). Разрешение изображений в датасете варьируется от 98х98 до 614х614.

Исследование для **MesoNet** проводилось для данных: [ROOP](Datasets/RoopData/fake), [GHOST](Datasets/GHOSTdata/fake), [FaceForensics](Datasets/FaceForensics/fake), [RealImage](Datasets/CelebaHQReal/Data/real), [SmileAttribute](Datasets/Encoder4EditingData/Smile/fake), [OldAgeAttribute](Datasets/Encoder4EditingData/OldAge/fake) и [Тестовой выборки Meso](AllModels/DeepFakeDetectionModels/MesoNet-DeepFakeDetection/data/train). Все базовые изображения тестовой выборки MesoNet имеют разрешение 128х128.  

Для предсказаний модели **EfficientNetB4** использовался Jupyter Notebook [EfficientNetB4]([AllModels/DeepFakeDetectionModels/EfficientNetB4 + EfficientNetB4ST + B4Att + B4AttST/EfficientNetAutoAttB4_myUpdate.ipynb](https://github.com/Dryg1214/DeepFakeRepos/blob/main/AllModels/DeepFakeDetectionModels/EfficientNetB4%20%2B%20EfficientNetB4ST%20%2B%20B4Att%20%2B%20B4AttST/EfficientNetAutoAttB4_myUpdate.ipynb)). Исследование проводилось для данных: [ROOP](Datasets/RoopData/fake), [GHOST](Datasets/GHOSTdata/fake), [FaceForensics](Datasets/FaceForensics/fake), [RealImage](Datasets/CelebaHQReal/Data/real), [SmileAttribute](Datasets/Encoder4EditingData/Smile/fake), [OldAgeAttribute](Datasets/Encoder4EditingData/OldAge/fake)

**Код каждой модели был обновлен для удобного сбора вероятности классификации к классу real/fake для отдельных групп изображений.**


**5. Результаты детектирования.**  
Для каждого типа данных, описанных в пункте 4 создан txt файл с несколькими параметрами.  

В названии файла будет указан тип данных (fake or real), название генерационной модели, а также используемый датасет.  

Предсказания для модели XceptionNet находятся в [XceptionPredict](AllModels/DeepFakeDetectionModels/XceptionNet) 
 
Предсказания для модели MesoNet находятся в [MesoNetPredict](AllModels/DeepFakeDetectionModels/MesoNet-DeepFakeDetection/notebook)  

Предсказания для модели EfficientNetB4 находятся в [EfficientNetB4Predict](https://github.com/Dryg1214/DeepFakeRepos/tree/main/AllModels/DeepFakeDetectionModels/EfficientNetB4%20%2B%20EfficientNetB4ST%20%2B%20B4Att%20%2B%20B4AttST)

Таблица результатов моделей:

| Модель                | FaceForensics | ROOP | GHOST | RealImage | SmileAttribute | OldAgeAttribute | BaseReal | BaseFake |
|-----------------------|---------------|------|-------|-----------|-----------------|------------------|----------|----------|
| XceptionNet Precision  |       0       |  0  | 0.024 |      1    |        0        |         0         |  0.995   |  0.925   |
| XceptionNet Recall     |               |      |       |           |                 |                  |          |          |
| MesoNet Precision      |      0.04     | 0.05 |  0.07 |    0.96   |         0       |         0        |   0.96   |   0.96   |
| MesoNet Recall         |       1       |   1  |   1   |    0.97   |         0       |         0        |   0.97   |   0.94   |
| EfficientNetB4 Precision|    0.406      |0.047 | 0.163 |   0.971   |       0         |        0.02      |     -    |   -      |
| EfficientNetB4 Recall   |               |      |       |           |                 |                  |     -    |   -      |

**Общая структура проекта**  

[Tools](Tools) - Все дополнительные методы, использованные для обработки изображений, таких как изменение разрешения и обрезка изображений.

[GenerateDeepFakeRoopScripts](GenerateDeepFakeRoopScripts) - Файл и скрипт для запуска автоматической генерации модели ROOP.

[Datasets](Datasets) - Наборы данных, используемые в проекте. 
- [Destination](Datasets/destination) - Изображения для полной смены лиц, являющиеся целевым объектом замены.
- [Source](Datasets/source) - Изображения для полной смены лиц, являющиеся источником контекста (лица).
- [RoopData](Datasets/RoopData) - Сгенерированные изображения модели ROOP. Внутри папки есть подпапки источника и целевого объекта изображения.
- [GHOSTdata](Datasets/GHOSTdata) - Сгенерированные изображения модели GHOST. Источники и целевые объекты использовались как у ROOP.
- [Encoder4EditingData](Datasets/Encoder4EditingData) - Сгенерированные модифицированные изображения модели Encoder4Editing.
- [CelebaHQReal](Datasets/CelebaHQReal) - Демонстрационный вариант части реальных изображений.
- [FaceForensicsFakeImage](Datasets/FaceForensics/fake) - Фейковые изображения бенчмарка FF.

[AllModels](AllModels) - Все модели, используемые в проекте, содержит в себе две подпапки для Генеративных и Детектирующих моделей
- [ModelGenerate](AllModels/ModelGenerate) - Папка со всеми генеративными моделями
- [DeepFakeDetectionModels](AllModels/DeepFakeDetectionModels) - Папка со всеми детектирующими моделями


