**ЭТАПЫ ВЫПОЛНЕНИЯ НАУЧНОЙ РАБОТЫ**  

**1. Скачиваем БД FaceForensics++**
Скачиваем фейковые изображения с официального репозитория FF 
https://github.com/ondyari/FaceForensics

После отправки заявки к авторам, на почту пришло письмо с ссылкой на скачивание бенчмарка с изображениями:
[FaceForensics_benchmark](http://kaldir.vc.in.tum.de/faceforensics_benchmark_images.zip)

В качестве реальных изображений использовались изображения 
[Celebrity-Face-Recognition-Dataset](https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset)

**2. Берем 2-3 готовых (других, не из списка FaceForensics++) решения по замене/изменению лиц:**  
Для замены лиц были выбраны сети **GHOST**, **ROOP**
- Сеть GHOST располагается в [AllModels/ModelGenerate/GHOST](AllModels/ModelGenerate/GHOST)
- Сеть GHOST располагается в [AllModels/ModelGenerate/roop](AllModels/ModelGenerate/roop)

Для преобразования использовалась сеть **Encoder4editing**
- Сеть Encoder4editing располагается в [AllModels/ModelGenerate/encoder4editing](AllModels/ModelGenerate/encoder4editing)
     
**3. Дополняем FaceForensics++ новыми генерациями, причем БД разбиваем на 2 блока - замена 
   лиц и изменение лица (в исходной версии БД это также есть)**
   
**4. Находим готовые решения (модель+веса) по детекции подделок из числа, указаных в таблице 2**


   
**5. Тренеруем готовые сети, проверяем результат. Совпал ли с указанными в статьях**


**Общая структура проекта**
[Tools](Tools) - Все дополнительные методы, использованные для обработки изображений, таких как изменение разрешения и обрезка изображений.

[GenerateDeepFakeRoopScripts](GenerateDeepFakeRoopScripts) - Файл и скрипт для запуска автоматической генерации модели ROOP.

[Datasets](Datasets) - Наборы данных, используемые в проекте. 
- [RoopData](Datasets/RoopData) - Сгенерированные изображения модели ROOP. Внутри папки есть подпапки источника и целевого объекта изображения.
- [Encoder4EditingData](Datasets/Encoder4EditingData) - Сгенерированные модифицированные изображения модели Encoder4Editing.
- [CelebaHQReal](Datasets/CelebaHQReal) - Демонстрационный вариант части реальных изображений.
- [FaceForensicsFakeImage](Datasets/FaceForensics/fake) - Фейковые изображения бенчмарка FF

[AllModels](AllModels) - Все модели, используемые в проекте, содержит в себе две подпапки для Генеративных и Детектирующих моделей





