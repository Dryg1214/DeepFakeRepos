import argparse
import os
import subprocess

def process_files(input_folder, destination_folder, output_folder):
    # Получаем список файлов в папке input_folder
    input_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    dis_files = [f for f in os.listdir(destination_folder) if os.path.isfile(os.path.join(destination_folder, f))]

    num_destinations = len(dis_files)
    num_d = 0
    for input_file in input_files:
        if num_d >= num_destinations:
            return 
        destination_file = dis_files[num_d]
        # Составляем полные пути к входному и выходному файлам
        input_source = os.path.join(input_folder, input_file)
        destination_path = os.path.join(destination_folder, destination_file)
        output_path = os.path.join(output_folder, input_file.replace(".jpg", "_processed.jpg"))

        # Ваш код обработки файлов, например, вызов команды из строки
        #command = f"python run.py -s {input_source} -t dis.jpg -o {output_path} --keep-frames --keep-fps --temp-frame-quality 1 --output-video-quality 1 --execution-provider cuda --frame-processor face_swapper face_enhancer"
        #command = f"'C:/Users/Dungeon Master/.conda/envs/generate_deepfake/python.exe' F:/DeepFake/roop/run.py -s {input_source} -t {destination_path} -o {output_path} --keep-frames --keep-fps --temp-frame-quality 1 --output-video-quality 1 --execution-provider cuda --frame-processor face_swapper face_enhancer"
        num_d = num_d + 1

        command = [
            'C:/Users/Dungeon Master/.conda/envs/generate_deepfake/python.exe',
            'F:/DeepFake/roop/run.py',
            '-s', input_source,
            '-t', destination_path,
            '-o', output_path,
            '--keep-frames',
            '--keep-fps',
            '--temp-frame-quality', '1',
            '--output-video-quality', '1',
            '--execution-provider', 'cuda',
            '--frame-processor', 'face_swapper', 'face_enhancer'
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        # Вывод результатов (если нужно)
        print(result.stdout)
        print(result.stderr)

        print("Running")
        #os.system(command)

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Process files in a folder.")
    parser.add_argument("-i", "--input-folder", required=True, help="Input folder path")
    parser.add_argument("-d", "--destination-folder", required=True, help="destination folder path")
    parser.add_argument("-o", "--output-folder", required=True, help="Output folder path")
    
    args = parser.parse_args()

    # Вызываем функцию обработки файлов
    process_files(args.input_folder, args.destination_folder, args.output_folder)

if __name__ == "__main__":
    main()