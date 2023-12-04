import os
import shutil
import argparse
import random
from tqdm import tqdm

def move_files(source_folder, destination_folder, move_half=False, file_extension=None):
    for root, dirs, files in os.walk(source_folder):
        files_to_move = []
        for file in files:
            if file_extension and not file.endswith(file_extension):
                continue
            files_to_move.append(os.path.join(root, file))

        random.shuffle(files_to_move)

        if move_half:
            files_to_move = files_to_move[:len(files_to_move)//2]

        for source_path in tqdm(files_to_move, desc="Moving files", unit="file"):
            relative_path = os.path.relpath(source_path, source_folder)
            destination_path = os.path.join(destination_folder, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.move(source_path, destination_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move files from a folder to a secondary folder.")
    parser.add_argument("source_folder", help="Path to the source folder")
    parser.add_argument("destination_folder", help="Path to the destination folder")
    parser.add_argument("-m", "--move_half", action="store_true", help="Move half of all files")
    parser.add_argument("-f", "--file_extension", help="Move only files with the specified extension type")

    args = parser.parse_args()
    print(f"Source folder: {args.source_folder}")
    print(f"Destination folder: {args.destination_folder}")
    move_files(args.source_folder, args.destination_folder, args.move_half, args.file_extension)
