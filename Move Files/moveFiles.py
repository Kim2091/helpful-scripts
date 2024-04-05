import os
import shutil
import argparse
import random
from tqdm import tqdm

def move_files(source_folder, destination_folder, move_percentage=100, file_extension=None, seed=None):
    # Set the random seed to ensure consistent file selection
    if seed is not None:
        random.seed(seed)

    for root, dirs, files in os.walk(source_folder):
        files_to_move = []
        for file in files:
            if file_extension and not file.endswith(file_extension):
                continue
            files_to_move.append(os.path.join(root, file))

        random.shuffle(files_to_move)
        num_files_to_move = int(len(files_to_move) * (move_percentage / 100))
        files_to_move = files_to_move[:num_files_to_move]

        for source_path in tqdm(files_to_move, desc="Moving files", unit="file"):
            relative_path = os.path.relpath(source_path, source_folder)
            destination_path = os.path.join(destination_folder, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.move(source_path, destination_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move files from a folder to a secondary folder.")
    parser.add_argument("source_folder", help="Path to the source folder")
    parser.add_argument("destination_folder", help="Path to the destination folder")
    parser.add_argument("-p", "--move_percentage", type=float, default=100, help="Percentage of files to move (default: 100%)")
    parser.add_argument("-f", "--file_extension", help="Move only files with the specified extension type")
    parser.add_argument("-s", "--seed", type=int, help="Seed value to ensure consistent file selection")
    args = parser.parse_args()

    print(f"Source folder: {args.source_folder}")
    print(f"Destination folder: {args.destination_folder}")
    move_files(args.source_folder, args.destination_folder, args.move_percentage, args.file_extension, args.seed)
