import os
import argparse
from PIL import Image
from tqdm import tqdm

# Supported image types
IMAGE_TYPES = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.tiff', '.webp']

def is_image_file(filename, file_type=None):
    if file_type:
        return filename.endswith(file_type)
    return any(filename.endswith(image_type) for image_type in IMAGE_TYPES)

def check_image(file_path, in_depth):
    try:
        with Image.open(file_path) as img:
            if in_depth:
                img.load()
            else:
                img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def search_for_corrupted_files(input_folder, file_type=None, in_depth=False):
    searched_files = []
    corrupted_files = []

    # Get the total number of files to be searched for the progress bar
    total_files = sum([len(files) for r, d, files in os.walk(input_folder)])

    with tqdm(total=total_files, desc="Processing files", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for dirpath, dirnames, filenames in os.walk(input_folder):
            for filename in filenames:
                if is_image_file(filename, file_type):
                    file_path = os.path.join(dirpath, filename)
                    if not check_image(file_path, in_depth):
                        corrupted_files.append(file_path)
                    searched_files.append(file_path)
                    pbar.update()  # update progress bar

    return searched_files, corrupted_files

def write_log(searched_files, corrupted_files):
    with open('searchlog.txt', 'w') as log_file:
        log_file.write('Searched Files:\n')
        log_file.write('\n'.join(searched_files))
        log_file.write('\n\nCorrupted Files:\n')
        log_file.write('\n'.join(corrupted_files))

    print('Search log has been created.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for corrupted files in a directory.')
    parser.add_argument('input_folder', type=str, help='The input folder to search.')
    parser.add_argument('-f', '--file_type', type=str, default=None, help='The file type to search for.')
    parser.add_argument('-d', '--deep', action='store_true', help='Perform an in-depth scan.')
    args = parser.parse_args()

    searched_files, corrupted_files = search_for_corrupted_files(args.input_folder, args.file_type, args.deep)
    write_log(searched_files, corrupted_files)
