import os
import argparse
from PIL import Image
from tqdm import tqdm

# Supported image types
IMAGE_TYPES = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.tiff', '.webp']

def is_image_file(filename, file_type=None):
    filename_lower = filename.lower()
    if file_type:
        return filename_lower.endswith(file_type.lower())
    return any(filename_lower.endswith(image_type) for image_type in IMAGE_TYPES)

def check_image(file_path, in_depth):
    try:
        with Image.open(file_path) as img:
            if in_depth:
                img.load()
            else:
                img.verify()
        return True
    except Exception:
        return False

def search_for_corrupted_files(input_folder, file_type=None, in_depth=False):
    searched_files = []
    corrupted_files = []

    # First pass: count only image files for accurate progress bar
    image_files = []
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for filename in filenames:
            if is_image_file(filename, file_type):
                image_files.append(os.path.join(dirpath, filename))

    # Second pass: check images with accurate progress bar
    with tqdm(total=len(image_files), desc="Processing images", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for file_path in image_files:
            if not check_image(file_path, in_depth):
                corrupted_files.append(file_path)
            searched_files.append(file_path)
            pbar.update()

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

    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        exit(1)
    
    if not os.path.isdir(args.input_folder):
        print(f"Error: '{args.input_folder}' is not a directory.")
        exit(1)

    print(f"Scanning directory: {args.input_folder}")
    if args.file_type:
        print(f"Filtering for file type: {args.file_type}")
    if args.deep:
        print("Performing deep scan...")
    print()

    searched_files, corrupted_files = search_for_corrupted_files(args.input_folder, args.file_type, args.deep)
    
    # Print summary
    print(f"\nScan complete!")
    print(f"Total images scanned: {len(searched_files)}")
    print(f"Corrupted images found: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\nCorrupted files:")
        for file in corrupted_files:
            print(f"  - {file}")
    
    write_log(searched_files, corrupted_files)