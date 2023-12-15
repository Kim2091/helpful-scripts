import os
import shutil
import cv2
import concurrent.futures
import numpy as np
import logging

def confirm_paths(hr_path, lr_path, output_path):
    # Find the first pair of images to ensure paths are correct
    try:
        hr_files = {os.path.relpath(os.path.join(root, file), hr_path) for root, dirs, files in os.walk(hr_path) for file in files}
        lr_files = {os.path.relpath(os.path.join(root, file), lr_path) for root, dirs, files in os.walk(lr_path) for file in files}
        common_files = hr_files & lr_files
        sample_file = next(iter(common_files))
    except StopIteration:
        sample_file = "No common files found in HR and LR paths"

    # Confirm paths are correct
    print(f"HR Path: {hr_path} (Sample file: {os.path.join(hr_path, sample_file)})")
    print(f"LR Path: {lr_path} (Sample file: {os.path.join(lr_path, sample_file)})")
    print(f"Output Path: {output_path}")
    confirm = input("Are these paths correct? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Please correct the paths and run the script again.")
        exit()

# Paths
hr_path = 'path\to\hr\folder'
lr_path = 'path\to\lr\folder'
output_path = 'path\to\output\folder'

confirm_paths(hr_path, lr_path, output_path)

# Set up logging
log_file_path = os.path.join(output_path, 'image_comparator.log')
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()])

# List to store 'Moved' messages
moved_files = []

class ImageComparator:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def overlay_images(self, img1_path, img2_path, dest_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        cv2.imwrite(dest_path, overlay)

    def compare_images(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        _, shift = cv2.phaseCorrelate(np.float32(img1), np.float32(img2))
        score = np.linalg.norm(shift)
        logging.info(f'{img1_path}, {img2_path}, Score: {score}')
        return score

    def process_image(self, filename, folder1, folder2, dest_folder):
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        if os.path.isfile(img1_path) and os.path.isfile(img2_path):
            score = self.compare_images(img1_path, img2_path)

            if score < self.threshold:
                # Get the relative paths of the source images
                rel_path1 = os.path.relpath(img1_path, folder1)
                rel_path2 = os.path.relpath(img2_path, folder2)

                # Create the destination paths using the relative paths
                dest_path1 = os.path.join(dest_folder, 'hr', rel_path1)
                dest_path2 = os.path.join(dest_folder, 'lr', rel_path2)
                overlay_path = os.path.join(dest_folder, 'overlays', filename)
                
                os.makedirs(os.path.dirname(dest_path1), exist_ok=True)
                os.makedirs(os.path.dirname(dest_path2), exist_ok=True)
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)

                shutil.move(img1_path, dest_path1)
                shutil.move(img2_path, dest_path2)
                self.overlay_images(dest_path1, dest_path2, overlay_path)
                moved_message = f'Moved {filename} from {img1_path} and {img2_path} to {dest_folder} due to low similarity score'
                moved_files.append(moved_message)
                logging.info(moved_message)

    def scan_and_compare(self, folder1, folder2, dest_folder):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for dirpath, _, filenames in os.walk(folder1):
                    # Get the relative path to the current directory from folder1
                    rel_dir = os.path.relpath(dirpath, folder1)
                    
                    # Construct the corresponding directory path in folder2
                    dirpath2 = os.path.join(folder2, rel_dir)
                    
                    # Check if the corresponding directory exists in folder2
                    if os.path.exists(dirpath2):
                        futures = list(executor.map(self.process_image, filenames, [dirpath]*len(filenames), [dirpath2]*len(filenames), [dest_folder]*len(filenames)))
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            return

# Usage
comparator = ImageComparator()
comparator.scan_and_compare(os.path.normpath(hr_path), os.path.normpath(lr_path), os.path.normpath(output_path))

# Log moved files at the end of the log file
if moved_files:
    for moved_file in moved_files:
        logging.info(moved_file)
else:
    logging.info("No images were moved.")
