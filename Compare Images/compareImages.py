import os
import shutil
import cv2
import concurrent.futures
import numpy as np
import logging

# Paths
hr_path = 'I:/Dataset/SwatKats/hr/1'
lr_path = 'I:/Dataset/SwatKats/lr/1'
output_path = 'I:/Dataset/SwatKats/Output1'

# Find the first pair of images to ensure paths are correct
try:
    hr_files = os.listdir(hr_path)
    lr_files = os.listdir(lr_path)
    common_files = set(hr_files) & set(lr_files)
    sample_file = next(iter(common_files))
    hr_sample_path = os.path.join(hr_path, sample_file)
    lr_sample_path = os.path.join(lr_path, sample_file)
except StopIteration:
    hr_sample_path = lr_sample_path = "No common files found in HR and LR paths"

# Confirm paths are correct
print(f"HR Path: {hr_path} (Sample file: {hr_sample_path})")
print(f"LR Path: {lr_path} (Sample file: {lr_sample_path})")
print(f"Output Path: {output_path}")
confirm = input("Are these paths correct? (yes/no): ")
if confirm.lower() != 'yes':
    print("Please correct the paths and run the script again.")
    exit()

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
                filenames = os.listdir(folder1)
                futures = list(executor.map(self.process_image, filenames, [folder1]*len(filenames), [folder2]*len(filenames), [dest_folder]*len(filenames)))
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
