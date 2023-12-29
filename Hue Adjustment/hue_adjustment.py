import argparse
import cv2
import os
import random
import numpy as np
from PIL import Image, ImageEnhance

# Define the directories
hr_dir = 'hr'
lr_dir = 'lr'
output_hr_dir = 'hr5'
output_lr_dir = 'lr5'

# Create output directories if they don't exist
os.makedirs(output_hr_dir, exist_ok=True)
os.makedirs(output_lr_dir, exist_ok=True)

# Define the range for the brightness adjustments
min_brightness = 0.6
max_brightness = 1.4

# Define the range for the contrast adjustments
min_contrast = 0.7
max_contrast = 1.3

# Define the range for the hue adjustments
min_hue = 0
max_hue = 180

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--duplicates', type=int, help='Number of duplicates to create', default=1)
parser.add_argument('-b', '--brightness', action='store_true', help='Enable brightness shifting')
parser.add_argument('-c', '--contrast', action='store_true', help='Enable contrast shifting')
parser.add_argument('-u', '--hue', action='store_true', help='Enable hue shifting')
args = parser.parse_args()

# Iterate over the images in the directories
for filename in os.listdir(hr_dir):
    if filename.endswith('.png'):
        # Load the images
        hr_image = cv2.imread(os.path.join(hr_dir, filename))
        lr_image = cv2.imread(os.path.join(lr_dir, filename))

        # Generate random adjustments and apply them
        for i in range(args.duplicates):
            print(f"Processing duplicate {i+1} for {filename}...")
            hr_image_copy = hr_image.copy()
            lr_image_copy = lr_image.copy()

            if args.brightness:
                brightness = random.uniform(min_brightness, max_brightness)
                hr_image_pil = Image.fromarray(cv2.cvtColor(hr_image_copy, cv2.COLOR_BGR2RGB))
                lr_image_pil = Image.fromarray(cv2.cvtColor(lr_image_copy, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Brightness(hr_image_pil)
                hr_image_copy = cv2.cvtColor(np.array(enhancer.enhance(brightness)), cv2.COLOR_RGB2BGR)
                enhancer = ImageEnhance.Brightness(lr_image_pil)
                lr_image_copy = cv2.cvtColor(np.array(enhancer.enhance(brightness)), cv2.COLOR_RGB2BGR)

            if args.contrast:
                contrast = random.uniform(min_contrast, max_contrast)
                hr_image_pil = Image.fromarray(cv2.cvtColor(hr_image_copy, cv2.COLOR_BGR2RGB))
                lr_image_pil = Image.fromarray(cv2.cvtColor(lr_image_copy, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Contrast(hr_image_pil)
                hr_image_copy = cv2.cvtColor(np.array(enhancer.enhance(contrast)), cv2.COLOR_RGB2BGR)
                enhancer = ImageEnhance.Contrast(lr_image_pil)
                lr_image_copy = cv2.cvtColor(np.array(enhancer.enhance(contrast)), cv2.COLOR_RGB2BGR)

            if args.hue:
                # Convert the image to HSV
                hr_image_hsv = cv2.cvtColor(hr_image_copy, cv2.COLOR_BGR2HSV)
                lr_image_hsv = cv2.cvtColor(lr_image_copy, cv2.COLOR_BGR2HSV)

                # Change the hue
                hue_shift = random.randint(min_hue, max_hue)
                hr_image_hsv[..., 0] = (hr_image_hsv[..., 0] + hue_shift) % 180
                lr_image_hsv[..., 0] = (lr_image_hsv[..., 0] + hue_shift) % 180

                # Convert the image back to BGR
                hr_image_copy = cv2.cvtColor(hr_image_hsv, cv2.COLOR_HSV2BGR)
                lr_image_copy = cv2.cvtColor(lr_image_hsv, cv2.COLOR_HSV2BGR)
                
            # Save the images
            base_filename, ext = os.path.splitext(filename)
            new_filename = f"{base_filename}_{i}{ext}"
            print(f"Saving new image: {new_filename}")
            cv2.imwrite(os.path.join(output_hr_dir, new_filename), hr_image_copy)
            cv2.imwrite(os.path.join(output_lr_dir, new_filename), lr_image_copy)