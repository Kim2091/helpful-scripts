import os
import sys
from PIL import Image

def has_alpha_layer(image_path):
    try:
        image = Image.open(image_path)
        return image.mode.endswith('A')
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def find_images_with_alpha(folder_path):
    images_with_alpha = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.tiff', '.bmp', '.gif')):
                image_path = os.path.join(root, file)
                if has_alpha_layer(image_path):
                    images_with_alpha.append(image_path)
    return images_with_alpha

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"{folder_path} is not a valid directory.")
        sys.exit(1)

    images_with_alpha = find_images_with_alpha(folder_path)
    if images_with_alpha:
        print("Images with alpha layers:")
        for image_path in images_with_alpha:
            print(image_path)
    else:
        print("No images with alpha layers found.")