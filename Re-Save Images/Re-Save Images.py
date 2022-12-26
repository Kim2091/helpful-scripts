import argparse
import concurrent.futures
import os
import cv2
from tqdm import tqdm

def process_image(input_path, dest_dir, grayscale):
    try:
        # Read the image
        image = cv2.imread(input_path)

        # Convert the image to grayscale if specified
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the base name of the file and split it to get the file name and extension
        filename_only = os.path.basename(input_path)
        file_name, file_ext = os.path.splitext(filename_only)

        # Create the output path
        new_path = os.path.join(dest_dir, file_name + ".png")

        # Write the image to the output path
        cv2.imwrite(new_path, image)
    except Exception as e:
        # Print an error message if there was an issue processing the image
        print(f"Error processing {input_path}: {e}")

def main(input_dir, dest_dir, grayscale):
    # Create a list of input paths
    input_paths = []
    with os.scandir(input_dir) as entries:
        for entry in entries:
            # Skip non-regular files and directories
            if not entry.is_file() or not entry.name.endswith(".png"):
                continue
            input_paths.append(entry.path)

    # Use a progress bar to track the progress of the image processing
    with tqdm(total=len(input_paths)) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Process the images in parallel
            futures = []
            for input_path in input_paths:
                futures.append(executor.submit(process_image, input_path, dest_dir, grayscale))
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing the input images")
    parser.add_argument("dest_dir", help="Directory to store the output images")
    parser.add_argument("--grayscale", action="store_true", help="Convert the output images to grayscale")
    args = parser.parse_args()

    # Call the main function
    main(args.input_dir, args.dest_dir, args.grayscale)
