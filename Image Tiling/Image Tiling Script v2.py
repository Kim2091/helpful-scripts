import time
import threading
from PIL import Image
import os
import math
import argparse

# Set up the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('folder', help='The folder containing the images')
parser.add_argument('--tile-size', '-t', type=int, default=512, help='The size of the tiles. Default is 512')
parser.add_argument('--skip-tiles', '-s', action='store_true', help='Skip tiles that are predominantly black or white')
parser.add_argument('--min-tile-size', '-m', type=int, help='Limit tile size to a minimum value')
parser.add_argument('--color-type', '-c', choices=['g', 'c'], help='Output grayscale or color tiles')
parser.add_argument('output_dir', help='The output directory')
args = parser.parse_args()

# Create the output directory if it does not already exist
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all image files in the folder and its subfolders
file_list = []
for root, dirs, files in os.walk(args.folder):
    for file in files:
        if file.endswith(('.jpg', '.png', '.gif')):
            file_list.append(os.path.join(root, file))

# Start the timer
start_time = time.perf_counter()

# Define a function that processes a single image
def process_image(file):
    
    # Load the image
    try:
        image = Image.open(os.path.join(args.folder, file))
    except OSError:
        print(f'Error: Could not open {file}. Skipping this image.')
        return
    
    # Calculate the size of the image
    width = math.ceil(image.width / args.tile_size) * args.tile_size
    height = math.ceil(image.height / args.tile_size) * args.tile_size

    # Remove the file extension from the file name
    file_name, file_extension = os.path.splitext(file)

    # Calculate the number of rows and columns
    num_rows = math.ceil(image.height / args.tile_size)
    num_cols = math.ceil(image.width / args.tile_size)

    # Iterate through the rows and columns, and save each tile as a separate image file
    for row in range(num_rows):
        for col in range(num_cols):
            x1 = col * args.tile_size
            y1 = row * args.tile_size
            x2 = min(x1 + args.tile_size, image.width)  # Adjust the right edge of the tile if it goes beyond the width of the image
            y2 = min(y1 + args.tile_size, image.height)  # Adjust the bottom edge of the tile if it goes beyond the height of the image
            tile = image.crop((x1, y1, x2, y2))
            if tile.width < args.tile_size or tile.height < args.tile_size:
                continue  # Skip this tile if it is smaller than the specified tile size

            # Convert the tile to grayscale if the output type is set to grayscale
            if args.color_type == 'g':
                tile = tile.convert("L")

            # Calculate the average pixel value
            avg_value = sum(sum(x) for x in tile.getdata()) / len(tile.getdata())

            # Skip the tile if it is predominantly black or white (if the skip_tiles flag is set)
            if args.skip_tiles and (avg_value < 5 or avg_value > 250):
                continue

            # Skip this tile if it is smaller than the specified minimum tile size (if the min_tile_size flag is set)
            if args.min_tile_size and (tile.width < args.min_tile_size or tile.height < args.min_tile_size):
                continue

            output_file = f'tile_{row}_{col}_{os.path.basename(file_name)}.png'
            tile.save(os.path.join(output_dir, output_file))

# Create a list of threads
threads = []

# Iterate through the list of files
for file in file_list:
    # Create a new thread and add it to the list
    thread = threading.Thread(target=process_image, args=(file,))
    threads.append(thread)

# Start all the threads
for thread in threads:
    thread.start()

# Wait for all the threads to complete
for thread in threads:
    thread.join()

# Stop the timer
end_time = time.perf_counter()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Display a confirmation message and the elapsed time
print(f'Done! Processed {len(file_list)} images in {elapsed_time:.2f} seconds.')
