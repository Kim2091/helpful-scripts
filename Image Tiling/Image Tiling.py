import time
import threading
from PIL import Image
import os
import math

# Specify the folder containing the images
folder = 'images'

# Get a list of all image files in the folder
file_list = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.gif'))]

# Prompt the user for the tile size
tile_size = int(input('Enter the tile size: '))

# Start the timer
start_time = time.perf_counter()

# Create the output directory if it does not already exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a function that processes a single image
def process_image(file):
    # Load the image
    image = Image.open(os.path.join(folder, file))
    
    # Calculate the size of the image
    width = math.ceil(image.width / tile_size) * tile_size
    height = math.ceil(image.height / tile_size) * tile_size

    # Remove the file extension from the file name
    file_name, file_extension = os.path.splitext(file)

    # Calculate the number of rows and columns
    num_rows = math.ceil(image.height / tile_size)
    num_cols = math.ceil(image.width / tile_size)

    # Iterate through the rows and columns, and save each tile as a separate image file
    for row in range(num_rows):
        for col in range(num_cols):
            x1 = col * tile_size
            y1 = row * tile_size
            x2 = min(x1 + tile_size, image.width)  # Adjust the right edge of the tile if it goes beyond the width of the image
            y2 = min(y1 + tile_size, image.height)  # Adjust the bottom edge of the tile if it goes beyond the height of the image
            tile = image.crop((x1, y1, x2, y2))
            output_file = f'tile_{row}_{col}_{file_name}.png'
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
