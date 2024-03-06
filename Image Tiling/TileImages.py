import os
import argparse
from PIL import Image
from multiprocessing import Pool
import numpy as np
from skimage import color, filters

def process_image(image_path, output_folder, tile_size, num_tiles, grayscale, min_size, edge_threshold=None):
    try:
        image = Image.open(image_path)
        width, height = image.size
        if grayscale:
            image = image.convert('L')
        tiles_per_row = width // tile_size[0]
        tiles_per_col = height // tile_size[1]
        if num_tiles == 0:
            num_tiles = tiles_per_row * tiles_per_col
        tiles_saved = 0
        tiles_skipped = 0
        tile_indices = np.arange(tiles_per_row * tiles_per_col)
        rng = np.random.default_rng()
        rng.shuffle(tile_indices)
        
        # Calculate the edge threshold for the entire image if not specified
        if edge_threshold:
            image_rgb = np.array(image)[..., :3]  # Discard the alpha channel if it exists
            image_gray = color.rgb2gray(image_rgb)  # Convert the RGB image to grayscale
            edges = filters.sobel(image_gray)  # Apply the Sobel operator
            edge_threshold = np.sum(edges) * 0.01  # Set the threshold to 1% of the total edge intensity
        else:
            edge_threshold = None
        
        for i in range(tiles_per_row * tiles_per_col):
            if tiles_saved >= num_tiles:
                break
            idx = tile_indices[i]
            row = idx // tiles_per_row
            col = idx % tiles_per_row
            x = col * tile_size[0]
            y = row * tile_size[1]
            tile = image.crop((x, y, x + tile_size[0], y + tile_size[1]))
            tile_rgb = np.array(tile)[..., :3]  # Discard the alpha channel if it exists
            tile_gray = color.rgb2gray(tile_rgb)  # Convert the RGB image to grayscale
            edges = filters.sobel(tile_gray)  # Apply the Sobel operator
            if edge_threshold is not None and np.sum(edges) < edge_threshold:
                tiles_skipped += 1
                continue
            if min_size and (tile.width < min_size[0] or tile.height < min_size[1]):
                continue
            output_path = f"{output_folder}/{os.path.basename(image_path)}_{row}_{col}.png"
            tile.save(output_path)
            tiles_saved += 1
        if edge_threshold:
            print(f"{tiles_saved} tiles saved and {tiles_skipped} tiles skipped due to low edge content from {image_path}")
        else:
            print(f"{tiles_saved} tiles saved from {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_folder(input_folder, output_folder, tile_size, num_tiles, grayscale, min_size, std_dev_threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pool = Pool()
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.webp')):
                image_path = os.path.join(root, file)
                pool.apply_async(process_image, (image_path, output_folder, tile_size, num_tiles, grayscale, min_size, std_dev_threshold))
    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tile images from a folder.')
    parser.add_argument('input_folder', type=str,
                        help='Input folder containing images')
    parser.add_argument('output_folder', type=str,
                        help='Output folder to save tiles')
    parser.add_argument('-t', '--tile-size', type=int, nargs=2,
                        help='Size of tiles (width height)', default=(512, 512))
    parser.add_argument('-n', '--num-tiles', type=int,
                        help='Number of tiles to save per image', default=0)
    parser.add_argument('-g', '--grayscale', action='store_true',
                        help='Convert tiles to grayscale')
    parser.add_argument('-m', '--min-size', type=int, nargs=2,
                        help='Minimum size of tiles to save (width height)')
    parser.add_argument('-e', '--edge-threshold', action='store_true',
                        help='Enable edge intensity threshold for skipping tiles with low content.')
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder,
                   args.tile_size, args.num_tiles,
                   args.grayscale, args.min_size,
                   args.edge_threshold)
