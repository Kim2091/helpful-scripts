import os
import argparse
from PIL import Image
from multiprocessing import Pool
import random
import numpy as np

def process_image(image_path, output_folder, tile_size, num_tiles, grayscale, min_size, skip_black_white):
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
        tile_indices = np.arange(tiles_per_row * tiles_per_col)
        rng = np.random.default_rng()
        rng.shuffle(tile_indices)
        for i in tile_indices:
            if tiles_saved >= num_tiles:
                break
            row = i // tiles_per_row
            col = i % tiles_per_row
            x = col * tile_size[0]
            y = row * tile_size[1]
            tile = image.crop((x, y, x + tile_size[0], y + tile_size[1]))
            if min_size and (tile.width < min_size[0] or tile.height < min_size[1]):
                continue
            if skip_black_white:
                extrema = tile.getextrema()
                if isinstance(extrema[0], tuple):
                    is_black = all([x == 0 for x in extrema[0]])
                    is_white = all([x == 255 for x in extrema[1]])
                else:
                    is_black = extrema[0] == 0
                    is_white = extrema[1] == 255
                if is_black or is_white:
                    continue
            output_path = f"{output_folder}/{os.path.basename(image_path)}_{row}_{col}.png"
            tile.save(output_path)
            tiles_saved += 1
        print(f"{tiles_saved} tiles saved from {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_folder(input_folder, output_folder, tile_size, num_tiles, grayscale, min_size, skip_black_white):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pool = Pool()
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.webp')):
                image_path = os.path.join(root, file)
                pool.apply_async(process_image, (image_path, output_folder, tile_size, num_tiles, grayscale, min_size, skip_black_white))
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
    parser.add_argument('-s', '--skip-black-white', action='store_true',
                        help='Skip black and white tiles')
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder,
                   args.tile_size, args.num_tiles,
                   args.grayscale, args.min_size,
                   args.skip_black_white)
