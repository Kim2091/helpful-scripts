import os
import argparse
import numpy as np
import cv2
from multiprocessing import Pool
import math

def best_tile(laplacian_abs, tile_size):
    kernel = np.ones((tile_size, tile_size), dtype=np.float32) / (tile_size * tile_size)
    conv = cv2.filter2D(laplacian_abs, -1, kernel)
    y, x = np.unravel_index(np.argmax(conv), conv.shape)
    return np.array([y, x])

def process_image(image_path, output_folder, tile_size, num_tiles, grayscale, min_size, seed=None, use_best_tile=False, scale=1):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Unable to read image {image_path}")
            return

        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = img.shape[:2]

        if use_best_tile:
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img
            
            # Calculate the number of segments in each dimension
            segments = math.ceil(math.sqrt(num_tiles))
            segment_height = height // segments
            segment_width = width // segments

            best_tiles = []
            for i in range(segments):
                for j in range(segments):
                    if len(best_tiles) >= num_tiles:
                        break
                    
                    y_start = i * segment_height
                    x_start = j * segment_width
                    y_end = min((i + 1) * segment_height, height)
                    x_end = min((j + 1) * segment_width, width)
                    
                    segment = img_gray[y_start:y_end, x_start:x_end]
                    laplacian_abs = np.abs(cv2.Laplacian(segment, -1))
                    
                    if scale > 1:
                        laplacian_abs = cv2.resize(laplacian_abs, (segment.shape[1] // scale, segment.shape[0] // scale), 
                                                   interpolation=cv2.INTER_AREA)
                        best_tile_coord = best_tile(laplacian_abs, tile_size[0] // scale) * scale
                    else:
                        best_tile_coord = best_tile(laplacian_abs, tile_size[0])
                    
                    global_y = y_start + best_tile_coord[0]
                    global_x = x_start + best_tile_coord[1]
                    
                    if global_y + tile_size[0] <= height and global_x + tile_size[1] <= width:
                        best_tiles.append((global_y, global_x))
            
            for i, (y, x) in enumerate(best_tiles):
                tile = img[y:y + tile_size[0], x:x + tile_size[1]]
                output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_best_tile_{i}.png")
                cv2.imwrite(output_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            
            print(f"{len(best_tiles)} best tiles saved from {image_path}")
        else:
            tiles_per_row = width // tile_size[0]
            tiles_per_col = height // tile_size[1]
            if num_tiles == 0:
                num_tiles = tiles_per_row * tiles_per_col
            tiles_saved = 0
            tile_indices = np.arange(tiles_per_row * tiles_per_col)
            rng = np.random.default_rng(seed)
            rng.shuffle(tile_indices)
            
            for i in range(tiles_per_row * tiles_per_col):
                if tiles_saved >= num_tiles:
                    break
                idx = tile_indices[i]
                row = idx // tiles_per_row
                col = idx % tiles_per_row
                x = col * tile_size[0]
                y = row * tile_size[1]
                tile = img[y:y+tile_size[1], x:x+tile_size[0]]
                if min_size and (tile.shape[1] < min_size[0] or tile.shape[0] < min_size[1]):
                    continue
                output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_{row}_{col}.png")
                cv2.imwrite(output_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                tiles_saved += 1
            
            print(f"{tiles_saved} tiles saved from {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_folder(input_folder, output_folder, tile_size, num_tiles, grayscale, min_size, seed, use_best_tile, scale):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pool = Pool()
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_path = os.path.join(root, file)
                pool.apply_async(process_image, (image_path, output_folder, tile_size, num_tiles, grayscale, min_size, seed, use_best_tile, scale))
    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tile images from a folder.')
    parser.add_argument('input_folder', type=str, help='Input folder containing images')
    parser.add_argument('output_folder', type=str, help='Output folder to save tiles')
    parser.add_argument('-t', '--tile-size', type=int, nargs=2, help='Size of tiles (width height)', default=(512, 512))
    parser.add_argument('-n', '--num-tiles', type=int, help='Number of tiles to save per image', default=1)
    parser.add_argument('-g', '--grayscale', action='store_true', help='Convert tiles to grayscale')
    parser.add_argument('-m', '--min-size', type=int, nargs=2, help='Minimum size of tiles to save (width height)')
    parser.add_argument('-s', '--seed', type=int, help='Seed for random number generator to ensure reproducible results')
    parser.add_argument('-b', '--best-tile', action='store_true', help='Use best tile selection based on Laplacian intensity')
    parser.add_argument('-c', '--scale', type=int, default=1, help='Scale factor for best tile selection')
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_folder, args.tile_size, args.num_tiles,
                   args.grayscale, args.min_size, args.seed, args.best_tile, args.scale)
