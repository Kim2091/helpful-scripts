from tqdm import tqdm
from .utils import img_size
import numpy as np
from numpy.random import randint


def lin_tile(input_folder: str, img_list: list[str], tile_size: int, number_tiles: int | None) -> (dict, list):
    """
    Generates a linear grid of tiles for each image in the provided list.

    Args:
        input_folder (str): Path to the folder containing images.
        img_list (list[str]): List of image filenames.
        tile_size (int): Size of each tile (both width and height).
        number_tiles (int | None): Maximum number of tiles to generate per image.
                                   If None, defaults to 10000.

    Returns:
        (dict, list): A dictionary with image filenames as keys and arrays of tile coordinates as values,
                      and a list of indices of generated tiles.
    """
    if number_tiles is None:
        number_tiles = 10000
    img_dict = {}
    list_index = []
    for img_name in tqdm(img_list, desc="Tile generation"):
        try:
            w, h = img_size(input_folder, img_name)
        except OSError as e:
            continue
        if w < tile_size or h < tile_size:
            continue
        x_cords = np.arange(0, w // tile_size) * tile_size
        y_cords = np.arange(0, h // tile_size) * tile_size
        x_grid, y_grid = np.meshgrid(x_cords, y_cords)

        tiles = np.column_stack((y_grid.ravel(), x_grid.ravel()))[:number_tiles]
        list_index.extend([img_name + str(tile) for tile in tiles])
        img_dict[img_name] = tiles
    return img_dict, list_index


def random_tile(input_folder: str, img_list: list[str], tile_size: int, number_tiles: int | None) -> (dict, list):
    """
    Generates a random set of tiles for each image in the provided list.

    Args:
        input_folder (str): Path to the folder containing images.
        img_list (list[str]): List of image filenames.
        tile_size (int): Size of each tile (both width and height).
        number_tiles (int | None): Number of tiles to generate per image.
                                   If None, defaults to 1.

    Returns:
        (dict, list): A dictionary with image filenames as keys and arrays of tile coordinates as values,
                      and a list of indices of generated tiles.
    """
    if number_tiles is None:
        number_tiles = 1
    img_dict = {}
    list_index = []
    for img_name in tqdm(img_list, desc="Tile generation"):
        w, h = img_size(input_folder, img_name)
        if w < tile_size or h < tile_size:
            continue
        x_cords = randint(0, w - tile_size - 1, number_tiles)
        y_cords = randint(0, h - tile_size - 1, number_tiles)

        tiles = np.column_stack((y_cords.ravel(), x_cords.ravel()))
        list_index.extend([img_name + str(tile) for tile in tiles])
        img_dict[img_name] = tiles
    return img_dict, list_index


def overlap_tile(input_folder: str, img_list: list[str], tile_size: int, number_tiles: int | None,
                 overlap: int = 0.25) -> (dict, list):
    """
    Generates a set of overlapping tiles for each image in the provided list.

    Args:
        input_folder (str): Path to the folder containing images.
        img_list (list[str]): List of image filenames.
        tile_size (int): Size of each tile (both width and height).
        number_tiles (int | None): Maximum number of tiles to generate per image.
                                   If None, defaults to 10000.
        overlap (int): Overlap fraction between tiles. Defaults to 0.25.

    Returns:
        (dict, list): A dictionary with image filenames as keys and arrays of tile coordinates as values,
                      and a list of indices of generated tiles.
    """
    if number_tiles is None:
        number_tiles = 10000
    img_dict = {}
    list_index = []
    for img_name in tqdm(img_list, desc="Tile generation"):
        w, h = img_size(input_folder, img_name)
        if w < tile_size or h < tile_size:
            continue
        x_cords = np.arange(0, w // tile_size, overlap) * tile_size
        y_cords = np.arange(0, h // tile_size, overlap) * tile_size
        x_grid, y_grid = np.meshgrid(x_cords.astype(np.uint32), y_cords.astype(np.uint32))

        tiles = np.column_stack((y_grid.ravel(), x_grid.ravel()))[:number_tiles]
        list_index.extend([img_name + str(tile) for tile in tiles])
        img_dict[img_name] = tiles
    return img_dict, list_index


def best_tile_list_index(input_folder: str, img_list: list[str], tile_size: int) -> list:
    """
    Generates a list of image names that can accommodate tiles of the specified size.

    Args:
        input_folder (str): Path to the folder containing images.
        img_list (list[str]): List of image filenames.
        tile_size (int): Size of each tile (both width and height).

    Returns:
        list: A list of image filenames that can accommodate tiles of the specified size.
    """
    list_index = []
    for img_name in tqdm(img_list, desc="Tile generation"):
        w, h = img_size(input_folder, img_name)
        if w < tile_size or h < tile_size:
            continue
        list_index.append(img_name)
    return list_index
