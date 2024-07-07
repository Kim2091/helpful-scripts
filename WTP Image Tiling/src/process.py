from os import listdir, makedirs
from os.path import exists, join

import cv2
import numpy as np
from chainner_ext import resize, ResizeFilter

from .tile_scripts import lin_tile, random_tile, overlap_tile, best_tile_list_index
from pepeline import read, save, cvt_color, CvtType, best_tile
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from numpy.random import shuffle

TILE_FUNC = {
    "linear": lin_tile,
    "random": random_tile
}


class Tiler:
    def __init__(self, config: dict):
        """
        Initializes the Tiler object with the given configuration.

        Args:
            config (dict): Configuration dictionary containing:
                - "in_folder" (str): Input folder path.
                - "out_folder" (str): Output folder path.
                - "tiler" (dict): Tiler configuration with type and other parameters.
                - "process" (str): Processing mode ("thread" or "process").
                - "num_work" (int): Number of workers.
                - "tile_size" (int, optional): Size of each tile. Defaults to 512.
                - "shuffle" (bool, optional): Whether to shuffle the tile indices. Defaults to False.

        Raises:
            ValueError: If mandatory keys are missing in the configuration.
        """
        self.in_folder = config.get("in_folder")
        self.out_folder = config.get("out_folder")
        tiler = config.get("tiler")
        self.process_map = config.get("process", "thread")
        self.num_work = config.get("num_work")
        self.real_name = config.get("real_name")
        if self.out_folder is None:
            raise ValueError("You didn't include out_folder in config")
        elif self.in_folder is None:
            raise ValueError("You didn't include in_folder in config")
        elif tiler is None:
            raise ValueError("You didn't include tiler in config")
        if not exists(self.out_folder):
            makedirs(self.out_folder)
        self.img_list = listdir(self.in_folder)
        self.tile_size = config.get("tile_size", 512)
        self.tiler_type = tiler.get("type", "linear")
        if self.tiler_type in ["linear", "random", "overlap"]:
            number_tiles = tiler.get("n_tiles")
            if self.tiler_type == "overlap":
                overlap = tiler.get("overlap", 0.25)
                self.img_dict, self.list_index = overlap_tile(self.in_folder, self.img_list, self.tile_size,
                                                              number_tiles, overlap)
            else:
                tile_func = TILE_FUNC[self.tiler_type]
                self.img_dict, self.list_index = tile_func(self.in_folder, self.img_list, self.tile_size, number_tiles)
            self.img_list = list(self.img_dict.keys())
        elif self.tiler_type == "best":
            self.scale = tiler.get("scale", 1)
            self.img_list = best_tile_list_index(self.in_folder, self.img_list, self.tile_size)
            self.list_index = self.img_list.copy()
        else:
            raise ValueError("Unknown type")
        if config.get("shuffle"):
            shuffle(self.list_index)

    def __tile(self, origin_img, tile_cord: list[int, int]) -> np.ndarray:
        """
        Extracts a tile from the original image based on given coordinates.

        Args:
            origin_img (np.ndarray): The original image array.
            tile_cord (list[int, int]): Coordinates of the top-left corner of the tile.

        Returns:
            np.ndarray: The extracted tile.
        """
        return origin_img[
               tile_cord[0]:tile_cord[0] + self.tile_size,
               tile_cord[1]:tile_cord[1] + self.tile_size
               ]

    def __name(self, img_name: str, cord: list | None) -> str:
        if self.real_name:
            base_name = ".".join(img_name.split(".")[:-1])
            if cord is None:
                name = base_name + ".png"
            else:
                name = f"{base_name}_{cord[0]}_{cord[1]}.png"

        else:
            if cord is None:
                name = str(self.list_index.index(img_name)) + ".png"
            else:
                name = str(self.list_index.index(img_name + str(cord))) + ".png"

        return name

    def best_tile(self, img_name: str) -> None:
        """
        Finds and saves the best tile for an image based on the Laplacian focus measure.

        Args:
            img_name (str): The image filename.
        """
        img = read(join(self.in_folder, img_name), 1, 0)
        img_shape = img.shape
        result_name = self.__name(img_name,None)
        if img_shape[0] == self.tile_size or img_shape[1] == self.tile_size:
            save(img, join(self.out_folder, result_name))
            return
        img_gray = cvt_color(img, CvtType.RGB2GrayBt2020)
        laplacian_abs = np.abs(cv2.Laplacian(img_gray, -1))
        if self.scale > 1:
            laplacian_abs = resize(laplacian_abs, (img_shape[1] // self.scale, img_shape[0] // self.scale),
                                   ResizeFilter.Box, False).squeeze()
            left_up_cord = best_tile(laplacian_abs, self.tile_size // self.scale) * self.scale
        else:
            left_up_cord = best_tile(laplacian_abs, self.tile_size // self.scale)
        save(img[left_up_cord[0]:left_up_cord[0] + self.tile_size, left_up_cord[1]:left_up_cord[1] + self.tile_size],
             join(self.out_folder, result_name))

    def process(self, img_name: str) -> None:
        """
        Processes an image by generating and saving tiles.

        Args:
            img_name (str): The image filename.
        """
        img_path = join(self.in_folder, img_name)
        img = read(img_path, None, 1)
        for tile_cord in self.img_dict[img_name]:
            out_name = self.__name(img_name, tile_cord)
            tile_img = self.__tile(img, tile_cord)
            save(tile_img, join(self.out_folder, out_name))

    def run(self):
        """
        Runs the tiling process using the specified processing method.
        """
        if self.tiler_type in ["linear", "random", "overlap"]:
            process = self.process
        elif self.tiler_type == "best":
            process = self.best_tile
        else:
            raise ValueError("Unknown type")
        if self.process_map == "thread":
            thread_map(process, self.img_list, max_workers=self.num_work, desc="Process")
        elif self.process_map == "process":
            process_map(process, self.img_list, max_workers=self.num_work, desc="Process")
        else:
            for img_name in tqdm(self.img_list, desc="Process"):
                process(img_name)
