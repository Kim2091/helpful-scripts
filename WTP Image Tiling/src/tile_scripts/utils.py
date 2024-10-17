from pepeline import read_size
from os.path import join


def img_size(in_folder: str, img_name: str):
    return read_size(join(in_folder, img_name))
