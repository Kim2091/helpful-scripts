*Written by ChatGPT*

This script allows quick tiling of your images with multithreading. Put your images in an `images` folder, and it'll output the results in `output`. ~2000% faster than MagickUtils.

**Features:**
* Split images into tiles of any size
* Optional Grayscale output
* Option to skip saving tiles based on their size
* Option to skip saving tiles that are mostly black or white

**Planned:**
* Progress bar

**Required Packages:**
* pillow
* argparse

`python Image Tiling Script v2.py /path/to/image/folder --tile-size 512 --color-type g --skip-tiles --min-tile-size 256 /path/to/output/folder`

    /path/to/image/folder: This is the path to the folder containing the images to be tiled. All subfolders will be searched recursively.
    --tile-size 512: This specifies the size of the tiles in pixels. The default value is 512. Tiles will be square.
    --color-type g: This specifies that the output tiles should be grayscale. Use "g" for grayscale or "c" for color. The default is color.
    --skip-tiles: This flag causes the script to skip tiles that are predominantly black or white. A tile is considered predominantly black or white if its average pixel value is less than 5 or greater than 250, respectively.
    --min-tile-size 256: This flag limits the size of the tiles to a minimum value in pixels. Any tiles smaller than this size will be skipped. For example, if the tile size is 512 and the minimum tile size is 256, any tiles that are 256x256 or smaller will be skipped.
    /path/to/output/folder: This is the path to the output directory where the tiled images will be saved. The directory will be created if it does not already exist.
