This script allows quick tiling of your images with multithreading. ~2000% faster than MagickUtils.

**Features:**

* Split images into tiles of any size
* Generate a specified number of tiles per image or generate as many non-overlapping tiles as possible
* Convert tiles to grayscale
* Skip tiles below a minimum size
* Choose between random and "best" tile selection methods
* Set a scale factor for "best" tile selection to improve performance

**Required Packages:**

* opencv-python
* numpy

How to use: `python TileImages.py /path/to/image/folder /path/to/output/folder`

**Additional Arguments:**
* `-t` -  Specify the size of the tiles to take from the image. Use like so: `-t 512 512`
* `-n` - The number of tiles to save per image. This will take a set amount of tiles from each image, pulled from random locations or using the "best" selection method. This helps with increasing variety in your dataset without saving unnecessary tiles
* `-g` - Saves your images in grayscale
* `-m` - Sets a minimum size for tiles. If any tiles are below the specified size, they will not be saved
* `--selection` - Choose the tile selection method: 'random' (default) or 'best'
* `-s` - Sets a seed for the random number generator. This is for usage with random selection method
* `-c` - Sets a scale factor for the "best" tile selection method to improve performance

**Credits:**
Thank you @umzi2 for the best_tile code, which was used as a basis for the "best" function
