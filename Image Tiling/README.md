*Written with the assistance of Bing*

This script allows quick tiling of your images with multithreading. Just provide and input and output folder. ~2000% faster than MagickUtils.

**Features:**

* Split images into tiles of any size
* Generate a specified number of tiles per image or generate as many non-overlapping tiles as possible
* Convert tiles to grayscale
* Skip black and white tiles
* Enforce a minimum size for the tiles

**Required Packages:**

* pillow

How to use: `python TileImages.py /path/to/image/folder /path/to/output/folder`

**Additional Arguments:**
* `-t` -  Specify the size of the tiles to take from the image. Use like so: `-t 512 512`
* `-n` - The number of tiles to save per image. This will take a set amount of tiles from each image, pulled from random locations. This helps with increasing variety in your dataset without saving unnecessary tiles
* `-g` - Saves your images in grayscale
* `-m` - Sets a minimum size for tiles. If any tiles are below the specified size, they will not be saved
* `-e` - Skip tiles based on edge intensity threshold calculated with Sobel edge detection. Allows you to skip tiles if they contain little valid content. Easily triggered by noise however.
* `-s` - Sets a seed for the random number generator. This is for usage with `-n`. Using `-e` with it will break compatibility
