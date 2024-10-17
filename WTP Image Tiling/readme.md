## Config:
```json 
{
  "in_folder": "input",
  "out_folder": "output",
  "tiler": {
  },
  "tile_size": 1024,
  "shuffle": true,
  "real_name": false,
  "process": "for"
}
 
```
optional = *

`in_folder` - input folder

`out_folder` - output folder

`tiler` - dictionary with tile type and settings
- `type` - one of the types "linear", "random", "overlap" or "best"
  * `type` = "linear" - standard division into tiles from the upper left to the lower right
    * `n_tiles` - maximum number of tiles default: 10k 
  * `type` = "random" - splits the image into n tiles at random coordinates
    * `n_tiles` - number of tiles default: 1  
  * `type` = "overlap" - splits the image into overlapping tiles
    * `n_tiles` - maximum number of tiles default: 10k 
    * `overlap` -overlap percentage 0.25 is approximately equal to 25% default: 0.25
  * `type` = "best" - finds the best tie for the image and saves it
    * `scale` - scale - reduces the image by the chosen factor, which speeds up the search for the best tile, although with some loss of accuracy. Additionally, the larger the value, the more the script focuses on larger details. default: 1

`tile_size` - final tile size 

`shuffle` - mixes tiles

`real_name` - saves the image not by index but by real name. Makes shuffle useless when enabled

`process` - Valid processing types: process, thread and for
* If you run into issues on Windows, swap to thread
