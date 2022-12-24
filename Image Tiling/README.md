All of these scripts were created with ChatGPT and allow quick tiling of your images with multithreading. Put your images in an `images` folder, and it'll output the results in `output`. There are multiple options depending on what you need:

The `Image Tiling with Simple Removal` script removes any solid white or black tiles with a small margin of error. It will also remove any output tiles that are below the size of your specified tile size.

The `Image Tiling with Tile Size Limit` script is the same as the one above, but it only removes tiles below the specified tile size. It will leave any solid black/white tiles.

The `Image Tiling Script` is just a plain tiling script with no filters. From a quick test, it's about 2250% faster than magickutils.
