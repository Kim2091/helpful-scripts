This script moves (or deletes) duplicate images from an Image Pair Dataset (HR/LR dataset).

It only checks for duplicates in the HR images. It then duplicates the move or delete operation in the LR images folder by matching the filename.
This means the LR files may not strictly beduplicates, but the differences would be caused by mis-alignment with HR or differences in compression.
I recommend ensuring the LR and HR dataset matches up correctly by filename.

It checks for duplicates by matching each image's SHA256 hash against all other images in the HR folder. This means images are only considered a
duplicate if they match exactly. Also note that by default it moves duplicate images to another folder so you can verify the operation, but if you
wish to delete them instead, then pass the `--delete` flag.

`De-dupe Images.py [-h] --hr HR_OR_GT_FOLDER --lr LR_OR_LQ_FOLDER [--delete]`

**Required Packages**
* pillow
