*Written with the assistance of GitHub Copilot*

This script allows you to compare images from two folders and move the ones with low similarity scores to a specified output folder

**Features:**

* Compare images from two folders based on their similarity scores
* Move images with low similarity scores to a specified output folder
* Overlay the moved images and save them in the output folder
* Multithreading support for faster image comparison and moving

**Required Packages:**
* OpenCV
* numpy

**How to use:** `python test.py`

Before running the script, make sure to set the `hr_path`, `lr_path`, and `output_path` paths. Set these to your hr, lr, and output folders respectively.

There's an adjustable `threshold` value in the script. Increasing this makes detection more strict, though it can lead to false positives. For cartoons an anime, the current value of 0.7 has been the most accurate in my experience.
