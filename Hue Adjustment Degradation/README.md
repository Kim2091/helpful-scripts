*Written with the assistance of GitHub Copilot*

This script allows you easily adjust hue, brightness, and contrast in a dataset. It is also capable of duplicating the original images with various adjustments applied to them.

You may consider using this script for use with particularly small image super resolution datasets. It can help reduce color based artifacts.

**Features:**

* Adjust hue, brightness, and contrast separately
* Duplicate images with adjustments applied to expand dataset

**Required Packages:**
* OpenCV
* numpy
* Pillow

**How to use:** `python hue_adjustment.py -b -c -u -d 10`

Before running the script, make sure to set the `hr_dir`, `lr_dir`, and `output_hr_dir` / `output_lr_dir` paths. Set these to your hr, lr, and output folders respectively.

There are adjustable range values in the script. Set these to the desired ranges. The defaults should be good enough
