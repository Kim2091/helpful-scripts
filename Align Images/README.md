*Partially written by ChatGPT*

This script handles basic image alignment. It looks at 2 folders, scans through and finds images with the same names, and performs projective transformations to align the image in the first folder to the image in the second folder.

`imageAlignmentTool.py --folder1 FOLDER1 --folder2 FOLDER2 --output OUTPUT`

**Features:**
* Transparency support
* Projective transformations using keypoints
* Batch image handling

**Required Packages**
* opencv-python
* numpy
* argparse
* time
* logging
