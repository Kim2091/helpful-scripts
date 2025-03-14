# Video Frame Extraction Script

This script enables precise frame extraction from paired high-resolution (HR) and low-resolution (LR) videos using scene detection. It's designed to extract synchronized frames at scene changes, making it ideal for training video super-resolution models.

Thank you to Bendel on Enhance Everything! for providing the original script as a baseline.

**Features:**
* Scene-based frame extraction using PySceneDetect
* Synchronized extraction from HR and LR video pairs
* Support for frame offset compensation between video pairs
* Configurable number of consecutive frames per scene
* Adjustable scene detection threshold

**Required Packages:**
* opencv-python
* numpy
* scenedetect

How to use: `python vidpair.py --hr /path/to/hr/video --lr /path/to/lr/video --output /path/to/output/folder`

**Required Arguments:**
* `--hr` - Path to the high-resolution video file
* `--lr` - Path to the low-resolution video file
* `--output` - Directory where extracted frames will be saved

**Optional Arguments:**
* `--frames` - Number of consecutive frames to extract per scene (default: 10)
* `--offset` - Frame offset between HR and LR videos (default: 0)
* `--threshold` - Scene detection threshold (default: 30.0)

**Notes:**
* Higher threshold values result in fewer detected scenes
* Adjust `--frames` based on your training requirements
* Use `--offset` if your HR and LR videos are not perfectly synchronized
* Frame pairs are saved as PNG files to preserve quality
* Frames are numbered sequentially within each detected scene
* Each scene's frames are prefixed with "showXXXX" where XXXX is the scene number

**Example Usage:**
```bash
# Extract 5 frames per scene with default settings
python vidpair.py --hr video_hr.mp4 --lr video_lr.mp4 --output ./frames --frames 5

# Compensate for 2-frame offset between videos
python vidpair.py --hr video_hr.mp4 --lr video_lr.mp4 --output ./frames --offset 2

# Adjust scene detection sensitivity
python vidpair.py --hr video_hr.mp4 --lr video_lr.mp4 --output ./frames --threshold 25.0
```
