This script analyzes video frame sequences to detect motion issues between consecutive frames, helping identify sequences with too much or too little motion.

**Features:**
* Detects low motion (nearly identical frames) and high motion (scene changes or jumps) between consecutive frames
* Groups frames by sequence name automatically
* Configurable minimum and maximum motion thresholds
* Option to automatically move problematic sequences to a separate directory
* Detailed per-sequence analysis with visual indicators (✓ for OK, ⚠️ for issues)
* Motion scoring based on percentage of changed pixels between frames

**Required Packages:**
* opencv-python (cv2)
* numpy

**How to use:** `python VideoFrameMotionDetect.py <input_directory> [options]`

**Arguments:**
* `input_dir` - Directory containing frame sequences (required)
* `--min-motion` - Minimum motion threshold percentage (default: 0.5). Frames below this are flagged as too similar. Use -1 to disable.
* `--max-motion` - Maximum motion threshold percentage (default: 15.0). Frames above this are flagged as having too much motion. Use -1 to disable.
* `--move-to` - Optional directory to move entire sequences that have motion issues

**Example:**
```bash
python VideoFrameMotionDetect.py ./demo --min-motion 0.5 --max-motion 15.0 --move-to ./out