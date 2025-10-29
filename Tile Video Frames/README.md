# Tile Video Frames

This script processes video frame sequences by extracting tiles from each frame, creating multiple new sequences from different regions of the original frames.

**Features:**
* Extracts non-overlapping tiles from video frame sequences
* Maintains temporal consistency - same tile position across all frames in a sequence
* Supports multiple scenes/sequences in a single run
* Parallel processing for improved performance
* Generates sliding window sequences from input frames

**Required Packages:**
* Pillow (PIL)

**How to use:** `python tileVideoFrames.py <input_dir> <output_dir> [options]`

**Arguments:**
* `input_dir` - Directory containing frame sequences (required)
* `output_dir` - Directory where tiled frames will be saved (required)
* `--sequence-length` - Number of consecutive frames in a sequence (default: 5)
* `--tile-width` - Width of each tile in pixels (default: 512)
* `--tile-height` - Height of each tile in pixels (default: 512)
* `--seed` - Random seed for tile selection order (default: 1024)
* `--workers` - Number of parallel workers (default: CPU count)

**Input Format:**
Frames must follow the naming pattern: `[prefix][number]_Frame[number].[ext]`
* Example: `show00279_Frame00001.png`, `scene00005_Frame00023.jpg`

**Example:**
```bash
python tileVideoFrames.py ./input ./output --sequence-length 5 --tile-width 512 --tile-height 512