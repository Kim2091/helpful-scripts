# Video Processing Scripts

This repository contains two complementary scripts for video processing: `encodeVideo.sh` and `extractFrames.sh`. These scripts work together to provide a comprehensive solution for creating multi-quality video datasets and extracting frames for analysis or machine learning tasks.

**Features:**

* Create multi-quality video datasets with `encodeVideo.sh`:
  - Encode videos with varying quality segments using multiple codecs
  - Support for various video codecs including h264, vp8, vp9, mpeg, mpeg2, svt-av1, and hevc
  - Customizable quality ranges for encoding
* Extract frames from the generated videos using `extractFrames.sh`:
  - Extract frames from both high and low quality videos at specified intervals
  - Flexible frame extraction intervals
* Organized output with detailed logging for both processes

**Required Packages:**

* ffmpeg
* bash

## How to Use

These scripts are designed to be used in sequence:

1. First, use `encodeVideo.sh` to create a multi-quality version of your input video.
2. Then, use `extractFrames.sh` to extract frames from both the original (high quality) and the generated multi-quality (low quality) videos.

### Step 1: Encode Video

```bash
./encodeVideo.sh <input_video> [options]
```

**Options:**
* `--codec` - Specify the codec to use (vp8, vp9, h264, mpeg, mpeg2, svt-av1, hevc). Default: h264
* `--min` - Set the minimum quality value. Default: 16
* `--max` - Set the maximum quality value. Default: 32

Example:
```bash
./encodeVideo.sh input.mp4 --codec vp9 --min 20 --max 40
```

This will generate a multi-quality video named `output_multi_quality_vp9.mkv`.

### Step 2: Extract Frames

```bash
./extractFrames.sh -h <high_quality_video> -l <low_quality_video> -s <seconds_between_frames>
```

**Options:**
* `-h` - Path to the high quality video file (your original input video)
* `-l` - Path to the low quality video file (the output from encodeVideo.sh)
* `-s` - Number of seconds between captured frames (default: 1)

Example:
```bash
./extractFrames.sh -h input.mp4 -l output_multi_quality_vp9.mkv -s 5
```

This will extract frames every 5 seconds from both the high quality (original) and low quality (encoded) videos.

## Output

Both scripts generate output in organized directories:

- `encodeVideo.sh` creates a multi-quality video file and a detailed encoding log.
- `extractFrames.sh` creates separate directories for high and low quality frames, named after the low quality video file.

These outputs can be used for various purposes, such as creating datasets for machine learning models that work with video quality enhancement or compression artifact reduction.

**Credits:**
These scripts were developed to provide a flexible and efficient pipeline for creating multi-quality video datasets and extracting frames for analysis or machine learning tasks.
