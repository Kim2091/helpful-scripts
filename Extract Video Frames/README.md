*Written by ChatGPT*

This script extracts frames from a given input video and saves them as images in a specified output folder. This was tested with videos in various formats, although it may not support more obscure formats or codecs.

Features:
* FPS Setting
* Start and End frame settings
* Output format option

Usage:
`python extract_frames.py [-h] [-f FPS] [-fmt FORMAT] [-s START] [-e END] input_video output_folder`

    input_video: path to the input video file from which the frames will be extracted.
    output_folder: path to the output folder where the extracted frames will be saved.
    -f, --fps: frame rate at which to extract the frames. This argument is optional. If not specified, the frames will be extracted at the frame rate of the input video.
    -fmt, --format: image format in which to save the frames. This argument is optional. If not specified, the frames will be saved as PNG images.
    -s, --start: start frame number. This argument is optional. If not specified, the extraction will start from the first frame.
    -e, --end: end frame number. This argument is optional. If not specified, the extraction will end at the last frame.
