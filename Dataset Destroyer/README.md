*Written with the help of Bing's GPT-4 chat*

This script's main usage is to generate datasets for your image models. 

Main features:
- The strength of each degradation is adjustable
- Optional randomization of the strength
- Optional randomization of degradation order
- Video compression support through ffmpeg-python
- Progress bar

<details><summary>Supported filters:</summary>

- Blur
  - average
  - gaussian
  - isotropic
  - anisotropic

- Noise
  - uniform
  - gaussian
  - color
  - gray

- Compression
  - H264
  - HEVC
  - MPEG
  - MPEG-2
  - JPEG
  - WEBP

- Scale
  - down_up
  - Bicubic
  - Bilinear
  - Box
  - Nearest
  - Lanczos
</details>

Usage: 
- Download the script and the config.ini file
- Edit config.ini to your liking. Make sure to add file paths! Comments within the file describe each function
- Run the .py file with python
- Enjoy!

This script requires the following libraries:
- cv2 (OpenCV)
- numpy
- ffmpeg-python
- scipy
- tqdm
- configparser
