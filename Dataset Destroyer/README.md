*Written with the help of multiple AI assistants*

This script's main usage is to generate datasets for your image models. 

Note: Avoid running all degradations at once in combination with ffmpeg options (mpeg, mpeg2, h264, hevc, vp9). It will likely cause errors

Main features:
- Adjustable degradations
- Supports: Blur, noise, compression, and scaling
- Randomization of strengths and order for every function
- Video compression support through ffmpeg-python
- Progress bar
  

<details><summary>Supported filters:</summary>

- Blur
  - Average
  - Gaussian
  - Isotropic
  - Anisotropic

- Noise
  - Uniform
  - Gaussian
  - Color
  - Gray
  - ISO
  - Salt and Pepper

- Quantization
  - Floyd-Steinberg
  - Jarvis-Judice-Ninke
  - Stucki
  - Atkinson
  - Burkes
  - Sierra
  - Two-Row Sierra
  - Sierra Lite

- Compression
  - H264
  - HEVC
  - VP9
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
