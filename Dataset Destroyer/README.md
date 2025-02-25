
This script's main purpose is to generate datasets for your image models. 

Note: Avoid running all degradations at once in combination with ffmpeg options (mpeg, mpeg2, h264, hevc, vp9). It will likely cause errors

Main features:
- Adjustable degradations
- Supports: Blur, noise, compression, scaling, quantization, and unsharp mask
- Adjustable strengths and order for every degradation, with a randomization option
- Probability control for each degradation
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
  - Box
  - Linear
  - Cubic_Catrom
  - Cubic_Mitchell
  - Cubic_BSpline
  - Lanczos
  - Gauss
 
- Unsharp Mask

- Chroma Blur

</details>

Usage: 
- Download the script and the config.ini file
- Edit config.ini to your liking. Make sure to add file paths! Comments within the file describe each function
- Run the .py file with python
- Enjoy!

__You may want to consider using [wtp_dataset_destroyer](https://github.com/umzi2/wtp_dataset_destroyer) instead. It builds on the concepts used in this original version and makes it a lot easier to use.__
