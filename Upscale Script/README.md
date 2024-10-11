# Image Upscaling Script

This script allows for quick upscaling of your images with support for [spandrel](https://github.com/chaiNNer-org/spandrel). Just provide an input & output folder and the model you want to use.

**Features:**
* Upscale images using AI models
* Support for [multiple AI architectures](https://github.com/chaiNNer-org/spandrel?tab=readme-ov-file#model-architecture-support) via [spandrel](https://github.com/chaiNNer-org/spandrel)
* Convert output to various formats (PNG, JPEG, WebP, etc.)
* Enforce precision settings (FP32, FP16, BF16, or auto)

**Required Packages:**
* torch
* spandrel
* spandrel_extra_arches
* pillow
* numpy
* tqdm
* chainner-ext

How to use: `python upscale-script.py --input /path/to/image/folder --output /path/to/output/folder --model /path/to/model/file`

**Additional Arguments:**
* `--input` - Path to an image file or directory containing images to upscale
* `--output` - Directory where upscaled images will be saved
* `--model` - Path to the AI model file used for upscaling. [Supported models here](https://github.com/chaiNNer-org/spandrel?tab=readme-ov-file#model-architecture-support)

**Configuration Options (in config.ini):**
* `TileSize` - Size of tiles for processing. Use "native" for no tiling or specify a number (e.g., 512)
* `Precision` - Set computation precision ("auto", "fp32", "fp16", or "bf16")
* `ThreadPoolWorkers` - Number of worker threads for CPU tasks
* `OutputFormat` - Output image format (e.g., "png", "jpg", "webp")
* `AlphaHandling` - Whether to resize, upscale, or discard the alpha channel
* `GammaCorrection` - Whether or not to gamma correct the resized alpha channel

**Notes:**
* Experiment with different `TileSize` values to find the optimal setting for your hardware
* Adjust `ThreadPoolWorkers` based on your CPU capabilities
* If you encounter memory issues with large images, try reducing the `TileSize`
