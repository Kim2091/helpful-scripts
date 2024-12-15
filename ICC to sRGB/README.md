# ICC to sRGB Conversion Script

This script converts images by applying their embedded ICC profiles and converting them to sRGB color space. It preserves alpha channels and supports batch processing through folder inputs.

**Features:**
* Apply embedded ICC profiles to images
* Convert images to sRGB color space
* Preserve transparency/alpha channels
* Support for batch processing via folders
* Convert multiple image formats (PNG, JPEG, TIFF, BMP)

**Required Packages:**
* pillow

How to use: `python icc_to_srgb.py -i /path/to/input -o /path/to/output`

**Arguments:**
* `-i, --input` - Path to an image file or directory containing images to process
* `-o, --output` - Path where processed images will be saved (file or directory)

**Supported Input Formats:**
* PNG
* JPEG/JPG
* TIFF
* BMP

**Notes:**
* All output images are saved as PNG to ensure alpha channel support
* Images without ICC profiles are simply converted to RGB/RGBA
* Original alpha channels are preserved during the conversion process
* Directory structure is preserved when processing folders

**Examples:**

Process a single image:
`python icc_to_srgb.py -i input_image.png -o output_image.png`

Process an entire folder:
`python icc_to_srgb.py -i input_folder -o output_folder`
