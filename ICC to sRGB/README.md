# ICC to sRGB Conversion Script

This script converts images by applying their embedded ICC profiles and converting them to sRGB color space. It preserves alpha channels and supports batch processing through folder inputs.

**Features:**
* Apply embedded ICC profiles to images
* Convert images to sRGB color space
* Preserve transparency/alpha channels
* Support for batch processing via folders
* Preserve directory structure in output
* Convert multiple image formats (PNG, JPEG, TIFF, BMP, WebP)

**Required Packages:**
* pillow

How to use: `python icc_to_srgb.py input_folder output_folder`

**Arguments:**
### Arguments:
* `input_path` - Path to a single image file or a directory containing images to process.
* `output_path` - Path to the output file (if processing a single image) or directory where processed images will be saved (if processing a folder).

**Supported Input Formats:**
* PNG
* JPEG/JPG
* TIFF
* BMP
* WebP

**Notes:**
* All output images are saved as PNG to ensure alpha channel support
* Images without ICC profiles are simply converted to RGB/RGBA
* Original alpha channels are preserved during the conversion process
* Directory structure is preserved when processing folders
* Subdirectories are automatically created in the output folder

**Examples:**

Process a folder of images:
```bash
python icc_to_srgb.py input_folder output_folder
```
Process a single image:
```bash
python icc_to_srgb.py input_image.jpg output_image.png
```
