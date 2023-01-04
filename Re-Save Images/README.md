This script was originally made by sudo. His github page can be found here: https://github.com/styler00dollar

The script's main function is to re-save images in a given directory. For example, this can be used to "fix" corrupted images to allow training to continue without finding the corrupted image manually. It doesn't actually fix the corruption, but it allows training software to read it as a valid file.

I asked ChatGPT to modify it and add additional functions such as:
* Grayscale output support
* Subfolder support
* Exception handling
* Progress bar
* Commandline Argument Support
* Parallel Processing

**Required Packages:**
* argparse
* opencv-python
* tqdm
