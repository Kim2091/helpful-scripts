*Written with the assistance of GitHub Copilot*

This script's main usage is to search for corrupted image files in a directory, with an additional argument for finer control.

Main features:
- Searches for corrupted image files in a specified directory
- By default searches for '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.tiff', '.webp'
- Option to search for a specific image file type
- Progress bar to track the file searching process
- Creates a log file ('searchlog.txt') in the script's directory listing all searched files and detected corrupted files

Usage: 
- Download the script
- Run the script with the directory to be searched as an argument
- Optional arguments include:
  - `-f` or `--file_type` to search only for a specific image file type
  - `-d` or `--deep` to do a "deep" scan. It loads the images individually to ensure they are not corrupted. Much slower than the default method, but it may catch more corrupted images
- Enjoy!

Example:
```bash
python test.py /path/to/directory -f .jpg
```
This will search for corrupted '.jpg' files in the specified directory. If no file type is specified, the script will search for all supported image types.
