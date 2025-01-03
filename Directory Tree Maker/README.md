# Directory Tree Maker

This script generates a visual representation of a directory, using emojis to represent different file types. It's designed to make trees that are easier to read 🙂.

## Features
- Generate a pretty tree structure for any directory.
- Categorize files with emojis
  - Folders: 📁
  - Images: 🖼️
  - Audio: 🎵
  - Videos: 🎬
  - Documents: 📄
  - Code files: 📝
  - Archives: 📦
  - Executables: ⚙️
  - Fonts: 🔤
  - Others: 📄
- Ignore common system folders (`.git`, `__pycache__`, etc.).
- Output is saved to a `directory_tree.md` file in the specified directory for later reference.

## Usage
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the script from the terminal:

```bash
python directory_tree.py [path] [--ignore PATTERN ...]
```

- `path`: Path to the directory to generate the tree for. Defaults to the current directory.
- `--ignore`: (Optional) Additional patterns to ignore (e.g., temporary or hidden files).
  - Example: `python directory_tree.py [path] --ignore .git node_modules`

## Example Output
Here’s what the tree output looks like:

```
Directory Tree for: /example/path

├── 📁 test1
│   ├── 📁 media
│   │   ├── 🖼️ image2.png
│   │   └── 🎬 video.mp4
│   └── 🖼️ image1.png
├── 📁 test2
│   ├── 🎬 video1.mp4
│   └── 🎬 video2.mp4
├── 📦 compressed.zip
├── 📄 random_text_doc.odt
└── 📄 text_files!.txt
```
