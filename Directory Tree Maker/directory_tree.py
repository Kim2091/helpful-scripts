import os
import pathlib
from typing import Optional

try:
    import magic
except ImportError:
    magic = None
    print("❌ Error: The 'python-magic' library is not installed. Please install it using 'pip install python-magic' before running this script.")
    exit(1)

class DirectoryTreeGenerator:
    def __init__(self):
        # Define categories with appropriate emojis
        self.categories = {
            'directory': '📁',
            'image': '🖼️',
            'audio': '🎵',
            'video': '🎬',
            'document': '📄',
            'executable': '⚙️',
            'archive': '📦',
            'code': '📝',
            'data': '📊',
            'web': '🌐',
            '3d': '💠',
            'font': '🔤',
            'other': '📄',  # Default category
        }

    def get_file_emoji(self, file_path: str) -> str:
        """Get the appropriate emoji for a file based on its type."""
        if os.path.isdir(file_path):
            return self.categories['directory']

        try:
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type.startswith('image'):
                return self.categories['image']
            elif mime_type.startswith('audio'):
                return self.categories['audio']
            elif mime_type.startswith('video'):
                return self.categories['video']
            elif mime_type.startswith('text'):
                if 'html' in mime_type or 'xml' in mime_type:
                    return self.categories['web']
                elif 'javascript' in mime_type or 'python' in mime_type:
                    return self.categories['code']
                else:
                    return self.categories['document']
            elif mime_type.startswith('application'):
                if 'zip' in mime_type or 'x-tar' in mime_type or 'x-rar' in mime_type:
                    return self.categories['archive']
                elif 'pdf' in mime_type:
                    return self.categories['document']
                elif 'json' in mime_type or 'xml' in mime_type:
                    return self.categories['data']
                elif 'octet-stream' in mime_type:
                    return self.categories['executable']
                elif 'font' in mime_type:
                    return self.categories['font']
                else:
                    return self.categories['other']
            else:
                return self.categories['other']
        except Exception as e:
            print(f"⚠️ Error detecting file type for {file_path}: {e}. Please ensure all required libraries are installed.")
            return self.categories['other']

    def generate_tree(self, root_path: str, prefix: str = '', ignore_patterns: Optional[list] = None, is_subdir: bool = False) -> str:
        """Generate a tree structure starting from the root path."""
        if ignore_patterns is None:
            ignore_patterns = ['.git', '__pycache__', 'node_modules', '.idea']

        output = []
        root_path = os.path.abspath(root_path)

        try:
            items = os.listdir(root_path)
        except PermissionError:
            return f"{prefix}└── ⛔ Permission Denied\n"

        items.sort(key=lambda x: (not os.path.isdir(os.path.join(root_path, x)), x.lower()))
        items = [item for item in items if not any(pattern in item for pattern in ignore_patterns)]

        for index, item in enumerate(items):
            item_path = os.path.join(root_path, item)
            is_last_item = index == len(items) - 1

            current_prefix = '└── ' if is_last_item else '├── '
            next_prefix = '    ' if is_last_item else '│   '

            if os.path.isdir(item_path):
                output.append(f"{prefix}{current_prefix}{self.get_file_emoji(item_path)} {item}\n")
                output.append(self.generate_tree(
                    item_path,
                    prefix + next_prefix,
                    ignore_patterns,
                    is_subdir=True
                ))
            else:
                emoji = self.get_file_emoji(item_path)
                output.append(f"{prefix}{current_prefix}{emoji} {item}\n")

        return ''.join(output)

def main():
    """Main function to run the directory tree generator."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate a directory tree with emojis')
    parser.add_argument('path', help='Path to generate tree from')
    parser.add_argument('--ignore', nargs='+', help='Patterns to ignore', default=[])

    args = parser.parse_args()

    if not args.path:
        print("❌ Error: The 'path' argument is required.")
        exit(1)

    tree_generator = DirectoryTreeGenerator()
    ignore_patterns = ['.git', '__pycache__', 'node_modules', '.idea'] + args.ignore

    print(f"Directory Tree for: {os.path.abspath(args.path)}\n")
    print(tree_generator.generate_tree(args.path, ignore_patterns=ignore_patterns))

    # Generate the tree and save to a Markdown file
    tree_output = tree_generator.generate_tree(args.path, ignore_patterns=ignore_patterns)
    markdown_file = os.path.join(os.path.abspath(args.path), "directory_tree.md")

    with open(markdown_file, "w", encoding="utf-8") as md_file:
        md_file.write(f"# Directory Tree for: {os.path.abspath(args.path)}\n\n")
        md_file.write(tree_output)

    print(f"\nDirectory tree saved to: {markdown_file}")

if __name__ == "__main__":
    main()
