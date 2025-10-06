import os
import shutil
import argparse
from pathlib import Path


def get_relative_files(directory):
    """
    Get all files in a directory with their relative paths.
    
    Args:
        directory: Path to the directory to scan
        
    Returns:
        Set of relative file paths
    """
    directory = Path(directory)
    files = set()
    
    if not directory.exists():
        print(f"Warning: Directory '{directory}' does not exist")
        return files
    
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            full_path = Path(root) / filename
            relative_path = full_path.relative_to(directory)
            files.add(relative_path)
    
    return files


def move_files(source_dir, relative_paths, destination_dir, label):
    """
    Move files from source directory to destination directory.
    
    Args:
        source_dir: Source directory path
        relative_paths: Set of relative file paths to move
        destination_dir: Destination directory path
        label: Label for logging (e.g., "baseline" or "secondary")
    """
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    
    moved_count = 0
    
    for rel_path in relative_paths:
        source_file = source_dir / rel_path
        dest_file = destination_dir / label / rel_path
        
        # Create destination directory if it doesn't exist
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(source_file), str(dest_file))
            print(f"Moved: {rel_path} -> {dest_file}")
            moved_count += 1
        except Exception as e:
            print(f"Error moving {rel_path}: {e}")
    
    return moved_count


def compare_and_move_folders(baseline_dir, secondary_dir, output_dir):
    """
    Compare two folders and move missing files to output directory.
    
    Args:
        baseline_dir: Path to baseline folder
        secondary_dir: Path to secondary folder
        output_dir: Path to output folder for missing files
    """
    print("=" * 60)
    print("Folder Comparison Tool")
    print("=" * 60)
    print(f"Baseline directory: {baseline_dir}")
    print(f"Secondary directory: {secondary_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Get all files from both directories
    print("\nScanning directories...")
    baseline_files = get_relative_files(baseline_dir)
    secondary_files = get_relative_files(secondary_dir)
    
    print(f"Files in baseline: {len(baseline_files)}")
    print(f"Files in secondary: {len(secondary_files)}")
    
    # Find missing files
    missing_from_baseline = secondary_files - baseline_files
    missing_from_secondary = baseline_files - secondary_files
    
    print(f"\nFiles in secondary but missing from baseline: {len(missing_from_baseline)}")
    print(f"Files in baseline but missing from secondary: {len(missing_from_secondary)}")
    
    if not missing_from_baseline and not missing_from_secondary:
        print("\nNo missing files found. Directories are in sync!")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Move missing files
    print("\n" + "=" * 60)
    print("Moving files...")
    print("=" * 60)
    
    total_moved = 0
    
    if missing_from_baseline:
        print(f"\nMoving {len(missing_from_baseline)} files from secondary (missing in baseline)...")
        count = move_files(secondary_dir, missing_from_baseline, output_dir, "missing_from_baseline")
        total_moved += count
    
    if missing_from_secondary:
        print(f"\nMoving {len(missing_from_secondary)} files from baseline (missing in secondary)...")
        count = move_files(baseline_dir, missing_from_secondary, output_dir, "missing_from_secondary")
        total_moved += count
    
    print("\n" + "=" * 60)
    print(f"Complete! Moved {total_moved} files to {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two folders and move missing files to a third directory"
    )
    parser.add_argument(
        "baseline",
        help="Path to the baseline folder"
    )
    parser.add_argument(
        "secondary",
        help="Path to the secondary folder"
    )
    parser.add_argument(
        "output",
        help="Path to the output folder for missing files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be moved\n")
        # For dry run, just show the differences without moving
        baseline_files = get_relative_files(args.baseline)
        secondary_files = get_relative_files(args.secondary)
        
        missing_from_baseline = secondary_files - baseline_files
        missing_from_secondary = baseline_files - secondary_files
        
        print(f"Files in secondary but missing from baseline ({len(missing_from_baseline)}):")
        for f in sorted(missing_from_baseline):
            print(f"  - {f}")
        
        print(f"\nFiles in baseline but missing from secondary ({len(missing_from_secondary)}):")
        for f in sorted(missing_from_secondary):
            print(f"  - {f}")
    else:
        compare_and_move_folders(args.baseline, args.secondary, args.output)


if __name__ == "__main__":
    main()