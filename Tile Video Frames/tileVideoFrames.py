import os
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict
import re
import random
from multiprocessing import Pool, cpu_count

def parse_filename(filename):
    """
    Parse filename to extract scene/show/sequence identifier and frame number.
    Expected format: show00279_Frame00001.ext (case-insensitive)
    The prefix (show/scene/sequence/etc.) can be any word.
    """
    match = re.match(r'([a-zA-Z]+)(\d+)_[fF]rame(\d+)', filename)
    if match:
        prefix = match.group(1)  # e.g., 'show', 'scene', 'sequence'
        scene_num = match.group(2)  # e.g., '00279'
        frame_num = match.group(3)  # e.g., '00001'
        scene_id = f"{prefix}{scene_num}"  # e.g., 'show00279'
        return scene_id, frame_num, prefix, len(scene_num)
    return None, None, None, None

def group_frames_by_scene(input_dir, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    """
    Group frames by scene number.
    Returns: dict with scene as key and list of (frame_number, filepath, filename) tuples as value,
             and the prefix and number width from the first file
    """
    scenes = defaultdict(list)
    prefix = None
    num_width = None
    
    for file in os.listdir(input_dir):
        if not any(file.lower().endswith(ext) for ext in extensions):
            continue
            
        scene, frame, file_prefix, file_num_width = parse_filename(file)
        if scene and frame:
            filepath = os.path.join(input_dir, file)
            scenes[scene].append((int(frame), filepath, file))
            if prefix is None:
                prefix = file_prefix
                num_width = file_num_width
    
    # Sort frames within each scene
    for scene in scenes:
        scenes[scene].sort(key=lambda x: x[0])
    
    return scenes, prefix, num_width

def get_tile_positions(img_width, img_height, tile_width, tile_height):
    """
    Calculate all possible tile positions for an image.
    Returns list of (x, y) coordinates for top-left corner of each tile.
    """
    positions = []
    for y in range(0, img_height - tile_height + 1, tile_height):
        for x in range(0, img_width - tile_width + 1, tile_width):
            positions.append((x, y))
    return positions

def extract_tile(image, x, y, tile_width, tile_height):
    """Extract a tile from the image at the specified position."""
    return image.crop((x, y, x + tile_width, y + tile_height))

def process_tile_sequence(args_tuple):
    """
    Worker function to process a single tile from a sequence.
    
    Args:
        args_tuple: Tuple containing (tile_idx, tile_position, sequence, output_dir, 
                    tile_width, tile_height, global_show_counter, prefix, num_width, 
                    sequence_length, extension)
    
    Returns:
        Number of frames processed
    """
    tile_idx, (x, y), sequence, output_dir, tile_width, tile_height, global_show_counter, prefix, num_width, sequence_length, extension = args_tuple
    
    # Calculate the show number for this tile
    show_num = global_show_counter + tile_idx
    
    # Process each frame in the sequence for this tile
    for frame_idx in range(sequence_length):
        frame_num, filepath, filename = sequence[frame_idx]
        
        # Open image and extract tile
        with Image.open(filepath) as img:
            tile = img.crop((x, y, x + tile_width, y + tile_height))
            
            # Generate output filename
            output_filename = f"{prefix}{show_num:0{num_width}d}_Frame{frame_idx + 1:05d}{extension}"
            output_path = os.path.join(output_dir, output_filename)
            
            tile.save(output_path)
    
    return sequence_length

def process_scene(scene_name, scene_frames, output_dir, sequence_length, tile_width, tile_height, global_show_counter, prefix, num_width, pool):
    """
    Process all frames in a scene, extracting tiles from sequences.
    
    Args:
        scene_name: Name of the scene (e.g., 'show00279')
        scene_frames: List of (frame_number, filepath, filename) tuples
        output_dir: Output directory path
        sequence_length: Number of consecutive frames in a sequence
        tile_width: Width of each tile
        tile_height: Height of each tile
        global_show_counter: Starting show/scene number for output
        prefix: Prefix to use (e.g., 'show', 'scene')
        num_width: Width of the number padding
        pool: Multiprocessing pool to use
        
    Returns:
        Updated global_show_counter
    """
    if len(scene_frames) < sequence_length:
        print(f"Skipping {scene_name} with only {len(scene_frames)} frames (need {sequence_length})")
        return global_show_counter
    
    # Get image dimensions and extension from first frame
    first_frame_path = scene_frames[0][1]
    with Image.open(first_frame_path) as img:
        img_width, img_height = img.size
    extension = Path(scene_frames[0][2]).suffix
    
    # Calculate tile positions
    tile_positions = get_tile_positions(img_width, img_height, tile_width, tile_height)
    
    if not tile_positions:
        print(f"Warning: Image dimensions ({img_width}x{img_height}) are smaller than tile size ({tile_width}x{tile_height})")
        return global_show_counter
    
    # Shuffle tile positions based on seed (already set in main)
    random.shuffle(tile_positions)
    
    # Process each sequence
    num_sequences = len(scene_frames) - sequence_length + 1
    
    # Batch all tasks for this scene
    all_task_args = []
    
    for seq_start_idx in range(num_sequences):
        sequence = scene_frames[seq_start_idx:seq_start_idx + sequence_length]
        
        # Prepare arguments for parallel processing
        for tile_idx, tile_pos in enumerate(tile_positions):
            all_task_args.append((
                tile_idx, tile_pos, sequence, output_dir, 
                tile_width, tile_height, global_show_counter, 
                prefix, num_width, sequence_length, extension
            ))
        
        # Update show counter for next sequence
        global_show_counter += len(tile_positions)
    
    # Process all tiles in parallel using the shared pool
    pool.map(process_tile_sequence, all_task_args)
    
    print(f"  Processed {num_sequences} sequences: {len(tile_positions)} tiles × {sequence_length} frames each = {len(all_task_args)} total tile sequences")
    
    return global_show_counter

def main():
    parser = argparse.ArgumentParser(
        description='Tile video frame sequences with consistent tile positions across frames.'
    )
    parser.add_argument('input_dir', type=str, help='Input directory containing frame sequences')
    parser.add_argument('output_dir', type=str, help='Output directory for tiled frames')
    parser.add_argument('--sequence-length', type=int, default=5, 
                        help='Number of consecutive frames in a sequence (default: 5)')
    parser.add_argument('--tile-width', type=int, default=512, 
                        help='Width of each tile in pixels (default: 512)')
    parser.add_argument('--tile-height', type=int, default=512, 
                        help='Height of each tile in pixels (default: 512)')
    parser.add_argument('--seed', type=int, default=1024, 
                        help='Random seed for tile selection (default: 1024)')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    # Set number of workers
    if args.workers is None:
        args.workers = cpu_count()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Tile size: {args.tile_width}x{args.tile_height}")
    print(f"Random seed: {args.seed}")
    print(f"Number of workers: {args.workers}")
    print("-" * 50)
    
    # Group frames by scene
    scenes, prefix, num_width = group_frames_by_scene(args.input_dir)
    
    if not scenes:
        print("No valid frame sequences found in input directory")
        return
    
    if prefix is None:
        print("Error: Could not determine prefix from filenames")
        return
    
    print(f"Found {len(scenes)} scene(s)")
    print(f"Using prefix: '{prefix}' with {num_width}-digit numbering")
    
    # Create a single pool for all processing
    with Pool(processes=args.workers) as pool:
        # Process each scene with continuous show numbering
        global_show_counter = 1
        for scene_name in sorted(scenes.keys()):
            scene_frames = scenes[scene_name]
            print(f"\nProcessing {scene_name}: {len(scene_frames)} input frames")
            global_show_counter = process_scene(scene_name, scene_frames, args.output_dir, 
                                                args.sequence_length, args.tile_width, 
                                                args.tile_height, global_show_counter, prefix, num_width, pool)
    
    print(f"\nTiling complete! Generated {global_show_counter - 1} show sequences")

if __name__ == "__main__":
    main()