import cv2
import numpy as np
import os
import re
import argparse
import shutil
from pathlib import Path
from collections import defaultdict


def parse_filename(filename):
    """Parse filename to extract sequence name and frame number."""
    match = re.match(r'(.+)_Frame(\d+)\.png', filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def calculate_motion_score(frame1, frame2, threshold=30):
    """Calculate motion score between two frames using percentage of changed pixels."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference to get binary mask of changed pixels
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of pixels that changed
    total_pixels = thresh.size
    changed_pixels = np.count_nonzero(thresh)
    motion_score = (changed_pixels / total_pixels) * 100
    
    return motion_score


def group_frames_by_sequence(input_dir):
    """Group frame files by their sequence name."""
    sequences = defaultdict(list)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            seq_name, frame_num = parse_filename(filename)
            if seq_name and frame_num:
                sequences[seq_name].append((frame_num, filename))
    
    # Sort frames within each sequence
    for seq_name in sequences:
        sequences[seq_name].sort(key=lambda x: x[0])
    
    return sequences


def analyze_sequence_motion(input_dir, sequence_name, frames, min_motion=0.5, max_motion=15.0, move_to_dir=None):
    """Analyze motion in a sequence and report issues."""
    print(f"\n{'='*60}")
    print(f"Analyzing sequence: {sequence_name}")
    print(f"{'='*60}")
    
    if len(frames) < 2:
        print(f"⚠️  WARNING: Only {len(frames)} frame(s) in sequence - cannot detect motion")
        return
    
    issues_found = False
    
    for i in range(len(frames) - 1):
        frame_num1, filename1 = frames[i]
        frame_num2, filename2 = frames[i + 1]
        
        # Load frames
        img1_path = os.path.join(input_dir, filename1)
        img2_path = os.path.join(input_dir, filename2)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"⚠️  WARNING: Could not load frames {filename1} or {filename2}")
            continue
        
        # Calculate motion
        motion_score = calculate_motion_score(img1, img2)
        
        # Check thresholds
        if min_motion is not None and motion_score < min_motion:
            print(f"⚠️  LOW MOTION: Frames {frame_num1:05d} → {frame_num2:05d} | Score: {motion_score:.2f}% (< {min_motion}%)")
            issues_found = True
        elif max_motion is not None and motion_score > max_motion:
            print(f"⚠️  HIGH MOTION: Frames {frame_num1:05d} → {frame_num2:05d} | Score: {motion_score:.2f}% (> {max_motion}%)")
            issues_found = True
        else:
            print(f"✓ OK: Frames {frame_num1:05d} → {frame_num2:05d} | Score: {motion_score:.2f}%")
    
    if not issues_found:
        print(f"✓ All frames in sequence have acceptable motion levels")
    elif move_to_dir:
        # Move entire sequence if any issues were found
        print(f"\n⚠️  Moving entire sequence ({len(frames)} frames) to: {move_to_dir}")
        for frame_num, filename in frames:
            src = os.path.join(input_dir, filename)
            dst = os.path.join(move_to_dir, filename)
            shutil.move(src, dst)
            print(f"  Moved: {filename}")
            
def main():
    parser = argparse.ArgumentParser(description='Detect motion issues in video frame sequences')
    parser.add_argument('input_dir', help='Directory containing frame sequences')
    parser.add_argument('--min-motion', type=float, default=-1, 
                        help='Minimum motion threshold percentage (default: -1, use -1 to disable)')
    parser.add_argument('--max-motion', type=float, default=15.0,
                        help='Maximum motion threshold percentage (default: 15.0, use -1 to disable)')
    parser.add_argument('--move-to', type=str, default=None,
                        help='Optional directory to move frames with motion issues')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist")
        return
    
    if args.move_to:
        os.makedirs(args.move_to, exist_ok=True)
        print(f"Frames with issues will be moved to: {args.move_to}")
    
    # Group frames by sequence
    sequences = group_frames_by_sequence(args.input_dir)
    
    if not sequences:
        print("No valid frame sequences found in the directory")
        return
    
    print(f"Found {len(sequences)} sequence(s)")
    
    # Build threshold message
    threshold_parts = []
    if args.min_motion is not None:
        threshold_parts.append(f"min: {args.min_motion}%")
    if args.max_motion is not None:
        threshold_parts.append(f"max: {args.max_motion}%")
    
    if threshold_parts:
        print(f"Motion thresholds: {', '.join(threshold_parts)}")
    else:
        print("Motion thresholds: disabled")
    
    # Analyze each sequence
    for seq_name in sorted(sequences.keys()):
        analyze_sequence_motion(args.input_dir, seq_name, sequences[seq_name], 
                                args.min_motion, args.max_motion, args.move_to)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()