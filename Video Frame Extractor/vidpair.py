import argparse
import cv2
import os
import numpy as np
from scenedetect import SceneManager, open_video, ContentDetector

def detect_scenes(video_path, threshold=30.0):
    """Detect scene changes in video using the current PySceneDetect API."""
    print(f"Detecting scenes in {video_path} with threshold {threshold}...")

    # Open video with the new API
    video = open_video(video_path)

    # Create scene manager and add detector
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Detect scenes
    scene_manager.detect_scenes(video)

    # Get list of scenes
    scene_list = scene_manager.get_scene_list()
    print(f"Detected {len(scene_list)} scenes")

    # Print scene information for debugging
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        print(f"Scene {i}: frames {start_frame} to {end_frame} (length: {end_frame - start_frame})")

    return scene_list

def extract_frames(hr_path, lr_path, output_dir, frames_per_scene=10, lr_offset=0, threshold=30.0):
    """Extract consecutive frames from HR and LR videos at scene changes."""
    # Create output directories
    hr_output = os.path.join(output_dir, 'hr')
    lr_output = os.path.join(output_dir, 'lr')
    os.makedirs(hr_output, exist_ok=True)
    os.makedirs(lr_output, exist_ok=True)

    print(f"Output directories created: {hr_output} and {lr_output}")

    # Detect scenes in HR video
    scene_list = detect_scenes(hr_path, threshold)

    if not scene_list:
        print("No scenes detected. Try adjusting the threshold value.")
        return

    # Open both videos
    hr_cap = cv2.VideoCapture(hr_path)
    lr_cap = cv2.VideoCapture(lr_path)

    if not hr_cap.isOpened():
        print(f"Error: Could not open HR video {hr_path}")
        return

    if not lr_cap.isOpened():
        print(f"Error: Could not open LR video {lr_path}")
        hr_cap.release()
        return

    # Get video properties
    hr_fps = hr_cap.get(cv2.CAP_PROP_FPS)
    lr_fps = lr_cap.get(cv2.CAP_PROP_FPS)
    hr_total_frames = int(hr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lr_total_frames = int(lr_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"HR video: {hr_total_frames} frames at {hr_fps} fps")
    print(f"LR video: {lr_total_frames} frames at {lr_fps} fps")
    print(f"LR offset: {lr_offset} frames")

    # Check if fps is the same
    if abs(hr_fps - lr_fps) > 0.1:
        print(f"Warning: HR and LR videos have different frame rates: {hr_fps} vs {lr_fps}")

    frames_extracted = 0

    # Initialize frame counters
    hr_current_frame = 0
    lr_current_frame = 0
    frames_extracted = 0
    frame_number = 1  # Start frame numbering from 1

    # Process each scene
    for i, scene in enumerate(scene_list):
        scene_start, scene_end = scene
        scene_start_frame = scene_start.get_frames()
        scene_end_frame = scene_end.get_frames()
        scene_length = scene_end_frame - scene_start_frame

        print(f"Processing scene {i}: {scene_length} frames")

        if scene_length < frames_per_scene:
            print(f"Scene {i} is too short ({scene_length} frames), skipping")
            continue

        # Seek to scene start by reading frames sequentially
        while hr_current_frame < scene_start_frame:
            ret = hr_cap.read()[0]
            if not ret:
                print(f"Error: Could not read HR frame while seeking to scene start")
                break
            hr_current_frame += 1

        # Extract consecutive frames from the beginning of the scene
        start_frame = scene_start_frame
        end_frame = min(scene_start_frame + frames_per_scene, scene_end_frame)

        print(f"Extracting consecutive frames from {start_frame} to {end_frame-1}")

        # Extract frames
        for frame_idx in range(start_frame, end_frame):
            # Read HR frame sequentially
            ret, hr_frame = hr_cap.read()
            if not ret:
                print(f"Warning: Could not read HR frame {frame_idx}")
                continue
            hr_current_frame += 1

            # Calculate LR frame index with offset
            lr_frame_idx = frame_idx - lr_offset

            # Seek LR frame by reading sequentially
            while lr_current_frame < lr_frame_idx:
                ret = lr_cap.read()[0]
                if not ret:
                    print(f"Error: Could not read LR frame while seeking")
                    break
                lr_current_frame += 1

            # Read LR frame
            ret, lr_frame = lr_cap.read()
            if not ret:
                print(f"Warning: Could not read LR frame {lr_frame_idx}")
                continue
            lr_current_frame += 1

            # Calculate frame index relative to the scene start
            relative_frame_idx = frame_idx - scene_start_frame

            # Save frames with new naming convention
            hr_frame_path = os.path.join(hr_output, f'show{i+1:04d}_frame{relative_frame_idx+1:04d}.png')
            lr_frame_path = os.path.join(lr_output, f'show{i+1:04d}_frame{relative_frame_idx+1:04d}.png')

            cv2.imwrite(hr_frame_path, hr_frame)
            cv2.imwrite(lr_frame_path, lr_frame)

            frames_extracted += 1
            frame_number += 1  # Increment frame number for next iteration

            if frames_extracted % 10 == 0:
                print(f"Saved {frames_extracted} frame pairs so far...")

        print(f"Extracted {end_frame - start_frame} consecutive frames from scene {i}")

    # Release video captures
    hr_cap.release()
    lr_cap.release()

    print(f"Extraction complete. Total frames extracted: {frames_extracted}")

def main():
    parser = argparse.ArgumentParser(description='Extract consecutive frames from HR and LR videos at scene changes')
    parser.add_argument('--hr', required=True, help='Path to high resolution video')
    parser.add_argument('--lr', required=True, help='Path to low resolution video')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--frames', type=int, default=10, help='Number of consecutive frames to extract per scene')
    parser.add_argument('--offset', type=int, default=0, help='Frame offset for LR video')
    parser.add_argument('--threshold', type=float, default=30.0, help='Scene detection threshold')

    args = parser.parse_args()

    extract_frames(
        args.hr,
        args.lr,
        args.output,
        frames_per_scene=args.frames,
        lr_offset=args.offset,
        threshold=args.threshold
    )

if __name__ == '__main__':
    main()
