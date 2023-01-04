import cv2
import os
import sys
import argparse
from tqdm import tqdm

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_video", help="path to the input video file")
parser.add_argument("output_folder", help="path to the output folder where the frames will be saved")
parser.add_argument("-f", "--fps", type=float, help="frame rate at which to extract the frames")
parser.add_argument("-fmt", "--format", help="image format in which to save the frames (e.g. jpg, png)")
parser.add_argument("-s", "--start", type=int, help="start frame number")
parser.add_argument("-e", "--end", type=int, help="end frame number")
args = parser.parse_args()

# Open the video file
video = cv2.VideoCapture(args.input_video)

# Check if the video file was opened successfully
if not video.isOpened():
    print(f"Error: unable to open video file {args.input_video}")
    sys.exit()

# Get the total number of frames in the video
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the start and end frame numbers
if args.start:
    start_frame = args.start
else:
    start_frame = 0
if args.end:
    end_frame = args.end
else:
    end_frame = total_frames

# Check if the start and end frame numbers are valid
if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
    print("Error: invalid start and end frame numbers")
    sys.exit()

# Create the output folder if it doesn't exist
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Get the frame rate of the video
video_fps = video.get(cv2.CAP_PROP_FPS)

# Calculate the number of frames to skip between each extracted frame
if args.fps:
    skip_frames = int(video_fps / args.fps)
else:
    skip_frames = 1

# Extract the frames and save them to the output folder
for frame_number in tqdm(range(start_frame, end_frame, skip_frames)):
    # Read the current frame
    success, frame = video.read()
    
    # Save the frame to the output folder
    if args.format:
        cv2.imwrite(f"{args.output_folder}/{frame_number}.{args.format}", frame)
    else:
        cv2.imwrite(f"{args.output_folder}/{frame_number}.png", frame)

# Release the video file
video.release()

