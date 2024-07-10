#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -h <high_quality_video> -l <low_quality_video> -s <seconds_between_frames> [-n <start_number>] [-o <output_folder>]"
    echo
    echo "Options:"
    echo "  -h    Path to high quality video file"
    echo "  -l    Path to low quality video file"
    echo "  -s    Number of seconds between captured frames (default: 1)"
    echo "  -n    Starting number for frame counting (default: 1)"
    echo "  -o    Custom output folder name (default: name of the low quality video file)"
    exit 1
}

# Function to create directory
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Parse command line arguments
while getopts "h:l:s:n:o:" opt; do
    case $opt in
        h) hq_video="$OPTARG" ;;
        l) lq_video="$OPTARG" ;;
        s) seconds_between_frames="$OPTARG" ;;
        n) start_number="$OPTARG" ;;
        o) output_folder="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if required arguments are provided
if [ -z "$hq_video" ] || [ -z "$lq_video" ]; then
    usage
fi

# Set default values if not provided
seconds_between_frames=${seconds_between_frames:-1}
start_number=${start_number:-1}
output_folder=${output_folder:-$(basename "$lq_video" .mp4)}

# Create temporary directories for frame extraction
temp_hq_dir=$(mktemp -d)
temp_lq_dir=$(mktemp -d)

# Set the frame selection filter based on seconds between frames
frame_select="select='not(mod(t,$seconds_between_frames))'"

# Run FFmpeg command for high quality video
ffmpeg -i "$hq_video" -vf "$frame_select,setpts=N/FRAME_RATE/TB" \
    -vsync 0 -start_number "$start_number" "$temp_hq_dir/%d.png"

# Run FFmpeg command for low quality video
ffmpeg -i "$lq_video" -vf "$frame_select,setpts=N/FRAME_RATE/TB" \
    -vsync 0 -start_number "$start_number" "$temp_lq_dir/%d.png"

# Count the number of frames in each directory
hq_frame_count=$(ls "$temp_hq_dir" | wc -l)
lq_frame_count=$(ls "$temp_lq_dir" | wc -l)

# Determine the minimum frame count
min_frame_count=$((hq_frame_count < lq_frame_count ? hq_frame_count : lq_frame_count))

# Create output directories
create_dir "$output_folder"
create_dir "$output_folder/hq"
create_dir "$output_folder/lq"

# Copy only paired frames to the output directories
for i in $(seq $start_number $((start_number + min_frame_count - 1))); do
    cp "$temp_hq_dir/$i.png" "$output_folder/hq/$i.png"
    cp "$temp_lq_dir/$i.png" "$output_folder/lq/$i.png"
done

# Print warning if frame counts don't match
if [ $hq_frame_count -ne $lq_frame_count ]; then
    echo "WARNING: Frame count mismatch detected."
    echo "High quality frames: $hq_frame_count"
    echo "Low quality frames: $lq_frame_count"
    echo "Only $min_frame_count paired frames were saved."
fi

# Clean up temporary directories
rm -rf "$temp_hq_dir" "$temp_lq_dir"

echo "Frame extraction complete. Output saved in $output_folder"
echo "Frames are numbered from $start_number to $((start_number + min_frame_count - 1))"
