#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -h <high_quality_video> -l <low_quality_video> -s <seconds_between_frames>"
    echo
    echo "Options:"
    echo "  -h    Path to high quality video file"
    echo "  -l    Path to low quality video file"
    echo "  -s    Number of seconds between captured frames (default: 1)"
    exit 1
}

# Function to create directory
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Parse command line arguments
while getopts "h:l:s:" opt; do
    case $opt in
        h) hq_video="$OPTARG" ;;
        l) lq_video="$OPTARG" ;;
        s) seconds_between_frames="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if required arguments are provided
if [ -z "$hq_video" ] || [ -z "$lq_video" ]; then
    usage
fi

# Set default seconds between frames if not provided
seconds_between_frames=${seconds_between_frames:-1}

# Create output directory based on LQ video filename
output_dir=$(basename "$lq_video" .mp4)
create_dir "$output_dir"
create_dir "$output_dir/hq"
create_dir "$output_dir/lq"

# Set the frame selection filter based on seconds between frames
frame_select="select='not(mod(t,$seconds_between_frames))'"

# Run FFmpeg command for high quality video
ffmpeg -i "$hq_video" -vf "$frame_select,setpts=N/FRAME_RATE/TB" \
    -vsync 0 "$output_dir/hq/%d.png"

# Run FFmpeg command for low quality video
ffmpeg -i "$lq_video" -vf "$frame_select,setpts=N/FRAME_RATE/TB" \
    -vsync 0 "$output_dir/lq/%d.png"

echo "Frame extraction complete. Output saved in $output_dir"