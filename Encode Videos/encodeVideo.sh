#!/bin/bash

# Default values
codec="h264"
min_quality=16
max_quality=32
min_segments=10
min_segment_duration=5  # Minimum segment duration in seconds

# Function to display usage
usage() {
    echo "Usage: $0 <input_video> [options]"
    echo "Options:"
    echo "  --codec <codec>      Codec to use (vp8, vp9, h264, mpeg, mpeg2, svt-av1, hevc) (default: h264)"
    echo "  --min <value>        Minimum quality value (default: 16)"
    echo "  --max <value>        Maximum quality value (default: 32)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --codec) codec="$2"; shift 2 ;;
        --min) min_quality="$2"; shift 2 ;;
        --max) max_quality="$2"; shift 2 ;;
        -*) echo "Unknown option $1"; usage ;;
        *) input_video="$1"; shift ;;
    esac
done

# Check if input file is provided
if [ -z "$input_video" ]; then
    usage
fi

# Set output container based on codec
case $codec in
    vp8|vp9) container="mkv" ;;
    *) container="mp4" ;;
esac

output_video="output_multi_quality_${codec}.${container}"
log_file="${input_video%.*}_encode_log.txt"

# Get video duration
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_video")
duration=${duration%.*}  # Remove decimal part

# Calculate number of segments
if [ $duration -le 60 ]; then
    num_segments=$min_segments
else
    num_segments=$((duration / 6))  # Divide by 6 to get at least 10 segments for 1-minute video
    if [ $num_segments -lt $min_segments ]; then
        num_segments=$min_segments
    fi
fi

# Calculate segment duration
segment_duration=$((duration / num_segments))
if [ $segment_duration -lt $min_segment_duration ]; then
    segment_duration=$min_segment_duration
    num_segments=$((duration / segment_duration))
fi

# Temporary files
temp_files=()

# Initialize log file
echo "Encoding log for $input_video" > "$log_file"
echo "Total duration: $duration seconds" >> "$log_file"
echo "Number of full segments: $num_segments" >> "$log_file"
echo "Approximate segment duration: $segment_duration seconds" >> "$log_file"
echo "Codec: $codec" >> "$log_file"
echo "Quality range: $min_quality - $max_quality" >> "$log_file"
echo "----------------------------------------" >> "$log_file"

# Encode each segment
start_time=0
segment_count=0

while [ $start_time -lt $duration ]; do
    end_time=$((start_time + segment_duration))
    if [ $end_time -ge $duration ]; then
        end_time=$duration
    fi
    
    # Generate random quality value between min_quality and max_quality
    quality=$((RANDOM % (max_quality - min_quality + 1) + min_quality))
    
    temp_file="temp_${segment_count}_quality${quality}.${container}"
    temp_files+=("$temp_file")
    
    encode_start=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Encode based on selected codec
    case $codec in
        vp8)
            ffmpeg -i "$input_video" -ss "$start_time" -to "$end_time" \
                   -c:v libvpx -crf "$quality" -b:v 0 \
                   -c:a copy "$temp_file"
            quality_param="CRF"
            ;;
        vp9)
            ffmpeg -i "$input_video" -ss "$start_time" -to "$end_time" \
                   -c:v libvpx-vp9 -crf "$quality" -b:v 0 \
                   -c:a copy "$temp_file"
            quality_param="CRF"
            ;;
        h264)
            ffmpeg -i "$input_video" -ss "$start_time" -to "$end_time" \
                   -c:v libx264 -crf "$quality" \
                   -c:a copy "$temp_file"
            quality_param="CRF"
            ;;
        mpeg)
            ffmpeg -i "$input_video" -ss "$start_time" -to "$end_time" \
                   -c:v mpeg1video -qscale:v "$quality" \
                   -c:a copy "$temp_file"
            quality_param="qscale"
            ;;
        mpeg2)
            ffmpeg -i "$input_video" -ss "$start_time" -to "$end_time" \
                   -c:v mpeg2video -qscale:v "$quality" \
                   -c:a copy "$temp_file"
            quality_param="qscale"
            ;;
        svt-av1)
            ffmpeg -i "$input_video" -ss "$start_time" -to "$end_time" \
                   -c:v libsvtav1 -crf "$quality" \
                   -c:a copy "$temp_file"
            quality_param="CRF"
            ;;
        hevc)
            ffmpeg -i "$input_video" -ss "$start_time" -to "$end_time" \
                   -c:v libx265 -crf "$quality" \
                   -c:a copy "$temp_file"
            quality_param="CRF"
            ;;
        *)
            echo "Unsupported codec: $codec"
            exit 1
            ;;
    esac
    
    encode_end=$(date +"%Y-%m-%d %H:%M:%S")
    
    echo "Segment $((segment_count+1)) encoded with $quality_param $quality"
    
    # Log segment information
    echo "Segment $((segment_count+1)):" >> "$log_file"
    echo "  Time range: ${start_time}s - ${end_time}s" >> "$log_file"
    echo "  $quality_param value: $quality" >> "$log_file"
    echo "  Encode start: $encode_start" >> "$log_file"
    echo "  Encode end: $encode_end" >> "$log_file"
    echo "----------------------------------------" >> "$log_file"
    
    start_time=$end_time
    segment_count=$((segment_count + 1))
done

# Prepare concat file
concat_file="concat_list.txt"
for temp_file in "${temp_files[@]}"; do
    echo "file '$temp_file'" >> "$concat_file"
done

# Concatenate all segments
ffmpeg -f concat -i "$concat_file" -c copy "$output_video"

# Clean up temporary files
rm "${temp_files[@]}" "$concat_file"

echo "Encoding complete. Output saved as $output_video"
echo "Log file saved as $log_file"
echo "Total segments encoded: $segment_count"
