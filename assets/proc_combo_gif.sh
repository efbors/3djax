#!/bin/bash

# Define input and output files
LEFT_VID="pam4_fixed.mp4"
RIGHT_VID="qam16_fixed.mp4" # Adjust to .mp4 if needed
OUTPUT_GIF="combo_plot.gif"

# Layout variables
GAP=10
GAP_COLOR="black" # Change to "white" if your plots have a white background

# Optimization variables (Tweak these to balance size vs. quality)
FPS=10          # 10 to 15 is usually perfect for data plots/animations
MAX_WIDTH=720   # Set a maximum pixel width for the final combined GIF

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg could not be found. Please install it to run this script."
    exit 1
fi

echo "Combining and optimizing $LEFT_VID and $RIGHT_VID..."

# Run FFmpeg
# 1. Pad the left video with the gap
# 2. Horizontally stack them
# 3. Apply FPS drop and scale the combined video down (maintains aspect ratio)
# 4. Generate palette and apply it with NO dithering (crucial for smaller file size)
ffmpeg -y -i "$LEFT_VID" -i "$RIGHT_VID" -filter_complex "
    [0:v]pad=width=iw+${GAP}:height=ih:x=0:y=0:color=${GAP_COLOR}[left_padded];
    [left_padded][1:v]hstack=inputs=2[stacked];
    [stacked]fps=${FPS},scale='min(${MAX_WIDTH},iw)':-1:flags=lanczos[scaled];
    [scaled]split[s0][s1];
    [s0]palettegen=stats_mode=diff[p];
    [s1][p]paletteuse=dither=none
" -loop 0 "$OUTPUT_GIF"

echo "Success! Optimized looping GIF saved as $OUTPUT_GIF"