#!/bin/bash

# Check if folder name is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 input_folder output_folder"
    exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Input folder does not exist!"
    exit 1
fi

# Loop through all .mkv files in the input folder
for video in "$INPUT_FOLDER"/*.mkv; do
    if [ -f "$video" ]; then
        # Get the base filename without extension
        base_name=$(basename "$video" .mkv)
        
        # Extract 5 frames per second as images
        ffmpeg -i "$video" -vf fps=5 "$OUTPUT_FOLDER/${base_name}_frame_%04d.jpg"
        
        echo "Processed: $video"
    fi
done

echo "Frame extraction complete!"