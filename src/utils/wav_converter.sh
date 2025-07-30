#!/bin/bash

# Directory containing the MP3 files
SOURCE_DIR="/data/EECS-MachineListeningLab/datasets/AudioCaps-E/subjective_eval/_enrich"
# Directory to save the WAV files
TARGET_DIR="/data/EECS-MachineListeningLab/datasets/AudioCaps-E/subjective_eval/enrich"

SAMPLING_RATE = 32000

# Create target directory if it does not exist
mkdir -p "$TARGET_DIR"

# Loop through all MP3 files in the source directory
for file in "$SOURCE_DIR"/*.mp3; do
    # Extract filename without extension
    filename=$(basename "$file" .mp3)
    
    # Convert to WAV
    ffmpeg -i "$file" -ar "$SAMPLING_RATE" "$TARGET_DIR/$filename.wav"
done

echo "Conversion complete."