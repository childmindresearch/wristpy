#!/bin/bash

# Source directories
dir1="/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/external_ID/"
dir2="/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/gt3x_ExternalID/"

# Destination directory
dest_dir="/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/data_to_process/"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Search for files with the name "*.gt3x" in both directories
files1=$(find "$dir1" -type f -name "*.gt3x")
files2=$(find "$dir2" -type f -name "*.gt3x")

# Combine the files from both directories
all_files="$files1"$'\n'"$files2"

# Loop through each file
while IFS= read -r file; do
    # Get the filename without the path
    filename=$(basename "$file")

    # Check if the file already exists in the destination directory
    if [[ ! -f "$dest_dir/$filename" ]]; then
        # Copy the file to the destination directory
        cp "$file" "$dest_dir"
    fi
done <<< "$all_files"
