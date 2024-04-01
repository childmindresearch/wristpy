#!/bin/bash

# Directory containing the "filename_output" folders
output_dir="/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/wristpy_out/"

# Directory to search and delete files
search_dir="/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/data_to_process/"

# Loop through the "filename_output" folders
for folder in "$output_dir"/*_output; do
    # Extract the "filename" part
    filename=$(basename "$folder" _output)

    # Search and delete files with the extracted filename and ".gt3x" extension
    find "$search_dir" -type f -name "$filename.gt3x" -delete
done
