#!/bin/bash

# Script to recursively scale down images by 20% (to 80% of original size)
# Creates a new parallel directory for each iteration
# Total 4 iterations (original -> 80% -> 64% -> 51.2% -> 40.96%)
# Uses ImageMagick with parallel processing for maximum speed

set -e

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    echo "Scales images down by 20% (to 80% size) iteratively 4 times"
    echo "Creates new parallel directories: <dir>_80, <dir>_64, <dir>_51, <dir>_41"
    exit 1
fi

INPUT_DIR="$1"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Get the base directory and name
PARENT_DIR=$(dirname "$INPUT_DIR")
BASE_NAME=$(basename "$INPUT_DIR")

# Define output directories for each iteration
DIR_80="${PARENT_DIR}/${BASE_NAME}_80"
DIR_64="${PARENT_DIR}/${BASE_NAME}_64"
DIR_51="${PARENT_DIR}/${BASE_NAME}_51"
DIR_41="${PARENT_DIR}/${BASE_NAME}_41"

# Function to scale images in a directory
scale_images() {
    local input_dir="$1"
    local output_dir="$2"
    local scale_factor="$3"
    local iteration="$4"
    
    echo "=== Iteration $iteration: Scaling images to ${scale_factor}% ==="
    echo "Input:  $input_dir"
    echo "Output: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Count total images
    local total_files=$(find "$input_dir" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tiff" -o -iname "*.raw" \) | wc -l)
    
    if [ "$total_files" -eq 0 ]; then
        echo "Warning: No image files found in $input_dir"
        return 1
    fi
    
    echo "Found $total_files images to process..."
    
    # Use GNU Parallel if available, otherwise fall back to xargs
    if command -v parallel &> /dev/null; then
        echo "Using GNU Parallel for maximum concurrency..."
        
        export OUTPUT_DIR="$output_dir"
        export SCALE_FACTOR="$scale_factor"
        
        find "$input_dir" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tiff" -o -iname "*.raw" \) | \
        parallel -j +0 \
            'magick "{}" -resize "'"${scale_factor}"'%" -quality 95 "'"$output_dir"'/{/}"'
    else
        echo "Using find with background jobs for parallel processing..."
        
        find "$input_dir" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tiff" -o -iname "*.raw" \) | \
        xargs -P 4 -I {} bash -c '
            filename=$(basename "{}")
            magick "{}" -resize "'"${scale_factor}"'%" -quality 95 "'"$output_dir"'/$filename"
        '
    fi
    
    local output_files=$(find "$output_dir" -maxdepth 1 -type f | wc -l)
    echo "✓ Iteration $iteration complete: Created $output_files scaled images"
    echo ""
}

# Function to rename JPB files to jpg
rename_jpb_to_jpg() {
    local dir="$1"
    
    local jpb_count=$(find "$dir" -maxdepth 1 -iname "*.jpb" | wc -l)
    
    if [ "$jpb_count" -gt 0 ]; then
        echo "=== Renaming JPB files to jpg ==="
        echo "Found $jpb_count JPB files in $dir"
        
        find "$dir" -maxdepth 1 -iname "*.jpb" | while read -r file; do
            new_name="${file%.*}.jpg"
            if mv "$file" "$new_name"; then
                echo "✓ Renamed: $(basename "$file") -> $(basename "$new_name")"
            else
                echo "✗ Failed to rename: $(basename "$file")"
            fi
        done
        
        echo "✓ JPB renaming complete"
        echo ""
    fi
}

# Main execution
echo "======================================"
echo "Image Scaling Pipeline (4 iterations)"
echo "======================================"
echo "Starting directory: $INPUT_DIR"
echo ""

# First, rename any JPB files to jpg
rename_jpb_to_jpg "$INPUT_DIR"

# Iteration 1: Original -> 80%
scale_images "$INPUT_DIR" "$DIR_80" "80" "1"

# Iteration 2: 80% -> 64% (80% of 80%)
scale_images "$DIR_80" "$DIR_64" "80" "2"

# Iteration 3: 64% -> 51.2% (80% of 64%)
scale_images "$DIR_64" "$DIR_51" "80" "3"

# Iteration 4: 51.2% -> 40.96% (80% of 51.2%)
scale_images "$DIR_51" "$DIR_41" "80" "4"

echo "======================================"
echo "✓ All iterations complete!"
echo "======================================"
echo ""
echo "Results:"
printf "%-15s: 100.0%% of original\n" "$BASE_NAME"
printf "%-15s: 80.0%% of original\n" "${BASE_NAME}_80"
printf "%-15s: 64.0%% of original\n" "${BASE_NAME}_64"
printf "%-15s: 51.2%% of original\n" "${BASE_NAME}_51"
printf "%-15s: 40.96%% of original\n" "${BASE_NAME}_41"
echo ""

# Show directory sizes
echo "Directory sizes:"
du -sh "$INPUT_DIR"
du -sh "$DIR_80"
du -sh "$DIR_64"
du -sh "$DIR_51"
du -sh "$DIR_41"
