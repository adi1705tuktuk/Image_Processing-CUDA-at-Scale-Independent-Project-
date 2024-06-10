#!/bin/bash

INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
OPERATION="canny"  # Default operation
KERNEL_SIZE=3      # Default kernel size for morphological operations (erosion/dilation)

mkdir -p $OUTPUT_DIR

# Function to display usage
usage() {
    echo "Usage: $0 [-o operation] [-k kernel_size] [-i input_dir] [-u output_dir]"
    echo "  -o operation: The operation to perform (canny, erosion, dilation, histogram, sobel)"
    echo "  -k kernel_size: Kernel size for morphological operations (default: 3)"
    echo "  -i input_dir: Directory containing input images (default: data/input)"
    echo "  -u output_dir: Directory to save output images (default: data/output)"
    exit 1
}

# Parse command-line arguments
while getopts ":o:k:i:u:" opt; do
    case ${opt} in
        o)
            OPERATION=$OPTARG
            ;;
        k)
            KERNEL_SIZE=$OPTARG
            ;;
        i)
            INPUT_DIR=$OPTARG
            ;;
        u)
            OUTPUT_DIR=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Process each image in the input directory
for INPUT_IMAGE in $INPUT_DIR/*; do
    BASENAME=$(basename $INPUT_IMAGE)
    OUTPUT_IMAGE="$OUTPUT_DIR/$BASENAME"
    echo "Processing $INPUT_IMAGE -> $OUTPUT_IMAGE"

    case $OPERATION in
        canny)
            ./image_processing canny $INPUT_IMAGE $OUTPUT_IMAGE
            ;;
        erosion)
            ./image_processing erosion $INPUT_IMAGE $OUTPUT_IMAGE $KERNEL_SIZE
            ;;
        dilation)
            ./image_processing dilation $INPUT_IMAGE $OUTPUT_IMAGE $KERNEL_SIZE
            ;;
        histogram)
            ./image_processing histogram $INPUT_IMAGE $OUTPUT_IMAGE
            ;;
        sobel)
            ./image_processing sobel $INPUT_IMAGE $OUTPUT_IMAGE
            ;;
        *)
            echo "Unknown operation: $OPERATION"
            usage
            ;;
    esac
done
