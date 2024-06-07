#!/bin/bash

INPUT_DIR="data/input"
OUTPUT_DIR="data/output"

mkdir -p $OUTPUT_DIR

for INPUT_IMAGE in $INPUT_DIR/*; do
    BASENAME=$(basename $INPUT_IMAGE)
    OUTPUT_IMAGE="$OUTPUT_DIR/$BASENAME"
    echo "Processing $INPUT_IMAGE -> $OUTPUT_IMAGE"
    ./edge_detection $INPUT_IMAGE $OUTPUT_IMAGE
done
