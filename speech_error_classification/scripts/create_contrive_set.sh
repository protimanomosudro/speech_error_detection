#!/bin/bash

CSV_DIR="data/metadata/"
OUTPUT_DIR="data/metadata/"
CONTRIVE_RATIO=0.5
SEED=42

# Create output directory if it doesn't exist
if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    mkdir -p "$(dirname "OUTPUT_DIR")"
fi

echo "Creating contrived datasets with balanced event and non-event samples..."

python src/feature_extraction/create_contrive_set.py \
    --csv_dir "$CSV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ratio "$CONTRIVE_RATIO" \
    --seed "$SEED"

if [ $? -ne 0 ]; then
    echo "Error: create_contrive_set.py failed."
    exit 1
fi

echo "Contrived datasets created successfully."
echo "Contrived data is located in: $OUTPUT_DIR"
