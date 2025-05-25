#!/bin/bash

set -e

GDRIVE_FILE_ID="1gZ0HUO8owebGXnaeFw_1mRcmSpYQuM6v"
OUTPUT_FILENAME="data.tar.gz"
EXTRACT_DIR="."

pip install gdown || { echo "Failed to install gdown. Exiting."; exit 1; }
gdown --id "$GDRIVE_FILE_ID" --output "$OUTPUT_FILENAME" || { echo "Failed to download file. Exiting."; exit 1; }

if [ ! -f "$OUTPUT_FILENAME" ]; then
    echo "Error: Downloaded file '$OUTPUT_FILENAME' not found."
    exit 1
fi

tar -xzvf "$OUTPUT_FILENAME" -C "$EXTRACT_DIR" || { echo "Failed to extract archive. Exiting."; exit 1; }

rm "$OUTPUT_FILENAME"
