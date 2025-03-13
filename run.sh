#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Streamlit is not installed. Installing required packages..."
    python3 -m pip install -r requirements.txt
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is not installed. Please install FFmpeg and try again."
    echo "On macOS: brew install ffmpeg"
    echo "On Ubuntu: sudo apt-get install ffmpeg"
    echo "On Windows: Download from https://ffmpeg.org/download.html"
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p data
mkdir -p exports

# Run the application
echo "Starting Clipper application..."
streamlit run app.py 