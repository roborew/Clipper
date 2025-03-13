# Installation Guide for Clipper

This guide will help you set up and run the Clipper application on your system.

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Get the Code

#### Option A: Clone the Repository (if you have Git installed)

```bash
git clone https://github.com/yourusername/clipper.git
cd clipper
```

#### Option B: Download the ZIP File

- Download the ZIP file from the repository
- Extract it to a folder of your choice
- Open a terminal/command prompt and navigate to the extracted folder

### 2. Set Up the Environment

#### Option A: Using pip (recommended for most users)

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda (alternative)

```bash
# Create a conda environment
conda env create -f environment.yml

# Activate the environment
conda activate clipper
```

### 3. Install FFmpeg

FFmpeg is required for video processing. Installation depends on your operating system:

#### Windows

1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract the ZIP file to a folder (e.g., `C:\ffmpeg`)
3. Add the `bin` folder to your PATH environment variable:
   - Right-click on "This PC" or "My Computer" and select "Properties"
   - Click on "Advanced system settings"
   - Click on "Environment Variables"
   - Under "System variables", find the "Path" variable, select it and click "Edit"
   - Click "New" and add the path to the bin folder (e.g., `C:\ffmpeg\bin`)
   - Click "OK" on all dialogs

#### macOS

Using Homebrew:

```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

### 4. Create Required Directories

The application needs certain directories to store data:

```bash
mkdir -p data
mkdir -p exports
```

## Running the Application

### Option 1: Using the Provided Scripts

#### On Windows:

Double-click on `run.bat` or run it from the command prompt:

```
run.bat
```

#### On macOS/Linux:

Make the script executable and run it:

```bash
chmod +x run.sh
./run.sh
```

### Option 2: Running Directly with Streamlit

```bash
streamlit run app.py
```

## Verifying the Installation

After starting the application:

1. The Streamlit server should start and automatically open a browser window
2. If it doesn't open automatically, you can access it at http://localhost:8501
3. You should see the Clipper interface with a sidebar for video selection

## Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError: No module named 'streamlit'"**

   - Make sure you've installed the requirements: `pip install -r requirements.txt`
   - If using conda, ensure you've activated the environment: `conda activate clipper`

2. **"FFmpeg is not installed or not in PATH"**

   - Make sure FFmpeg is installed and added to your PATH
   - Try running `ffmpeg -version` in a terminal to verify

3. **"No video files found"**

   - Place your video files in the `data` directory
   - Check the configuration in `config.yaml` for the correct video directories

4. **Browser shows "This site can't be reached"**
   - Check if Streamlit is running in the terminal
   - Try accessing the application at http://localhost:8501
   - Check if another application is using port 8501

For more help, please open an issue on the GitHub repository.
