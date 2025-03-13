@echo off
echo Checking Python installation...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3 and try again.
    pause
    exit /b 1
)

echo Checking Streamlit installation...
python -c "import streamlit" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Streamlit is not installed. Installing required packages...
    python -m pip install -r requirements.txt
)

echo Checking FFmpeg installation...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo FFmpeg is not installed or not in PATH.
    echo Please download FFmpeg from https://ffmpeg.org/download.html
    echo Add the bin directory to your PATH environment variable.
    pause
    exit /b 1
)

echo Creating necessary directories...
if not exist data mkdir data
if not exist exports mkdir exports

echo Starting Clipper application...
streamlit run app.py

pause 