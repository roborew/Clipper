# Clipper - Video Clipping and Cropping Tool

A Streamlit application for efficiently clipping and cropping videos with the ability to save configurations for batch processing.

## Features

- Load and preview video files
- Set in-points and out-points for precise clip selection
- Define crop regions that can be animated across the clip duration
- Save clip configurations to JSON files for later use
- Batch export multiple clips in one run
- Distribute rendering across multiple machines
- Select from common output resolutions (1080p, 720p, etc.)
- Edit or remove previously created clips

## Requirements

- Python 3.11
- Conda (for environment management)
- OpenCV
- Streamlit
- FFmpeg

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/clipper.git
   cd clipper
   ```

2. Create and activate the conda environment:

   ```
   conda env create -f environment.yml
   conda activate clipper
   ```

3. Create the necessary directories:

   ```
   mkdir -p data
   ```

4. Creating Symbolic Links

To create symbolic links to external storage:

1.  For the source data directory (if stored externally):

```bash
ln -s "/Volumes/ExternalDrive/02_CALIBRATED_FOOTAGE" data/source
```

2.  For the processed data directory:

```bash
ln -s "/path/to/StorageLocationls/00_SURF_FOOTAGE_PREPT" data/prept
```

#### Verifying Symbolic Links

To verify that your symbolic links are correctly set up:

```bash
ls -la data/
```

## Usage

1. Start the Streamlit application:

   ```
   streamlit run app.py
   ```

2. Use the sidebar to select and load a video file.

3. Navigate through the video using the player controls.

4. Set in-points and out-points for your clip.

5. Define crop regions (start and end positions if you want the crop to animate).

6. Choose an output resolution and give your clip a name.

7. Click "Add/Update Clip" to save the clip configuration.

8. Repeat steps 3-7 for additional clips.

9. Click "Export All Clips" to process and save all clips to the output directory.

## Automated Processing

Clipper includes a clip status system and automated processing capabilities:

- **Clip Status System**: Mark clips as "Draft", "Process", or "Complete"
- **Processing Script**: Automatically process clips marked as "Process"
- **Camera Filtering**: Process clips only from specific camera types
- **Batch Processing**: Control parallel processing with workers and batch sizes

For detailed information about these features, see [Automated Processing Documentation](docs/AUTOMATED_PROCESSING.md).

## Workflow for Distributed Rendering

1. Create clip configurations on one machine.
2. Share the JSON configuration file and source video with other machines.
3. On each machine, load the video and configuration file.
4. Distribute the rendering workload by exporting different clips on different machines.

## Directory Structure

- `app.py`: Main application file
- `environment.yml`: Conda environment configuration
- `data/`: Directory for source video files
- `configs/`: Directory for saved clip configurations
- `output/`: Directory for exported video clips

## License

MIT
