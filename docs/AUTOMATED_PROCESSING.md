# Automated Clip Processing

This document explains how to use the clip status system and automated processing features in Clipper.

## Clip Status System

The Clipper application now supports a status system for clips, which helps manage the workflow from creation to processing and completion. Each clip can have one of three statuses:

1. **Draft** - The default status for new clips. Indicates that the clip is still being edited or refined.
2. **Process** - Indicates that the clip is ready for processing by automated scripts.
3. **Complete** - Indicates that the clip has been processed and is ready for use.

### Managing Clip Status

You can change a clip's status using the dropdown selector in the clip management panel:

1. Select a video in the sidebar's "Videos" tab
2. Go to the "Clips" tab
3. Expand a clip from the list
4. Use the "Status" dropdown to select a new status
5. Click the "Save" button to apply the changes

## Automated Processing

The application includes a utility for automated processing of clips. This allows you to set up a dedicated machine to watch for clips with "Process" status and process them automatically.

### Processing Script

The `scripts/process_clips.py` script is provided as an example of how to set up automated processing. This script:

1. Scans all clip configuration files for clips with "Process" status
2. Processes each clip according to the defined processing function
3. Updates the clip status to "Complete" when processing is successful

### Running the Processing Script

You can run the processing script in several modes:

#### One-time scan:

```bash
python scripts/process_clips.py
```

This will scan for clips with "Process" status, process them, and exit.

#### Daemon mode:

```bash
python scripts/process_clips.py --daemon --interval 60
```

This will run continuously, scanning for new clips to process every 60 seconds (or other interval you specify).

#### Camera Filtering:

```bash
python scripts/process_clips.py --camera SONY
```

This will only process clips from videos captured with the specified camera type. This is useful for setting up dedicated processing machines for different camera sources.

#### Export Formats:

By default, clips are exported in standard H.264 format. You can choose from two additional export options:

```bash
# CV-optimized export only (FFV1 codec, optimized for computer vision)
python scripts/process_clips.py --cv-optimized

# Export both formats (both H.264 and FFV1)
python scripts/process_clips.py --both-formats
```

When using `--both-formats`, the script will:

- Create two separate files for each clip (one in H.264 format and one in FFV1 format)
- Store both file paths as an array in the clip's JSON configuration
- Mark the clip as "Complete" only after both formats are successfully created

#### Parallel Processing and Batch Controls:

```bash
python scripts/process_clips.py --max-workers 2 --batch-size 10
```

- `--max-workers` controls how many clips are processed in parallel (default: 1)
- `--batch-size` limits how many clips are processed in one batch (0 for unlimited)

**Note on Resource Usage**: The default `--max-workers` value is 1, which is recommended for most systems. This is because each FFmpeg process can use up to 16 threads, so processing multiple clips in parallel can quickly oversubscribe your CPU. Only increase this value if you have a high-end workstation or server with many CPU cores.

You can combine these options as needed:

```bash
python scripts/process_clips.py --daemon --interval 120 --camera GOPRO --max-workers 2 --batch-size 5 --both-formats
```

This would run as a daemon, checking every 2 minutes for GoPro clips, processing 5 clips at a time with 2 parallel workers, and creating both regular and CV-optimized versions of each clip.

### Output File Structure

When processing clips, the script creates the following folder structure:

```
data/prept/03_CLIPPED/
├── h264/                   # Standard H.264 clips
│   └── [camera]/
│       └── [session]/
│           └── clip_files.mp4
└── ffv1/                   # CV-optimized clips
    └── [camera]/
        └── [session]/
            └── clip_files.mkv
```

This organized structure ensures clips are properly categorized by codec type, camera, and session folder.

### Customizing Processing Logic

To customize the processing logic, modify the `process_clip` function in `scripts/process_clips.py`. The default implementation:

1. Resolves the full path to the source video
2. Creates an output path for the processed clip
3. Uses `proxy_service.export_clip` to export the clip with the specified crop region

You can replace this with your own processing logic, such as applying effects, encoding with specific settings, or uploading to a server.

## Setting Up a Processing Server

To set up a dedicated processing server:

1. Install Clipper on the server
2. Configure the server to access the same video sources and configuration files
   - You can use shared network drives or a synchronization tool like rsync
3. Run the processing script in daemon mode
4. Optionally, set up the script to start automatically on boot

## Workflow Example

1. Videographer captures footage
2. Editor uses Clipper to:
   - Create clips from the footage
   - Set crop regions and keyframes
   - Change clip status to "Process" when ready
   - Save the configuration
3. Processing server:
   - Detects clips with "Process" status
   - Processes them according to the defined logic
   - Updates their status to "Complete"
4. Editor can then:
   - See which clips have been processed (status: "Complete")
   - Make further edits if needed, changing status back to "Draft"
   - Set clips back to "Process" for re-processing

## Monitoring

The processing script logs all activity to both the console and a log file (`scripts/clip_processor.log`). You can monitor this log to track the progress of clip processing.

## Extending the System

The status system is designed to be easily extended. Some possible enhancements:

- Add more statuses (e.g., "Rejected", "Archived")
- Add additional metadata fields (e.g., "Processing Notes", "Quality Rating")
- Integrate with external systems via webhooks or APIs
- Create a web interface for monitoring processing status
