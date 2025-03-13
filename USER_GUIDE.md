# Clipper User Guide

This guide will help you use the Clipper application to create and export video clips with precise control over crop regions.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Working with Videos](#working-with-videos)
4. [Creating Clips](#creating-clips)
5. [Crop Regions](#crop-regions)
6. [Exporting Clips](#exporting-clips)
7. [Tips and Tricks](#tips-and-tricks)

## Getting Started

After [installing](INSTALL.md) and launching Clipper, you'll see the main interface with a sidebar on the left and the main content area on the right.

If this is your first time using the application, you'll need to:

1. Place your video files in the `data` directory
2. Configure the video directories in `config.yaml` if needed
3. Restart the application if you've added new videos while it was running

## Interface Overview

The Clipper interface consists of:

### Sidebar

- **Videos Tab**: Select videos to work with
- **Clips Tab**: Manage your saved clips
- **Settings Tab**: Configure application settings

### Main Content Area

- **Video Player**: View and navigate through the video
- **Controls**: Navigate frames, set clip points, and manage crop regions
- **Clip Information**: View details about the current clip

## Working with Videos

### Selecting a Video

1. Click on the "Videos" tab in the sidebar
2. Choose a video from the dropdown menu
3. The video will load in the main content area

### Video Information

After selecting a video, you'll see information about it:

- Resolution
- Duration
- FPS (frames per second)
- Total frames

### Proxy Videos

For better performance, Clipper can create proxy (lower resolution) versions of your videos:

1. If a proxy doesn't exist, you'll see a "Create Proxy" button
2. Click it to generate a proxy video (this may take some time)
3. Once created, the proxy will be used for playback

### Navigating the Video

Use the player controls to navigate through the video:

- First/Last Frame buttons
- Previous/Next Frame buttons
- Jump buttons (-10/+10 frames)
- Frame slider
- Timecode display

## Creating Clips

A clip is a section of video defined by start and end frames, with optional crop regions.

### Creating a New Clip

1. Navigate to the desired start frame
2. Click "Set Start Frame"
3. Navigate to the desired end frame
4. Click "Set End Frame"
5. Click "Create New Clip" if you want to create a new clip

### Managing Clips

In the "Clips" tab of the sidebar, you can:

- View all your saved clips
- Select a clip to work on
- Delete clips you no longer need
- Save changes to clips

## Crop Regions

Crop regions allow you to select a portion of the frame to include in your exported clip.

### Setting a Crop Region

1. Navigate to the frame where you want to set a crop region
2. Click "Select Crop at Current Frame"
3. Use the controls to position and size the crop region:
   - Arrow buttons to move the region
   - Size buttons to make it larger or smaller
   - "Center" button to center the region
4. Click "Set Crop Position" to confirm

### Keyframe Animation

You can set different crop regions at different frames to create animated panning and zooming effects:

1. Navigate to the first keyframe position
2. Set a crop region
3. Navigate to another frame
4. Set a different crop region
5. Clipper will automatically interpolate between keyframes

### Managing Keyframes

- View all keyframes in the "Keyframes" section
- Click "Go to" to navigate to a specific keyframe
- Click "Delete" to remove a keyframe
- Click "Clear Crop Keyframe" to remove the keyframe at the current frame

## Exporting Clips

### Previewing a Clip

Before exporting, you can preview your clip:

1. Select the clip you want to preview
2. Click "Play Clip" to see how it will look with the current settings

### Export Settings

In the "Settings" tab of the sidebar, you can configure:

- Output resolution (2160p, 1080p, 720p, etc.)

### Exporting a Clip

1. Select the clip you want to export
2. Click "Export Clip"
3. The clip will be exported to the `exports` directory
4. You'll see a success message when the export is complete

## Tips and Tricks

### Keyboard Shortcuts

- **Left/Right Arrow**: Previous/Next frame
- **Home/End**: First/Last frame
- **Page Up/Down**: Jump -10/+10 frames

### Efficient Workflow

1. Create proxy videos for all your source material first
2. Set start and end frames for all your clips
3. Then go back and add crop regions
4. Preview each clip before exporting
5. Export all clips at once

### Troubleshooting

If you encounter issues:

- Check the logs in the sidebar (enable in Settings)
- Make sure your video files are in the correct format
- Ensure FFmpeg is properly installed
- Restart the application if it becomes unresponsive

For more detailed information, refer to the [README.md](README.md) file.
