"""
Video playback and frame manipulation services for the Clipper application.
"""

import os
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import logging
import base64

logger = logging.getLogger("clipper.video")


def get_frame(video_path, frame_number, config_manager=None):
    """
    Get a specific frame from a video file

    Args:
        video_path: Path to the video file
        frame_number: Frame number to retrieve (0-indexed)
        config_manager: ConfigManager instance

    Returns:
        The frame as a numpy array, or None if the frame could not be retrieved
    """
    try:
        # Check if we should use proxy video
        if "proxy_path" in st.session_state and st.session_state.proxy_path:
            # Use proxy video if available
            proxy_path = st.session_state.proxy_path
            logger.debug(f"Using proxy video for frame extraction: {proxy_path}")
            video_path = proxy_path

        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate frame number
        if frame_number < 0 or frame_number >= total_frames:
            logger.warning(
                f"Invalid frame number: {frame_number}, total frames: {total_frames}"
            )
            # Clamp to valid range
            frame_number = max(0, min(frame_number, total_frames - 1))

        # Seek to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        # Release the video capture
        cap.release()

        if not ret:
            logger.error(f"Failed to read frame {frame_number} from {video_path}")
            return None

        # Convert from BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame_rgb

    except Exception as e:
        logger.exception(
            f"Error getting frame {frame_number} from {video_path}: {str(e)}"
        )
        return None


def get_video_info(video_path):
    """
    Get information about a video file

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video information or None if the video could not be opened
    """
    try:
        # Open the video file with the exact path provided
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return None

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Release the video capture
        cap.release()

        # Create info dictionary
        video_info = {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "duration_formatted": format_duration(duration),
            "aspect_ratio": width / height if height > 0 else 0,
        }

        return video_info

    except Exception as e:
        logger.exception(f"Error getting video info for {video_path}: {str(e)}")
        return None


def format_duration(seconds):
    """
    Format duration in seconds to HH:MM:SS format

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def format_timecode(frame_number, fps):
    """
    Format frame number to timecode (HH:MM:SS:FF)

    Args:
        frame_number: Frame number
        fps: Frames per second

    Returns:
        Timecode string
    """
    if fps <= 0:
        return "00:00:00:00"

    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames = int((total_seconds - int(total_seconds)) * fps)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def frame_to_timecode(frame_number, fps):
    """
    Convert frame number to timecode

    Args:
        frame_number: Frame number
        fps: Frames per second

    Returns:
        Timecode in seconds
    """
    if fps <= 0:
        return 0
    return frame_number / fps


def timecode_to_frame(timecode, fps):
    """
    Convert timecode to frame number

    Args:
        timecode: Timecode in seconds
        fps: Frames per second

    Returns:
        Frame number
    """
    if fps <= 0:
        return 0
    return int(timecode * fps)


def parse_timecode_to_frame(timecode_str, fps):
    """
    Parse a timecode string in the format HH:MM:SS:FF or HH:MM:SS or MM:SS
    and convert it to a frame number.

    Args:
        timecode_str: Timecode string in format HH:MM:SS:FF, HH:MM:SS, or MM:SS
        fps: Frames per second

    Returns:
        Frame number
    """
    if fps <= 0:
        return 0

    # Split the timecode string
    parts = timecode_str.strip().split(":")

    if len(parts) == 4:  # HH:MM:SS:FF
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        frames = int(parts[3])
    elif len(parts) == 3:  # HH:MM:SS
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        frames = 0
    elif len(parts) == 2:  # MM:SS
        hours = 0
        minutes = int(parts[0])
        seconds = int(parts[1])
        frames = 0
    else:
        raise ValueError(f"Invalid timecode format: {timecode_str}")

    # Calculate total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + (frames / fps)

    # Convert to frame number
    frame_number = int(total_seconds * fps)

    return frame_number


def draw_crop_overlay(frame, crop_region, alpha=0.3, color=(0, 255, 0), thickness=2):
    """
    Draw a semi-transparent overlay on the frame to indicate the crop region

    Args:
        frame: The frame to draw on (numpy array)
        crop_region: Tuple of (x, y, width, height) defining the crop region
        alpha: Transparency level (0-1)
        color: RGB color tuple for the overlay
        thickness: Line thickness for the border

    Returns:
        Frame with overlay
    """
    if frame is None or crop_region is None:
        return frame

    # Make a copy of the frame to avoid modifying the original
    overlay = frame.copy()

    # Extract crop region coordinates
    x, y, width, height = crop_region

    # Ensure coordinates are within frame boundaries
    h, w = frame.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = max(1, min(width, w - x))
    height = max(1, min(height, h - y))

    # Draw filled rectangle with transparency
    cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)

    # Blend the overlay with the original frame
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw border around crop region
    cv2.rectangle(result, (x, y), (x + width, y + height), color, thickness)

    return result


def apply_crop(frame, crop_region):
    """
    Apply crop to a frame

    Args:
        frame: The frame to crop (numpy array)
        crop_region: Tuple of (x, y, width, height) defining the crop region

    Returns:
        Cropped frame
    """
    if frame is None or crop_region is None:
        return frame

    # Extract crop region coordinates
    x, y, width, height = crop_region

    # Ensure coordinates are within frame boundaries
    h, w = frame.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = max(1, min(width, w - x))
    height = max(1, min(height, h - y))

    # Apply crop
    cropped_frame = frame[y : y + height, x : x + width]

    return cropped_frame


def resize_frame(frame, width=None, height=None, maintain_aspect_ratio=True):
    """
    Resize a frame to the specified dimensions

    Args:
        frame: The frame to resize (numpy array)
        width: Target width (if None, will be calculated from height)
        height: Target height (if None, will be calculated from width)
        maintain_aspect_ratio: Whether to maintain the aspect ratio

    Returns:
        Resized frame
    """
    if frame is None:
        return None

    # Get original dimensions
    h, w = frame.shape[:2]

    # If both width and height are None, return original frame
    if width is None and height is None:
        return frame

    # Calculate new dimensions
    if maintain_aspect_ratio:
        if width is not None and height is None:
            # Calculate height based on width
            height = int(h * (width / w))
        elif height is not None and width is None:
            # Calculate width based on height
            width = int(w * (height / h))
        elif width is not None and height is not None:
            # Use the smaller scale factor to ensure the entire frame fits
            scale_w = width / w
            scale_h = height / h
            scale = min(scale_w, scale_h)
            width = int(w * scale)
            height = int(h * scale)
    else:
        # Use provided dimensions or original if not provided
        width = width if width is not None else w
        height = height if height is not None else h

    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    return resized_frame


def calculate_crop_dimensions(output_resolution, aspect_ratio=None):
    """
    Calculate crop dimensions based on output resolution and aspect ratio

    Args:
        output_resolution: String like "1080p", "720p", etc.
        aspect_ratio: Target aspect ratio (width/height), if None uses the output resolution's aspect ratio

    Returns:
        Tuple of (width, height)
    """
    # Define standard resolutions
    resolutions = {
        "2160p": (3840, 2160),  # 4K UHD
        "1440p": (2560, 1440),  # 2K QHD
        "1080p": (1920, 1080),  # Full HD
        "720p": (1280, 720),  # HD
        "480p": (854, 480),  # SD
        "360p": (640, 360),  # Low
    }

    # Get dimensions for the selected resolution
    if output_resolution in resolutions:
        width, height = resolutions[output_resolution]
    else:
        # Default to 1080p if resolution not recognized
        width, height = resolutions["1080p"]

    # If aspect ratio is provided, adjust dimensions
    if aspect_ratio is not None:
        # Calculate new dimensions based on aspect ratio
        current_ratio = width / height

        if aspect_ratio > current_ratio:
            # Wider than standard, adjust height
            height = int(width / aspect_ratio)
        elif aspect_ratio < current_ratio:
            # Taller than standard, adjust width
            width = int(height * aspect_ratio)

    return (width, height)


def export_clip(
    source_path,
    output_path,
    start_frame,
    end_frame,
    crop_region=None,
    output_resolution="1080p",
    config_manager=None,
):
    """
    Export a clip from a video with optional cropping and resizing

    Args:
        source_path: Path to the source video
        output_path: Path to save the exported clip
        start_frame: Starting frame number (inclusive)
        end_frame: Ending frame number (inclusive)
        crop_region: Optional tuple of (x, y, width, height) for cropping
        output_resolution: Target resolution for the output
        config_manager: ConfigManager instance

    Returns:
        True if export was successful, False otherwise
    """
    try:
        # Check if we should use proxy video
        if "proxy_path" in st.session_state and st.session_state.proxy_path:
            # Use proxy video if available
            proxy_path = st.session_state.proxy_path
            logger.debug(f"Using proxy video for clip export: {proxy_path}")
            source_path = proxy_path

        logger.info(f"Exporting clip from {source_path} to {output_path}")
        logger.info(f"Frames: {start_frame} to {end_frame}")

        if crop_region:
            logger.info(f"Crop region: {crop_region}")

        # Open the source video
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            logger.error(f"Could not open source video: {source_path}")
            return False

        # Get video properties
        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate frame range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))

        # Calculate output dimensions
        if crop_region:
            # If cropping, use the crop dimensions
            x, y, crop_width, crop_height = crop_region
            # Ensure crop region is within frame boundaries
            x = max(0, min(x, src_width - 1))
            y = max(0, min(y, src_height - 1))
            crop_width = max(1, min(crop_width, src_width - x))
            crop_height = max(1, min(crop_height, src_height - y))

            # Calculate aspect ratio of the crop
            crop_aspect_ratio = crop_width / crop_height

            # Calculate output dimensions based on the target resolution and crop aspect ratio
            out_width, out_height = calculate_crop_dimensions(
                output_resolution, crop_aspect_ratio
            )
        else:
            # If not cropping, use the source aspect ratio
            src_aspect_ratio = src_width / src_height
            out_width, out_height = calculate_crop_dimensions(
                output_resolution, src_aspect_ratio
            )

        logger.info(f"Output dimensions: {out_width}x{out_height}")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v codec for MP4 files
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))

        if not out.isOpened():
            logger.error(f"Could not create output video: {output_path}")
            cap.release()
            return False

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Process frames
        frame_count = 0
        total_frames_to_process = end_frame - start_frame + 1

        for i in range(total_frames_to_process):
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"End of video reached at frame {start_frame + i}")
                break

            # Apply crop if specified
            if crop_region:
                frame = apply_crop(frame, crop_region)

            # Resize to output dimensions
            frame = resize_frame(
                frame, out_width, out_height, maintain_aspect_ratio=False
            )

            # Write the frame
            out.write(frame)
            frame_count += 1

            # Log progress periodically
            if frame_count % 100 == 0 or frame_count == total_frames_to_process:
                progress = frame_count / total_frames_to_process * 100
                logger.info(
                    f"Export progress: {progress:.1f}% ({frame_count}/{total_frames_to_process})"
                )

        # Release resources
        cap.release()
        out.release()

        # Verify the output file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Export completed: {output_path} ({file_size:.2f} MB)")
            return True
        else:
            logger.error(f"Export failed: Output file not created")
            return False

    except Exception as e:
        logger.exception(f"Error exporting clip: {str(e)}")
        return False


def frame_to_base64(frame):
    """
    Convert a frame (numpy array) to a base64 encoded string for HTML embedding

    Args:
        frame: The frame as a numpy array

    Returns:
        Base64 encoded string representation of the frame
    """
    try:
        # Ensure the frame is in BGR format (OpenCV default)
        if frame is None:
            logger.error("Cannot convert None frame to base64")
            return ""

        # Convert to RGB if needed (assuming BGR input)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # Encode the frame as JPEG
        success, buffer = cv2.imencode(".jpg", frame_rgb)
        if not success:
            logger.error("Failed to encode frame as JPEG")
            return ""

        # Convert to base64
        encoded = base64.b64encode(buffer).decode("utf-8")
        return encoded

    except Exception as e:
        logger.exception(f"Error converting frame to base64: {str(e)}")
        return ""
