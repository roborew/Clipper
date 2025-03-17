"""
Video processor service for the Clipper application.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger("clipper.processor")


class VideoProcessor:
    """Class for handling video processing operations"""

    def __init__(self):
        """Initialize the video processor"""
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        logger.info("VideoProcessor initialized")

    def load_video(self, video_path):
        """Load a video file and initialize video properties"""
        logger.info(f"Attempting to load video: {video_path}")
        logger.info(f"File exists: {os.path.exists(video_path)}")

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False, f"Video file not found: {video_path}"

        # Check file size and extension
        file_size = os.path.getsize(video_path)
        file_ext = os.path.splitext(video_path)[1].lower()
        logger.info(f"File size: {file_size/1024/1024:.2f} MB, Extension: {file_ext}")

        # Check if file is a valid video using ffprobe if available
        try:
            import subprocess

            # Create command list
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]

            # Convert command list to a string for shell=True
            # Ensure proper quoting for paths with spaces
            cmd_str = " ".join(
                (
                    f'"{arg}"'
                    if " " in str(arg) or "+" in str(arg) or ":" in str(arg)
                    else str(arg)
                )
                for arg in cmd
            )
            logger.info(f"Running shell command: {cmd_str}")

            result = subprocess.run(
                cmd_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
            )
            if result.stdout.strip():
                duration = float(result.stdout.strip())
                logger.info(f"Video duration (ffprobe): {duration:.2f} seconds")
            else:
                logger.warning(f"ffprobe couldn't determine duration: {result.stderr}")
        except Exception as e:
            logger.warning(f"ffprobe check failed: {str(e)}")

        try:
            logger.info("Creating VideoCapture object...")
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Error opening video file: {video_path}")
                return False, f"Error opening video file: {video_path}"

            # Get video properties
            logger.info("Reading video properties...")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

            logger.info(f"Video codec: {codec_str}")
            logger.info(
                f"Video properties: frames={total_frames}, fps={fps}, size={width}x{height}"
            )

            # Calculate duration
            duration = total_frames / fps if fps > 0 else 0
            logger.info(f"Video duration (calculated): {duration:.2f} seconds")

            # Test reading first frame
            logger.info("Testing frame reading...")
            ret, test_frame = cap.read()
            if not ret:
                logger.error("Failed to read first frame")
                cap.release()
                return False, "Failed to read video frames"

            logger.info("First frame read successfully")

            # Reset to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Update instance variables
            self.cap = cap
            self.video_path = video_path
            self.total_frames = total_frames
            self.fps = fps
            self.width = width
            self.height = height

            logger.info("Video loaded successfully")
            return True, "Video loaded successfully"

        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            logger.exception("Full traceback:")
            return False, f"Error loading video: {str(e)}"

    def get_frame(self, frame_number):
        """Get a specific frame from the video"""
        if self.cap is None:
            logger.error("No video loaded")
            return None

        try:
            # Ensure frame number is within valid range
            frame_number = max(0, min(frame_number, self.total_frames - 1))

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()

            if not ret:
                logger.error(f"Failed to read frame {frame_number}")
                return None

            # Convert BGR to RGB for display in Streamlit
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error getting frame {frame_number}: {str(e)}")
            return None

    def export_clip(
        self,
        output_path,
        start_frame,
        end_frame,
        crop_region=None,
        output_resolution="1080p",
        progress_callback=None,
    ):
        """
        Export a clip based on the provided parameters

        Args:
            output_path (str): Path to save the exported clip
            start_frame (int): Starting frame number
            end_frame (int): Ending frame number
            crop_region (callable or tuple): Function that returns crop region for a frame or static crop region
            output_resolution (str): Output resolution preset
            progress_callback (function): Callback function for progress updates

        Returns:
            bool, str: Success status and message
        """
        if self.cap is None:
            return False, "No video loaded"

        try:
            # Define resolution presets
            resolution_presets = {
                "2160p": (3840, 2160),  # 4K UHD
                "1440p": (2560, 1440),  # 2K QHD
                "1080p": (1920, 1080),  # Full HD
                "720p": (1280, 720),  # HD
                "480p": (854, 480),  # SD
                "360p": (640, 360),  # Low
            }

            output_res = resolution_presets.get(output_resolution, (1920, 1080))
            logger.info(
                f"Exporting clip to {output_path} with resolution {output_resolution} ({output_res[0]}x{output_res[1]})"
            )
            logger.info(f"Frame range: {start_frame} to {end_frame}")

            # Create a new video capture to avoid interfering with the UI
            cap = cv2.VideoCapture(self.video_path)

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, self.fps, output_res)

            # Process frames
            total_clip_frames = end_frame - start_frame + 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_idx}, stopping export")
                    break

                # Get crop region for current frame
                current_crop = None
                if crop_region:
                    if callable(crop_region):
                        # If crop_region is a function, call it with the current frame
                        current_crop = crop_region(frame_idx)
                    else:
                        # Otherwise, use it directly
                        current_crop = crop_region

                # Apply crop if specified
                if current_crop:
                    x, y, w, h = current_crop

                    # Ensure crop region is within frame boundaries
                    x = max(0, min(x, self.width - 10))
                    y = max(0, min(y, self.height - 10))
                    w = max(10, min(w, self.width - x))
                    h = max(10, min(h, self.height - y))

                    cropped_frame = frame[y : y + h, x : x + w]
                else:
                    cropped_frame = frame

                # Resize to output resolution
                resized_frame = cv2.resize(cropped_frame, output_res)

                # Write frame
                out.write(resized_frame)

                # Update progress
                if progress_callback and total_clip_frames > 0:
                    progress = (frame_idx - start_frame + 1) / total_clip_frames
                    progress_callback(progress)

                # Log progress periodically
                if (frame_idx - start_frame) % 100 == 0 or frame_idx == end_frame:
                    progress = (frame_idx - start_frame + 1) / total_clip_frames * 100
                    logger.info(
                        f"Export progress: {progress:.1f}% ({frame_idx - start_frame + 1}/{total_clip_frames})"
                    )

            # Release resources
            cap.release()
            out.release()

            # Verify the output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                logger.info(
                    f"Clip exported successfully to {output_path} ({file_size:.2f} MB)"
                )
                return True, f"Clip exported successfully to {output_path}"
            else:
                logger.error("Export failed: Output file not created")
                return False, "Export failed: Output file not created"

        except Exception as e:
            logger.exception(f"Error exporting clip: {str(e)}")
            return False, f"Error exporting clip: {str(e)}"

    def close(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video capture resources released")
