import cv2
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime


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

    def load_video(self, video_path):
        """Load a video file and initialize video properties"""
        if not os.path.exists(video_path):
            return False, f"Video file not found: {video_path}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Error opening video file: {video_path}"

        self.cap = cap
        self.video_path = video_path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return True, "Video loaded successfully"

    def get_frame(self, frame_number):
        """Get a specific frame from the video"""
        if self.cap is None:
            return None

        # Ensure frame number is within valid range
        frame_number = max(0, min(frame_number, self.total_frames - 1))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert BGR to RGB for display in Streamlit
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def export_clip(self, clip_data, output_path, progress_callback=None):
        """
        Export a single clip based on the provided configuration

        Args:
            clip_data (dict): Clip configuration data
            output_path (str): Path to save the exported clip
            progress_callback (function): Callback function for progress updates

        Returns:
            bool, str: Success status and message
        """
        if self.cap is None:
            return False, "No video loaded"

        try:
            in_point = clip_data["in_point"]
            out_point = clip_data["out_point"]
            crop_start = clip_data["crop_start"]
            crop_end = clip_data.get("crop_end", crop_start)
            output_resolution = clip_data.get("output_resolution", "1080p")

            # Define resolution presets
            resolution_presets = {
                "1080p": (1920, 1080),
                "720p": (1280, 720),
                "480p": (854, 480),
                "360p": (640, 360),
                "240p": (426, 240),
            }

            output_res = resolution_presets.get(output_resolution, (1920, 1080))

            # Create a new video capture to avoid interfering with the UI
            cap = cv2.VideoCapture(self.video_path)

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, self.fps, output_res)

            # Process frames
            total_clip_frames = out_point - in_point
            cap.set(cv2.CAP_PROP_POS_FRAMES, in_point)

            for frame_idx in range(in_point, out_point):
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate interpolated crop region for current frame
                if crop_start != crop_end:
                    progress = (frame_idx - in_point) / max(1, total_clip_frames - 1)
                    current_crop = [
                        int(crop_start[0] + (crop_end[0] - crop_start[0]) * progress),
                        int(crop_start[1] + (crop_end[1] - crop_start[1]) * progress),
                        int(crop_start[2] + (crop_end[2] - crop_start[2]) * progress),
                        int(crop_start[3] + (crop_end[3] - crop_start[3]) * progress),
                    ]
                else:
                    current_crop = crop_start

                # Apply crop
                x, y, w, h = current_crop

                # Ensure crop region is within frame boundaries
                x = max(0, min(x, self.width - 10))
                y = max(0, min(y, self.height - 10))
                w = max(10, min(w, self.width - x))
                h = max(10, min(h, self.height - y))

                cropped_frame = frame[y : y + h, x : x + w]

                # Resize to output resolution
                resized_frame = cv2.resize(cropped_frame, output_res)

                # Write frame
                out.write(resized_frame)

                # Update progress
                if progress_callback:
                    progress_callback((frame_idx - in_point + 1) / total_clip_frames)

            # Release resources
            cap.release()
            out.release()

            return True, f"Clip exported successfully to {output_path}"

        except Exception as e:
            return False, f"Error exporting clip: {str(e)}"

    def export_clips(
        self,
        clips,
        output_directory,
        progress_callback=None,
        clip_progress_callback=None,
    ):
        """
        Export multiple clips based on the provided configurations

        Args:
            clips (list): List of clip configurations
            output_directory (str): Directory to save the exported clips
            progress_callback (function): Callback function for overall progress updates
            clip_progress_callback (function): Callback function for individual clip progress updates

        Returns:
            list: List of (success, message) tuples for each clip
        """
        if not clips:
            return [(False, "No clips to export")]

        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        results = []

        for i, clip in enumerate(clips):
            clip_name = clip["name"]

            # Create output filename
            video_filename = os.path.basename(self.video_path)
            base_name = os.path.splitext(video_filename)[0]
            output_path = os.path.join(output_directory, f"{base_name}_{clip_name}.mp4")

            # Update overall progress
            if progress_callback:
                progress_callback(i / len(clips))

            # Export clip
            success, message = self.export_clip(
                clip, output_path, progress_callback=clip_progress_callback
            )

            results.append((success, message))

        # Final progress update
        if progress_callback:
            progress_callback(1.0)

        return results

    def release(self):
        """Release video resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def save_config(config_path, video_path, clips):
    """Save clip configurations to a JSON file"""
    config_data = {
        "video_path": video_path,
        "clips": clips,
        "timestamp": datetime.now().isoformat(),
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    return True, f"Configuration saved to {config_path}"


def load_config(config_path):
    """Load clip configurations from a JSON file"""
    if not os.path.exists(config_path):
        return False, f"Config file not found: {config_path}", None, []

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        video_path = config_data.get("video_path")
        clips = config_data.get("clips", [])

        return True, "Configuration loaded successfully", video_path, clips
    except Exception as e:
        return False, f"Error loading configuration: {str(e)}", None, []
