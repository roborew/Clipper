import streamlit as st
import os
import time
from pathlib import Path
from video_processor import VideoProcessor, save_config, load_config
from config_manager import ConfigManager
import logging
import base64
import cv2
import yaml
import re
import subprocess
import threading
from queue import Queue
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Clipper - Video Clipping Tool", page_icon="🎬", layout="wide"
)

# Initialize configuration manager
if "config_manager" not in st.session_state:
    st.session_state.config_manager = ConfigManager()

# Initialize proxy generation state
if "proxy_generation_active" not in st.session_state:
    st.session_state.proxy_generation_active = False
if "proxy_current_video" not in st.session_state:
    st.session_state.proxy_current_video = None
if "proxy_current_index" not in st.session_state:
    st.session_state.proxy_current_index = 0
if "proxy_total_videos" not in st.session_state:
    st.session_state.proxy_total_videos = 0
if "proxy_videos_to_process" not in st.session_state:
    st.session_state.proxy_videos_to_process = []
if "proxy_completed_videos" not in st.session_state:
    st.session_state.proxy_completed_videos = []
if "proxy_failed_videos" not in st.session_state:
    st.session_state.proxy_failed_videos = []


# Add function for interactive crop region selection
def select_crop_region_opencv(frame, output_resolution):
    """
    Select a crop region with fixed dimensions based on the selected output resolution
    using a simplified button-based approach.

    Args:
        frame: The current video frame
        output_resolution: The selected output resolution (e.g., "1080p")

    Returns:
        [x, y, w, h]: Coordinates of the selected crop region
    """
    # Get the dimensions for the selected resolution
    target_width, target_height = RESOLUTION_PRESETS[output_resolution]

    # Calculate aspect ratio
    aspect_ratio = target_width / target_height

    # Make a copy of the frame to avoid modifying the original
    img = frame.copy()
    frame_height, frame_width = img.shape[:2]

    # Calculate default crop region (centered)
    default_width = min(frame_width, int(frame_height * aspect_ratio))
    default_height = int(default_width / aspect_ratio)
    default_x = (frame_width - default_width) // 2
    default_y = (frame_height - default_height) // 2

    # Create a session state for the crop position if it doesn't exist
    if "temp_crop_x" not in st.session_state:
        st.session_state.temp_crop_x = default_x
    if "temp_crop_y" not in st.session_state:
        st.session_state.temp_crop_y = default_y
    if "temp_crop_width" not in st.session_state:
        st.session_state.temp_crop_width = default_width
    if "temp_crop_height" not in st.session_state:
        st.session_state.temp_crop_height = default_height

    # Create a container for the crop selection UI
    st.subheader(f"Select Crop Region - Target: {target_width}x{target_height}")

    # Draw the selection on the frame
    preview_frame = img.copy()
    cv2.rectangle(
        preview_frame,
        (st.session_state.temp_crop_x, st.session_state.temp_crop_y),
        (
            st.session_state.temp_crop_x + st.session_state.temp_crop_width,
            st.session_state.temp_crop_y + st.session_state.temp_crop_height,
        ),
        (0, 255, 0),
        2,
    )

    # Display the preview
    st.image(
        preview_frame,
        caption=f"Preview of selected crop region: x={st.session_state.temp_crop_x}, y={st.session_state.temp_crop_y}, width={st.session_state.temp_crop_width}, height={st.session_state.temp_crop_height}",
        use_container_width=True,
    )

    # Movement step size
    step = 10

    # Create a simple grid of buttons for movement
    st.write("Position Controls:")

    # Row 1: Up
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("⬆️ Move Up"):
            st.session_state.temp_crop_y = max(0, st.session_state.temp_crop_y - step)
            st.rerun()

    # Row 2: Left, Center, Right
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Move Left"):
            st.session_state.temp_crop_x = max(0, st.session_state.temp_crop_x - step)
            st.rerun()
    with col2:
        if st.button("Center"):
            st.session_state.temp_crop_x = default_x
            st.session_state.temp_crop_y = default_y
            st.rerun()
    with col3:
        if st.button("➡️ Move Right"):
            max_x = frame_width - st.session_state.temp_crop_width
            st.session_state.temp_crop_x = min(
                max_x, st.session_state.temp_crop_x + step
            )
            st.rerun()

    # Row 3: Down
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("⬇️ Move Down"):
            max_y = frame_height - st.session_state.temp_crop_height
            st.session_state.temp_crop_y = min(
                max_y, st.session_state.temp_crop_y + step
            )
            st.rerun()

    # Size controls
    st.write("Size Controls:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➖ Smaller"):
            # Decrease size while maintaining aspect ratio
            new_width = max(100, st.session_state.temp_crop_width - 50)
            new_height = int(new_width / aspect_ratio)

            # Make sure it fits within the frame
            if new_height <= frame_height:
                # Adjust position to keep centered
                x_diff = (st.session_state.temp_crop_width - new_width) // 2
                y_diff = (st.session_state.temp_crop_height - new_height) // 2

                st.session_state.temp_crop_width = new_width
                st.session_state.temp_crop_height = new_height
                st.session_state.temp_crop_x += x_diff
                st.session_state.temp_crop_y += y_diff

                # Ensure within bounds
                st.session_state.temp_crop_x = min(
                    max(0, st.session_state.temp_crop_x), frame_width - new_width
                )
                st.session_state.temp_crop_y = min(
                    max(0, st.session_state.temp_crop_y), frame_height - new_height
                )
            st.rerun()
    with col2:
        if st.button("➕ Larger"):
            # Increase size while maintaining aspect ratio
            new_width = min(frame_width, st.session_state.temp_crop_width + 50)
            new_height = int(new_width / aspect_ratio)

            # Make sure it fits within the frame
            if new_height <= frame_height:
                # Adjust position to keep centered
                x_diff = (new_width - st.session_state.temp_crop_width) // 2
                y_diff = (new_height - st.session_state.temp_crop_height) // 2

                new_x = st.session_state.temp_crop_x - x_diff
                new_y = st.session_state.temp_crop_y - y_diff

                # Ensure within bounds
                if (
                    new_x >= 0
                    and new_y >= 0
                    and new_x + new_width <= frame_width
                    and new_y + new_height <= frame_height
                ):
                    st.session_state.temp_crop_width = new_width
                    st.session_state.temp_crop_height = new_height
                    st.session_state.temp_crop_x = new_x
                    st.session_state.temp_crop_y = new_y
            st.rerun()

    # Reset button
    if st.button("Reset Size"):
        # Reset to default size based on the selected output resolution
        target_width, target_height = RESOLUTION_PRESETS[
            st.session_state.output_resolution
        ]
        aspect_ratio = target_width / target_height

        # If the frame is wider than the target aspect ratio, constrain by height
        # Otherwise, constrain by width
        if (frame_width / frame_height) > aspect_ratio:
            # Frame is wider than target aspect ratio, so constrain by height
            default_height = min(frame_height, target_height)
            default_width = int(default_height * aspect_ratio)
        else:
            # Frame is taller than target aspect ratio, so constrain by width
            default_width = min(frame_width, target_width)
            default_height = int(default_width / aspect_ratio)

        # Keep the current position but reset the size
        x_diff = (default_width - st.session_state.temp_crop_width) // 2
        y_diff = (default_height - st.session_state.temp_crop_height) // 2

        new_x = st.session_state.temp_crop_x - x_diff
        new_y = st.session_state.temp_crop_y - y_diff

        # Ensure within bounds
        if (
            new_x >= 0
            and new_y >= 0
            and new_x + default_width <= frame_width
            and new_y + default_height <= frame_height
        ):
            st.session_state.temp_crop_width = default_width
            st.session_state.temp_crop_height = default_height
            st.session_state.temp_crop_x = new_x
            st.session_state.temp_crop_y = new_y
        else:
            # If out of bounds, center it
            st.session_state.temp_crop_width = default_width
            st.session_state.temp_crop_height = default_height
            st.session_state.temp_crop_x = (frame_width - default_width) // 2
            st.session_state.temp_crop_y = (frame_height - default_height) // 2

        st.rerun()

    # Confirm or cancel buttons
    col1, col2 = st.columns(2)
    with col1:
        confirm = st.button("Confirm Selection")
    with col2:
        cancel = st.button("Cancel")

    # Process the user's choice
    if confirm:
        crop_coords = [
            st.session_state.temp_crop_x,
            st.session_state.temp_crop_y,
            st.session_state.temp_crop_width,
            st.session_state.temp_crop_height,
        ]
        # Clean up session state
        if "temp_crop_x" in st.session_state:
            del st.session_state.temp_crop_x
        if "temp_crop_y" in st.session_state:
            del st.session_state.temp_crop_y
        if "temp_crop_width" in st.session_state:
            del st.session_state.temp_crop_width
        if "temp_crop_height" in st.session_state:
            del st.session_state.temp_crop_height
        return crop_coords
    elif cancel:
        # Clean up session state
        if "temp_crop_x" in st.session_state:
            del st.session_state.temp_crop_x
        if "temp_crop_y" in st.session_state:
            del st.session_state.temp_crop_y
        if "temp_crop_width" in st.session_state:
            del st.session_state.temp_crop_width
        if "temp_crop_height" in st.session_state:
            del st.session_state.temp_crop_height
        return None

    # If we get here, the user hasn't made a choice yet
    return None


# Function to draw crop overlay on frame
def draw_crop_overlay(frame, crop_coords, output_resolution):
    """
    Draw a crop region overlay on the frame

    Args:
        frame: The video frame
        crop_coords: [x, y, w, h] coordinates of the crop region
        output_resolution: The selected output resolution

    Returns:
        Frame with crop overlay
    """
    if crop_coords is None:
        return frame

    # Make a copy of the frame
    overlay = frame.copy()

    # Extract coordinates
    x, y, w, h = crop_coords

    # Draw rectangle
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add text with resolution
    cv2.putText(
        overlay,
        f"Crop: {output_resolution} ({w}x{h})",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return overlay


# Add function to check if a proxy exists for a video
def proxy_exists_for_video(video_path):
    """Check if a proxy video exists for the given video path"""
    proxy_path = st.session_state.config_manager.get_proxy_path(Path(video_path))
    # Check if the proxy file exists
    exists = proxy_path.exists()
    if exists:
        logger.debug(f"Proxy exists for {video_path} at {proxy_path}")
    else:
        logger.debug(f"No proxy found for {video_path}, would be at {proxy_path}")
    return exists


# Function to display proxy generation progress
def display_proxy_generation_progress():
    """Display the progress of proxy generation if active"""
    if st.session_state.proxy_generation_active:
        st.subheader("Generating Proxy Videos")

        # Show current video being processed
        if st.session_state.proxy_current_video:
            current_video_name = os.path.basename(st.session_state.proxy_current_video)
            st.info(
                f"Processing video {st.session_state.proxy_current_index + 1} of {st.session_state.proxy_total_videos}: {current_video_name}"
            )

        # Progress placeholder for the current video
        st.session_state.proxy_progress_placeholder = st.empty()

        # Individual video progress bar - this will be updated by create_proxy_video
        st.session_state.proxy_progress_bar = st.progress(0)

        # Overall progress
        st.text("Overall Progress:")
        st.progress(
            (st.session_state.proxy_current_index)
            / max(1, st.session_state.proxy_total_videos)
        )

        # Completed videos
        if st.session_state.proxy_completed_videos:
            with st.expander(
                f"Completed Videos ({len(st.session_state.proxy_completed_videos)})"
            ):
                for video in st.session_state.proxy_completed_videos:
                    st.success(f"✅ {os.path.basename(video)}")

        # Failed videos
        if st.session_state.proxy_failed_videos:
            with st.expander(
                f"Failed Videos ({len(st.session_state.proxy_failed_videos)})"
            ):
                for video in st.session_state.proxy_failed_videos:
                    st.error(f"❌ {os.path.basename(video)}")

        # Cancel button
        if st.button("Cancel Proxy Generation"):
            st.session_state.proxy_generation_active = False
            st.rerun()

        return True
    return False


# Add function to generate proxies for all videos
def generate_all_proxies():
    """Generate proxy videos for all videos that don't have one yet"""
    try:
        # Check if proxy generation is already active
        if st.session_state.proxy_generation_active:
            st.warning("Proxy generation is already in progress")
            return

        # Log that the function was called
        logger.info("generate_all_proxies function called")

        video_files = st.session_state.config_manager.get_video_files()
        if not video_files:
            st.warning("No video files found to generate proxies for")
            logger.warning("No video files found to generate proxies for")
            return

        # Count videos that need proxies
        videos_without_proxies = [
            v for v in video_files if not proxy_exists_for_video(v)
        ]
        total_videos = len(videos_without_proxies)

        logger.info(
            f"Found {total_videos} videos without proxies out of {len(video_files)} total videos"
        )

        if total_videos == 0:
            st.success("All videos already have proxy versions")
            logger.info("All videos already have proxy versions")
            return

        # Set up proxy generation state
        st.session_state.proxy_generation_active = True
        st.session_state.proxy_videos_to_process = videos_without_proxies
        st.session_state.proxy_current_index = 0
        st.session_state.proxy_total_videos = total_videos
        st.session_state.proxy_completed_videos = []
        st.session_state.proxy_failed_videos = []

        # Start processing the first video
        if videos_without_proxies:
            st.session_state.proxy_current_video = videos_without_proxies[0]

        # Force a rerun to show the progress UI
        st.rerun()

    except Exception as e:
        logger.exception(f"Error in proxy generation: {str(e)}")
        st.error(f"Error in proxy generation: {str(e)}")


# Custom progress callback for proxy generation
def proxy_progress_callback(progress_info):
    """Update progress information for the current video being processed"""
    try:
        progress = progress_info.get("progress", 0.0)
        remaining = progress_info.get("remaining", 0)

        # Update session state with progress info
        st.session_state.proxy_current_progress = progress

        # Format the time remaining
        if remaining > 0:
            minutes = int(remaining / 60)
            seconds = int(remaining % 60)
            st.session_state.proxy_time_remaining = f"{minutes} min {seconds} sec"
        else:
            st.session_state.proxy_time_remaining = "Calculating..."
    except Exception as e:
        if "ScriptRunContext" not in str(e):
            logger.error(f"Error in proxy progress callback: {str(e)}")


# Initialize session state variables
if "video_processor" not in st.session_state:
    st.session_state.video_processor = VideoProcessor()
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "clips" not in st.session_state:
    st.session_state.clips = []
if "current_clip_index" not in st.session_state:
    st.session_state.current_clip_index = -1
if "config_file" not in st.session_state:
    st.session_state.config_file = None
if "crop_start" not in st.session_state:
    st.session_state.crop_start = None
if "crop_end" not in st.session_state:
    st.session_state.crop_end = None
if "crop_keyframes" not in st.session_state:
    st.session_state.crop_keyframes = {}  # Dictionary of frame -> crop coordinates
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "output_resolution" not in st.session_state:
    st.session_state.output_resolution = "1080p"

# Resolution presets
RESOLUTION_PRESETS = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
    "240p": (426, 240),
}


def create_proxy_video(source_path, progress_placeholder=None, progress_callback=None):
    """Create a proxy (web-compatible) version of the video for faster playback"""
    try:
        # Get proxy settings from config manager
        config_manager = st.session_state.config_manager
        proxy_settings = config_manager.get_proxy_settings()

        # Check if proxy creation is enabled
        if not proxy_settings["enabled"]:
            logger.info("Proxy video creation is disabled in config")
            return None

        # Get proxy path from config manager
        proxy_path = config_manager.get_proxy_path(Path(source_path))

        # Check if proxy already exists
        if proxy_path.exists():
            logger.info(f"Proxy already exists: {proxy_path}")

            # If this is part of batch processing, mark as completed and move to next video
            if (
                st.session_state.proxy_generation_active
                and st.session_state.proxy_current_video == source_path
            ):
                st.session_state.proxy_completed_videos.append(source_path)
                move_to_next_proxy_video()

            return str(proxy_path)

        # Ensure the parent directory exists
        proxy_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring proxy directory exists: {proxy_path.parent}")

        # Use the provided progress placeholder or create one if in batch mode
        actual_progress_placeholder = progress_placeholder
        if st.session_state.proxy_generation_active and not progress_placeholder:
            actual_progress_placeholder = st.session_state.proxy_progress_placeholder
            progress_bar = st.session_state.proxy_progress_bar
        else:
            # Show progress
            if actual_progress_placeholder:
                actual_progress_placeholder.text(
                    "Creating proxy video for faster playback..."
                )
                progress_bar = st.progress(0)

        # Use ffmpeg to create a proxy version
        import subprocess
        import threading
        import re
        import time
        from queue import Queue

        # Get video duration for progress calculation
        duration_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ]

        try:
            duration_result = subprocess.run(
                duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            duration = float(duration_result.stdout.strip())
            logger.info(f"Video duration: {duration:.2f} seconds")
        except Exception as e:
            logger.warning(f"Could not determine video duration: {str(e)}")
            duration = 0

        # Command to create a lower-resolution, web-compatible proxy using settings from config
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(proxy_settings["quality"]),
            "-vf",
            f"scale={proxy_settings['width']}:-2",  # Use width from config
            "-c:a",
            "aac",
            "-b:a",
            proxy_settings["audio_bitrate"],
            str(proxy_path),
        ]

        # Create a queue to communicate between threads
        progress_queue = Queue()
        stop_event = threading.Event()

        # Function to monitor progress
        def monitor_progress(process, duration, queue, stop_event):
            pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

            while process.poll() is None and not stop_event.is_set():
                try:
                    # Read line from stderr
                    line = process.stderr.readline()

                    # Convert to string if it's bytes
                    if isinstance(line, bytes):
                        output = line.decode("utf-8", errors="replace")
                    else:
                        output = line

                    # Check for progress information
                    match = pattern.search(output)
                    if match and duration > 0:
                        h, m, s = map(float, match.groups())
                        current_time = h * 3600 + m * 60 + s
                        progress = min(current_time / duration, 1.0)

                        # Calculate remaining time
                        if current_time > 0:
                            remaining = (
                                (duration - current_time) / current_time * current_time
                            )
                        else:
                            remaining = 0

                        # Put progress info in queue
                        progress_info = {
                            "progress": progress,
                            "remaining": remaining,
                            "current_time": current_time,
                        }
                        queue.put(progress_info)

                        # Call the progress callback if provided
                        if progress_callback:
                            progress_callback(progress_info)
                except Exception as e:
                    # Don't log Streamlit context errors as they're expected in threads
                    if "ScriptRunContext" not in str(e):
                        logger.error(f"Error in progress monitoring: {str(e)}")
                    continue

            # Signal completion
            final_progress = {"progress": 1.0, "remaining": 0, "current_time": duration}
            queue.put(final_progress)
            if progress_callback:
                progress_callback(final_progress)
            logger.info("Monitor thread completed")

        # Start the conversion process with pipe for stderr
        logger.info(f"Starting ffmpeg conversion: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        # Monitor progress in a separate thread
        if duration > 0:
            monitor_thread = threading.Thread(
                target=monitor_progress,
                args=(process, duration, progress_queue, stop_event),
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            # Update progress from main thread
            last_update_time = time.time()
            try:
                while process.poll() is None:
                    # Check queue for progress updates (non-blocking)
                    try:
                        progress_info = progress_queue.get(block=False)
                        if actual_progress_placeholder:
                            try:
                                progress_bar.progress(progress_info["progress"])
                                remaining = progress_info["remaining"]
                                actual_progress_placeholder.text(
                                    f"Creating proxy video: {int(progress_info['progress'] * 100)}% complete "
                                    f"(approx. {int(remaining/60)} min {int(remaining%60)} sec remaining)"
                                )
                            except Exception:
                                # Ignore Streamlit context errors
                                pass
                        progress_queue.task_done()
                    except Exception:
                        # No progress update available, sleep briefly
                        pass

                    # Sleep to avoid consuming too much CPU
                    time.sleep(0.1)

                    # Periodically update UI even if no progress info
                    current_time = time.time()
                    if current_time - last_update_time > 2.0:
                        if actual_progress_placeholder:
                            try:
                                actual_progress_placeholder.text(
                                    f"Creating proxy video... (ffmpeg is running)"
                                )
                            except Exception:
                                # Ignore Streamlit context errors
                                pass
                        last_update_time = current_time
            finally:
                # Signal monitor thread to stop
                stop_event.set()

                # Wait for process to complete
                logger.info("Waiting for ffmpeg process to complete...")
                process.wait()
                logger.info("ffmpeg process completed")

                # Final progress update
                if actual_progress_placeholder:
                    try:
                        progress_bar.progress(1.0)
                        actual_progress_placeholder.text(
                            "Proxy video created successfully!"
                        )
                    except Exception:
                        # Ignore Streamlit context errors
                        pass
        else:
            # If duration couldn't be determined, just wait for completion
            logger.info(
                "No duration available, waiting for process without progress updates"
            )
            process.wait()
            if actual_progress_placeholder:
                progress_bar.progress(1.0)
                actual_progress_placeholder.text("Proxy video created successfully!")

        if os.path.exists(proxy_path):
            proxy_size = os.path.getsize(proxy_path) / (1024 * 1024)
            logger.info(
                f"Proxy created successfully: {proxy_path} ({proxy_size:.2f} MB)"
            )

            # If this is part of batch processing, mark as completed and move to next video
            if (
                st.session_state.proxy_generation_active
                and st.session_state.proxy_current_video == source_path
            ):
                st.session_state.proxy_completed_videos.append(source_path)
                move_to_next_proxy_video()

            return str(proxy_path)
        else:
            logger.error("Failed to create proxy video")
            if actual_progress_placeholder:
                actual_progress_placeholder.error("Failed to create proxy video")

            # If this is part of batch processing, mark as failed and move to next video
            if (
                st.session_state.proxy_generation_active
                and st.session_state.proxy_current_video == source_path
            ):
                st.session_state.proxy_failed_videos.append(source_path)
                move_to_next_proxy_video()

            return None

    except Exception as e:
        logger.exception(f"Error creating proxy: {str(e)}")
        if progress_placeholder:
            progress_placeholder.error(f"Error creating proxy: {str(e)}")

        # If this is part of batch processing, mark as failed and move to next video
        if (
            st.session_state.proxy_generation_active
            and st.session_state.proxy_current_video == source_path
        ):
            st.session_state.proxy_failed_videos.append(source_path)
            move_to_next_proxy_video()

        return None


# Function to move to the next video in the proxy generation queue
def move_to_next_proxy_video():
    """Move to the next video in the proxy generation queue"""
    st.session_state.proxy_current_index += 1
    if st.session_state.proxy_current_index < len(
        st.session_state.proxy_videos_to_process
    ):
        st.session_state.proxy_current_video = st.session_state.proxy_videos_to_process[
            st.session_state.proxy_current_index
        ]
    else:
        st.session_state.proxy_generation_active = False
    st.rerun()


def load_video_file(video_path):
    """Load a video file and initialize video properties"""
    if video_path:
        logger.info(f"Starting video load process for: {video_path}")

        # Clean up any existing temporary video symlink
        temp_video_path = Path("temp_video.mp4")
        if temp_video_path.exists():
            try:
                os.unlink(temp_video_path)
                logger.info("Cleaned up existing temporary video symlink")
            except Exception as e:
                error_msg = f"Error cleaning up temporary video: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)

        # Check video format
        video_ext = Path(video_path).suffix.lower()
        if video_ext not in [".mp4", ".webm", ".ogg"]:
            warning_msg = f"Video format {video_ext} might not play in the browser. For best results, use MP4 format."
            logger.warning(warning_msg)
            st.warning(warning_msg)

        # Create a proxy video for faster playback
        progress_placeholder = st.empty()
        proxy_path = create_proxy_video(video_path, progress_placeholder)

        # Load video using VideoProcessor - use original for processing but proxy for display
        logger.info("Attempting to load video with VideoProcessor")
        success, message = st.session_state.video_processor.load_video(video_path)

        if success:
            logger.info("Video loaded successfully")
            st.session_state.video_path = video_path
            st.session_state.proxy_path = (
                proxy_path  # Store proxy path in session state
            )
            st.session_state.current_frame = 0

            # Generate config file path using config manager
            try:
                st.session_state.config_file = str(
                    st.session_state.config_manager.get_config_path(Path(video_path))
                )
                logger.info(
                    f"Generated config file path: {st.session_state.config_file}"
                )
            except Exception as e:
                logger.error(f"Error generating config file path: {str(e)}")
                st.error(f"Error generating config file path: {str(e)}")
                return False

            # Try to load existing config if it exists
            load_config_file(st.session_state.config_file)

            # Clear the progress placeholder
            progress_placeholder.empty()

            return True
        else:
            error_msg = f"Failed to load video: {message}"
            logger.error(error_msg)
            st.error(error_msg)
            return False
    return False


def load_config_file(config_path):
    """Load clip configurations from a JSON file"""
    success, message, video_path, clips = load_config(config_path)

    if success:
        # Only load clips if the video path matches
        if video_path == st.session_state.video_path:
            st.session_state.clips = clips
            st.success(f"Loaded {len(clips)} clips from configuration")
        else:
            st.warning(
                "Config file exists but is for a different video. Starting with empty clips."
            )
            st.session_state.clips = []
    else:
        st.error(message)
        st.session_state.clips = []


def save_config_file(config_path):
    """Save clip configurations to a JSON file"""
    if st.session_state.video_path:
        success, message = save_config(
            config_path, st.session_state.video_path, st.session_state.clips
        )
        if success:
            st.success(message)
        else:
            st.error(message)
    else:
        st.error("No video loaded")


def add_or_update_clip():
    """Add a new clip or update an existing one"""
    if st.session_state.in_point is None or st.session_state.out_point is None:
        st.error("Please set both in and out points")
        return

    if st.session_state.crop_start is None:
        st.error("Please set crop region")
        return

    # Filter keyframes to only include those within the clip range
    clip_keyframes = {}
    for frame, coords in st.session_state.crop_keyframes.items():
        if st.session_state.in_point <= frame <= st.session_state.out_point:
            clip_keyframes[frame] = coords

    clip_data = {
        "name": st.session_state.clip_name,
        "in_point": st.session_state.in_point,
        "out_point": st.session_state.out_point,
        "crop_start": st.session_state.crop_start,
        "crop_end": st.session_state.crop_end
        or st.session_state.crop_start,  # Use crop_start if crop_end not set
        "crop_keyframes": clip_keyframes,  # Add keyframes to clip data
        "output_resolution": st.session_state.output_resolution,
    }

    if (
        st.session_state.current_clip_index >= 0
        and st.session_state.current_clip_index < len(st.session_state.clips)
    ):
        # Update existing clip
        st.session_state.clips[st.session_state.current_clip_index] = clip_data
        st.success(f"Updated clip: {clip_data['name']}")
    else:
        # Add new clip
        st.session_state.clips.append(clip_data)
        st.success(f"Added new clip: {clip_data['name']}")

    # Save configuration
    save_config_file(st.session_state.config_file)


def delete_clip(index):
    """Delete a clip from the list"""
    if 0 <= index < len(st.session_state.clips):
        deleted_clip = st.session_state.clips.pop(index)
        st.success(f"Deleted clip: {deleted_clip['name']}")
        save_config_file(st.session_state.config_file)

        # Reset current clip index if needed
        if st.session_state.current_clip_index >= len(st.session_state.clips):
            st.session_state.current_clip_index = -1


def export_clips(output_directory):
    """Export all clips based on the configuration"""
    if not st.session_state.clips:
        st.error("No clips to export")
        return

    if not st.session_state.video_path:
        st.error("No video loaded")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    def overall_progress_callback(progress):
        progress_bar.progress(progress)

    def clip_progress_callback(progress):
        status_text.text(f"Processing clip... {int(progress * 100)}%")

    status_text.text("Starting export...")

    results = st.session_state.video_processor.export_clips(
        st.session_state.clips,
        output_directory,
        progress_callback=overall_progress_callback,
        clip_progress_callback=clip_progress_callback,
    )

    # Display results
    for success, message in results:
        if success:
            st.success(message)
        else:
            st.error(message)

    status_text.text("Export complete!")
    progress_bar.progress(1.0)


# Add cleanup function
def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        # Get proxy directory from config manager
        proxy_dir = st.session_state.config_manager.proxy_dir

        # Clean up proxy_videos directory
        if proxy_dir.exists() and st.checkbox("Clean up proxy videos?", value=False):
            # Recursively find and delete all files in proxy directory
            for file in proxy_dir.glob("**/*"):
                if file.is_file():
                    try:
                        file.unlink()
                        logger.info(f"Deleted proxy file: {file}")
                    except Exception as e:
                        logger.error(f"Error deleting proxy file {file}: {str(e)}")

            # Remove empty directories (except the root proxy directory)
            for dir_path in sorted(
                [p for p in proxy_dir.glob("**/*") if p.is_dir()], reverse=True
            ):
                try:
                    if dir_path != proxy_dir and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        logger.info(f"Removed empty proxy directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error removing proxy directory {dir_path}: {str(e)}")

        # Clean up preview_frames directory
        preview_dir = Path("preview_frames")
        if preview_dir.exists():
            for file in preview_dir.glob("*"):
                try:
                    file.unlink()
                    logger.info(f"Deleted preview frame: {file}")
                except Exception as e:
                    logger.error(f"Error deleting preview frame {file}: {str(e)}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")

    # Reset proxy path in session state if it was deleted
    if hasattr(st.session_state, "proxy_path") and st.session_state.proxy_path:
        if not os.path.exists(st.session_state.proxy_path):
            st.session_state.proxy_path = None


# Main application UI
st.title("Clipper - Video Clipping Tool")

# Check if proxy generation is active and display progress
proxy_ui_shown = display_proxy_generation_progress()

# Only show the rest of the UI if proxy generation isn't active or we've already shown the proxy UI
if not proxy_ui_shown:
    # Sidebar for controls
    with st.sidebar:
        st.header("Video Selection")

        # Video file selection from calibrated footage directory
        video_files = st.session_state.config_manager.get_video_files()
        if video_files:
            # Convert paths to strings for display, keeping them relative to calibrated directory
            display_paths = [
                str(st.session_state.config_manager.get_relative_source_path(f))
                for f in video_files
            ]

            # Check which videos have proxies
            has_proxy = [proxy_exists_for_video(v) for v in video_files]

            # Create formatted display options with proxy indicators
            display_options = [
                f"{'✅ ' if has_proxy[i] else '⬜ '}{display_paths[i]}"
                for i in range(len(display_paths))
            ]

            # Display a legend for the indicators
            st.caption("✅ = Proxy available | ⬜ = No proxy")

            selected_index = st.selectbox(
                "Select Video",
                range(len(display_options)),
                format_func=lambda x: display_options[x],
            )

            if st.button("Load Video"):
                load_video_file(str(video_files[selected_index]))
        else:
            st.warning(
                f"No video files found in {st.session_state.config_manager.source_calibrated}"
            )

        st.divider()

        # Display proxy generation progress in the sidebar
        if st.session_state.proxy_generation_active:
            display_proxy_generation_progress()

        # Configuration file controls
        st.header("Configuration")
        if st.session_state.config_file:
            config_display = Path(st.session_state.config_file).relative_to(
                st.session_state.config_manager.configs_dir
            )
        else:
            config_display = ""
        config_path = st.text_input("Config File Path", value=str(config_display))

        if config_path:
            full_config_path = st.session_state.config_manager.configs_dir / config_path
        else:
            full_config_path = st.session_state.config_file or ""

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Config"):
                if os.path.exists(full_config_path):
                    st.session_state.config_file = str(full_config_path)
                    load_config_file(full_config_path)
                else:
                    st.error("Config file does not exist")

        with col2:
            if st.button("Save Config"):
                save_config_file(full_config_path)

        st.divider()

        # Proxy settings
        st.header("Proxy Settings")
        proxy_settings = st.session_state.config_manager.get_proxy_settings()

        # Display current proxy directory
        st.info(f"Proxy directory: {st.session_state.config_manager.proxy_dir}")

        # Add info about directory structure
        if st.session_state.config_manager.config["export"]["preserve_structure"]:
            st.info(
                "Proxies will be stored in a directory structure that mirrors the source videos (camera/session folders)."
            )
        else:
            st.info("Proxies will be stored in a flat directory structure.")

        # Toggle for proxy creation
        proxy_enabled = st.checkbox(
            "Enable proxy videos", value=proxy_settings["enabled"]
        )

        # Only show these settings if proxy is enabled
        if proxy_enabled:
            proxy_width = st.number_input(
                "Proxy width",
                min_value=320,
                max_value=1920,
                value=proxy_settings["width"],
                help="Width of proxy videos (height will be calculated to maintain aspect ratio)",
            )

            proxy_quality = st.slider(
                "Proxy quality",
                min_value=18,
                max_value=35,
                value=proxy_settings["quality"],
                help="CRF value (18=high quality/larger file, 35=low quality/smaller file)",
            )

            # Update proxy settings in config if changed
            if (
                proxy_enabled != proxy_settings["enabled"]
                or proxy_width != proxy_settings["width"]
                or proxy_quality != proxy_settings["quality"]
            ):
                # Update config in memory
                st.session_state.config_manager.config["proxy"][
                    "enabled"
                ] = proxy_enabled
                st.session_state.config_manager.config["proxy"]["width"] = proxy_width
                st.session_state.config_manager.config["proxy"][
                    "quality"
                ] = proxy_quality

                # Save config to file
                with open(st.session_state.config_manager.config_path, "w") as f:
                    yaml.dump(
                        st.session_state.config_manager.config,
                        f,
                        default_flow_style=False,
                    )

                st.success("Proxy settings updated")

        # Add button to generate all proxies (moved below the configuration)
        if st.button("Generate All Missing Proxies"):
            generate_all_proxies()

        st.divider()

        # Export controls
        st.header("Export")
        if st.button("Export All Clips"):
            if st.session_state.video_path:
                try:
                    # Get the output directory based on the video path
                    video_path = Path(st.session_state.video_path)
                    for clip in st.session_state.clips:
                        output_path = st.session_state.config_manager.get_output_path(
                            video_path, clip["name"]
                        )
                        # Ensure output directory exists
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        # Export the clip
                        export_clips(str(output_path.parent))
                except ValueError as e:
                    st.error(str(e))
            else:
                st.error("No video loaded")

        # Add cleanup button to sidebar
        st.divider()
        st.header("Maintenance")
        if st.button("Clean Up Temp Files"):
            cleanup_temp_files()
            st.success("Temporary files cleaned up")

# Main content area
if st.session_state.video_path is not None:
    # Video preview with sound
    st.subheader("Video Preview")
    video_preview_col, controls_col = st.columns([3, 1])

    with video_preview_col:
        try:
            # Get video file information
            video_info = f"Video path: {st.session_state.video_path}"
            st.info(video_info)

            # Check if file exists and is readable
            if os.path.exists(st.session_state.video_path):
                file_size = os.path.getsize(st.session_state.video_path)
                st.success(f"File exists and is {file_size/1024/1024:.2f} MB")

                # Use proxy video for playback if available
                display_path = (
                    st.session_state.proxy_path
                    if hasattr(st.session_state, "proxy_path")
                    and st.session_state.proxy_path
                    else st.session_state.video_path
                )

                # Show which video is being used for display
                if display_path != st.session_state.video_path:
                    try:
                        # Show the relative path of the proxy
                        proxy_rel_path = Path(display_path).relative_to(
                            st.session_state.config_manager.proxy_dir
                        )
                        st.success(f"Using proxy video for playback: {proxy_rel_path}")
                    except ValueError:
                        # Fall back to just the filename if relative path can't be determined
                        st.success(
                            f"Using proxy video for playback: {os.path.basename(display_path)}"
                        )

                # Try direct video playback with the proxy
                st.video(display_path)

            else:
                st.error(f"File does not exist: {st.session_state.video_path}")
        except Exception as e:
            st.error(f"Error displaying video: {str(e)}")
            logger.exception("Video display error:")

    with controls_col:
        st.write("Video Controls")
        st.write("Use the video player above for preview with sound.")
        st.write("Use the controls below for precise frame selection:")

    st.divider()

    # Frame-by-frame controls
    st.subheader("Frame Controls")

    # Add custom CSS to reduce spacing between buttons
    st.markdown(
        """
    <style>
    div.row-widget.stButton {
        margin: 0px 1px;
        padding: 0px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create a container for the buttons
    button_container = st.container()
    with button_container:
        # Use a single row with equal columns for the buttons
        cols = st.columns([0.2, 0.2, 0.2, 1])

        with cols[0]:
            if st.button("⏮️ Previous"):
                st.session_state.current_frame = max(
                    0, st.session_state.current_frame - 1
                )

        with cols[1]:
            play_pause = "⏸️ Pause" if st.session_state.is_playing else "▶️ Play"
            if st.button(play_pause):
                st.session_state.is_playing = not st.session_state.is_playing
                # Force a rerun to update the button state immediately
                st.rerun()

        with cols[2]:
            if st.button("⏭️ Next"):
                st.session_state.current_frame = min(
                    st.session_state.video_processor.total_frames - 1,
                    st.session_state.current_frame + 1,
                )

        with cols[3]:
            st.write(
                f"Frame: {st.session_state.current_frame}/{st.session_state.video_processor.total_frames}"
            )

    # Frame slider
    st.session_state.current_frame = st.slider(
        "Frame Position",
        min_value=0,
        max_value=max(0, st.session_state.video_processor.total_frames - 1),
        value=st.session_state.current_frame,
    )

    # Display current frame for precise control
    current_frame = st.session_state.video_processor.get_frame(
        st.session_state.current_frame
    )
    if current_frame is not None:
        frame_height, frame_width = current_frame.shape[:2]

        # Create a copy of the frame to draw on
        display_frame = current_frame.copy()

        # Initialize crop region if not already set
        if (
            "temp_crop_x" not in st.session_state
            and st.session_state.crop_start is None
        ):
            # Calculate default crop region (centered) based on the selected output resolution
            target_width, target_height = RESOLUTION_PRESETS[
                st.session_state.output_resolution
            ]

            # If the frame is wider than the target aspect ratio, constrain by height
            # Otherwise, constrain by width
            if (frame_width / frame_height) > (target_width / target_height):
                # Frame is wider than target aspect ratio, so constrain by height
                default_height = min(frame_height, target_height)
                default_width = int(default_height * (target_width / target_height))
            else:
                # Frame is taller than target aspect ratio, so constrain by width
                default_width = min(frame_width, target_width)
                default_height = int(default_width * (target_height / target_width))

            # Center the crop region
            default_x = (frame_width - default_width) // 2
            default_y = (frame_height - default_height) // 2

            st.session_state.temp_crop_x = default_x
            st.session_state.temp_crop_y = default_y
            st.session_state.temp_crop_width = default_width
            st.session_state.temp_crop_height = default_height
            st.session_state.crop_selection_active = False

        # Draw crop overlay if crop regions are set
        if st.session_state.crop_selection_active:
            # Draw the active crop selection
            cv2.rectangle(
                display_frame,
                (st.session_state.temp_crop_x, st.session_state.temp_crop_y),
                (
                    st.session_state.temp_crop_x + st.session_state.temp_crop_width,
                    st.session_state.temp_crop_y + st.session_state.temp_crop_height,
                ),
                (0, 255, 0),
                2,
            )
            # Add text with resolution
            cv2.putText(
                display_frame,
                f"Crop: {st.session_state.output_resolution} ({st.session_state.temp_crop_width}x{st.session_state.temp_crop_height})",
                (st.session_state.temp_crop_x, st.session_state.temp_crop_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        elif st.session_state.current_frame in st.session_state.crop_keyframes:
            # If there's a keyframe at the current frame, show it
            display_frame = draw_crop_overlay(
                display_frame,
                st.session_state.crop_keyframes[st.session_state.current_frame],
                st.session_state.output_resolution,
            )
        elif st.session_state.crop_start:
            # Otherwise show the start crop
            display_frame = draw_crop_overlay(
                display_frame,
                st.session_state.crop_start,
                st.session_state.output_resolution,
            )

        # Display frame with annotations
        st.image(
            display_frame,
            caption=f"Frame {st.session_state.current_frame}"
            + (
                f" | Crop: x={st.session_state.temp_crop_x}, y={st.session_state.temp_crop_y}, w={st.session_state.temp_crop_width}, h={st.session_state.temp_crop_height}"
                if st.session_state.crop_selection_active
                else ""
            ),
            use_container_width=True,
        )

        # Clip editing controls
        st.subheader("Clip Controls")

        col1, col2 = st.columns(2)

        with col1:
            # In/Out point controls
            if "in_point" not in st.session_state:
                st.session_state.in_point = None

            if "out_point" not in st.session_state:
                st.session_state.out_point = None

            in_out_col1, in_out_col2 = st.columns(2)

            with in_out_col1:
                if st.button("Set In Point"):
                    st.session_state.in_point = st.session_state.current_frame

                    # If crop_start is not set, prompt user to set it
                    if st.session_state.crop_start is None:
                        st.info("Please select a start crop region")
                        # We'll let the user manually click the Select Start Crop button

            with in_out_col2:
                if st.button("Set Out Point"):
                    st.session_state.out_point = st.session_state.current_frame

                    # If crop_end is not set and we're not using the same crop as start,
                    # prompt user to set it
                    if (
                        st.session_state.crop_end is None
                        and st.session_state.crop_start is not None
                    ):
                        # Default to using the same crop as start
                        st.session_state.crop_end = None
                        st.info(
                            "Using same crop region for end point. Change this in crop settings if needed."
                        )

            st.write(f"In Point: {st.session_state.in_point}")
            st.write(f"Out Point: {st.session_state.out_point}")

        with col2:
            # Crop region controls
            st.write("Crop Region")

            # Output resolution selection (moved up from below)
            st.subheader("Output Settings")
            st.session_state.output_resolution = st.selectbox(
                "Output Resolution",
                list(RESOLUTION_PRESETS.keys()),
                index=list(RESOLUTION_PRESETS.keys()).index(
                    st.session_state.output_resolution
                ),
            )

            # Get the dimensions for the selected resolution
            target_width, target_height = RESOLUTION_PRESETS[
                st.session_state.output_resolution
            ]
            st.info(f"Selected resolution: {target_width}x{target_height}")

            crop_col1, crop_col2 = st.columns(2)

            with crop_col1:
                if not st.session_state.crop_selection_active:
                    if st.button("Select Crop at Current Frame"):
                        # Start crop selection mode
                        st.session_state.crop_selection_active = True

                        # Initialize with default or existing crop
                        if (
                            st.session_state.current_frame
                            in st.session_state.crop_keyframes
                        ):
                            x, y, w, h = st.session_state.crop_keyframes[
                                st.session_state.current_frame
                            ]
                            st.session_state.temp_crop_x = x
                            st.session_state.temp_crop_y = y
                            st.session_state.temp_crop_width = w
                            st.session_state.temp_crop_height = h
                        elif st.session_state.crop_start is not None:
                            x, y, w, h = st.session_state.crop_start
                            st.session_state.temp_crop_x = x
                            st.session_state.temp_crop_y = y
                            st.session_state.temp_crop_width = w
                            st.session_state.temp_crop_height = h
                        else:
                            # Calculate default crop region (centered)
                            target_width, target_height = RESOLUTION_PRESETS[
                                st.session_state.output_resolution
                            ]
                            aspect_ratio = target_width / target_height

                            # If the frame is wider than the target aspect ratio, constrain by height
                            # Otherwise, constrain by width
                            if (frame_width / frame_height) > aspect_ratio:
                                # Frame is wider than target aspect ratio, so constrain by height
                                default_height = min(frame_height, target_height)
                                default_width = int(default_height * aspect_ratio)
                            else:
                                # Frame is taller than target aspect ratio, so constrain by width
                                default_width = min(frame_width, target_width)
                                default_height = int(default_width / aspect_ratio)

                            # Center the crop region
                            default_x = (frame_width - default_width) // 2
                            default_y = (frame_height - default_height) // 2

                            st.session_state.temp_crop_x = default_x
                            st.session_state.temp_crop_y = default_y
                            st.session_state.temp_crop_width = default_width
                            st.session_state.temp_crop_height = default_height

                        st.rerun()
                else:
                    # Show crop adjustment controls when in crop selection mode
                    if st.button("Confirm Crop"):
                        crop_coords = [
                            st.session_state.temp_crop_x,
                            st.session_state.temp_crop_y,
                            st.session_state.temp_crop_width,
                            st.session_state.temp_crop_height,
                        ]

                        # Store as a keyframe
                        st.session_state.crop_keyframes[
                            st.session_state.current_frame
                        ] = crop_coords

                        # Also set as start crop if this is the first keyframe or at in_point
                        if (
                            st.session_state.in_point is not None
                            and st.session_state.current_frame
                            == st.session_state.in_point
                        ):
                            st.session_state.crop_start = crop_coords

                        # Also set as end crop if at out_point
                        if (
                            st.session_state.out_point is not None
                            and st.session_state.current_frame
                            == st.session_state.out_point
                        ):
                            st.session_state.crop_end = crop_coords

                        # Exit crop selection mode
                        st.session_state.crop_selection_active = False
                        st.success(
                            f"Crop keyframe added at frame {st.session_state.current_frame}"
                        )
                        st.rerun()

            with crop_col2:
                if not st.session_state.crop_selection_active:
                    if st.button("Remove Crop at Current Frame"):
                        if (
                            st.session_state.current_frame
                            in st.session_state.crop_keyframes
                        ):
                            del st.session_state.crop_keyframes[
                                st.session_state.current_frame
                            ]
                            st.success(
                                f"Crop keyframe removed from frame {st.session_state.current_frame}"
                            )

                            # If we removed the in_point or out_point keyframe, update crop_start/crop_end
                            if (
                                st.session_state.in_point is not None
                                and st.session_state.current_frame
                                == st.session_state.in_point
                            ):
                                st.session_state.crop_start = None

                            if (
                                st.session_state.out_point is not None
                                and st.session_state.current_frame
                                == st.session_state.out_point
                            ):
                                st.session_state.crop_end = None

                            # Force rerun to update the display
                            st.rerun()
                        else:
                            st.warning(
                                f"No crop keyframe at frame {st.session_state.current_frame}"
                            )
                else:
                    if st.button("Cancel"):
                        # Exit crop selection mode without saving
                        st.session_state.crop_selection_active = False
                        st.rerun()

            # Show crop adjustment controls when in crop selection mode
            if st.session_state.crop_selection_active:
                st.write("Adjust Crop Position:")

                # Movement controls in a grid
                move_col1, move_col2, move_col3 = st.columns(3)

                with move_col2:
                    if st.button("⬆️ Up"):
                        st.session_state.temp_crop_y = max(
                            0, st.session_state.temp_crop_y - 10
                        )
                        st.rerun()

                with move_col1:
                    if st.button("⬅️ Left"):
                        st.session_state.temp_crop_x = max(
                            0, st.session_state.temp_crop_x - 10
                        )
                        st.rerun()

                with move_col2:
                    if st.button("Center"):
                        # Center the crop region
                        st.session_state.temp_crop_x = (
                            frame_width - st.session_state.temp_crop_width
                        ) // 2
                        st.session_state.temp_crop_y = (
                            frame_height - st.session_state.temp_crop_height
                        ) // 2
                        st.rerun()

                with move_col3:
                    if st.button("➡️ Right"):
                        max_x = frame_width - st.session_state.temp_crop_width
                        st.session_state.temp_crop_x = min(
                            max_x, st.session_state.temp_crop_x + 10
                        )
                        st.rerun()

                with move_col2:
                    if st.button("⬇️ Down"):
                        max_y = frame_height - st.session_state.temp_crop_height
                        st.session_state.temp_crop_y = min(
                            max_y, st.session_state.temp_crop_y + 10
                        )
                        st.rerun()

                st.write("Adjust Size:")
                size_col1, size_col2 = st.columns(2)

                with size_col1:
                    if st.button("➖ Smaller"):
                        # Calculate aspect ratio
                        target_width, target_height = RESOLUTION_PRESETS[
                            st.session_state.output_resolution
                        ]
                        aspect_ratio = target_width / target_height

                        # Decrease size while maintaining aspect ratio
                        new_width = max(100, st.session_state.temp_crop_width - 50)
                        new_height = int(new_width / aspect_ratio)

                        # Make sure it fits within the frame
                        if new_height <= frame_height:
                            # Adjust position to keep centered
                            x_diff = (st.session_state.temp_crop_width - new_width) // 2
                            y_diff = (
                                st.session_state.temp_crop_height - new_height
                            ) // 2

                            st.session_state.temp_crop_width = new_width
                            st.session_state.temp_crop_height = new_height
                            st.session_state.temp_crop_x += x_diff
                            st.session_state.temp_crop_y += y_diff

                            # Ensure within bounds
                            st.session_state.temp_crop_x = min(
                                max(0, st.session_state.temp_crop_x),
                                frame_width - new_width,
                            )
                            st.session_state.temp_crop_y = min(
                                max(0, st.session_state.temp_crop_y),
                                frame_height - new_height,
                            )
                        st.rerun()

                with size_col2:
                    if st.button("➕ Larger"):
                        # Calculate aspect ratio
                        target_width, target_height = RESOLUTION_PRESETS[
                            st.session_state.output_resolution
                        ]
                        aspect_ratio = target_width / target_height

                        # Increase size while maintaining aspect ratio
                        new_width = min(
                            frame_width, st.session_state.temp_crop_width + 50
                        )
                        new_height = int(new_width / aspect_ratio)

                        # Make sure it fits within the frame
                        if new_height <= frame_height:
                            # Adjust position to keep centered
                            x_diff = (new_width - st.session_state.temp_crop_width) // 2
                            y_diff = (
                                new_height - st.session_state.temp_crop_height
                            ) // 2

                            new_x = st.session_state.temp_crop_x - x_diff
                            new_y = st.session_state.temp_crop_y - y_diff

                            # Ensure within bounds
                            if (
                                new_x >= 0
                                and new_y >= 0
                                and new_x + new_width <= frame_width
                                and new_y + new_height <= frame_height
                            ):
                                st.session_state.temp_crop_width = new_width
                                st.session_state.temp_crop_height = new_height
                                st.session_state.temp_crop_x = new_x
                                st.session_state.temp_crop_y = new_y
                        st.rerun()

                if st.button("Reset Size"):
                    # Reset to default size based on the selected output resolution
                    target_width, target_height = RESOLUTION_PRESETS[
                        st.session_state.output_resolution
                    ]
                    aspect_ratio = target_width / target_height

                    # If the frame is wider than the target aspect ratio, constrain by height
                    # Otherwise, constrain by width
                    if (frame_width / frame_height) > aspect_ratio:
                        # Frame is wider than target aspect ratio, so constrain by height
                        default_height = min(frame_height, target_height)
                        default_width = int(default_height * aspect_ratio)
                    else:
                        # Frame is taller than target aspect ratio, so constrain by width
                        default_width = min(frame_width, target_width)
                        default_height = int(default_width / aspect_ratio)

                    # Keep the current position but reset the size
                    x_diff = (default_width - st.session_state.temp_crop_width) // 2
                    y_diff = (default_height - st.session_state.temp_crop_height) // 2

                    new_x = st.session_state.temp_crop_x - x_diff
                    new_y = st.session_state.temp_crop_y - y_diff

                    # Ensure within bounds
                    if (
                        new_x >= 0
                        and new_y >= 0
                        and new_x + default_width <= frame_width
                        and new_y + default_height <= frame_height
                    ):
                        st.session_state.temp_crop_width = default_width
                        st.session_state.temp_crop_height = default_height
                        st.session_state.temp_crop_x = new_x
                        st.session_state.temp_crop_y = new_y
                    else:
                        # If out of bounds, center it
                        st.session_state.temp_crop_width = default_width
                        st.session_state.temp_crop_height = default_height
                        st.session_state.temp_crop_x = (
                            frame_width - default_width
                        ) // 2
                        st.session_state.temp_crop_y = (
                            frame_height - default_height
                        ) // 2

                    st.rerun()

            # Display keyframes
            if st.session_state.crop_keyframes:
                with st.expander("Crop Keyframes", expanded=True):
                    st.write(
                        "Keyframes define how the crop region changes throughout the clip."
                    )

                    # Display all keyframes in a table
                    keyframes_data = []
                    for frame, coords in sorted(
                        st.session_state.crop_keyframes.items()
                    ):
                        x, y, w, h = coords
                        keyframes_data.append(
                            {"Frame": frame, "X": x, "Y": y, "Width": w, "Height": h}
                        )

                    if keyframes_data:
                        st.dataframe(keyframes_data)

                        if st.button("Clear All Keyframes"):
                            st.session_state.crop_keyframes = {}
                            st.session_state.crop_start = None
                            st.session_state.crop_end = None
                            st.success("All crop keyframes cleared")
                            st.rerun()

        # Clip name and save controls
        if "clip_name" not in st.session_state:
            st.session_state.clip_name = f"clip_{len(st.session_state.clips) + 1}"

        st.session_state.clip_name = st.text_input(
            "Clip Name", value=st.session_state.clip_name
        )

        save_col1, save_col2 = st.columns(2)

        with save_col1:
            if st.button("Add/Update Clip"):
                add_or_update_clip()

        with save_col2:
            if st.button("Clear Clip Data"):
                st.session_state.in_point = None
                st.session_state.out_point = None
                st.session_state.crop_start = None
                st.session_state.crop_end = None
                st.session_state.crop_keyframes = {}  # Clear keyframes
                st.session_state.current_clip_index = -1
                st.session_state.clip_name = f"clip_{len(st.session_state.clips) + 1}"

        # List of saved clips
        st.subheader("Saved Clips")

        if st.session_state.clips:
            for i, clip in enumerate(st.session_state.clips):
                with st.expander(
                    f"{i+1}. {clip['name']} ({clip['in_point']} - {clip['out_point']})"
                ):
                    st.write(f"In Point: {clip['in_point']}")
                    st.write(f"Out Point: {clip['out_point']}")
                    st.write(f"Crop Start: {clip['crop_start']}")
                    st.write(
                        f"Crop End: {clip['crop_end'] if clip['crop_end'] else 'Same as start'}"
                    )
                    st.write(f"Output Resolution: {clip['output_resolution']}")

                    # Display keyframes if present
                    if "crop_keyframes" in clip and clip["crop_keyframes"]:
                        st.write(f"Keyframes: {len(clip['crop_keyframes'])}")
                        keyframes_data = []
                        for frame, coords in sorted(clip["crop_keyframes"].items()):
                            x, y, w, h = coords
                            keyframes_data.append(
                                {
                                    "Frame": frame,
                                    "X": x,
                                    "Y": y,
                                    "Width": w,
                                    "Height": h,
                                }
                            )
                        st.dataframe(keyframes_data)

                    clip_col1, clip_col2 = st.columns(2)

                    with clip_col1:
                        if st.button(f"Edit Clip #{i+1}"):
                            st.session_state.current_clip_index = i
                            st.session_state.in_point = clip["in_point"]
                            st.session_state.out_point = clip["out_point"]
                            st.session_state.crop_start = clip["crop_start"]
                            st.session_state.crop_end = clip["crop_end"]
                            st.session_state.clip_name = clip["name"]
                            st.session_state.output_resolution = clip[
                                "output_resolution"
                            ]

                            # Load keyframes if present
                            if "crop_keyframes" in clip and clip["crop_keyframes"]:
                                st.session_state.crop_keyframes = clip[
                                    "crop_keyframes"
                                ].copy()
                            else:
                                st.session_state.crop_keyframes = {}

                            # Jump to in point
                            st.session_state.current_frame = clip["in_point"]

                    with clip_col2:
                        if st.button(f"Delete Clip #{i+1}"):
                            delete_clip(i)
        else:
            st.info("No clips saved yet")

        # Auto-play functionality
        if st.session_state.is_playing:
            time.sleep(1 / st.session_state.video_processor.fps)
            st.session_state.current_frame += 1
            if (
                st.session_state.current_frame
                >= st.session_state.video_processor.total_frames
            ):
                st.session_state.current_frame = 0
                st.session_state.is_playing = False
            st.rerun()

    # Add debug information display
    with st.expander("Debug Information"):
        st.write("Video Information:")
        st.json(
            {
                "Path": str(st.session_state.video_path),
                "Total Frames": st.session_state.video_processor.total_frames,
                "FPS": st.session_state.video_processor.fps,
                "Resolution": f"{st.session_state.video_processor.width}x{st.session_state.video_processor.height}",
                "Format": Path(st.session_state.video_path).suffix,
            }
        )

        # Add log display
        st.subheader("Recent Logs")
        if "log_messages" not in st.session_state:
            st.session_state.log_messages = []

        # Create a custom handler to capture logs
        class StreamlitHandler(logging.Handler):
            def emit(self, record):
                log_entry = self.format(record)
                st.session_state.log_messages.append(f"{record.levelname}: {log_entry}")
                # Keep only last 100 messages
                if len(st.session_state.log_messages) > 100:
                    st.session_state.log_messages.pop(0)

        # Add our custom handler to the logger
        handler = StreamlitHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(handler)

        # Display logs in reverse chronological order
        for msg in reversed(
            st.session_state.log_messages[-20:]
        ):  # Show last 20 messages
            st.text(msg)
else:
    st.info("Please load a video file from the sidebar to get started.")

# Instructions
with st.expander("How to Use"):
    st.markdown(
        """
    ## Instructions
    
    1. **Load a Video**: Select a video file from the calibrated footage directory (02_CALIBRATED_FOOTAGE) and click "Load Video".
       - A proxy video will automatically be created for smoother playback
       - You can configure proxy settings in the sidebar
    
    2. **Navigate the Video**: Use the play/pause button, next/previous frame buttons, or the slider to navigate through the video.
    
    3. **Create a Clip**:
        - Set the in-point (start) and out-point (end) of your clip
        - Define the crop region by setting start and end crop coordinates
        - Choose an output resolution
        - Give your clip a name and click "Add/Update Clip"
    
    4. **Manage Clips**:
        - View all saved clips in the "Saved Clips" section
        - Edit or delete existing clips as needed
    
    5. **Export**:
        - Click "Export All Clips" to process and save all clips
        - Clips will be saved in the corresponding camera/session folders under 03_CLIPPED
    
    ## Directory Structure
    - `data/source/02_CALIBRATED_FOOTAGE/`: Source videos (calibrated footage)
    - `data/prept/03_CLIPPED/`: Processed clips output (preserves camera/session structure)
    - `data/prept/03_CLIPPED/_configs/`: Clip configurations
    - `proxy_videos/`: Proxy videos for faster playback (configurable in config.yaml)
    
    ## Configuration
    The application uses a `config.yaml` file for configuration:
    - You can change the proxy video settings in the sidebar
    - Proxy videos are automatically created when loading a video
    - For very large files, you can extract preview frames instead of playing the video
    
    The application will preserve the folder structure of your footage when exporting clips,
    making it easy to maintain organization across multiple cameras and sessions.
    """
    )

# Footer
st.markdown("---")
st.markdown("Clipper - Video Clipping Tool | Created with Streamlit and OpenCV")

# Process the current video if proxy generation is active
if (
    st.session_state.proxy_generation_active
    and st.session_state.proxy_current_video
    and st.session_state.proxy_current_index < st.session_state.proxy_total_videos
):
    # Create proxy for the current video
    create_proxy_video(st.session_state.proxy_current_video)
