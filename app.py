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


# Add function to check if a proxy exists for a video
def proxy_exists_for_video(video_path):
    """Check if a proxy video exists for the given video path"""
    proxy_path = st.session_state.config_manager.get_proxy_path(Path(video_path))
    return proxy_path.exists()


# Function to display proxy generation progress
def display_proxy_generation_progress():
    """Display the progress of proxy generation if active"""
    # This function now returns a boolean indicating if progress is active
    if (
        "proxy_generation_progress" in st.session_state
        and st.session_state.proxy_generation_progress["active"]
    ):
        progress = st.session_state.proxy_generation_progress
        return True
    return False


# Add function to generate proxies for all videos
def generate_all_proxies():
    """Generate proxy videos for all videos that don't have one yet"""
    try:
        video_files = st.session_state.config_manager.get_video_files()
        if not video_files:
            st.warning("No video files found to generate proxies for")
            return

        # Count videos that need proxies
        videos_without_proxies = [
            v for v in video_files if not proxy_exists_for_video(v)
        ]
        total_videos = len(videos_without_proxies)

        if total_videos == 0:
            st.success("All videos already have proxy versions")
            return

        # Create progress indicators
        if "proxy_generation_progress" not in st.session_state:
            st.session_state.proxy_generation_progress = {
                "active": False,
                "total": 0,
                "current": 0,
                "current_video": "",
                "completed": 0,
                "errors": 0,
                "last_error": "",
                "videos_to_process": [],
                "last_update_time": time.time(),
            }

        # Set up the progress tracking
        progress_state = st.session_state.proxy_generation_progress
        progress_state["active"] = True
        progress_state["total"] = total_videos
        progress_state["current"] = 0
        progress_state["completed"] = 0
        progress_state["errors"] = 0
        progress_state["last_error"] = ""
        progress_state["videos_to_process"] = videos_without_proxies
        progress_state["current_video"] = (
            os.path.basename(videos_without_proxies[0])
            if videos_without_proxies
            else ""
        )
        progress_state["last_update_time"] = time.time()

        # Force a rerun to immediately show the progress UI
        st.rerun()

    except Exception as e:
        logger.exception(f"Error in proxy generation setup: {str(e)}")
        if "proxy_generation_progress" in st.session_state:
            st.session_state.proxy_generation_progress["active"] = False
            st.session_state.proxy_generation_progress["last_error"] = str(e)
        st.error(f"Error in proxy generation setup: {str(e)}")


# Function to process videos for proxy generation
def process_proxy_generation():
    """Process one video for proxy generation when active"""
    if (
        "proxy_generation_progress" not in st.session_state
        or not st.session_state.proxy_generation_progress["active"]
    ):
        return

    progress_state = st.session_state.proxy_generation_progress

    # Only process one video every few seconds to avoid blocking the UI
    current_time = time.time()
    if current_time - progress_state.get("last_update_time", 0) < 0.5:
        # Don't process too frequently - allow UI to remain responsive
        return

    # Update the last update time
    progress_state["last_update_time"] = current_time

    # Get the videos to process
    videos_to_process = progress_state.get("videos_to_process", [])

    # Check if we're done
    if not videos_to_process or progress_state["current"] >= len(videos_to_process):
        progress_state["active"] = False
        return

    # Get the current video to process
    current_idx = progress_state["current"]
    video_path = videos_to_process[current_idx]
    progress_state["current_video"] = os.path.basename(video_path)

    try:
        # Generate proxy for this video - no UI placeholder needed as we show progress in sidebar
        proxy_path = create_proxy_video(video_path, None)

        if proxy_path:
            logger.info(
                f"Successfully created proxy for {os.path.basename(video_path)}"
            )
            progress_state["completed"] += 1
        else:
            logger.error(f"Failed to create proxy for {os.path.basename(video_path)}")
            progress_state["errors"] += 1
            progress_state["last_error"] = (
                f"Failed to create proxy for {os.path.basename(video_path)}"
            )
    except Exception as e:
        logger.exception(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        progress_state["errors"] += 1
        progress_state["last_error"] = (
            f"Error processing {os.path.basename(video_path)}: {str(e)}"
        )

    # Move to the next video
    progress_state["current"] = current_idx + 1

    # If we've processed all videos, mark as complete
    if progress_state["current"] >= len(videos_to_process):
        progress_state["active"] = False

    # Force a rerun to update the UI with the latest progress
    st.rerun()


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


def create_proxy_video(source_path, progress_placeholder=None):
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
            return str(proxy_path)

        # Show progress
        if progress_placeholder:
            progress_placeholder.text("Creating proxy video for faster playback...")
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
                        queue.put(
                            {
                                "progress": progress,
                                "remaining": remaining,
                                "current_time": current_time,
                            }
                        )
                except Exception as e:
                    logger.error(f"Error in progress monitoring: {str(e)}")
                    continue

            # Signal completion
            queue.put({"progress": 1.0, "remaining": 0, "current_time": duration})
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
                        if progress_placeholder:
                            progress_bar.progress(progress_info["progress"])
                            remaining = progress_info["remaining"]
                            progress_placeholder.text(
                                f"Creating proxy video: {int(progress_info['progress'] * 100)}% complete "
                                f"(approx. {int(remaining/60)} min {int(remaining%60)} sec remaining)"
                            )
                        progress_queue.task_done()
                    except Exception:
                        # No progress update available, sleep briefly
                        pass

                    # Sleep to avoid consuming too much CPU
                    time.sleep(0.1)

                    # Periodically update UI even if no progress info
                    current_time = time.time()
                    if current_time - last_update_time > 2.0:
                        if progress_placeholder:
                            progress_placeholder.text(
                                f"Creating proxy video... (ffmpeg is running)"
                            )
                        last_update_time = current_time
            finally:
                # Signal monitor thread to stop
                stop_event.set()

                # Wait for process to complete
                logger.info("Waiting for ffmpeg process to complete...")
                process.wait()
                logger.info("ffmpeg process completed")

                # Final progress update
                if progress_placeholder:
                    progress_bar.progress(1.0)
                    progress_placeholder.text("Proxy video created successfully!")
        else:
            # If duration couldn't be determined, just wait for completion
            logger.info(
                "No duration available, waiting for process without progress updates"
            )
            process.wait()
            if progress_placeholder:
                progress_bar.progress(1.0)
                progress_placeholder.text("Proxy video created successfully!")

        if os.path.exists(proxy_path):
            proxy_size = os.path.getsize(proxy_path) / (1024 * 1024)
            logger.info(
                f"Proxy created successfully: {proxy_path} ({proxy_size:.2f} MB)"
            )
            return str(proxy_path)
        else:
            logger.error("Failed to create proxy video")
            if progress_placeholder:
                progress_placeholder.error("Failed to create proxy video")
            return None

    except Exception as e:
        logger.exception(f"Error creating proxy: {str(e)}")
        if progress_placeholder:
            progress_placeholder.error(f"Error creating proxy: {str(e)}")
        return None


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

    clip_data = {
        "name": st.session_state.clip_name,
        "in_point": st.session_state.in_point,
        "out_point": st.session_state.out_point,
        "crop_start": st.session_state.crop_start,
        "crop_end": st.session_state.crop_end
        or st.session_state.crop_start,  # Use crop_start if crop_end not set
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
            for file in proxy_dir.glob("*"):
                try:
                    file.unlink()
                    logger.info(f"Deleted proxy file: {file}")
                except Exception as e:
                    logger.error(f"Error deleting proxy file {file}: {str(e)}")

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

# Process proxy generation if active
process_proxy_generation()

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

    # Toggle for proxy creation
    proxy_enabled = st.checkbox("Enable proxy videos", value=proxy_settings["enabled"])

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
            st.session_state.config_manager.config["proxy"]["enabled"] = proxy_enabled
            st.session_state.config_manager.config["proxy"]["width"] = proxy_width
            st.session_state.config_manager.config["proxy"]["quality"] = proxy_quality

            # Save config to file
            with open(st.session_state.config_manager.config_path, "w") as f:
                yaml.dump(
                    st.session_state.config_manager.config, f, default_flow_style=False
                )

            st.success("Proxy settings updated")

    # Add button to generate all proxies (moved below the configuration)
    if st.button("Generate All Missing Proxies"):
        generate_all_proxies()

    # Show proxy generation progress in the sidebar
    if (
        "proxy_generation_progress" in st.session_state
        and st.session_state.proxy_generation_progress["active"]
    ):
        progress = st.session_state.proxy_generation_progress

        # Create a container for the progress information
        with st.container():
            st.subheader("Proxy Generation Progress")

            # Show progress bar with a distinctive color
            progress_percentage = progress["current"] / max(1, progress["total"])
            progress_bar = st.progress(progress_percentage)

            # Show status text with more details
            st.info(f"Processing: {progress['current']}/{progress['total']} videos")
            st.caption(f"Current video: {progress['current_video']}")
            st.caption(f"Successfully completed: {progress['completed']} videos")

            if progress.get("errors", 0) > 0:
                st.warning(f"Errors encountered: {progress['errors']}")
                if "last_error" in progress and progress["last_error"]:
                    st.caption(f"Last error: {progress['last_error']}")

            # Add a stop button
            if st.button("Stop Proxy Generation", key="stop_proxy_gen_sidebar"):
                st.session_state.proxy_generation_progress["active"] = False
                st.rerun()

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

    # Create a container for the frame information
    frame_info = st.container()
    frame_info.write(
        f"Frame: {st.session_state.current_frame}/{st.session_state.video_processor.total_frames}"
    )

    # Create a horizontal container for the buttons with minimal spacing
    button_container = st.container()
    with button_container:
        # Place all three buttons in a single row with minimal spacing
        cols = st.columns([1, 1, 1])

        with cols[0]:
            if st.button("⏮️ Previous"):
                st.session_state.current_frame = max(
                    0, st.session_state.current_frame - 1
                )

        with cols[1]:
            play_pause = "⏸️ Pause" if st.session_state.is_playing else "▶️ Play"
            if st.button(play_pause):
                st.session_state.is_playing = not st.session_state.is_playing

        with cols[2]:
            if st.button("⏭️ Next"):
                st.session_state.current_frame = min(
                    st.session_state.video_processor.total_frames - 1,
                    st.session_state.current_frame + 1,
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

        # Display frame with annotations
        st.image(
            current_frame,
            caption=f"Frame {st.session_state.current_frame}",
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

            with in_out_col2:
                if st.button("Set Out Point"):
                    st.session_state.out_point = st.session_state.current_frame

            st.write(f"In Point: {st.session_state.in_point}")
            st.write(f"Out Point: {st.session_state.out_point}")

        with col2:
            # Crop region controls
            st.write("Crop Region (x, y, width, height)")

            crop_col1, crop_col2 = st.columns(2)

            with crop_col1:
                if st.button("Set Start Crop"):
                    # Default to full frame if not set
                    st.session_state.crop_start = [0, 0, frame_width, frame_height]

            with crop_col2:
                if st.button("Set End Crop"):
                    # Default to start crop if not set
                    if st.session_state.crop_start:
                        st.session_state.crop_end = st.session_state.crop_start.copy()
                    else:
                        st.session_state.crop_end = [0, 0, frame_width, frame_height]

            # Crop region input fields
            if st.session_state.crop_start:
                crop_x = st.slider(
                    "Crop X", 0, frame_width - 10, st.session_state.crop_start[0]
                )
                crop_y = st.slider(
                    "Crop Y", 0, frame_height - 10, st.session_state.crop_start[1]
                )
                crop_w = st.slider(
                    "Crop Width",
                    10,
                    frame_width - crop_x,
                    st.session_state.crop_start[2],
                )
                crop_h = st.slider(
                    "Crop Height",
                    10,
                    frame_height - crop_y,
                    st.session_state.crop_start[3],
                )

                st.session_state.crop_start = [crop_x, crop_y, crop_w, crop_h]

                if st.checkbox("Same end crop as start"):
                    st.session_state.crop_end = None
                elif st.session_state.crop_end:
                    st.write("End Crop (if different from start):")
                    end_crop_x = st.slider(
                        "End Crop X", 0, frame_width - 10, st.session_state.crop_end[0]
                    )
                    end_crop_y = st.slider(
                        "End Crop Y", 0, frame_height - 10, st.session_state.crop_end[1]
                    )
                    end_crop_w = st.slider(
                        "End Crop Width",
                        10,
                        frame_width - end_crop_x,
                        st.session_state.crop_end[2],
                    )
                    end_crop_h = st.slider(
                        "End Crop Height",
                        10,
                        frame_height - end_crop_y,
                        st.session_state.crop_end[3],
                    )

                    st.session_state.crop_end = [
                        end_crop_x,
                        end_crop_y,
                        end_crop_w,
                        end_crop_h,
                    ]

        # Output resolution selection
        st.subheader("Output Settings")
        st.session_state.output_resolution = st.selectbox(
            "Output Resolution",
            list(RESOLUTION_PRESETS.keys()),
            index=list(RESOLUTION_PRESETS.keys()).index(
                st.session_state.output_resolution
            ),
        )

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
