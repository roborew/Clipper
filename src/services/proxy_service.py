"""
Proxy video generation services for the Clipper application.
"""

import os
import re
import time
import threading
import subprocess
import tempfile
from pathlib import Path
from queue import Queue
import streamlit as st
import logging
import collections
import cv2
import numpy as np
from datetime import datetime
import shutil

# Configure logging to suppress Streamlit context warnings
streamlit_logger = logging.getLogger("streamlit")
streamlit_logger.setLevel(logging.ERROR)


# Create a filter to remove the specific warning message
class ScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext" not in record.getMessage()


# Apply filter to the streamlit.runtime.scriptrunner_utils.script_run_context logger
script_run_context_logger = logging.getLogger(
    "streamlit.runtime.scriptrunner_utils.script_run_context"
)
script_run_context_logger.addFilter(ScriptRunContextFilter())

logger = logging.getLogger("clipper.proxy")

# Import calibration service
from . import calibration_service


def monitor_progress(process, duration, progress_queue, stop_event):
    """
    Monitor FFmpeg process and provide progress updates.

    Args:
        process: Subprocess running FFmpeg
        duration: Expected duration of the video in seconds
        progress_queue: Queue to put progress updates into
        stop_event: Event to signal when monitoring should stop
    """
    try:
        last_update_time = time.time()
        start_time = time.time()

        while process.poll() is None and not stop_event.is_set():
            # Get FFmpeg stderr output line by line
            line = process.stderr.readline().strip()
            if not line:
                time.sleep(0.1)
                continue

            # Look for time information in FFmpeg output
            time_match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
            if time_match:
                # Extract current timestamp
                hours, minutes, seconds = time_match.groups()
                current_time = (
                    (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
                )

                # Calculate progress as percentage
                progress = min(current_time / duration, 1.0) if duration > 0 else 0

                # Calculate time remaining
                elapsed = time.time() - start_time
                if progress > 0:
                    total_estimated = elapsed / progress
                    remaining = total_estimated - elapsed
                else:
                    remaining = 0

                # Look for speed information
                speed_match = re.search(r"speed=(\d+\.?\d*)x", line)
                encoding_speed = float(speed_match.group(1)) if speed_match else 0

                # Update progress no more than once per 0.5 seconds to avoid flooding
                current_time = time.time()
                if current_time - last_update_time >= 0.5 or progress >= 1.0:
                    progress_queue.put(
                        {
                            "progress": progress,
                            "time": current_time,
                            "elapsed": elapsed,
                            "remaining": remaining,
                            "encoding_speed": encoding_speed,
                        }
                    )
                    last_update_time = current_time

            # Check for stop event periodically
            if stop_event.is_set():
                break

    except Exception as e:
        logger.warning(f"Error in progress monitoring: {str(e)}")
    finally:
        # Send a final progress update if we haven't reached 100%
        try:
            progress_queue.put(
                {
                    "progress": 1.0,
                    "time": time.time(),
                    "elapsed": time.time() - start_time,
                    "remaining": 0,
                    "encoding_speed": 0,
                }
            )
        except:
            pass


def is_streamlit_context():
    """Check if the code is running in a Streamlit context

    Returns:
        bool: True if running in Streamlit context, False otherwise
    """
    try:
        import streamlit as st

        # Try to access session_state to verify we're in a Streamlit context
        _ = st.session_state
        return True
    except Exception:
        return False


def _progress_callback_stage1(
    progress, progress_placeholder, progress_bar, main_callback, in_streamlit
):
    """Helper function to scale and forward progress updates for the first stage of calibration.

    Args:
        progress: Progress value from calibration (0.0 to 1.0)
        progress_placeholder: Streamlit placeholder for progress text
        progress_bar: Streamlit progress bar
        main_callback: Main progress callback function
        in_streamlit: Whether we're in a Streamlit context
    """
    try:
        # Scale progress to 0-50% for first stage
        scaled_progress = progress * 0.5

        # Update progress bar if available
        if progress_bar and in_streamlit:
            progress_bar.progress(scaled_progress)
            percent_complete = int(scaled_progress * 100)
            progress_placeholder.text(
                f"Stage 1/2: Applying calibration: {percent_complete}% complete"
            )

        # Forward to main callback if available
        if main_callback:
            main_callback({"progress": scaled_progress})

        # Update progress tracking for polling
        if in_streamlit:
            import streamlit as st

            st.session_state.proxy_last_progress_time = time.time()
            st.session_state.proxy_last_progress_value = scaled_progress
    except Exception as e:
        # Silently ignore errors in callback - these are expected in some contexts
        pass


def create_proxy_video(
    source_path,
    progress_placeholder=None,
    progress_callback=None,
    config_manager=None,
    is_clip=False,
):
    """
    Create a proxy (web-compatible) version of the video for faster playback

    Args:
        source_path: Path to the source video
        progress_placeholder: Streamlit placeholder for progress updates
        progress_callback: Callback function for progress updates
        config_manager: ConfigManager instance
        is_clip: Whether this is a clip preview (True) or raw video proxy (False)

    Returns:
        Path to the proxy video or None if creation failed
    """
    try:
        in_streamlit = is_streamlit_context()

        # Initialize progress queue
        progress_queue = Queue()
        stop_event = threading.Event()

        # Get proxy settings from config manager
        if not config_manager:
            # Check if we're in a Streamlit context
            try:
                import streamlit as st

                config_manager = st.session_state.config_manager
            except:
                logger.warning(
                    "Not in Streamlit context and no config_manager provided"
                )
                return None

        proxy_settings = config_manager.get_proxy_settings()

        # Check if proxy creation is enabled
        if not proxy_settings["enabled"]:
            logger.info("Proxy video creation is disabled in config")
            return None

        # Get proxy path from config manager
        proxy_path = config_manager.get_proxy_path(Path(source_path), is_clip)

        # Check if proxy already exists
        if proxy_path.exists():
            logger.info(f"Proxy already exists: {proxy_path}")

            # If this is part of batch processing in Streamlit context, mark as completed
            if in_streamlit:
                try:
                    if (
                        st.session_state.get("proxy_generation_active", False)
                        and st.session_state.get("proxy_current_video") == source_path
                    ):
                        logger.info(
                            f"Marking {os.path.basename(source_path)} as completed and moving to next video"
                        )
                        st.session_state.proxy_completed_videos.append(source_path)
                        if "proxy_calibration_stage" in st.session_state:
                            del st.session_state.proxy_calibration_stage
                        if "proxy_calibration_start_time" in st.session_state:
                            del st.session_state.proxy_calibration_start_time
                        st.session_state.proxy_needs_rerun = True
                        move_to_next_proxy_video()
                except:
                    pass  # Error accessing session state

            return str(proxy_path)

        # Create progress bar if not already done
        progress_bar = None
        if progress_placeholder and in_streamlit:
            progress_bar = progress_placeholder.progress(0.0)

        # Get video duration for progress tracking
        duration = get_video_duration(source_path)
        if duration is None:
            logger.error("Could not determine video duration")
            if progress_placeholder and in_streamlit:
                progress_placeholder.error("Could not determine video duration")
            return None

        # Check if this video needs calibration - using calibration_service directly
        camera_type = calibration_service.get_camera_type_from_path(
            source_path, config_manager
        )

        if camera_type and camera_type != "none":
            logger.info(f"Video requires {camera_type} calibration")
            camera_matrix, dist_coeffs, _ = calibration_service.load_calibration(
                camera_type, config_manager=config_manager, video_path=source_path
            )

            if camera_matrix is None or dist_coeffs is None:
                logger.error(f"Missing calibration data for camera type: {camera_type}")
                if progress_placeholder and in_streamlit:
                    progress_placeholder.error(
                        f"Missing calibration data for camera type: {camera_type}"
                    )
                return None

            # Use two-stage calibration process
            return _create_proxy_with_calibration_two_stage(
                source_path=source_path,
                proxy_path=proxy_path,
                camera_type=camera_type,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                proxy_settings=proxy_settings,
                duration=duration,
                progress_placeholder=progress_placeholder,
                progress_callback=progress_callback,
                config_manager=config_manager,
            )

        # For non-calibrated videos, proceed with direct proxy creation
        logger.info("Creating proxy without calibration")
        if progress_placeholder and in_streamlit:
            progress_placeholder.text("Creating proxy video...")

        # Ensure the parent directory exists
        proxy_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring proxy directory exists: {proxy_path.parent}")

        # Prepare FFmpeg command
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
            f"scale={proxy_settings['width']}:-2",
            "-c:a",
            "aac",
            "-b:a",
            proxy_settings["audio_bitrate"],
            str(proxy_path),
        ]

        # Log the command for debugging
        logger.info(f"Starting ffmpeg conversion: {' '.join(cmd)}")

        try:
            # Convert command list to a string for shell=True
            cmd_str = " ".join(
                (
                    f'"{arg}"'
                    if " " in str(arg) or "+" in str(arg) or ":" in str(arg)
                    else str(arg)
                )
                for arg in cmd
            )
            logger.info(f"Running shell command: {cmd_str}")

            process = subprocess.Popen(
                cmd_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=True,
            )
        except Exception as e:
            logger.error(f"Failed to start ffmpeg process: {str(e)}")
            if progress_placeholder and in_streamlit:
                progress_placeholder.error(f"Failed to start ffmpeg process: {str(e)}")
            return None

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
                    try:
                        progress_info = progress_queue.get_nowait()
                        if progress_placeholder and progress_bar and in_streamlit:
                            progress_bar.progress(progress_info["progress"])

                            # Calculate time remaining
                            remaining = progress_info.get("remaining", 0)
                            minutes_remaining = int(remaining / 60)
                            seconds_remaining = int(remaining % 60)
                            percent_complete = int(progress_info["progress"] * 100)

                            # Format message with encoding speed if available
                            encoding_speed = progress_info.get("encoding_speed", 0)
                            speed_text = (
                                f" at {encoding_speed:.2f}x speed"
                                if encoding_speed > 0
                                else ""
                            )

                            message = f"Creating proxy: {percent_complete}% complete{speed_text}"
                            message += f" (approx. {minutes_remaining}m {seconds_remaining}s remaining)"
                            progress_placeholder.text(message)

                        if progress_callback:
                            progress_callback(progress_info["progress"])

                        # Update progress tracking for polling (Streamlit only)
                        if in_streamlit:
                            st.session_state.proxy_last_progress_time = time.time()
                            st.session_state.proxy_last_progress_value = progress_info[
                                "progress"
                            ]

                        progress_queue.task_done()
                    except Exception:
                        # No progress update available, sleep briefly
                        time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error monitoring progress: {str(e)}")
            finally:
                # Stop progress monitoring
                stop_event.set()
                if monitor_thread and monitor_thread.is_alive():
                    monitor_thread.join(timeout=1.0)

            # Wait for process to complete
            logger.info("Waiting for ffmpeg process to complete...")
            return_code = process.wait()
            logger.info(f"ffmpeg process completed with return code {return_code}")

            if return_code != 0:
                logger.error(f"ffmpeg process failed with return code {return_code}")
                if progress_placeholder and in_streamlit:
                    progress_placeholder.error(
                        f"Failed to create proxy video (error code {return_code})"
                    )
                return None

        # Verify the proxy file exists and is valid
        if os.path.exists(proxy_path):
            proxy_size = os.path.getsize(proxy_path) / (1024 * 1024)
            logger.info(
                f"Proxy created successfully: {proxy_path} ({proxy_size:.2f} MB)"
            )

            # Final progress update
            if progress_placeholder and progress_bar and in_streamlit:
                progress_bar.progress(1.0)
                progress_placeholder.success("Proxy video created successfully!")

            # If this is part of batch processing in Streamlit context, mark as completed
            if in_streamlit:
                try:
                    if (
                        st.session_state.get("proxy_generation_active", False)
                        and st.session_state.get("proxy_current_video") == source_path
                    ):
                        st.session_state.proxy_completed_videos.append(source_path)
                        move_to_next_proxy_video()
                except:
                    pass  # Error accessing session state

            return str(proxy_path)
        else:
            logger.error("Failed to create proxy video - output file does not exist")
            if progress_placeholder and in_streamlit:
                progress_placeholder.error("Failed to create proxy video")
            return None

    except Exception as e:
        logger.exception(f"Error creating proxy: {str(e)}")
        if progress_placeholder and in_streamlit:
            progress_placeholder.error(f"Error creating proxy: {str(e)}")

        # If this is part of batch processing in Streamlit context, mark as failed
        if in_streamlit:
            try:
                if (
                    st.session_state.get("proxy_generation_active", False)
                    and st.session_state.get("proxy_current_video") == source_path
                ):
                    st.session_state.proxy_failed_videos.append(source_path)
                    move_to_next_proxy_video()
            except:
                pass  # Error accessing session state

        return None


def move_to_next_proxy_video():
    """Move to the next video in the proxy generation queue"""
    try:
        import streamlit as st

        logger.info("Moving to next video in proxy queue")

        # Check if we need to increment or if we're already at the end
        if (
            "proxy_current_index" not in st.session_state
            or "proxy_videos_to_process" not in st.session_state
        ):
            logger.warning(
                "No proxy queue found in session state - batch processing may not be active"
            )
            return

        st.session_state.proxy_current_index += 1
        current_index = st.session_state.proxy_current_index
        total_videos = len(st.session_state.proxy_videos_to_process)

        logger.info(f"Next video index: {current_index} of {total_videos}")

        # Reset progress tracking for the next video
        if "proxy_last_progress_time" in st.session_state:
            del st.session_state.proxy_last_progress_time
        if "proxy_last_progress_value" in st.session_state:
            del st.session_state.proxy_last_progress_value

        if current_index < total_videos:
            # Move to the next video
            st.session_state.proxy_current_video = (
                st.session_state.proxy_videos_to_process[current_index]
            )

            # Set timestamp for when this video started processing
            st.session_state.proxy_process_start_time = time.time()

            logger.info(
                f"Moving to next video: {os.path.basename(st.session_state.proxy_current_video)}"
            )

            # Force a rerun to start processing the next video
            logger.info("Triggering Streamlit rerun to process next video")
            st.rerun()
        else:
            # We've completed all videos
            logger.info("All videos processed - marking batch process as complete")
            st.session_state.proxy_generation_active = False

            # Force a final rerun to update the UI
            logger.info("Triggering final Streamlit rerun to update UI")
            st.rerun()
    except Exception as e:
        # Gracefully handle errors and non-Streamlit contexts
        logger.warning(f"Error in move_to_next_proxy_video: {str(e)}")
        # In CLI context, we just continue without rerunning


def check_proxy_progress():
    """
    Check if the current proxy generation process is stuck and advance if needed.
    This should be called periodically by the display function.
    """
    # Only check if proxy generation is active and we have a current video
    if not st.session_state.get(
        "proxy_generation_active", False
    ) or not st.session_state.get("proxy_current_video"):
        return False

    current_time = time.time()

    # Initialize tracking variables if they don't exist
    if "proxy_last_progress_time" not in st.session_state:
        st.session_state.proxy_last_progress_time = current_time
        st.session_state.proxy_last_progress_value = 0.0
        return False

    # Get current progress value
    current_progress = st.session_state.get("proxy_current_progress", 0.0)

    # Check if we're in a calibration process
    in_calibration = "proxy_calibration_stage" in st.session_state
    calibration_stage = st.session_state.get("proxy_calibration_stage", "")
    calibration_timeout = 300  # 5 minutes max for calibration

    # If we're in calibration, use a different timeout strategy
    if in_calibration:
        calibration_start_time = st.session_state.get(
            "proxy_calibration_start_time", current_time
        )
        calibration_duration = current_time - calibration_start_time

        # Log detailed calibration status
        logger.info(
            f"Calibration in progress. Stage: {calibration_stage}, Duration: {calibration_duration:.1f}s"
        )

        # If calibration is completed or has been running too long
        if (
            calibration_stage == "completed"
            or calibration_duration > calibration_timeout
        ):
            if calibration_duration > calibration_timeout:
                logger.warning(
                    f"Calibration timeout exceeded ({calibration_timeout}s). Forcing progress."
                )
            else:
                logger.info(
                    "Calibration completed, checking if we need to move to next video"
                )

            # Check if the proxy file exists for the current video
            config_manager = st.session_state.config_manager
            current_video = st.session_state.proxy_current_video
            proxy_path = config_manager.get_proxy_path(Path(current_video))

            if proxy_path.exists():
                logger.info(
                    f"Proxy file exists for current video. Moving to next video."
                )
                # Add to completed videos if not already there
                if current_video not in st.session_state.proxy_completed_videos:
                    st.session_state.proxy_completed_videos.append(current_video)
                # Move to next video
                move_to_next_proxy_video()
                return True

    # Standard progress stall detection
    progress_timeout = (
        120 if in_calibration else 60
    )  # Longer timeout during calibration
    time_since_last_progress = current_time - st.session_state.proxy_last_progress_time

    # If progress has changed, update the last progress time
    if (
        abs(current_progress - st.session_state.proxy_last_progress_value) > 0.01
    ):  # 1% change
        st.session_state.proxy_last_progress_time = current_time
        st.session_state.proxy_last_progress_value = current_progress
        return False

    # If too much time has passed without progress, consider it stalled
    if time_since_last_progress > progress_timeout:
        if in_calibration:
            logger.warning(
                f"Calibration for {os.path.basename(st.session_state.proxy_current_video)} appears stalled (no progress for {progress_timeout}s in stage {calibration_stage}). Moving to next video."
            )
        else:
            logger.warning(
                f"Proxy generation for {os.path.basename(st.session_state.proxy_current_video)} appears stalled (no progress for {progress_timeout}s). Moving to next video."
            )

        # Add current video to failed list if it's not already there
        if (
            st.session_state.proxy_current_video
            not in st.session_state.proxy_failed_videos
        ):
            st.session_state.proxy_failed_videos.append(
                st.session_state.proxy_current_video
            )

        # Move to next video
        move_to_next_proxy_video()
        return True

    # Also check for overall timeout - if a video has been processing too long
    if "proxy_process_start_time" in st.session_state:
        total_process_time = current_time - st.session_state.proxy_process_start_time
        max_process_time = 3600  # 1 hour max for a single video

        if total_process_time > max_process_time:
            logger.warning(
                f"Proxy generation for {os.path.basename(st.session_state.proxy_current_video)} exceeded maximum time ({max_process_time/60}m). Moving to next video."
            )
            # Add current video to failed list if it's not already there
            if (
                st.session_state.proxy_current_video
                not in st.session_state.proxy_failed_videos
            ):
                st.session_state.proxy_failed_videos.append(
                    st.session_state.proxy_current_video
                )
            # Move to next video
            move_to_next_proxy_video()
            return True

    # Force a rerun periodically to ensure UI updates during long processes
    if (int(current_time) % 10) == 0:  # Every 10 seconds
        if in_calibration:
            # More frequent updates during calibration
            return True

    # Check if the polling mechanism should trigger a rerun
    if st.session_state.get("proxy_needs_rerun", False):
        st.session_state.proxy_needs_rerun = False
        return True

    return False


def generate_all_proxies(config_manager=None):
    """Generate proxy videos for all videos that don't have one yet"""
    try:
        # Check if proxy generation is already active
        if st.session_state.proxy_generation_active:
            st.warning("Proxy generation is already in progress")
            return

        # Log that the function was called
        logger.info("generate_all_proxies function called")

        if not config_manager:
            config_manager = st.session_state.config_manager

        video_files = config_manager.get_video_files()
        if not video_files:
            st.warning("No video files found to generate proxies for")
            logger.warning("No video files found to generate proxies for")
            return

        # Count videos that need proxies
        videos_without_proxies = [
            v for v in video_files if not proxy_exists_for_video(v, config_manager)
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

        # Initialize polling tracking variables
        st.session_state.proxy_needs_rerun = False
        if "proxy_last_progress_time" in st.session_state:
            del st.session_state.proxy_last_progress_time
        if "proxy_last_progress_value" in st.session_state:
            del st.session_state.proxy_last_progress_value
        st.session_state.proxy_process_start_time = time.time()

        # Start processing the first video
        if videos_without_proxies:
            st.session_state.proxy_current_video = videos_without_proxies[0]

        # Force a rerun to show the progress UI
        st.rerun()

    except Exception as e:
        logger.exception(f"Error in proxy generation: {str(e)}")
        st.error(f"Error in proxy generation: {str(e)}")


def proxy_exists_for_video(video_path, config_manager=None):
    """Check if a proxy video exists for the given video path"""
    if not config_manager:
        config_manager = st.session_state.config_manager

    proxy_path = config_manager.get_proxy_path(Path(video_path))

    # Check if the proxy file exists
    if not proxy_path.exists():
        logger.debug(f"No proxy found for {video_path}, would be at {proxy_path}")
        return False

    # Verify the proxy file is valid
    try:
        verify_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(proxy_path),
        ]

        logger.debug(f"Running verification command: {' '.join(verify_cmd)}")
        # Convert command list to a string for shell=True
        # Ensure proper quoting for paths with spaces
        verify_cmd_str = " ".join(
            (
                f'"{arg}"'
                if " " in str(arg) or "+" in str(arg) or ":" in str(arg)
                else str(arg)
            )
            for arg in verify_cmd
        )
        logger.debug(f"Running shell command: {verify_cmd_str}")

        verify_result = subprocess.run(
            verify_cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            shell=True,
        )

        if verify_result.returncode != 0 or "video" not in verify_result.stdout.strip():
            logger.warning(f"Proxy file exists but is not valid: {proxy_path}")
            logger.warning(f"ffprobe stderr: {verify_result.stderr}")

            # Remove the invalid file
            try:
                os.remove(proxy_path)
                logger.info(f"Removed invalid proxy file: {proxy_path}")
            except Exception as e:
                logger.error(f"Failed to remove invalid proxy file: {str(e)}")

            return False

        logger.debug(f"Proxy exists and is valid for {video_path} at {proxy_path}")
        return True

    except Exception as e:
        logger.warning(f"Could not verify proxy file {proxy_path}: {str(e)}")
        # Assume it's valid if we can't verify
        logger.debug(f"Proxy exists for {video_path} at {proxy_path} (not verified)")
        return True


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

        # Add process monitoring info with last update time
        if "proxy_last_progress_time" in st.session_state:
            time_since_update = time.time() - st.session_state.proxy_last_progress_time
            if time_since_update > 30:  # Show warning if no updates for 30+ seconds
                st.warning(f"⚠️ No progress updates for {int(time_since_update)}s")

        # Check if we need to move to the next video (using polling mechanism)
        needs_rerun = check_proxy_progress()
        if needs_rerun:
            st.rerun()

        # Cancel button
        if st.button("Cancel Proxy Generation"):
            st.session_state.proxy_generation_active = False
            st.rerun()

        return True
    return False


def proxy_progress_callback(progress_info):
    """Update progress information for the current video being processed"""
    try:
        progress = progress_info.get("progress", 0.0)
        remaining = progress_info.get("remaining", 0)
        encoding_speed = progress_info.get("encoding_speed", 0)

        # Update session state with progress info
        st.session_state.proxy_current_progress = progress

        # Track progress updates for polling mechanism
        st.session_state.proxy_last_progress_time = time.time()
        st.session_state.proxy_last_progress_value = progress

        # Store encoding speed if available
        if encoding_speed > 0:
            st.session_state.proxy_encoding_speed = encoding_speed

        # Format the time remaining
        if remaining > 0:
            minutes = int(remaining / 60)
            seconds = int(remaining % 60)
            st.session_state.proxy_time_remaining = f"{minutes}m {seconds}s"
        else:
            st.session_state.proxy_time_remaining = "Calculating..."
    except Exception as e:
        # Silently ignore all errors in this callback, especially Streamlit context errors
        # These are expected when running in a background thread
        pass


def cleanup_proxy_files(config_manager=None):
    """Clean up proxy video files

    Returns:
        bool: True if cleanup was performed, False otherwise
    """
    try:
        # Get proxy directory from config manager
        if not config_manager:
            config_manager = st.session_state.config_manager

        # Get both proxy directories
        proxy_raw_dir = config_manager.proxy_raw_dir
        proxy_clipped_dir = config_manager.proxy_clipped_dir

        # Display information about the proxy directories
        if not proxy_raw_dir.exists() and not proxy_clipped_dir.exists():
            st.info("No proxy directories exist")
            return False

        # Count proxy files in both directories
        raw_files = (
            list(proxy_raw_dir.glob("**/*.mp4")) if proxy_raw_dir.exists() else []
        )
        clipped_files = (
            list(proxy_clipped_dir.glob("**/*.mp4"))
            if proxy_clipped_dir.exists()
            else []
        )

        raw_count = len(raw_files)
        clipped_count = len(clipped_files)
        total_count = raw_count + clipped_count

        if total_count == 0:
            st.info("No proxy files to clean up.")
            return False

        # Initialize session state for cleanup confirmation
        if "cleanup_confirm_state" not in st.session_state:
            st.session_state.cleanup_confirm_state = "initial"

        # Show button with file count breakdown
        if st.button(
            f"Clean Up Proxy Videos ({raw_count} raw, {clipped_count} clipped)",
            key="cleanup_proxy_btn",
        ):
            # Set state to confirmation
            st.session_state.cleanup_confirm_state = "confirm"
            st.rerun()

        # Show confirmation dialog if in confirm state
        if st.session_state.cleanup_confirm_state == "confirm":
            st.warning("⚠️ This will delete all proxy videos. Are you sure?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete All Proxies", key="confirm_cleanup"):
                    deleted_count = 0
                    dir_count = 0

                    # Function to clean up a directory
                    def cleanup_directory(directory):
                        nonlocal deleted_count, dir_count
                        if not directory.exists():
                            return

                        # Delete all files
                        for file in directory.glob("**/*"):
                            if file.is_file():
                                try:
                                    file.unlink()
                                    deleted_count += 1
                                    logger.info(f"Deleted proxy file: {file}")
                                except Exception as e:
                                    logger.error(
                                        f"Error deleting proxy file {file}: {str(e)}"
                                    )

                        # Remove empty directories (except the root directory)
                        for dir_path in sorted(
                            [p for p in directory.glob("**/*") if p.is_dir()],
                            reverse=True,
                        ):
                            try:
                                if dir_path != directory and not any(
                                    dir_path.iterdir()
                                ):
                                    dir_path.rmdir()
                                    dir_count += 1
                                    logger.info(
                                        f"Removed empty proxy directory: {dir_path}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error removing proxy directory {dir_path}: {str(e)}"
                                )

                    # Clean up both directories
                    cleanup_directory(proxy_raw_dir)
                    cleanup_directory(proxy_clipped_dir)

                    # Reset proxy path in session state if it was deleted
                    if (
                        hasattr(st.session_state, "proxy_path")
                        and st.session_state.proxy_path
                    ):
                        if not os.path.exists(st.session_state.proxy_path):
                            st.session_state.proxy_path = None

                    # Reset confirmation state
                    st.session_state.cleanup_confirm_state = "initial"

                    st.success(
                        f"Cleanup complete: {deleted_count} files and {dir_count} empty directories removed."
                    )
                    return True

            with col2:
                if st.button("Cancel", key="cancel_cleanup"):
                    # Reset confirmation state
                    st.session_state.cleanup_confirm_state = "initial"
                    st.info("Cleanup cancelled.")
                    st.rerun()
                    return False

        return False
    except Exception as e:
        logger.error(f"Error cleaning up proxy files: {str(e)}")
        st.error(f"Error cleaning up proxy files: {str(e)}")
        return False


def create_clip_preview(
    source_path,
    clip_name,
    start_frame,
    end_frame,
    crop_region=None,
    progress_placeholder=None,
    config_manager=None,
    crop_keyframes=None,
    crop_keyframes_proxy=None,
):
    """Create a preview video for a clip using the proxy footage and crop_keyframes_proxy values.

    Args:
        source_path: Path to the source video (proxy path)
        clip_name: Name of the clip
        start_frame: Starting frame number
        end_frame: Ending frame number
        crop_region: Optional static crop region (x, y, width, height)
        progress_placeholder: Streamlit placeholder for progress updates
        config_manager: ConfigManager instance
        crop_keyframes: Optional keyframes for dynamic cropping (not used)
        crop_keyframes_proxy: Optional keyframes for dynamic cropping on proxy video

    Returns:
        Path to the preview video or None if creation failed
    """
    try:
        if not config_manager:
            config_manager = st.session_state.config_manager

        # Convert source path to Path object
        if isinstance(source_path, str):
            source_path = Path(source_path)

        # Get preview path maintaining folder structure
        preview_path = config_manager.get_clip_preview_path(source_path, clip_name)
        logger.info(f"Preview path: {preview_path}")

        # Ensure the parent directory exists
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created preview directory: {preview_path.parent}")

        # Get video FPS for timestamp calculation
        fps_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ]

        fps = 30.0  # Default fallback
        try:
            fps_output = subprocess.check_output(fps_cmd).decode("utf-8").strip()
            if fps_output:
                # Parse fraction format (e.g., "30000/1001")
                if "/" in fps_output:
                    num, den = fps_output.split("/")
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_output)
        except Exception as e:
            logger.error(f"Error getting FPS: {str(e)}")

        logger.info(f"Video FPS: {fps}")

        # Calculate timestamps for the start and end frames
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        logger.info(
            f"Start time: {start_time}s, Duration: {duration}s for frames {start_frame} to {end_frame}"
        )

        # Get proxy settings
        proxy_settings = config_manager.get_proxy_settings()
        logger.info(f"Proxy settings: {proxy_settings}")

        # Build FFmpeg command for creating a preview clip
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss",
            str(start_time),  # Start time
            "-t",
            str(duration),  # Duration
            "-i",
            str(source_path),  # Input file
            "-c:v",
            "libx264",  # Video codec
            "-preset",
            "fast",  # Faster encoding
            "-crf",
            str(proxy_settings["quality"]),  # Quality factor from settings
        ]

        # Add video filters
        vf_filters = []

        # Process crop keyframes if provided
        if crop_keyframes_proxy:
            # Filter out keyframes outside the clip range and sort by frame number
            valid_keyframes = {
                k: v
                for k, v in crop_keyframes_proxy.items()
                if start_frame <= int(k) <= end_frame
            }

            # If no valid keyframes in range, use all keyframes but adjust timings
            if not valid_keyframes:
                logger.warning(
                    f"No keyframes in clip range {start_frame}-{end_frame}. Using all available keyframes."
                )
                valid_keyframes = crop_keyframes_proxy

            sorted_keyframes = sorted([(int(k), v) for k, v in valid_keyframes.items()])

            # Debug log the keyframes we're using
            logger.info(f"Using {len(sorted_keyframes)} crop keyframes for preview")
            for frame_num, crop_vals in sorted_keyframes:
                logger.info(f"Keyframe {frame_num}: {crop_vals}")

            # Handle case with no keyframes
            if not sorted_keyframes and crop_region:
                logger.warning(
                    "No keyframes available, falling back to static crop region"
                )
                x, y, width, height = crop_region
                vf_filters.append(f"crop={width}:{height}:{x}:{y}")
            elif not sorted_keyframes:
                logger.warning(
                    "No keyframes available and no static crop region. Skipping crop."
                )
            else:
                # Get the constant width and height from first keyframe
                _, first_crop = sorted_keyframes[0]
                width = first_crop[2]  # Width is the 3rd value
                height = first_crop[3]  # Height is the 4th value

                logger.info(f"Using crop dimensions width={width}, height={height}")

                # Build expressions for x and y positions
                expressions = []
                expressions.append(f"w='{width}'")
                expressions.append(f"h='{height}'")

                # Build x position expression (interpolating)
                x_expr = []
                for i in range(len(sorted_keyframes)):
                    curr_frame, curr_crop = sorted_keyframes[i]
                    curr_t = max(0, (curr_frame - start_frame) / fps)
                    curr_x = curr_crop[0]  # X is the 1st value

                    logger.info(
                        f"Keyframe {curr_frame} maps to time {curr_t:.3f}s with x={curr_x}"
                    )

                    if i == 0:
                        # Before first keyframe
                        x_expr.append(f"if(lt(t,{curr_t}),{curr_x}")
                    if i < len(sorted_keyframes) - 1:
                        # Between keyframes
                        next_frame, next_crop = sorted_keyframes[i + 1]
                        next_t = max(0, (next_frame - start_frame) / fps)
                        next_x = next_crop[0]

                        # Skip keyframes with identical timestamps
                        if abs(next_t - curr_t) < 0.001:  # If less than 1ms apart
                            logger.warning(
                                f"Skipping keyframe at {next_frame} because timestamp {next_t:.3f}s is too close to previous {curr_t:.3f}s"
                            )
                            continue

                        logger.info(
                            f"Interpolating x from {curr_x} to {next_x} between {curr_t:.3f}s and {next_t:.3f}s"
                        )
                        x_expr.append(
                            f",if(between(t,{curr_t},{next_t}),{curr_x}+({next_x}-{curr_x})*(0.5-0.5*cos(PI*(t-{curr_t})/({next_t}-{curr_t})))"
                        )

                # After last keyframe
                last_frame, last_crop = sorted_keyframes[-1]
                last_x = last_crop[0]
                x_expr.append(f",{last_x}")
                x_expr.append(")" * len(sorted_keyframes))
                x_expr_final = f"x='{''.join(x_expr)}'"
                expressions.append(x_expr_final)
                logger.info(f"X expression: {x_expr_final}")

                # Build y position expression (interpolating)
                y_expr = []
                for i in range(len(sorted_keyframes)):
                    curr_frame, curr_crop = sorted_keyframes[i]
                    curr_t = max(0, (curr_frame - start_frame) / fps)
                    curr_y = curr_crop[1]  # Y is the 2nd value

                    logger.info(
                        f"Keyframe {curr_frame} maps to time {curr_t:.3f}s with y={curr_y}"
                    )

                    if i == 0:
                        # Before first keyframe
                        y_expr.append(f"if(lt(t,{curr_t}),{curr_y}")
                    if i < len(sorted_keyframes) - 1:
                        # Between keyframes
                        next_frame, next_crop = sorted_keyframes[i + 1]
                        next_t = max(0, (next_frame - start_frame) / fps)
                        next_y = next_crop[1]

                        # Skip keyframes with identical timestamps
                        if abs(next_t - curr_t) < 0.001:  # If less than 1ms apart
                            logger.warning(
                                f"Skipping keyframe at {next_frame} because timestamp {next_t:.3f}s is too close to previous {curr_t:.3f}s"
                            )
                            continue

                        logger.info(
                            f"Interpolating y from {curr_y} to {next_y} between {curr_t:.3f}s and {next_t:.3f}s"
                        )
                        y_expr.append(
                            f",if(between(t,{curr_t},{next_t}),{curr_y}+({next_y}-{curr_y})*(0.5-0.5*cos(PI*(t-{curr_t})/({next_t}-{curr_t})))"
                        )

                # After last keyframe
                last_frame, last_crop = sorted_keyframes[-1]
                last_y = last_crop[1]
                y_expr.append(f",{last_y}")
                y_expr.append(")" * len(sorted_keyframes))
                y_expr_final = f"y='{''.join(y_expr)}'"
                expressions.append(y_expr_final)
                logger.info(f"Y expression: {y_expr_final}")

                # Handle special case: if all keyframes have the same frame number
                if len({frame for frame, _ in sorted_keyframes}) == 1:
                    logger.info("Only one unique keyframe, using static crop")
                    x, y, width, height = first_crop
                    vf_filters.append(f"crop={width}:{height}:{x}:{y}")
                else:
                    # Create the dynamic crop filter
                    crop_filter = f"crop={':'.join(expressions)}"
                    vf_filters.append(crop_filter)
                    logger.info(f"Using dynamic crop filter: {crop_filter}")

        elif crop_region:
            x, y, width, height = crop_region
            vf_filters.append(f"crop={width}:{height}:{x}:{y}")
            logger.info(f"Using static crop region: {x}, {y}, {width}, {height}")

        # Add scale filter
        vf_filters.append(f"scale={proxy_settings['width']}:-2")

        # Combine filters
        if vf_filters:
            vf_string = ",".join(vf_filters)
            cmd.extend(["-vf", vf_string])
            logger.info(f"Video filters: {vf_string}")

        # Add video codec and quality settings
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-crf",
                "18",  # Lower CRF for higher quality
                "-preset",
                "veryslow",  # Better compression
                "-pix_fmt",
                "yuv420p",  # Explicit pixel format
                "-color_range",
                "tv",  # Explicit color range
            ]
        )

        # Add output path
        cmd.append(str(preview_path))

        # Convert command to string with proper quoting
        cmd_str = " ".join(
            f'"{arg}"' if " " in str(arg) or ":" in str(arg) else str(arg)
            for arg in cmd
        )
        logger.info(f"Running ffmpeg command: {cmd_str}")

        # Run ffmpeg command
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
        )

        # Initialize progress tracking
        progress_queue = Queue()
        stop_event = threading.Event()
        monitor_thread = None

        # Monitor progress
        try:
            if duration > 0:
                monitor_thread = threading.Thread(
                    target=monitor_progress,
                    args=(process, duration, progress_queue, stop_event),
                )
                monitor_thread.daemon = True
                monitor_thread.start()

                # Update progress via callback
                while process.poll() is None:
                    try:
                        progress_info = progress_queue.get_nowait()
                        if progress_callback:
                            progress_callback(progress_info["progress"])
                        progress_queue.task_done()
                    except Exception:
                        # No progress update available, sleep briefly
                        time.sleep(0.1)
        except Exception as e:
            logger.warning(f"Error monitoring progress: {str(e)}")
        finally:
            # Stop progress monitoring
            stop_event.set()
            if monitor_thread and monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)

        # Wait for process to complete
        logger.info("Waiting for ffmpeg process to complete...")
        return_code = process.wait()
        logger.info(f"ffmpeg process completed with return code {return_code}")

        if return_code != 0:
            logger.error(f"ffmpeg process failed with return code {return_code}")
            return None

        # Verify the exported file exists and is valid
        if os.path.exists(preview_path):
            preview_size = os.path.getsize(preview_path) / (1024 * 1024)
            logger.info(
                f"Preview created successfully: {preview_path} ({preview_size:.2f} MB)"
            )
            return str(preview_path)
        else:
            logger.error("Preview file was not created")
            return None

    except Exception as e:
        logger.error(f"Error in preview creation: {str(e)}")
        if progress_placeholder:
            progress_placeholder.error(f"Error creating preview: {str(e)}")
        return None


def interpolate_crop_keyframes(current_frame, keyframes):
    """
    Interpolate crop region between keyframes for the current frame

    Args:
        current_frame: Current frame number
        keyframes: Dictionary of {frame: crop_region} keyframes

    Returns:
        Interpolated crop region (x, y, width, height) or None
    """
    if not keyframes or len(keyframes) == 0:
        return None

    # Convert keyframes from dict to sorted list
    sorted_keyframes = sorted([(int(f), keyframes[f]) for f in keyframes])

    # If current frame is before first keyframe, use first keyframe
    if current_frame <= sorted_keyframes[0][0]:
        return sorted_keyframes[0][1]

    # If current frame is after last keyframe, use last keyframe
    if current_frame >= sorted_keyframes[-1][0]:
        return sorted_keyframes[-1][1]

    # Find the two keyframes to interpolate between
    prev_kf = None
    next_kf = None

    for i, (frame, crop) in enumerate(sorted_keyframes):
        if frame <= current_frame:
            prev_kf = (frame, crop)
        if frame > current_frame and next_kf is None:
            next_kf = (frame, crop)
            break

    if prev_kf is None or next_kf is None:
        return sorted_keyframes[0][1]  # Fallback

    # Calculate interpolation factor
    prev_frame, prev_crop = prev_kf
    next_frame, next_crop = next_kf
    factor = (current_frame - prev_frame) / (next_frame - prev_frame)

    # Interpolate crop region
    x = int(prev_crop[0] + factor * (next_crop[0] - prev_crop[0]))
    y = int(prev_crop[1] + factor * (next_crop[1] - prev_crop[1]))
    w = int(prev_crop[2] + factor * (next_crop[2] - prev_crop[2]))
    h = int(prev_crop[3] + factor * (next_crop[3] - prev_crop[3]))

    return (x, y, w, h)


def generate_crop_keyframe_filters(
    crop_keyframes, start_frame, end_frame, calibration_offset=0
):
    """Generate crop filter with keyframes for dynamic cropping"""
    # ... existing implementation ...


def calculate_wider_crop(original_crop, factor, frame_dimensions):
    """
    Calculate a wider crop region centered around the original crop,
    with intelligent edge boundary handling.

    Args:
        original_crop: Original crop region (x, y, width, height)
        factor: Multiplier for crop size (1.5 = 50% larger)
        frame_dimensions: (width, height) of the source frame

    Returns:
        New crop region (x, y, width, height) or None if original_crop is None
    """
    if original_crop is None:
        return None

    x, y, width, height = original_crop
    frame_width, frame_height = frame_dimensions

    # Calculate new dimensions - ensure they don't exceed frame dimensions
    new_width = min(int(width * factor), frame_width)
    new_height = min(int(height * factor), frame_height)

    # Calculate the ideal centered position
    ideal_x = x - (new_width - width) // 2
    ideal_y = y - (new_height - height) // 2

    # Adjust position to stay within frame boundaries
    final_x = max(0, min(ideal_x, frame_width - new_width))
    final_y = max(0, min(ideal_y, frame_height - new_height))

    return (final_x, final_y, new_width, new_height)


def get_video_duration(source_path):
    """Get the duration of a video file in seconds using ffprobe.

    Args:
        source_path: Path to the video file

    Returns:
        Duration in seconds or None if duration could not be determined
    """
    try:
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

        logger.info(f"Running duration command: {' '.join(duration_cmd)}")
        # Convert command list to a string for shell=True
        # Ensure proper quoting for paths with spaces
        duration_cmd_str = " ".join(
            (
                f'"{arg}"'
                if " " in str(arg) or "+" in str(arg) or ":" in str(arg)
                else str(arg)
            )
            for arg in duration_cmd
        )
        logger.info(f"Running shell command: {duration_cmd_str}")

        duration_result = subprocess.run(
            duration_cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )
        if duration_result.returncode != 0:
            logger.warning(
                f"ffprobe returned non-zero exit code: {duration_result.returncode}"
            )
            logger.warning(f"ffprobe stderr: {duration_result.stderr}")
            return None
        else:
            duration = float(duration_result.stdout.strip())
            logger.info(f"Video duration: {duration:.2f} seconds")
            return duration
    except Exception as e:
        logger.warning(f"Could not determine video duration: {str(e)}")
        return None


def export_clip(
    source_path,
    clip_name,
    start_frame,
    end_frame,
    crop_region=None,
    crop_keyframes=None,
    output_resolution="1080p",
    cv_optimized=False,
    progress_callback=None,
    config_manager=None,
    clean_up_existing=True,
):
    """
    Export a clip from a source video with optional cropping.

    Args:
        source_path: Path to the source video
        clip_name: Name for the exported clip
        start_frame: Starting frame number
        end_frame: Ending frame number
        crop_region: Optional static crop region (x, y, width, height)
        crop_keyframes: Optional keyframes for dynamic cropping
        output_resolution: Target resolution for export (e.g., "1080p")
        cv_optimized: Whether to use CV-optimized encoding
        progress_callback: Callback function for progress updates
        config_manager: ConfigManager instance
        clean_up_existing: Whether to delete existing export files before creating a new one

    Returns:
        Path to the exported clip or None if export failed
    """
    try:
        if not config_manager:
            try:
                config_manager = st.session_state.config_manager
            except:
                logger.warning("No config manager provided for export")
                return None

        # Convert source path to Path object if it's a string
        if isinstance(source_path, str):
            source_path = Path(source_path)

        # Get export directory from config - use configured clips directory
        export_dir = config_manager.clips_dir

        # Use FFV1 codec for CV-optimized export
        if cv_optimized:
            export_dir = export_dir / "ffv1"
            file_ext = ".mkv"
        else:
            export_dir = export_dir / "h264"
            file_ext = ".mp4"

        # Create subdirectories based on source path structure
        video_dirname = os.path.basename(os.path.dirname(source_path))
        parent_dirname = os.path.basename(os.path.dirname(os.path.dirname(source_path)))

        # Create camera type and session directories
        export_dir = export_dir / parent_dirname / video_dirname
        export_dir.mkdir(parents=True, exist_ok=True)

        # Create export file path
        export_path = (
            export_dir
            / f"{os.path.splitext(os.path.basename(source_path))[0]}_{clip_name}{file_ext}"
        )
        logger.info(f"Export path: {export_path}")

        # Clean up existing files if requested
        if clean_up_existing and export_path.exists():
            logger.info(f"Deleting existing export file: {export_path}")
            try:
                os.remove(export_path)
            except Exception as e:
                logger.warning(f"Failed to delete existing file {export_path}: {e}")
        else:
            logger.info(
                f"File will be overwritten by ffmpeg if it exists: {export_path}"
            )

        # =============================
        # CALIBRATION INTEGRATION STEP
        # =============================

        # Check if we should skip calibration (using pre-calibrated footage)
        if calibration_service.should_skip_calibration(config_manager):
            logger.info(
                "Using pre-calibrated footage, skipping calibration during export"
            )
            actual_source_path = source_path
        else:
            # Check if this video needs calibration
            camera_type = calibration_service.get_camera_type_from_path(
                source_path, config_manager
            )

            if camera_type and camera_type != "none":
                logger.info(f"Video requires {camera_type} calibration before export")

                # Load calibration parameters
                camera_matrix, dist_coeffs, _ = calibration_service.load_calibration(
                    camera_type, config_manager=config_manager, video_path=source_path
                )

                if camera_matrix is not None and dist_coeffs is not None:
                    logger.info(
                        "Applying calibration to full-frame source before cropping/export"
                    )

                    # Create a temporary file for the calibrated video
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp4", delete=False
                    ) as temp_file:
                        calibrated_temp_path = temp_file.name

                    # Apply calibration to the full source video first
                    calibration_success = (
                        calibration_service.apply_calibration_to_video(
                            input_path=source_path,
                            output_path=calibrated_temp_path,
                            camera_type=camera_type,
                            camera_matrix=camera_matrix,
                            dist_coeffs=dist_coeffs,
                            quality_preset=(
                                "lossless" if cv_optimized else "high"
                            ),  # Use lossless for CV, high for regular
                            progress_callback=lambda p: (
                                progress_callback(p * 0.5)
                                if progress_callback
                                else None
                            ),  # First 50% of progress
                            config_manager=config_manager,
                        )
                    )

                    if calibration_success:
                        logger.info(
                            "Calibration applied successfully, proceeding with calibrated source"
                        )
                        actual_source_path = calibrated_temp_path
                    else:
                        logger.error(
                            "Failed to apply calibration, using original source"
                        )
                        actual_source_path = source_path
                        # Clean up temp file
                        try:
                            os.unlink(calibrated_temp_path)
                        except:
                            pass
                else:
                    logger.warning(
                        f"Missing calibration data for camera type: {camera_type}, proceeding without calibration"
                    )
                    actual_source_path = source_path
            else:
                logger.info("No calibration required for this video")
                actual_source_path = source_path

        # =============================
        # END CALIBRATION INTEGRATION
        # =============================

        # IMPORTANT: All cleanup of previous exports is handled at the clip level in process_clip
        # We should NOT delete any files here except the one being directly overwritten

        # Get video FPS from the actual source (calibrated or original)
        fps_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(actual_source_path),
        ]

        fps = 30.0  # Default fallback
        try:
            fps_output = subprocess.check_output(fps_cmd).decode("utf-8").strip()
            if fps_output:
                # Parse fraction format (e.g., "30000/1001")
                if "/" in fps_output:
                    num, den = fps_output.split("/")
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_output)
        except Exception as e:
            logger.error(f"Error getting FPS: {str(e)}")

        logger.info(f"Video FPS: {fps}")

        # Calculate timestamps for the start and end frames
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        logger.info(
            f"Start time: {start_time}s, Duration: {duration}s for frames {start_frame} to {end_frame}"
        )

        # Build FFmpeg command using the actual source (calibrated if needed)
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss",
            str(start_time),  # Start time
            "-t",
            str(duration),  # Duration
            "-i",
            str(actual_source_path),  # Input file (calibrated or original)
        ]

        # Add video filters
        vf_filters = []

        # Handle crop settings
        if crop_keyframes and len(crop_keyframes) > 1:
            # Dynamic cropping with keyframes
            logger.info(f"Using dynamic crop with {len(crop_keyframes)} keyframes")

            # Sort keyframes by frame number
            sorted_keyframes = sorted([(int(k), v) for k, v in crop_keyframes.items()])

            # Get the width and height from the first keyframe
            _, first_crop = sorted_keyframes[0]
            crop_width = first_crop[2]
            crop_height = first_crop[3]

            # Build expressions for x and y positions
            expressions = []
            expressions.append(f"w='{crop_width}'")
            expressions.append(f"h='{crop_height}'")

            # Build x position expression (interpolating)
            x_expr = []
            for i in range(len(sorted_keyframes)):
                curr_frame, curr_crop = sorted_keyframes[i]
                # Convert frame number to timestamp relative to the start point
                curr_t = max(0, (curr_frame - start_frame) / fps)
                curr_x = curr_crop[0]  # X is the 1st value in crop tuple

                logger.info(
                    f"Keyframe {curr_frame} maps to time {curr_t:.3f}s with x={curr_x}"
                )

                if i == 0:
                    # Before first keyframe
                    x_expr.append(f"if(lt(t,{curr_t}),{curr_x}")
                if i < len(sorted_keyframes) - 1:
                    # Between keyframes
                    next_frame, next_crop = sorted_keyframes[i + 1]
                    next_t = max(0, (next_frame - start_frame) / fps)
                    next_x = next_crop[0]

                    # Skip identical timestamps
                    if abs(next_t - curr_t) < 0.001:
                        continue

                    # Use smooth cosine interpolation
                    x_expr.append(
                        f",if(between(t,{curr_t},{next_t}),{curr_x}+({next_x}-{curr_x})*(0.5-0.5*cos(PI*(t-{curr_t})/({next_t}-{curr_t})))"
                    )

            # After last keyframe
            last_frame, last_crop = sorted_keyframes[-1]
            last_t = max(0, (last_frame - start_frame) / fps)
            last_x = last_crop[0]
            x_expr.append(f",{last_x}")
            x_expr.append(")" * len(sorted_keyframes))
            x_expr_final = f"x='{''.join(x_expr)}'"
            expressions.append(x_expr_final)

            # Build y position expression (interpolating)
            y_expr = []
            for i in range(len(sorted_keyframes)):
                curr_frame, curr_crop = sorted_keyframes[i]
                curr_t = max(0, (curr_frame - start_frame) / fps)
                curr_y = curr_crop[1]  # Y is the 2nd value

                if i == 0:
                    # Before first keyframe
                    y_expr.append(f"if(lt(t,{curr_t}),{curr_y}")
                if i < len(sorted_keyframes) - 1:
                    # Between keyframes
                    next_frame, next_crop = sorted_keyframes[i + 1]
                    next_t = max(0, (next_frame - start_frame) / fps)
                    next_y = next_crop[1]

                    # Skip identical timestamps
                    if abs(next_t - curr_t) < 0.001:
                        continue

                    # Use smooth cosine interpolation
                    y_expr.append(
                        f",if(between(t,{curr_t},{next_t}),{curr_y}+({next_y}-{curr_y})*(0.5-0.5*cos(PI*(t-{curr_t})/({next_t}-{curr_t})))"
                    )

            # After last keyframe
            last_y = last_crop[1]
            y_expr.append(f",{last_y}")
            y_expr.append(")" * len(sorted_keyframes))
            y_expr_final = f"y='{''.join(y_expr)}'"
            expressions.append(y_expr_final)

            # Create the dynamic crop filter
            crop_filter = f"crop={':'.join(expressions)}"
            vf_filters.append(crop_filter)
            logger.info(f"Dynamic crop filter: {crop_filter}")

        elif crop_keyframes and len(crop_keyframes) == 1:
            # Just one keyframe, use static crop
            frame_num = list(crop_keyframes.keys())[0]
            crop = crop_keyframes[frame_num]
            x, y, width, height = crop
            vf_filters.append(f"crop={width}:{height}:{x}:{y}")
            logger.info(
                f"Static crop from single keyframe: {x}, {y}, {width}, {height}"
            )
        elif crop_region:
            # Static crop region provided directly
            x, y, width, height = crop_region
            vf_filters.append(f"crop={width}:{height}:{x}:{y}")
            logger.info(f"Static crop region: {x}, {y}, {width}, {height}")

        # Add output resolution scaling
        if output_resolution == "1080p":
            target_height = 1080
        elif output_resolution == "720p":
            target_height = 720
        else:
            target_height = 1080  # Default to 1080p

        vf_filters.append(f"scale=-2:{target_height}")

        # Combine filters
        if vf_filters:
            vf_string = ",".join(vf_filters)
            cmd.extend(["-vf", vf_string])
            logger.info(f"Video filters: {vf_string}")

        # Add codec settings based on export type
        if cv_optimized:
            # FFV1 lossless codec for CV applications
            cmd.extend(
                [
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",
                    "-g",
                    "1",  # All frames are keyframes for precise seeking
                    "-context",
                    "1",  # Better error recovery
                    "-slices",
                    "24",  # Good for multithreading
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        else:
            # H.264 for regular exports
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",  # High quality
                    "-preset",
                    "slow",  # Better compression
                    "-pix_fmt",
                    "yuv420p",
                ]
            )

        # Add audio codec
        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )

        # Add output path
        cmd.append(str(export_path))

        # Convert command to string with proper quoting
        cmd_str = " ".join(
            f'"{arg}"' if " " in str(arg) or ":" in str(arg) else str(arg)
            for arg in cmd
        )
        logger.info(f"Running ffmpeg command: {cmd_str}")

        # Run ffmpeg command
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
        )

        # Initialize progress tracking
        progress_queue = Queue()
        stop_event = threading.Event()
        monitor_thread = None

        # Monitor progress
        try:
            if duration > 0:
                monitor_thread = threading.Thread(
                    target=monitor_progress,
                    args=(process, duration, progress_queue, stop_event),
                )
                monitor_thread.daemon = True
                monitor_thread.start()

                # Update progress via callback (second 50% of progress if calibration was applied)
                base_progress = 0.5 if actual_source_path != source_path else 0.0
                progress_multiplier = 0.5 if actual_source_path != source_path else 1.0

                while process.poll() is None:
                    try:
                        progress_info = progress_queue.get_nowait()
                        if progress_callback:
                            # Adjust progress based on whether calibration was applied
                            adjusted_progress = base_progress + (
                                progress_info["progress"] * progress_multiplier
                            )
                            progress_callback(adjusted_progress)
                        progress_queue.task_done()
                    except Exception:
                        # No progress update available, sleep briefly
                        time.sleep(0.1)
        except Exception as e:
            logger.warning(f"Error monitoring progress: {str(e)}")
        finally:
            # Stop progress monitoring
            stop_event.set()
            if monitor_thread and monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)

        # Wait for process to complete
        logger.info("Waiting for ffmpeg process to complete...")
        return_code = process.wait()
        logger.info(f"ffmpeg process completed with return code {return_code}")

        # Clean up temporary calibrated file if used
        if actual_source_path != source_path:
            try:
                os.unlink(actual_source_path)
                logger.info("Cleaned up temporary calibrated file")
            except Exception as e:
                logger.warning(f"Failed to delete temporary calibrated file: {e}")

        if return_code != 0:
            logger.error(f"ffmpeg process failed with return code {return_code}")
            return None

        # Verify the exported file exists and is valid
        if os.path.exists(export_path):
            export_size = os.path.getsize(export_path) / (1024 * 1024)
            logger.info(
                f"Export created successfully: {export_path} ({export_size:.2f} MB)"
            )
            return str(export_path)
        else:
            logger.error("Export file was not created")
            return None

    except Exception as e:
        logger.error(f"Error in clip export: {str(e)}")
        return None


def _create_proxy_with_calibration_two_stage(
    source_path,
    proxy_path,
    camera_type,
    camera_matrix,
    dist_coeffs,
    proxy_settings,
    duration,
    progress_placeholder=None,
    progress_callback=None,
    config_manager=None,
):
    """
    Create a proxy video with calibration applied in two stages.

    Args:
        source_path: Path to the source video
        proxy_path: Path where the proxy video should be saved
        camera_type: Type of camera (GP1, GP2, SONY_70, SONY_300)
        camera_matrix: Camera matrix from calibration file
        dist_coeffs: Distortion coefficients from calibration file
        proxy_settings: Proxy video settings from config
        duration: Expected duration of the video in seconds
        progress_placeholder: Streamlit placeholder for progress updates
        progress_callback: Callback function for progress updates
        config_manager: ConfigManager instance

    Returns:
        Path to the proxy video or None if creation failed
    """
    try:
        in_streamlit = is_streamlit_context()

        # Create a temporary file for the calibrated video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            calibrated_path = temp_file.name

        # Stage 1: Apply calibration
        logger.info("Stage 1: Applying calibration")
        if progress_placeholder and in_streamlit:
            progress_placeholder.text("Stage 1/2: Applying calibration...")

        # Use calibration service to apply calibration
        calibration_success = calibration_service.apply_calibration_to_video(
            input_path=source_path,
            output_path=calibrated_path,
            camera_type=camera_type,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            quality_preset="high",  # Use high quality for calibration
            progress_callback=lambda p: _progress_callback_stage1(
                p, progress_placeholder, None, progress_callback, in_streamlit
            ),
            config_manager=config_manager,
        )

        if not calibration_success:
            logger.error("Failed to apply calibration")
            if progress_placeholder and in_streamlit:
                progress_placeholder.error("Failed to apply calibration")
            return None

        # Stage 2: Create proxy from calibrated video
        logger.info("Stage 2: Creating proxy from calibrated video")
        if progress_placeholder and in_streamlit:
            progress_placeholder.text("Stage 2/2: Creating proxy video...")

        # Ensure the parent directory exists
        proxy_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare FFmpeg command for proxy creation
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(calibrated_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(proxy_settings["quality"]),
            "-vf",
            f"scale={proxy_settings['width']}:-2",
            "-c:a",
            "aac",
            "-b:a",
            proxy_settings["audio_bitrate"],
            str(proxy_path),
        ]

        # Convert command list to a string for shell=True
        cmd_str = " ".join(
            (
                f'"{arg}"'
                if " " in str(arg) or "+" in str(arg) or ":" in str(arg)
                else str(arg)
            )
            for arg in cmd
        )

        # Initialize progress tracking
        progress_queue = Queue()
        stop_event = threading.Event()

        # Start FFmpeg process
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
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
                    try:
                        progress_info = progress_queue.get_nowait()
                        # Scale progress to 50-100% for second stage
                        scaled_progress = 0.5 + (progress_info["progress"] * 0.5)

                        if progress_placeholder and in_streamlit:
                            percent_complete = int(scaled_progress * 100)
                            progress_placeholder.text(
                                f"Stage 2/2: Creating proxy: {percent_complete}% complete"
                            )

                        if progress_callback:
                            progress_callback(scaled_progress)

                        # Update progress tracking for polling
                        if in_streamlit:
                            st.session_state.proxy_last_progress_time = time.time()
                            st.session_state.proxy_last_progress_value = scaled_progress

                        progress_queue.task_done()
                    except Exception:
                        # No progress update available, sleep briefly
                        time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error monitoring progress: {str(e)}")
            finally:
                # Stop progress monitoring
                stop_event.set()
                if monitor_thread and monitor_thread.is_alive():
                    monitor_thread.join(timeout=1.0)

        # Wait for process to complete
        return_code = process.wait()

        # Clean up temporary file
        try:
            os.unlink(calibrated_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary calibrated file: {e}")

        if return_code != 0:
            logger.error(f"FFmpeg process failed with return code {return_code}")
            if progress_placeholder and in_streamlit:
                progress_placeholder.error("Failed to create proxy video")
            return None

        # Verify the proxy file exists and is valid
        if os.path.exists(proxy_path):
            proxy_size = os.path.getsize(proxy_path) / (1024 * 1024)
            logger.info(
                f"Proxy created successfully: {proxy_path} ({proxy_size:.2f} MB)"
            )
            return str(proxy_path)
        else:
            logger.error("Failed to create proxy video - output file does not exist")
            if progress_placeholder and in_streamlit:
                progress_placeholder.error("Failed to create proxy video")
            return None

    except Exception as e:
        logger.exception(f"Error in two-stage proxy creation: {str(e)}")
        if progress_placeholder and in_streamlit:
            progress_placeholder.error(f"Error creating proxy: {str(e)}")
        return None
