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

logger = logging.getLogger("clipper.proxy")

# Import calibration service
from src.services import calibration_service


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
        # Initialize progress queue
        progress_queue = Queue()
        stop_event = threading.Event()

        # Get proxy settings from config manager
        if not config_manager:
            config_manager = st.session_state.config_manager

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
                duration = 0
            else:
                duration = float(duration_result.stdout.strip())
                logger.info(f"Video duration: {duration:.2f} seconds")
        except Exception as e:
            logger.warning(f"Could not determine video duration: {str(e)}")
            duration = 0

        # Check if we're using pre-calibrated footage
        if calibration_service.should_skip_calibration(config_manager):
            logger.info("Using pre-calibrated footage - skipping calibration step")
            # Use standard FFmpeg approach for proxy generation
            camera_matrix = None
            dist_coeffs = None
        else:
            # Try to determine camera type from path and load calibration
            camera_type = calibration_service.get_camera_type_from_path(
                source_path, config_manager
            )
            camera_matrix = None
            dist_coeffs = None

            if camera_type:
                try:
                    logger.info(
                        f"Loading calibration parameters for camera type: {camera_type}"
                    )
                    camera_matrix, dist_coeffs, _ = (
                        calibration_service.load_calibration(
                            camera_type,
                            config_manager=config_manager,
                            video_path=source_path,
                        )
                    )
                    if camera_matrix is not None and dist_coeffs is not None:
                        logger.info(
                            f"Successfully loaded calibration parameters for {camera_type}"
                        )
                    else:
                        logger.warning(
                            f"Could not load calibration parameters for {camera_type}"
                        )
                except Exception as e:
                    logger.error(f"Error loading calibration parameters: {e}")
                    camera_matrix = None
                    dist_coeffs = None

        # If calibration parameters were loaded, use the two-stage approach
        if camera_matrix is not None and dist_coeffs is not None:
            logger.info(
                "Using two-stage approach with calibration to create proxy video"
            )

            # Try the two-stage approach first
            proxy_result = _create_proxy_with_calibration_two_stage(
                source_path,
                proxy_path,
                camera_type,
                camera_matrix,
                dist_coeffs,
                proxy_settings,
                duration,
                actual_progress_placeholder,
                progress_callback,
                config_manager,
            )

            # If successful, return the result
            if proxy_result:
                return proxy_result

            # If we get here, the two-stage approach failed
            logger.warning(
                "Two-stage calibration approach failed, falling back to standard FFmpeg proxy creation"
            )
            if actual_progress_placeholder:
                actual_progress_placeholder.warning(
                    "Calibration failed - trying standard approach"
                )
                # Reset progress bar
                progress_bar.progress(0)

            # Fall back to standard approach without calibration
            camera_matrix = None
            dist_coeffs = None

        # If we reach this point, either calibration parameters were not available or there was an error
        # Fall back to the standard FFmpeg approach without calibration
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
            f"scale={proxy_settings['width']}:-2",  # Use width from config (now fixed to 960)
            "-c:a",
            "aac",
            "-b:a",
            proxy_settings["audio_bitrate"],
            str(proxy_path),
        ]

        # Log the command for debugging
        logger.info(f"Starting ffmpeg conversion: {' '.join(cmd)}")

        # Start the conversion process with pipe for stderr
        try:
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

            process = subprocess.Popen(
                cmd_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
                shell=True,
            )
        except Exception as e:
            logger.error(f"Failed to start ffmpeg process: {str(e)}")
            if actual_progress_placeholder:
                actual_progress_placeholder.error(
                    f"Failed to start ffmpeg process: {str(e)}"
                )
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
                    # Check queue for progress updates (non-blocking)
                    try:
                        progress_info = progress_queue.get(block=False)
                        if actual_progress_placeholder:
                            try:
                                progress_bar.progress(progress_info["progress"])
                                remaining = progress_info["remaining"]

                                # Calculate encoding speed (how fast we're processing the video)
                                if "current_time" in progress_info and duration > 0:
                                    percent_complete = int(
                                        progress_info["progress"] * 100
                                    )
                                    minutes_remaining = int(remaining / 60)
                                    seconds_remaining = int(remaining % 60)

                                    # Get encoding speed if available
                                    encoding_speed = progress_info.get(
                                        "encoding_speed", 0
                                    )
                                    speed_text = ""
                                    if encoding_speed > 0:
                                        speed_ratio = encoding_speed
                                        speed_text = f" at {speed_ratio:.2f}x speed"

                                    # Format the message with more accurate time estimate
                                    message = f"Creating proxy video: {percent_complete}% complete{speed_text} "
                                    message += f"(approx. {minutes_remaining}m {seconds_remaining}s remaining)"

                                    actual_progress_placeholder.text(message)
                                else:
                                    actual_progress_placeholder.text(
                                        f"Creating proxy video: {int(progress_info['progress'] * 100)}% complete"
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
                                elapsed_time = current_time - start_time
                                minutes_elapsed = int(elapsed_time / 60)
                                seconds_elapsed = int(elapsed_time % 60)

                                actual_progress_placeholder.text(
                                    f"Creating proxy video... (processing for {minutes_elapsed}m {seconds_elapsed}s)"
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
                return_code = process.wait()
                logger.info(f"ffmpeg process completed with return code {return_code}")

                # Check if process was successful
                if return_code != 0:
                    logger.error(
                        f"ffmpeg process failed with return code {return_code}"
                    )
                    if actual_progress_placeholder:
                        actual_progress_placeholder.error(
                            f"ffmpeg process failed with return code {return_code}"
                        )

                    # If the file was created but is incomplete, remove it
                    if os.path.exists(proxy_path):
                        try:
                            os.remove(proxy_path)
                            logger.info(f"Removed incomplete proxy file: {proxy_path}")
                        except Exception as e:
                            logger.error(
                                f"Failed to remove incomplete proxy file: {str(e)}"
                            )

                    # If this is part of batch processing, mark as failed and move to next video
                    if (
                        st.session_state.proxy_generation_active
                        and st.session_state.proxy_current_video == source_path
                    ):
                        st.session_state.proxy_failed_videos.append(source_path)
                        move_to_next_proxy_video()

                    return None

                # Final progress update
                if actual_progress_placeholder:
                    try:
                        progress_bar.progress(1.0)
                        # Calculate total processing time
                        total_time = time.time() - start_time
                        minutes = int(total_time / 60)
                        seconds = int(total_time % 60)

                        actual_progress_placeholder.text(
                            f"Proxy video created successfully in {minutes}m {seconds}s!"
                        )
                    except Exception:
                        # Ignore Streamlit context errors
                        pass
        else:
            # If duration couldn't be determined, just wait for completion
            logger.info(
                "No duration available, waiting for process without progress updates"
            )
            start_time = time.time()
            return_code = process.wait()
            logger.info(f"ffmpeg process completed with return code {return_code}")

            # Check if process was successful
            if return_code != 0:
                logger.error(f"ffmpeg process failed with return code {return_code}")
                if actual_progress_placeholder:
                    actual_progress_placeholder.error(
                        f"ffmpeg process failed with return code {return_code}"
                    )

                # If the file was created but is incomplete, remove it
                if os.path.exists(proxy_path):
                    try:
                        os.remove(proxy_path)
                        logger.info(f"Removed incomplete proxy file: {proxy_path}")
                    except Exception as e:
                        logger.error(
                            f"Failed to remove incomplete proxy file: {str(e)}"
                        )

                # If this is part of batch processing, mark as failed and move to next video
                if (
                    st.session_state.proxy_generation_active
                    and st.session_state.proxy_current_video == source_path
                ):
                    st.session_state.proxy_failed_videos.append(source_path)
                    move_to_next_proxy_video()

                return None

        if os.path.exists(proxy_path):
            proxy_size = os.path.getsize(proxy_path) / (1024 * 1024)
            logger.info(
                f"Proxy created successfully: {proxy_path} ({proxy_size:.2f} MB)"
            )

            # Verify the proxy file is valid
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

            try:
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
                if (
                    verify_result.returncode != 0
                    or "video" not in verify_result.stdout.strip()
                ):
                    logger.error(f"Created proxy file is not valid: {proxy_path}")
                    logger.error(f"ffprobe stderr: {verify_result.stderr}")
                    if actual_progress_placeholder:
                        actual_progress_placeholder.error(
                            "Created proxy file is not valid. Removing and trying again."
                        )

                    # Remove the invalid file
                    try:
                        os.remove(proxy_path)
                        logger.info(f"Removed invalid proxy file: {proxy_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove invalid proxy file: {str(e)}")

                    # If this is part of batch processing, mark as failed and move to next video
                    if (
                        st.session_state.proxy_generation_active
                        and st.session_state.proxy_current_video == source_path
                    ):
                        st.session_state.proxy_failed_videos.append(source_path)
                        move_to_next_proxy_video()

                    return None

                logger.info(f"Proxy file verified as valid: {proxy_path}")
            except Exception as e:
                logger.warning(f"Could not verify proxy file: {str(e)}")
                # Continue anyway since the file exists

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
    crop_keyframes=None,  # Original keyframes (not used for preview)
    crop_keyframes_proxy=None,  # Proxy-specific keyframes (used for preview)
):
    """
    Create a preview of a clip with optional crop region

    Args:
        source_path: Path to the source video
        clip_name: Name of the clip
        start_frame: Start frame of the clip
        end_frame: End frame of the clip
        crop_region: Optional static crop region (x, y, width, height)
        progress_placeholder: Streamlit placeholder for progress updates
        config_manager: ConfigManager instance
        crop_keyframes: Original keyframes (not used for preview)
        crop_keyframes_proxy: Proxy-specific keyframes (used for preview)

    Returns:
        Path to the preview clip or None if creation failed
    """
    try:
        logger.info(f"Creating preview for {clip_name}")
        logger.info(f"Source: {source_path}")
        logger.info(f"Frames: {start_frame} to {end_frame}")
        logger.info(f"Static crop region: {crop_region}")
        logger.info(f"Crop keyframes proxy: {crop_keyframes_proxy}")

        # Get configurations
        if not config_manager:
            config_manager = st.session_state.config_manager
        proxy_settings = config_manager.get_proxy_settings()
        logger.info(f"Proxy settings: {proxy_settings}")

        # Get preview path
        preview_path = config_manager.get_clip_preview_path(
            Path(source_path), clip_name
        )
        logger.info(f"Preview path: {preview_path}")

        # Check if preview already exists and processing is requested to be skipped
        if preview_path.exists() and proxy_settings.get("skip_existing", False):
            logger.info(f"Preview already exists: {preview_path}")
            if progress_placeholder:
                progress_placeholder.success("Preview file already exists!")
            return str(preview_path)

        # Delete existing preview file if it exists
        if preview_path.exists():
            try:
                preview_path.unlink()
                logger.info(f"Deleted existing preview file: {preview_path}")
            except Exception as e:
                logger.error(f"Error deleting existing preview file: {str(e)}")
                # Continue anyway as ffmpeg will overwrite the file

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

        try:
            fps_output = subprocess.check_output(fps_cmd).decode("utf-8").strip()
            if fps_output:
                # Parse fraction format (e.g., "30000/1001")
                if "/" in fps_output:
                    num, den = fps_output.split("/")
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_output)
            else:
                fps = 30.0  # Default fallback
        except Exception as e:
            logger.error(f"Error getting FPS: {str(e)}")
            fps = 30.0  # Default fallback

        logger.info(f"Video FPS: {fps}")

        # Calculate timestamps for the start and end frames
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        logger.info(
            f"Start time: {start_time}s, Duration: {duration}s for frames {start_frame} to {end_frame}"
        )

        # Check if we're using pre-calibrated footage
        if calibration_service.should_skip_calibration(config_manager):
            logger.info("Using pre-calibrated footage - skipping calibration step")
            # Use standard FFmpeg approach for preview generation
            camera_matrix = None
            dist_coeffs = None
        else:
            # Try to detect camera type and load calibration
            camera_type = calibration_service.get_camera_type_from_path(
                source_path, config_manager
            )
            camera_matrix = None
            dist_coeffs = None

            if camera_type:
                try:
                    logger.info(
                        f"Loading calibration parameters for camera type: {camera_type}"
                    )
                    camera_matrix, dist_coeffs, _ = (
                        calibration_service.load_calibration(
                            camera_type,
                            config_manager=config_manager,
                            video_path=source_path,
                        )
                    )
                    if camera_matrix is not None and dist_coeffs is not None:
                        logger.info(
                            f"Successfully loaded calibration parameters for {camera_type}"
                        )
                    else:
                        logger.warning(
                            f"Could not load calibration parameters for {camera_type}"
                        )
                except Exception as e:
                    logger.error(f"Error loading calibration parameters: {e}")
                    camera_matrix = None
                    dist_coeffs = None

        # If we have calibration parameters and crop region or keyframes, we need to process with OpenCV
        if (camera_matrix is not None and dist_coeffs is not None) and (
            crop_region or crop_keyframes_proxy
        ):
            logger.info(
                "Using OpenCV with calibration for preview generation with crop"
            )

            try:
                # Open the input video
                cap = cv2.VideoCapture(str(source_path))
                if not cap.isOpened():
                    logger.error(f"Could not open video: {source_path}")
                    if progress_placeholder:
                        progress_placeholder.error(
                            f"Could not open video: {source_path}"
                        )
                    return None

                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Validate frame range
                valid_start = max(0, min(start_frame, total_frames - 1))
                valid_end = max(valid_start, min(end_frame, total_frames - 1))

                # Get calibration alpha from settings
                calib_settings = (
                    config_manager.get_calibration_settings()
                    if config_manager
                    else {"alpha": 0.5}
                )
                alpha = calib_settings.get("alpha", 0.5)

                # Prepare undistortion maps
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    camera_matrix, dist_coeffs, (width, height), alpha
                )
                map1, map2 = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    dist_coeffs,
                    None,
                    new_camera_matrix,
                    (width, height),
                    cv2.CV_32FC1,
                )

                # Set up output video writer
                proxy_width = proxy_settings["width"]

                # Initialize video writer
                if crop_region:
                    # If using static crop
                    crop_w = crop_region[2]
                    crop_h = crop_region[3]
                    # Calculate aspect ratio-preserving dimensions
                    output_h = int(proxy_width * (crop_h / crop_w))
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        str(preview_path), fourcc, fps, (proxy_width, output_h)
                    )
                else:
                    # If using proxy-specific keyframes or no crop
                    output_h = int(proxy_width * (height / width))
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        str(preview_path), fourcc, fps, (proxy_width, output_h)
                    )

                if not out.isOpened():
                    logger.error("Failed to create output video writer")
                    return None

                # Set frame position to start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, valid_start)

                # Process frames
                frames_to_process = valid_end - valid_start + 1
                for frame_idx in range(frames_to_process):
                    # Read the frame
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Apply calibration
                    undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

                    # Apply crop
                    if crop_region:
                        # Static crop
                        x, y, w, h = crop_region
                        cropped = undistorted[y : y + h, x : x + w]
                    elif crop_keyframes_proxy:
                        # Animated crop - interpolate between keyframes
                        current_frame = valid_start + frame_idx
                        current_crop = interpolate_crop_keyframes(
                            current_frame, crop_keyframes_proxy
                        )
                        if current_crop:
                            x, y, w, h = current_crop
                            cropped = undistorted[y : y + h, x : x + w]
                        else:
                            cropped = undistorted
                    else:
                        cropped = undistorted

                    # Resize to proxy width
                    if crop_region or crop_keyframes_proxy:
                        resized = cv2.resize(cropped, (proxy_width, output_h))
                    else:
                        resized = cv2.resize(undistorted, (proxy_width, output_h))

                    # Write the frame
                    out.write(resized)

                    # Update progress
                    if progress_placeholder and frame_idx % 10 == 0:
                        progress = (frame_idx + 1) / frames_to_process
                        progress_placeholder.progress(progress)

                # Release resources
                cap.release()
                out.release()

                logger.info(f"Preview created successfully: {preview_path}")
                if progress_placeholder:
                    progress_placeholder.success("Preview created!")
                return str(preview_path)

            except Exception as e:
                logger.error(f"Error in OpenCV preview generation: {e}")
                # Fall back to FFmpeg approach
                logger.info("Falling back to FFmpeg approach for preview generation")

        # If we only have calibration but no crop, we can use the FFmpeg pipeline
        elif camera_matrix is not None and dist_coeffs is not None:
            logger.info("Using calibration with FFmpeg pipeline for preview generation")

            try:
                # Create a temporary calibrated version of the clip
                temp_dir = tempfile.mkdtemp()
                temp_file = os.path.join(temp_dir, "temp_calibrated.mp4")

                # Limit the input to just the frames we need using seeking
                input_args = ["-ss", str(start_time), "-t", str(duration)]

                # Extract the segment we need and apply calibration
                cap = cv2.VideoCapture(str(source_path))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                # Get calibration alpha from settings
                calib_settings = (
                    config_manager.get_calibration_settings()
                    if config_manager
                    else {"alpha": 0.5}
                )
                alpha = calib_settings.get("alpha", 0.5)

                # Prepare for calibration
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    camera_matrix, dist_coeffs, (width, height), alpha
                )

                # Use OpenCV to calibrate each frame and save to temporary file
                success = calibration_service.apply_calibration_to_video(
                    input_path=source_path,
                    output_path=temp_file,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    alpha=alpha,
                    progress_callback=lambda p: (
                        progress_placeholder.progress(p / 2)
                        if progress_placeholder
                        else None
                    ),
                    config_manager=config_manager,
                )

                if not success:
                    logger.error("Failed to calibrate video segment")
                    return None

                # Now use the calibrated segment for the preview
                source_path = temp_file
                start_time = 0  # The temp file starts at what was originally start_time

                # Continue with normal FFmpeg processing below, but use the calibrated temp file
            except Exception as e:
                logger.error(f"Error in calibration preprocessing: {e}")
                # Fall back to using the original source without calibration
                logger.info(
                    "Falling back to standard FFmpeg processing without calibration"
                )

                # Combine filters
                if vf_filters:
                    vf_string = ",".join(vf_filters)
                    cmd.extend(["-vf", vf_string])
                    logger.info(f"Video filters: {vf_string}")

                try:
                    # Run ffmpeg command
                    process = subprocess.Popen(
                        cmd_str,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                    )
                except Exception as e:
                    logger.exception(f"Error running ffmpeg command: {str(e)}")
                    if progress_placeholder:
                        progress_placeholder.error(
                            f"Error running ffmpeg command: {str(e)}"
                        )
                    return None

        # FFmpeg command for creating a preview clip
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

        # Add crop filter if needed
        if crop_region:
            x, y, width, height = crop_region
            vf_filters.append(f"crop={width}:{height}:{x}:{y}")
        elif crop_keyframes_proxy:
            # Use static crop from first keyframe as fallback
            first_frame = min(crop_keyframes_proxy.keys())
            x, y, width, height = crop_keyframes_proxy[first_frame]
            vf_filters.append(f"crop={width}:{height}:{x}:{y}")
            logger.warning(
                "Dynamic crop not supported in FFmpeg fallback mode, using first keyframe"
            )

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
                "18",  # Lower CRF for higher quality (0-51, lower is better)
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

        if progress_placeholder:
            progress_placeholder.text("Creating preview...")

        try:
            # Run ffmpeg command
            process = subprocess.Popen(
                cmd_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                universal_newlines=True,
            )
        except Exception as e:
            logger.exception(f"Error running ffmpeg command: {str(e)}")
            if progress_placeholder:
                progress_placeholder.error(f"Error running ffmpeg command: {str(e)}")
            return None

        # Collect all stderr output
        stderr_output = []

        # Monitor progress
        while True:
            line = process.stderr.readline()
            if not line:
                break
            stderr_output.append(line.strip())
            if progress_placeholder and "time=" in line:
                # Extract current time from ffmpeg output
                time_match = re.search(r"time=(\d+:\d+:\d+.\d+)", line)
                if time_match:
                    current_time = time_match.group(1)
                    progress_placeholder.text(
                        f"Creating preview... Time: {current_time}"
                    )
            logger.debug(f"ffmpeg output: {line.strip()}")

        # Wait for process to complete
        process.wait()

        if process.returncode == 0:
            logger.info(f"Successfully created preview: {preview_path}")
            if progress_placeholder:
                progress_placeholder.success("Preview generated successfully!")
            return str(preview_path)
        else:
            error_output = "\n".join(stderr_output)
            error_msg = f"Error creating preview. ffmpeg error:\n{error_output}"
            logger.error(error_msg)
            if progress_placeholder:
                progress_placeholder.error(error_msg)
            return None

    except Exception as e:
        logger.exception(f"Error creating preview: {str(e)}")
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


def export_clip(
    source_path,
    clip_name,
    start_frame,
    end_frame,
    crop_region=None,
    progress_placeholder=None,
    config_manager=None,
    crop_keyframes=None,
    output_resolution="1080p",
    cv_optimized=False,
    progress_callback=None,
):
    try:
        logger.info(f"Exporting clip for {clip_name}")
        logger.info(f"Source: {source_path}")
        logger.info(f"Frames: {start_frame} to {end_frame}")
        logger.info(f"Static crop region: {crop_region}")
        logger.info(f"Crop keyframes: {crop_keyframes}")
        logger.info(f"Output resolution: {output_resolution}")
        logger.info(f"CV optimized: {cv_optimized}")

        # Verify source video exists
        if not os.path.exists(source_path):
            error_msg = f"Source video not found: {source_path}"
            logger.error(error_msg)
            if progress_placeholder:
                progress_placeholder.error(error_msg)
            return None

        if not config_manager:
            config_manager = st.session_state.config_manager

        # Get export path with codec type
        codec_type = "ffv1" if cv_optimized else "h264"
        export_path = config_manager.get_output_path(
            Path(source_path), clip_name, codec_type
        )

        # The get_output_path method now handles the extension, no need to modify it here
        logger.info(f"Export path: {export_path}")

        # Delete existing export file if it exists
        if export_path.exists():
            try:
                export_path.unlink()
                logger.info(f"Deleted existing export file: {export_path}")
            except Exception as e:
                logger.error(f"Error deleting existing export file: {str(e)}")
                # Continue anyway as ffmpeg will overwrite the file

        # Ensure the parent directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created export directory: {export_path.parent}")

        # Get video FPS
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

        try:
            fps_output = subprocess.check_output(fps_cmd).decode("utf-8").strip()
            if fps_output:
                # Parse fraction format (e.g., "30000/1001")
                if "/" in fps_output:
                    num, den = fps_output.split("/")
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_output)
            else:
                fps = 30.0  # Default fallback
        except Exception as e:
            logger.warning(f"Could not determine video FPS: {str(e)}")
            fps = 30.0

        # Calculate start and end times in seconds
        start_time = start_frame / fps
        end_time = end_frame / fps
        logger.info(f"Time range: {start_time} to {end_time} seconds")
        duration = end_time - start_time

        # Check if we're using pre-calibrated footage
        if calibration_service.should_skip_calibration(config_manager):
            logger.info("Using pre-calibrated footage - skipping calibration step")
            # Use standard FFmpeg approach for export
            camera_matrix = None
            dist_coeffs = None
        else:
            # Try to detect camera type and load calibration
            camera_type = calibration_service.get_camera_type_from_path(
                source_path, config_manager
            )
            camera_matrix = None
            dist_coeffs = None

            if camera_type:
                try:
                    logger.info(
                        f"Loading calibration parameters for camera type: {camera_type}"
                    )
                    camera_matrix, dist_coeffs, _ = (
                        calibration_service.load_calibration(
                            camera_type,
                            config_manager=config_manager,
                            video_path=source_path,
                        )
                    )
                    if camera_matrix is not None and dist_coeffs is not None:
                        logger.info(
                            f"Successfully loaded calibration parameters for {camera_type}"
                        )
                    else:
                        logger.warning(
                            f"Could not load calibration parameters for {camera_type}"
                        )
                except Exception as e:
                    logger.error(f"Error loading calibration parameters: {e}")

        # Get output dimensions based on requested resolution
        # Import inside the function to avoid circular imports
        from src.services import video_service

        output_width, output_height = video_service.calculate_crop_dimensions(
            output_resolution
        )

        # Process clip with OpenCV and calibration if possible
        if camera_matrix is not None and dist_coeffs is not None:
            logger.info("Using OpenCV with calibration for export")

            try:
                # Open the input video
                cap = cv2.VideoCapture(str(source_path))
                if not cap.isOpened():
                    logger.error(f"Could not open video: {source_path}")
                    if progress_placeholder:
                        progress_placeholder.error(
                            f"Could not open video: {source_path}"
                        )
                    return None

                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Validate frame range
                valid_start = max(0, min(start_frame, total_frames - 1))
                valid_end = max(valid_start, min(end_frame, total_frames - 1))

                # Prepare undistortion maps
                # Get calibration alpha from settings
                calib_settings = (
                    config_manager.get_calibration_settings()
                    if config_manager
                    else {"alpha": 0.5}
                )
                alpha = calib_settings.get("alpha", 0.5)
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    camera_matrix, dist_coeffs, (width, height), alpha
                )
                map1, map2 = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    dist_coeffs,
                    None,
                    new_camera_matrix,
                    (width, height),
                    cv2.CV_32FC1,
                )

                # Handle animated crop keyframes if provided
                has_animated_crop = crop_keyframes and len(crop_keyframes) > 1
                has_static_crop = crop_region is not None

                # Set up named pipe for FFmpeg
                tmp_dir = tempfile.mkdtemp()
                fifo_path = os.path.join(
                    tmp_dir, f"ffmpeg_pipe_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )

                # Create the named pipe
                os.mkfifo(fifo_path)

                # Determine output codec settings based on cv_optimized flag
                if cv_optimized:
                    codec_args = [
                        "-c:v",
                        "ffv1",
                        "-level",
                        "3",  # Level 3 for multithreaded encoding
                        "-pix_fmt",
                        "yuv444p",  # Full chroma quality (better for CV)
                        "-g",
                        "1",  # Every frame is a keyframe (GOP=1)
                        "-threads",
                        str(min(16, os.cpu_count())),  # Use available cores
                        "-slices",
                        "24",  # Divide frame into slices for parallel encoding
                        "-slicecrc",
                        "1",  # Add CRC for each slice
                        "-context",
                        "1",  # Use larger context for better compression
                        "-an",  # No audio
                    ]
                    # For CV, we typically use .mkv extension
                    final_export_path = os.path.splitext(str(export_path))[0] + ".mkv"
                else:
                    codec_args = [
                        "-c:v",
                        "libx264",
                        "-crf",
                        "16",  # High quality
                        "-preset",
                        "slow",  # Good balance between quality and speed
                        "-pix_fmt",
                        "yuv420p",
                        "-color_range",
                        "tv",
                        # Include audio from source
                        "-c:a",
                        "aac",
                        "-b:a",
                        "320k",
                    ]
                    # Keep original export path with .mp4 extension
                    final_export_path = str(export_path)

                # Set up FFmpeg command
                cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output file if it exists
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{output_width}x{output_height}",  # Use the target resolution
                    "-pix_fmt",
                    "bgr24",
                    "-r",
                    str(fps),
                    "-i",
                    fifo_path,
                ]

                # Add audio input if not CV optimized
                if not cv_optimized:
                    # Calculate the start time in seconds
                    cmd.extend(
                        [
                            "-ss",
                            str(start_time),
                            "-t",
                            str(duration),
                            "-i",
                            str(source_path),  # Source for audio
                            "-map",
                            "0:v",  # Video from pipe
                            "-map",
                            "1:a",  # Audio from original
                        ]
                    )

                # Add codec settings
                cmd.extend(codec_args)

                # Add output path
                cmd.append(final_export_path)

                # Convert command to string
                cmd_str = " ".join(
                    (
                        f'"{arg}"'
                        if (" " in str(arg) or "+" in str(arg) or ":" in str(arg))
                        else str(arg)
                    )
                    for arg in cmd
                )

                logger.info(f"Starting FFmpeg with calibrated frames: {cmd_str}")

                # Start FFmpeg process
                ffmpeg_process = subprocess.Popen(
                    cmd_str,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    universal_newlines=True,
                )

                # Open the named pipe for writing
                pipe_out = open(fifo_path, "wb")

                # Set frame position to start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, valid_start)

                # Create a queue for progress information
                progress_queue = Queue()
                stop_event = threading.Event()

                # Create monitor thread for stderr output
                monitor_thread = threading.Thread(
                    target=monitor_ffmpeg_stderr,
                    args=(ffmpeg_process.stderr, duration, progress_queue, stop_event),
                )
                monitor_thread.daemon = True
                monitor_thread.start()

                try:
                    # Process frames
                    frames_to_process = valid_end - valid_start + 1
                    for frame_idx in range(frames_to_process):
                        # Read the frame
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Apply calibration
                        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LANCZOS4)

                        # Apply crop if needed
                        if has_animated_crop:
                            # Get current absolute frame number
                            current_frame = valid_start + frame_idx
                            # Interpolate crop region for current frame
                            current_crop = interpolate_crop_keyframes(
                                current_frame, crop_keyframes
                            )
                            if current_crop:
                                x, y, w, h = current_crop
                                cropped = undistorted[y : y + h, x : x + w]
                            else:
                                cropped = undistorted
                        elif has_static_crop:
                            # Static crop
                            x, y, w, h = crop_region
                            cropped = undistorted[y : y + h, x : x + w]
                        else:
                            cropped = undistorted

                        # Resize to target resolution with high quality interpolation
                        if has_animated_crop or has_static_crop:
                            resized = cv2.resize(
                                cropped,
                                (output_width, output_height),
                                interpolation=cv2.INTER_LANCZOS4,
                            )
                        else:
                            resized = cv2.resize(
                                undistorted,
                                (output_width, output_height),
                                interpolation=cv2.INTER_LANCZOS4,
                            )

                        # Write frame to pipe
                        pipe_out.write(resized.tobytes())

                        # Update progress
                        if frame_idx % 10 == 0:
                            # Calculate progress percentage
                            progress = frame_idx / frames_to_process

                            # Update UI with progress information from queue
                            try:
                                while not progress_queue.empty():
                                    progress_info = progress_queue.get(block=False)
                                    if progress_placeholder:
                                        progress_placeholder.progress(
                                            progress_info["progress"]
                                        )

                                        # Format progress message
                                        percent_complete = int(
                                            progress_info["progress"] * 100
                                        )
                                        remaining = progress_info.get("remaining", 0)
                                        minutes_remaining = int(remaining / 60)
                                        seconds_remaining = int(remaining % 60)

                                        if cv_optimized:
                                            message = f"Exporting CV-optimized clip with calibration: {percent_complete}% "
                                        else:
                                            message = f"Exporting clip with calibration: {percent_complete}% "

                                        message += f"(approx. {minutes_remaining}m {seconds_remaining}s remaining)"
                                        progress_placeholder.text(message)

                                    # Call progress callback if provided
                                    if progress_callback:
                                        progress_callback(progress_info["progress"])
                            except Exception as e:
                                logger.warning(f"Error updating progress: {e}")
                                continue

                    # Close pipe
                    pipe_out.close()

                    # Wait for FFmpeg to finish
                    return_code = ffmpeg_process.wait()

                    if return_code != 0:
                        error_output = (
                            ffmpeg_process.stderr.read()
                            if ffmpeg_process.stderr
                            else ""
                        )
                        logger.error(
                            f"FFmpeg process failed with code {return_code}: {error_output}"
                        )
                        if progress_placeholder:
                            progress_placeholder.error(f"Error exporting clip")
                        return None

                    logger.info(
                        f"Successfully exported calibrated clip: {final_export_path}"
                    )

                    # Update clip's export path in session state
                    if (
                        "clips" in st.session_state
                        and "current_clip_index" in st.session_state
                    ):
                        current_clip = st.session_state.clips[
                            st.session_state.current_clip_index
                        ]
                        current_clip.export_path = final_export_path
                        current_clip.update()  # Mark as modified
                        st.session_state.clip_modified = True

                    if progress_placeholder:
                        if cv_optimized:
                            progress_placeholder.success(
                                "CV-optimized export completed successfully!"
                            )
                        else:
                            progress_placeholder.success(
                                "Export completed successfully!"
                            )

                    return final_export_path

                finally:
                    # Clean up resources
                    stop_event.set()
                    if cap.isOpened():
                        cap.release()
                    try:
                        if os.path.exists(fifo_path):
                            os.unlink(fifo_path)
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)
                    except Exception as e:
                        logger.error(f"Error cleaning up resources: {e}")

            except Exception as e:
                logger.error(f"Error in OpenCV processing: {str(e)}")
                if progress_placeholder:
                    progress_placeholder.error(f"Error in OpenCV processing: {str(e)}")
                return None

            finally:
                # Clean up resources
                stop_event.set()
                if cap.isOpened():
                    cap.release()
                try:
                    if os.path.exists(fifo_path):
                        os.unlink(fifo_path)
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                except Exception as e:
                    logger.error(f"Error cleaning up resources: {e}")

        # If we get here, either calibration parameters weren't available or there was an error
        # Proceed with standard FFmpeg approach

        # Build the video filter string
        vf_filters = []

        # Add crop filter based on keyframes or static crop region
        if crop_keyframes:
            # Filter out keyframes outside the clip range and sort by frame number
            valid_keyframes = {
                k: v
                for k, v in crop_keyframes.items()
                if start_frame <= int(k) <= end_frame
            }

            # If no valid keyframes in range, use all keyframes but adjust timings
            if not valid_keyframes:
                logger.warning(
                    f"No keyframes in clip range {start_frame}-{end_frame}. Using all available keyframes."
                )
                valid_keyframes = crop_keyframes

            sorted_keyframes = sorted([(int(k), v) for k, v in valid_keyframes.items()])

            # Debug log the keyframes we're using
            logger.info(f"Using {len(sorted_keyframes)} crop keyframes for export")
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

                # Width and height are constant
                expressions.append(f"w='{width}'")
                expressions.append(f"h='{height}'")

                # Build x position expression (interpolating)
                x_expr = []
                for i in range(len(sorted_keyframes)):
                    curr_frame, curr_crop = sorted_keyframes[i]
                    # Make sure we're using the frame relative to the clip start time
                    # Negative times can cause issues in ffmpeg filtering
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

                        # Skip keyframes with identical timestamps (can happen if keyframes are too close)
                        if (
                            abs(next_t - curr_t) < 0.001
                        ):  # If less than 1ms apart, consider them identical
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

                # Build y position expression (interpolating) - with same improvements as x expression
                y_expr = []
                for i in range(len(sorted_keyframes)):
                    curr_frame, curr_crop = sorted_keyframes[i]
                    # Make sure we're using the frame relative to the clip start time
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

                        # Skip keyframes with identical timestamps (can happen if keyframes are too close)
                        if (
                            abs(next_t - curr_t) < 0.001
                        ):  # If less than 1ms apart, consider them identical
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
                if len(set([frame for frame, _ in sorted_keyframes])) == 1:
                    logger.info("Only one unique keyframe, using static crop")
                    x, y, width, height = first_crop
                    vf_filters.append(f"crop={width}:{height}:{x}:{y}")
                else:
                    # Create the dynamic crop filter
                    crop_filter = f"crop={':'.join(expressions)}"
                    vf_filters.append(crop_filter)
                    logger.info(f"Using dynamic crop filter: {crop_filter}")

        elif crop_region:
            # Static crop
            x, y, width, height = crop_region
            crop_filter = f"crop={width}:{height}:{x}:{y}"
            vf_filters.append(crop_filter)
            logger.info(f"Using static crop filter: {crop_filter}")

        # For CV exports, use high-quality scaler
        if cv_optimized:
            # Use lanczos scaling algorithm for higher quality
            vf_filters.append(f"scale={output_width}:{output_height}:flags=lanczos")
        else:
            # Use default scaling
            vf_filters.append(f"scale={output_width}:{output_height}")

        # Combine filters
        vf_string = ",".join(vf_filters)
        logger.info(f"Video filters: {vf_string}")

        # Base command list
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(source_path),
            "-t",
            str(end_time - start_time),
            "-vf",
            vf_string,
        ]

        # Add video codec settings
        if cv_optimized:
            # For computer vision, use near-lossless encoding
            cmd.extend(
                [
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",  # Level 3 for multithreaded encoding
                    "-pix_fmt",
                    "yuv444p",  # Full chroma quality (better for CV)
                    "-g",
                    "1",  # Every frame is a keyframe (GOP=1)
                    "-threads",
                    str(min(16, os.cpu_count())),  # Use all available CPU cores
                    "-slices",
                    "24",  # Divide each frame into slices for parallel encoding
                    "-slicecrc",
                    "1",  # Add CRC for each slice for integrity checking
                    "-context",
                    "1",  # Use larger context for better compression
                    "-an",  # No audio
                ]
            )

            # For CV, we typically don't need audio
            cmd.extend(["-an"])
        else:
            # Standard high quality settings for normal exports
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-crf",
                    "16",  # High quality
                    "-preset",
                    "slow",  # Good balance between quality and speed
                    "-pix_fmt",
                    "yuv420p",
                    "-color_range",
                    "tv",
                    # Include audio at high quality
                    "-c:a",
                    "aac",
                    "-b:a",
                    "320k",
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

        if progress_placeholder:
            if cv_optimized:
                progress_placeholder.text(
                    "Exporting CV-optimized clip (this may take longer)..."
                )
            else:
                progress_placeholder.text("Exporting clip...")

        # Run ffmpeg command
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
        )

        # Collect all stderr output
        stderr_output = []

        # Monitor progress
        while True:
            line = process.stderr.readline()
            if not line:
                break
            stderr_output.append(line.strip())

            # Extract progress information
            if "time=" in line:
                # Extract current time from ffmpeg output
                time_match = re.search(r"time=(\d+:\d+:\d+.\d+)", line)
                if time_match:
                    current_time = time_match.group(1)

                    # Update progress placeholder if available
                    if progress_placeholder:
                        progress_placeholder.text(
                            f"Exporting clip... Time: {current_time}"
                        )

                    # Calculate progress percentage for callback
                    if progress_callback:
                        # Parse the time string (HH:MM:SS.ms)
                        time_parts = current_time.split(":")
                        hours = int(time_parts[0])
                        minutes = int(time_parts[1])
                        seconds = float(time_parts[2])

                        # Convert to seconds
                        current_seconds = hours * 3600 + minutes * 60 + seconds

                        # Calculate progress as a fraction (0-1)
                        clip_duration = (end_frame - start_frame) / fps
                        if clip_duration > 0:
                            progress = min(1.0, current_seconds / clip_duration)
                            progress_callback(progress)

            logger.debug(f"ffmpeg output: {line.strip()}")

        # Wait for process to complete
        process.wait()

        if process.returncode == 0:
            logger.info(f"Successfully exported clip to: {export_path}")
            # Update clip's export path in session state
            if "clips" in st.session_state and "current_clip_index" in st.session_state:
                current_clip = st.session_state.clips[
                    st.session_state.current_clip_index
                ]
                current_clip.export_path = str(export_path)
                current_clip.update()  # Mark as modified
                st.session_state.clip_modified = True

            if progress_placeholder:
                if cv_optimized:
                    progress_placeholder.success(
                        "CV-optimized export completed successfully!"
                    )
                else:
                    progress_placeholder.success("Export completed successfully!")
            return str(export_path)
        else:
            error_output = "\n".join(stderr_output)
            error_msg = f"Error exporting clip. ffmpeg error:\n{error_output}"
            logger.error(error_msg)
            if progress_placeholder:
                progress_placeholder.error(error_msg)
            return None

    except Exception as e:
        logger.exception(f"Error exporting clip: {str(e)}")
        if progress_placeholder:
            progress_placeholder.error(f"Error exporting clip: {str(e)}")
        return None


def monitor_ffmpeg_stderr(stderr_pipe, duration, queue, stop_event):
    """
    Monitor FFmpeg stderr output for progress updates.

    Args:
        stderr_pipe: FFmpeg process stderr pipe
        duration: Video duration in seconds
        queue: Queue to put progress updates
        stop_event: Event to signal thread termination
    """
    pattern = re.compile(r"time=(\d+):(\d+):(\d+.\d+)")
    encoding_speed_pattern = re.compile(r"speed=(\d+.\d+)x")

    for line in iter(stderr_pipe.readline, ""):
        if stop_event.is_set():
            break

        line = line.strip()

        # Extract time information
        time_match = pattern.search(line)
        if time_match:
            hours, minutes, seconds = time_match.groups()
            current_time = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            progress = min(current_time / duration, 1.0) if duration > 0 else 0
            remaining = max(0, duration - current_time) if duration > 0 else 0

            # Extract encoding speed if available
            encoding_speed = 1.0
            speed_match = encoding_speed_pattern.search(line)
            if speed_match:
                encoding_speed = float(speed_match.group(1))

            # Put progress info in queue
            queue.put(
                {
                    "progress": progress,
                    "current_time": current_time,
                    "remaining": remaining,
                    "encoding_speed": encoding_speed,
                }
            )

    # Signal completion
    queue.put(
        {
            "progress": 1.0,
            "remaining": 0,
            "current_time": duration,
            "encoding_speed": 1.0,
        }
    )


def monitor_progress(process, duration, queue, stop_event):
    """
    Monitor FFmpeg process output for progress updates.

    Args:
        process: FFmpeg process
        duration: Video duration in seconds
        queue: Queue to put progress updates
        stop_event: Event to signal thread termination
    """
    pattern = re.compile(r"time=(\d+):(\d+):(\d+.\d+)")
    encoding_speed_pattern = re.compile(r"speed=(\d+.\d+)x")

    for line in iter(process.stderr.readline, ""):
        if stop_event.is_set():
            break

        line = line.strip()

        # Extract time information
        time_match = pattern.search(line)
        if time_match:
            hours, minutes, seconds = time_match.groups()
            current_time = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            progress = min(current_time / duration, 1.0) if duration > 0 else 0
            remaining = max(0, duration - current_time) if duration > 0 else 0

            # Extract encoding speed if available
            encoding_speed = 1.0
            speed_match = encoding_speed_pattern.search(line)
            if speed_match:
                encoding_speed = float(speed_match.group(1))

            # Put progress info in queue
            queue.put(
                {
                    "progress": progress,
                    "current_time": current_time,
                    "remaining": remaining,
                    "encoding_speed": encoding_speed,
                }
            )

    # Signal completion
    queue.put(
        {
            "progress": 1.0,
            "remaining": 0,
            "current_time": duration,
            "encoding_speed": 1.0,
        }
    )


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
    Create a proxy video using a two-stage approach:
    1. Create a temporary calibrated video
    2. Generate proxy from that calibrated video

    This method is more reliable than the pipe-based approach which often fails.

    Args:
        source_path: Path to the source video
        proxy_path: Path where the proxy should be saved
        camera_type: Type of camera (used for logging)
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        proxy_settings: Proxy settings from config
        duration: Video duration in seconds
        progress_placeholder: Streamlit placeholder for progress updates
        progress_callback: Callback function for progress updates
        config_manager: ConfigManager instance

    Returns:
        Path to the proxy video or None if creation failed
    """
    try:
        # Create a temporary file for the calibrated output
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            temp_calibrated_path = tmp_file.name

        logger.info(
            f"Stage 1/2: Creating temporary calibrated video at {temp_calibrated_path}"
        )

        # Create progress bar if not already done
        progress_bar = None
        if progress_placeholder:
            progress_placeholder.text("Stage 1/2: Calibrating footage...")
            progress_bar = progress_placeholder.progress(0.0)

        # Create a calibration progress wrapper function that scales to 0-50%
        def calibration_progress_wrapper(progress):
            # Scale progress to 0-50% for first stage
            scaled_progress = progress * 0.5
            if progress_placeholder and progress_bar:
                progress_bar.progress(scaled_progress)
                percent = int(scaled_progress * 100)
                progress_placeholder.text(
                    f"Stage 1/2: Calibrating footage: {percent}% complete"
                )
            if progress_callback:
                progress_callback(scaled_progress)

        # Apply calibration to create temporary file
        success = calibration_service.apply_calibration_to_video(
            input_path=str(source_path),
            output_path=temp_calibrated_path,
            camera_type=camera_type,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            config_manager=config_manager,
            progress_callback=calibration_progress_wrapper,
        )

        if not success:
            logger.error("Failed to calibrate video segment")
            if progress_placeholder:
                progress_placeholder.error(
                    "Failed to calibrate video - aborting proxy creation"
                )
            return None

        # Now use the calibrated segment for the proxy
        logger.info("Stage 2/2: Creating proxy from calibrated video")
        if progress_placeholder:
            progress_placeholder.text("Stage 2/2: Creating proxy video...")

        # Initialize progress tracking
        progress_queue = Queue()
        stop_event = threading.Event()
        start_time = time.time()

        # Continue with normal FFmpeg processing below, but use the calibrated temp file
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_calibrated_path),
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
            if progress_placeholder:
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
                        progress_info = progress_queue.get(block=False)
                        if progress_placeholder and progress_bar:
                            # Scale progress to 50-100% for second stage
                            scaled_progress = 0.5 + (progress_info["progress"] * 0.5)
                            progress_bar.progress(scaled_progress)

                            # Calculate remaining time
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

                            message = f"Stage 2/2: Creating proxy video: {percent_complete}% complete{speed_text}"
                            message += f" (approx. {minutes_remaining}m {seconds_remaining}s remaining)"
                            progress_placeholder.text(message)

                        if progress_callback:
                            # Scale progress to 50-100% for second stage
                            scaled_progress = 0.5 + (progress_info["progress"] * 0.5)
                            progress_callback(scaled_progress)

                        progress_queue.task_done()
                    except Exception:
                        # No progress update available, sleep briefly
                        time.sleep(0.1)

            finally:
                # Signal monitor thread to stop
                stop_event.set()

                # Wait for process to complete
                logger.info("Waiting for ffmpeg process to complete...")
                return_code = process.wait()
                logger.info(f"ffmpeg process completed with return code {return_code}")

                if return_code != 0:
                    logger.error(
                        f"ffmpeg process failed with return code {return_code}"
                    )
                    if progress_placeholder:
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
            if progress_placeholder and progress_bar:
                progress_bar.progress(1.0)
                total_time = time.time() - start_time
                minutes = int(total_time / 60)
                seconds = int(total_time % 60)
                progress_placeholder.success(
                    f"Proxy video created successfully in {minutes}m {seconds}s!"
                )

            return str(proxy_path)
        else:
            logger.error("Failed to create proxy video - output file does not exist")
            if progress_placeholder:
                progress_placeholder.error("Failed to create proxy video")
            return None

    except Exception as e:
        logger.exception(f"Error creating proxy: {str(e)}")
        if progress_placeholder:
            progress_placeholder.error(f"Error creating proxy: {str(e)}")
        return None

    finally:
        # Clean up temporary file
        try:
            if "temp_calibrated_path" in locals() and os.path.exists(
                temp_calibrated_path
            ):
                os.unlink(temp_calibrated_path)
                logger.debug(
                    f"Cleaned up temporary calibrated video: {temp_calibrated_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file: {str(e)}")
