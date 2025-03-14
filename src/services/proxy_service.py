"""
Proxy video generation services for the Clipper application.
"""

import os
import re
import time
import threading
import subprocess
from pathlib import Path
from queue import Queue
import streamlit as st
import logging
import collections

# Configure logging to suppress Streamlit context warnings
streamlit_logger = logging.getLogger("streamlit")
streamlit_logger.setLevel(logging.ERROR)

logger = logging.getLogger("clipper.proxy")


def create_proxy_video(
    source_path, progress_placeholder=None, progress_callback=None, config_manager=None
):
    """
    Create a proxy (web-compatible) version of the video for faster playback

    Args:
        source_path: Path to the source video
        progress_placeholder: Streamlit placeholder for progress updates
        progress_callback: Callback function for progress updates
        config_manager: ConfigManager instance

    Returns:
        Path to the proxy video or None if creation failed
    """
    try:
        # Get proxy settings from config manager
        if not config_manager:
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

        # Create a queue to communicate between threads
        progress_queue = Queue()
        stop_event = threading.Event()

        # Function to monitor progress
        def monitor_progress(process, duration, queue, stop_event):
            # Suppress Streamlit context warnings in this thread
            import logging
            import collections

            logging.getLogger("streamlit").setLevel(logging.ERROR)

            pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

            # Keep track of recent encoding speeds for better time estimation
            encoding_speeds = collections.deque(
                maxlen=10
            )  # Store last 10 speed measurements
            last_time = None
            last_current_time = 0
            start_time = time.time()

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

                        # Calculate encoding speed (seconds of video processed per second of real time)
                        now = time.time()
                        if last_time is not None and current_time > last_current_time:
                            elapsed_real_time = now - last_time
                            processed_video_time = current_time - last_current_time
                            if elapsed_real_time > 0:
                                encoding_speed = (
                                    processed_video_time / elapsed_real_time
                                )
                                encoding_speeds.append(encoding_speed)

                        last_time = now
                        last_current_time = current_time

                        # Calculate remaining time based on average recent encoding speed
                        if encoding_speeds:
                            avg_speed = sum(encoding_speeds) / len(encoding_speeds)
                            remaining_video_time = duration - current_time
                            remaining = (
                                remaining_video_time / avg_speed if avg_speed > 0 else 0
                            )
                        else:
                            # Fallback to simple estimation if we don't have speed data yet
                            elapsed_total = now - start_time
                            if current_time > 0 and elapsed_total > 0:
                                remaining = (duration - current_time) / (
                                    current_time / elapsed_total
                                )
                            else:
                                remaining = 0

                        # Put progress info in queue
                        progress_info = {
                            "progress": progress,
                            "remaining": remaining,
                            "current_time": current_time,
                            "encoding_speed": avg_speed if encoding_speeds else 0,
                        }
                        queue.put(progress_info)

                        # Call the progress callback if provided
                        if progress_callback:
                            try:
                                progress_callback(progress_info)
                            except Exception as callback_error:
                                # Silently ignore Streamlit context errors
                                pass
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

            # Final progress update for non-duration case
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

        proxy_dir = config_manager.proxy_dir

        # Display information about the proxy directory
        if not proxy_dir.exists():
            st.info(f"Proxy directory does not exist: {proxy_dir}")
            return False

        # Count proxy files
        proxy_files = list(proxy_dir.glob("**/*.mp4"))
        file_count = len(proxy_files)

        if file_count == 0:
            st.info("No proxy files to clean up.")
            return False

        # Initialize session state for cleanup confirmation
        if "cleanup_confirm_state" not in st.session_state:
            st.session_state.cleanup_confirm_state = "initial"

        # Show button with file count
        if st.button(f"Clean Up {file_count} Proxy Videos", key="cleanup_proxy_btn"):
            # Set state to confirmation
            st.session_state.cleanup_confirm_state = "confirm"
            st.rerun()

        # Show confirmation dialog if in confirm state
        if st.session_state.cleanup_confirm_state == "confirm":
            st.warning("⚠️ This will delete all proxy videos. Are you sure?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete All Proxies", key="confirm_cleanup"):
                    # Recursively find and delete all files in proxy directory
                    deleted_count = 0
                    for file in proxy_dir.glob("**/*"):
                        if file.is_file():
                            try:
                                file.unlink()
                                deleted_count += 1
                                logger.info(f"Deleted proxy file: {file}")
                            except Exception as e:
                                logger.error(
                                    f"Error deleting proxy file {file}: {str(e)}"
                                )

                    # Remove empty directories (except the root proxy directory)
                    dir_count = 0
                    for dir_path in sorted(
                        [p for p in proxy_dir.glob("**/*") if p.is_dir()], reverse=True
                    ):
                        try:
                            if dir_path != proxy_dir and not any(dir_path.iterdir()):
                                dir_path.rmdir()
                                dir_count += 1
                                logger.info(
                                    f"Removed empty proxy directory: {dir_path}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error removing proxy directory {dir_path}: {str(e)}"
                            )

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
