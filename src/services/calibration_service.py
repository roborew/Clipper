"""
Camera calibration service for the Clipper application.

This module provides functions to apply lens distortion correction to videos.
"""

import os
import cv2
import numpy as np
import logging
import json
from pathlib import Path
import shutil
import tempfile
from datetime import datetime
import subprocess
import time

logger = logging.getLogger("clipper.calibration")

# Cache for camera types to avoid redundant directory scanning
_camera_types_cache = {}
_camera_types_cache_time = 0
_CACHE_TIMEOUT = 60  # Seconds before cache expires

# Cache for camera type detection to avoid redundant lookups
_camera_type_path_cache = {}


def get_calibration_paths(config_manager=None):
    """
    Get calibration paths from config.

    Args:
        config_manager: ConfigManager instance

    Returns:
        Tuple of (base_dir, params_dir)
    """
    # Try to get config_manager from streamlit session if not provided
    if config_manager is None:
        try:
            import streamlit as st

            if "config_manager" in st.session_state:
                config_manager = st.session_state.config_manager
                logger.debug(
                    "Got config_manager from streamlit session state for calibration paths"
                )
        except ImportError:
            logger.debug("Could not import streamlit for calibration paths")

    if config_manager:
        try:
            calib_settings = config_manager.get_calibration_settings()
            if "directories" in calib_settings:
                dirs = calib_settings["directories"]
                if "base" in dirs and "parameters" in dirs:
                    base_dir = dirs["base"]
                    params_dir = dirs["parameters"]
                    logger.info(
                        f"Using calibration paths from config: {base_dir}/{params_dir}"
                    )
                    return base_dir, params_dir
                else:
                    logger.error(
                        "Missing base or parameters in calibration directories config"
                    )
            else:
                logger.error("Missing 'directories' in calibration settings")
        except Exception as e:
            logger.error(f"Error getting calibration paths from config: {e}")
    else:
        logger.error("No config_manager provided, cannot determine calibration paths")

    # If we reach here, we couldn't get valid paths from config
    logger.error(
        "Failed to get calibration paths from config, calibration will not work"
    )
    return None, None


def should_skip_calibration(config_manager=None):
    """
    Check if calibration should be skipped based on settings.

    Args:
        config_manager: ConfigManager instance

    Returns:
        True if calibration should be skipped (using pre-calibrated footage), False otherwise
    """
    if config_manager is None:
        import streamlit as st

        if "config_manager" in st.session_state:
            config_manager = st.session_state.config_manager
        else:
            # Fallback default
            return False

    # Check calibration settings
    calib_settings = config_manager.get_calibration_settings()
    return calib_settings.get("use_calibrated_footage", False)


def load_calibration(
    camera_type=None,
    session_id=None,
    calib_file=None,
    config_manager=None,
    video_path=None,
):
    """Load calibration parameters for a specific camera type.

    Args:
        camera_type: Type of camera (GP1, GP2, SONY_70, SONY_300), can be None if video_path is provided
        session_id: Optional session ID for session-specific calibration
        calib_file: Optional specific calibration file to use (overrides session_id)
        config_manager: ConfigManager instance
        video_path: Optional path to video file to determine camera type from path

    Returns:
        Tuple of (camera_matrix, dist_coeffs, calibration_data)
    """
    # Auto-detect camera type if not provided but video_path is available
    if camera_type is None and video_path is not None:
        camera_type = get_camera_type_from_path(video_path, config_manager)
        if camera_type:
            logger.info(
                f"Auto-detected camera type '{camera_type}' from video path: {video_path}"
            )
        else:
            logger.warning(
                f"Could not auto-detect camera type from video path: {video_path}"
            )
            return None, None, None

    # Ensure camera_type is not None at this point
    if camera_type is None:
        logger.error("Camera type is required but not provided")
        return None, None, None

    # Get calibration directories from config
    calibration_base_dir, calibration_params_dir = get_calibration_paths(config_manager)

    # Check if we got valid paths
    if calibration_base_dir is None or calibration_params_dir is None:
        logger.error("Could not get valid calibration paths from config")
        return None, None, None

    # Determine calibration file path
    if calib_file:
        # If a specific calibration file is provided, use it
        if os.path.isabs(calib_file):
            # If it's an absolute path, use it directly
            calib_path = calib_file
        else:
            # If it's a relative path, look in the camera type's parameters directory
            calib_path = (
                Path(calibration_base_dir)
                / calibration_params_dir
                / camera_type
                / calib_file
            )
    elif session_id:
        # If session ID is provided, look for session-specific calibration
        calib_path = (
            Path(calibration_base_dir)
            / calibration_params_dir
            / camera_type
            / f"{session_id}_calibration.json"
        )

        # If session-specific calibration doesn't exist, fall back to default
        if not calib_path.exists():
            logger.warning(f"Session-specific calibration not found: {calib_path}")
            logger.info(f"Falling back to default calibration")
            calib_path = (
                Path(calibration_base_dir)
                / calibration_params_dir
                / camera_type
                / "calibration.json"
            )
    else:
        # Otherwise, use default calibration
        calib_path = (
            Path(calibration_base_dir)
            / calibration_params_dir
            / camera_type
            / "calibration.json"
        )

    logger.info(f"Loading calibration from: {calib_path}")

    # Check if calibration file exists
    if not calib_path.exists():
        logger.error(f"Calibration file not found: {calib_path}")
        return None, None, None

    # Load calibration data
    try:
        with open(calib_path, "r") as f:
            calibration_data = json.load(f)

        # Extract camera matrix and distortion coefficients
        camera_matrix = np.array(calibration_data["camera_matrix"])
        dist_coeffs = np.array(calibration_data["dist_coeffs"])

        return camera_matrix, dist_coeffs, calibration_data
    except Exception as e:
        logger.error(f"Error loading calibration: {e}")
        return None, None, None


def get_camera_type_from_path(video_path, config_manager=None):
    """
    Determine the camera type from a video path.

    Args:
        video_path: Path to the video file
        config_manager: ConfigManager instance

    Returns:
        Camera type string or None if not detected
    """
    # Convert to Path object if it's a string
    if isinstance(video_path, str):
        video_path = Path(video_path)

    # Check cache first
    path_str = str(video_path)
    if path_str in _camera_type_path_cache:
        logger.debug(f"Using cached camera type for path: {path_str}")
        return _camera_type_path_cache[path_str]

    # Get available camera types
    camera_types = get_camera_types(config_manager)

    # If no camera types are available, we need to rely on pattern detection only
    if not camera_types:
        logger.debug(
            "No camera types available for lookup, will try to detect from path patterns"
        )
        detected_type = _detect_camera_type_from_patterns(video_path)
        _camera_type_path_cache[path_str] = detected_type
        return detected_type

    # Check if any camera type is in the path
    path_parts = video_path.parts
    path_str_upper = str(video_path).upper()

    # First check for exact matches in path components
    for part in path_parts:
        part_upper = part.upper()
        if part in camera_types or any(
            cam_type.upper() == part_upper for cam_type in camera_types
        ):
            logger.debug(f"Found camera type {part} in path: {video_path}")
            _camera_type_path_cache[path_str] = part
            return part

    # Next check for partial matches in the path string
    for cam_type in camera_types:
        if cam_type.upper() in path_str_upper:
            logger.debug(f"Found camera type {cam_type} in path: {video_path}")
            _camera_type_path_cache[path_str] = cam_type
            return cam_type

    # If no match found, try pattern detection
    detected_type = _detect_camera_type_from_patterns(video_path)
    _camera_type_path_cache[path_str] = detected_type
    return detected_type


def _detect_camera_type_from_patterns(video_path):
    """
    Detect camera type based on common naming patterns in the path.

    Args:
        video_path: Path to the video file

    Returns:
        Detected camera type or None
    """
    path_parts = Path(video_path).parts

    # Try to detect based on common patterns
    for part in path_parts:
        part_upper = part.upper()
        # Check for common camera patterns
        if (
            part_upper.startswith("GP")
            and len(part_upper) <= 3
            and part_upper[2:].isdigit()
        ):
            logger.info(f"Detected GoPro camera type {part} from path: {video_path}")
            return part
        elif "SONY" in part_upper and any(char.isdigit() for char in part_upper):
            logger.info(f"Detected Sony camera type {part} from path: {video_path}")
            return part
        elif any(cam in part_upper for cam in ["CAM", "CANON", "NIKON", "GOPRO"]):
            logger.info(f"Detected camera type {part} from path: {video_path}")
            return part

    logger.warning(f"Could not detect camera type from path: {video_path}")
    return None


def calibrate_frame(frame, camera_matrix, dist_coeffs, alpha=0.5):
    """
    Apply calibration to a single frame

    Args:
        frame: OpenCV image frame
        camera_matrix: Camera matrix from calibration file
        dist_coeffs: Distortion coefficients from calibration file
        alpha: Alpha parameter for undistortion (0.0-1.0)
               - 1.0: All pixels from original image are retained
               - 0.0: No black borders in undistorted image
               - 0.5: Good balance (recommended)

    Returns:
        Calibrated frame
    """
    if frame is None or camera_matrix is None or dist_coeffs is None:
        return frame

    height, width = frame.shape[:2]

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), alpha
    )

    # Apply undistortion using high-quality interpolation
    undistorted = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    # Crop the image if needed (when alpha < 1)
    if alpha < 1.0 and roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        undistorted = undistorted[y : y + h, x : x + w]
        undistorted = cv2.resize(
            undistorted, (width, height), interpolation=cv2.INTER_LANCZOS4
        )

    return undistorted


def apply_calibration_to_video(
    input_path,
    output_path,
    camera_type=None,
    session_id=None,
    alpha=0.5,
    calib_file=None,
    camera_matrix=None,
    dist_coeffs=None,
    quality_preset="high",
    progress_callback=None,
    config_manager=None,
):
    """Apply calibration to a video file.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        camera_type: Type of camera (GP1, GP2, SONY_70, SONY_300)
        session_id: Optional session ID for session-specific calibration
        alpha: Alpha parameter for undistortion (0.0-1.0)
        calib_file: Optional specific calibration file to use
        camera_matrix: Optional camera matrix (if already loaded)
        dist_coeffs: Optional distortion coefficients (if already loaded)
        quality_preset: Quality preset - "lossless", "high", "medium" (default: "high")
        progress_callback: Optional callback function for progress updates
        config_manager: ConfigManager instance for checking settings

    Returns:
        True if successful, False otherwise
    """
    # Check if we should skip calibration (using pre-calibrated footage)
    if should_skip_calibration(config_manager):
        logger.info("Using pre-calibrated footage, skipping calibration")
        # Simply copy the input file to the output path
        try:
            shutil.copy2(input_path, output_path)
            logger.info(f"Copied file from {input_path} to {output_path}")
            # Still call progress callback to indicate completion
            if progress_callback:
                progress_callback(1.0)
            return True
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return False

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try to determine camera type if not provided
    if camera_type is None and camera_matrix is None:
        camera_type = get_camera_type_from_path(input_path, config_manager)

    # Load calibration parameters if not provided
    if camera_matrix is None or dist_coeffs is None:
        if camera_type is None:
            logger.error(
                "Error: Either camera_type or camera_matrix/dist_coeffs must be provided"
            )
            return False

        camera_matrix, dist_coeffs, _ = load_calibration(
            camera_type, session_id, calib_file, config_manager, video_path=input_path
        )
        if camera_matrix is None or dist_coeffs is None:
            logger.error(f"Failed to load calibration for {camera_type}")
            return False

    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {input_path}")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), alpha
    )

    # Prepare undistortion maps
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )

    # For lossless quality with FFV1 level 3, we need to use FFmpeg directly
    if quality_preset == "lossless" and shutil.which("ffmpeg") is not None:
        logger.info(
            f"Using lossless FFV1 codec with level 3 (multithreaded, faster encoding)"
        )
        output_path = _apply_calibration_ffmpeg(
            input_path,
            output_path,
            cap,
            map1,
            map2,
            roi,
            alpha,
            width,
            height,
            fps,
            frame_count,
            progress_callback,
        )
        return output_path is not None

    # Select codec and quality settings based on preset (OpenCV approach)
    if quality_preset == "lossless":
        # FFV1 is a lossless codec that works well with OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        output_path = (
            os.path.splitext(output_path)[0] + ".mkv"
        )  # Switch to MKV container
        logger.info(f"Using lossless FFV1 codec (highest quality, larger files)")
    elif quality_preset == "high":
        # H.264 with high quality settings
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        output_path = os.path.splitext(output_path)[0] + ".mp4"
        logger.info(f"Using H264 codec with high quality settings")
    else:
        # Default to mp4v with medium quality
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        logger.info(f"Using mp4v codec with medium quality settings")

    # Try to create VideoWriter with selected codec
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # If the codec fails, fall back to a more universally supported one
    if not out.isOpened():
        logger.warning(
            f"Warning: Could not initialize {fourcc} codec, falling back to mp4v"
        )
        output_path = os.path.splitext(output_path)[0] + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error(f"Error: Could not create video writer for {output_path}")
            return False

    # Process each frame
    frame_idx = 0
    progress_report_interval = max(
        1, min(frame_count // 100, 100)
    )  # Report progress about every 1%
    last_progress_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply undistortion using high-quality interpolation
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LANCZOS4)

        # Crop the image if needed (when alpha < 1)
        if alpha < 1.0 and roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            undistorted = undistorted[y : y + h, x : x + w]
            undistorted = cv2.resize(
                undistorted, (width, height), interpolation=cv2.INTER_LANCZOS4
            )

        # Write frame to output video
        out.write(undistorted)

        # Update progress
        frame_idx += 1

        # Report progress at regular intervals
        if frame_idx % progress_report_interval == 0 or frame_idx == frame_count:
            progress = frame_idx / frame_count

            # Call progress callback if provided
            if progress_callback:
                progress_callback(progress)

            # Log progress at reasonable intervals (not too frequent)
            current_time = time.time()
            if current_time - last_progress_time > 2.0:  # Log every 2 seconds at most
                logger.info(
                    f"Progress: {progress*100:.1f}% ({frame_idx}/{frame_count})"
                )
                last_progress_time = current_time

    # Report final progress
    if progress_callback:
        progress_callback(1.0)

    # Release resources
    cap.release()
    out.release()

    logger.info(f"Calibration applied to {input_path} -> {output_path}")
    return True


def _apply_calibration_ffmpeg(
    input_path,
    output_path,
    cap,
    map1,
    map2,
    roi,
    alpha,
    width,
    height,
    fps,
    frame_count,
    progress_callback=None,
):
    """Apply calibration using FFmpeg for better FFV1 encoding with level 3.

    This method processes frames with OpenCV and pipes them to FFmpeg for encoding.
    """
    # Prepare the output file path with .mkv extension
    output_path = os.path.splitext(output_path)[0] + ".mkv"

    # Set up a named pipe (fifo) for sending frames to ffmpeg
    tmp_dir = tempfile.mkdtemp()
    fifo_path = os.path.join(
        tmp_dir, f"ffmpeg_pipe_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    try:
        # Create the named pipe
        os.mkfifo(fifo_path)

        # Set up FFmpeg command for FFV1 level 3 encoding
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            fifo_path,
            "-c:v",
            "ffv1",
            "-level",
            "3",  # FFV1 level 3 for multithreaded encoding
            "-pix_fmt",
            "yuv444p",  # Full chroma quality (better for CV)
            "-g",
            "1",  # Every frame is a keyframe (GOP=1)
            "-threads",
            str(min(16, os.cpu_count())),  # Use up to 16 cores
            "-slices",
            "24",  # Divide each frame into slices for parallel encoding
            "-slicecrc",
            "1",  # Add CRC for each slice for integrity checking
            "-context",
            "1",  # Use larger context for better compression
            output_path,
        ]

        # Start FFmpeg process
        logger.info(f"Starting FFmpeg with FFV1 level 3 encoding (CV-optimized)...")
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        # Open the named pipe for writing
        pipe_out = open(fifo_path, "wb")

        # Process each frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply undistortion using high-quality interpolation
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LANCZOS4)

            # Crop the image if needed (when alpha < 1)
            if alpha < 1.0 and roi[2] > 0 and roi[3] > 0:
                x, y, w, h = roi
                undistorted = undistorted[y : y + h, x : x + w]
                undistorted = cv2.resize(
                    undistorted, (width, height), interpolation=cv2.INTER_LANCZOS4
                )

            # Write the frame to the pipe
            pipe_out.write(undistorted.tobytes())

            # Update progress
            frame_idx += 1
            if frame_idx % 100 == 0:
                progress = frame_idx / frame_count
                if progress_callback:
                    progress_callback(progress)
                else:
                    logger.info(
                        f"Progress: {progress:.1f}% ({frame_idx}/{frame_count})"
                    )

        # Close the pipe and wait for FFmpeg to finish
        pipe_out.close()
        ffmpeg_process.wait()

        logger.info(f"Calibration applied to {input_path} -> {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error in FFmpeg encoding: {e}")
        return None

    finally:
        # Clean up the temporary directory and fifo
        if os.path.exists(fifo_path):
            os.unlink(fifo_path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

        # Make sure to release the video capture
        if cap.isOpened():
            cap.release()


def get_camera_types(config_manager=None):
    """
    Get camera types from the source raw directory, then calibration directory structure,
    then fallback to config setting.

    Args:
        config_manager: ConfigManager instance

    Returns:
        List of camera types based on directory names
    """
    global _camera_types_cache, _camera_types_cache_time

    # Check if we have a recent cache entry
    current_time = time.time()
    if (
        _camera_types_cache
        and (current_time - _camera_types_cache_time) < _CACHE_TIMEOUT
    ):
        logger.debug(
            f"Using cached camera types (age: {current_time - _camera_types_cache_time:.1f}s)"
        )
        return _camera_types_cache.get("types", [])

    # Try to get config_manager from streamlit session if not provided
    if config_manager is None:
        try:
            import streamlit as st

            if "config_manager" in st.session_state:
                config_manager = st.session_state.config_manager
                logger.debug("Got config_manager from streamlit session state")
            else:
                logger.warning("No config_manager in streamlit session state")
        except ImportError:
            logger.warning("Could not import streamlit, no config_manager available")

    camera_types = []

    # PRIORITY 1: Check source raw directory first
    if config_manager and hasattr(config_manager, "source_raw"):
        logger.debug("Checking source raw directory for camera types")
        source_raw = config_manager.source_raw

        if source_raw and os.path.exists(source_raw):
            logger.debug(f"Checking source raw directory: {source_raw}")
            try:
                # Get subdirectories of the source raw
                subdirs = [d.name for d in Path(source_raw).iterdir() if d.is_dir()]

                # Filter for likely camera type directories
                for subdir in subdirs:
                    # Check for common camera patterns
                    upper_name = subdir.upper()
                    if any(
                        pattern in upper_name
                        for pattern in [
                            "GP",
                            "SONY",
                            "CAM",
                            "CANON",
                            "NIKON",
                            "GOPRO",
                        ]
                    ):
                        camera_types.append(subdir)

                if camera_types:
                    logger.info(
                        f"Found camera types from source raw directory: {camera_types}"
                    )
                    # Update cache
                    _camera_types_cache = {"types": camera_types}
                    _camera_types_cache_time = current_time
                    return camera_types
            except Exception as e:
                logger.warning(f"Error checking source raw directory: {e}")

    # PRIORITY 2: Check calibration directory structure
    calibration_base_dir, calibration_params_dir = get_calibration_paths(config_manager)

    if calibration_base_dir and calibration_params_dir:
        params_path = Path(calibration_base_dir) / calibration_params_dir

        try:
            # Log the path we're checking
            logger.info(f"Looking for camera types in: {params_path}")

            # If the directory exists, list subdirectories as camera types
            if params_path.exists() and params_path.is_dir():
                camera_types = [d.name for d in params_path.iterdir() if d.is_dir()]
                if camera_types:
                    logger.info(
                        f"Found camera types from calibration directory: {camera_types}"
                    )
                    # Update cache
                    _camera_types_cache = {"types": camera_types}
                    _camera_types_cache_time = current_time
                    return camera_types
                else:
                    logger.warning(f"No subdirectories found in {params_path}")
            else:
                logger.warning(
                    f"Path does not exist or is not a directory: {params_path}"
                )

        except Exception as e:
            logger.warning(
                f"Error detecting camera types from calibration directories: {e}"
            )
    else:
        logger.warning("No valid calibration paths available to check for camera types")

    # PRIORITY 3: Check config for camera_types setting
    if config_manager:
        try:
            calib_settings = config_manager.get_calibration_settings()
            if "camera_types" in calib_settings:
                config_camera_types = calib_settings["camera_types"]
                # Handle both string (comma-separated) and list formats
                if isinstance(config_camera_types, str):
                    camera_types = [t.strip() for t in config_camera_types.split(",")]
                elif isinstance(config_camera_types, list):
                    camera_types = config_camera_types

                if camera_types:
                    logger.info(f"Using camera types from config: {camera_types}")
                    # Update cache
                    _camera_types_cache = {"types": camera_types}
                    _camera_types_cache_time = current_time
                    return camera_types
        except Exception as e:
            logger.warning(f"Error getting camera types from config: {e}")

    # PRIORITY 4: Extract from video paths as last resort
    if config_manager:
        try:
            logger.info("Extracting camera types from all video file paths")
            video_files = config_manager.get_video_files()

            # Extract potential camera types from paths
            extracted_types = set()
            for video_path in video_files:
                path_str = str(video_path).upper()
                parts = Path(video_path).parts

                # Check each path component
                for part in parts:
                    part_upper = part.upper()
                    # Check for common camera types (GP1, GP2, etc)
                    if (
                        part_upper.startswith("GP")
                        and len(part_upper) <= 3
                        and part_upper[2:].isdigit()
                    ):
                        extracted_types.add(part)
                    # Check for SONY cameras with numbers
                    elif "SONY" in part_upper and any(
                        char.isdigit() for char in part_upper
                    ):
                        extracted_types.add(part)
                    # Check for other common camera identifiers
                    elif any(
                        cam in part_upper for cam in ["CAM", "CANON", "NIKON", "GOPRO"]
                    ):
                        extracted_types.add(part)

            if extracted_types:
                camera_types = list(extracted_types)
                logger.info(f"Extracted camera types from file paths: {camera_types}")
                # Update cache
                _camera_types_cache = {"types": camera_types}
                _camera_types_cache_time = current_time
                return camera_types
        except Exception as e:
            logger.warning(f"Error extracting camera types from video paths: {e}")

    # No camera types found - update cache with empty list
    _camera_types_cache = {"types": []}
    _camera_types_cache_time = current_time
    logger.error(
        "No camera types found from any source. Calibration will not work correctly."
    )
    return []
