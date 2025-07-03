"""
Dedicated service for efficient clip export with frame-range calibration.

This service handles:
- Frame-range calibration (much faster than full-video calibration)
- Multi-crop generation (original, wide, full)
- Multi-format export (h264, ffv1)
"""

import os
import cv2
import tempfile
import logging
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue
import numpy as np
from src.services import calibration_service

logger = logging.getLogger("clipper.clip_export")


def is_calibration_enabled(config_manager):
    """
    Check if calibration is enabled in config.

    Args:
        config_manager: ConfigManager instance

    Returns:
        bool: True if calibration should be applied
    """
    if not config_manager:
        return False

    calib_settings = config_manager.get_calibration_settings()
    # Check for new 'enabled' setting first, then fall back to old setting
    if "enabled" in calib_settings:
        return calib_settings.get("enabled", False)
    else:
        # For backwards compatibility, invert the old confusing setting
        return not calib_settings.get("use_calibrated_footage", False)


def apply_calibration_to_clip(
    input_path,
    output_path,
    start_frame,
    end_frame,
    camera_type=None,
    alpha=0.5,
    quality_preset="high",
    progress_callback=None,
    config_manager=None,
):
    """
    Apply calibration to a specific frame range (MUCH more efficient).

    Instead of calibrating 44,760 frames, this calibrates only the 59 frames needed.
    This provides ~760x speedup for small clips!

    Args:
        input_path: Path to source video
        output_path: Path for calibrated output
        start_frame: Starting frame number
        end_frame: Ending frame number
        camera_type: Camera type for calibration
        alpha: Calibration alpha parameter (0.5 is optimal)
        quality_preset: "high", "lossless", or "medium"
        progress_callback: Callback for progress updates
        config_manager: ConfigManager instance

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(
            f"Applying calibration to frames {start_frame}-{end_frame} (frame-range only)"
        )

        # Load calibration parameters
        camera_matrix, dist_coeffs, _ = calibration_service.load_calibration(
            camera_type, config_manager=config_manager, video_path=input_path
        )

        if camera_matrix is None or dist_coeffs is None:
            logger.warning(
                "No calibration data available, extracting without calibration"
            )
            return extract_clip_frames(
                input_path, output_path, start_frame, end_frame, progress_callback
            )

        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {input_path}")
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = end_frame - start_frame + 1

        logger.info(
            f"Processing {total_frames} frames instead of entire video ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames)"
        )

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Get optimal camera matrix for calibration
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

        # Set up output codec
        if quality_preset == "lossless":
            fourcc = cv2.VideoWriter_fourcc(*"FFV1")
            output_path = os.path.splitext(output_path)[0] + ".mkv"
        else:
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            output_path = os.path.splitext(output_path)[0] + ".mp4"

        # Create video writer
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            logger.warning(f"Could not initialize H264 codec, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = os.path.splitext(output_path)[0] + ".mp4"
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            if not out.isOpened():
                logger.error(f"Could not create video writer for {output_path}")
                cap.release()
                return False

        # Process only the needed frames
        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply calibration using precomputed maps
            calibrated_frame = cv2.remap(frame, map1, map2, cv2.INTER_LANCZOS4)

            # Crop if needed (when alpha < 1)
            if alpha < 1.0 and roi[2] > 0 and roi[3] > 0:
                x, y, w, h = roi
                calibrated_frame = calibrated_frame[y : y + h, x : x + w]
                calibrated_frame = cv2.resize(
                    calibrated_frame, (width, height), interpolation=cv2.INTER_LANCZOS4
                )

            # Write calibrated frame
            out.write(calibrated_frame)

            # Update progress
            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:  # Update every 10 frames
                progress = frame_idx / total_frames
                progress_callback(progress)

        # Cleanup
        cap.release()
        out.release()

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Calibration applied to {total_frames} frames -> {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error in frame-range calibration: {e}")
        return False


def convert_to_final_format_with_crop(
    input_path,
    output_path,
    crop_region=None,
    crop_keyframes=None,
    start_frame=0,
    fps=30.0,
    is_cv_format=False,
    progress_callback=None,
):
    """
    Convert calibrated clip to final format with optional cropping and keyframe animation.

    Args:
        input_path: Path to calibrated clip
        output_path: Path for final export
        crop_region: Optional static crop region (x, y, width, height)
        crop_keyframes: Optional keyframes dict for dynamic cropping
        start_frame: Start frame of the clip for keyframe timing
        fps: Video FPS for keyframe timing calculations
        is_cv_format: Whether to use CV-optimized (FFV1 lossless) format
        progress_callback: Progress callback function

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(
            f"Converting to final format: {'FFV1 LOSSLESS' if is_cv_format else 'H.264'}"
        )

        # Build FFmpeg command with appropriate codec settings
        cmd = ["ffmpeg", "-y", "-i", str(input_path)]

        # Add crop and scale filters
        vf_filters = []

        # Handle dynamic crop keyframes (prioritized over static crop)
        if crop_keyframes and len(crop_keyframes) > 1:
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
            logger.info(f"Applying crop: {width}x{height} at ({x},{y})")

        # Scale to exactly 1920x1080 for consistent output resolution across all variations
        vf_filters.append("scale=1920:1080")
        logger.info("Scaling output to 1920x1080 (1080p)")

        # Add video filters
        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

        if is_cv_format:
            # FFV1 lossless codec for CV applications (matches original proxy_service settings)
            cmd.extend(
                [
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",  # FFV1 level 3 for better features
                    "-g",
                    "1",  # All frames are keyframes for precise seeking
                    "-context",
                    "1",  # Better error recovery
                    "-slices",
                    "24",  # Good for multithreading
                    "-pix_fmt",
                    "yuv420p",  # Standard pixel format
                ]
            )
        else:
            # H.264 for regular exports (matches original proxy_service settings)
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",  # High quality
                    "-preset",
                    "slow",  # Better compression
                    "-pix_fmt",
                    "yuv420p",  # Standard pixel format
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
        cmd.append(str(output_path))

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run conversion
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode == 0:
            # Verify file was created and get size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Successfully created {output_path} ({file_size:.2f} MB)")
                if progress_callback:
                    progress_callback(1.0)
                return True
            else:
                logger.error("Output file was not created")
                return False
        else:
            logger.error(f"FFmpeg conversion failed: {process.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error converting to final format: {e}")
        return False


def convert_to_final_format(
    input_path, output_path, is_cv_format=False, progress_callback=None
):
    """
    Convert calibrated clip to final format with proper codec settings.
    (Legacy function - calls convert_to_final_format_with_crop with no crop)

    Args:
        input_path: Path to calibrated clip
        output_path: Path for final export
        is_cv_format: Whether to use CV-optimized (FFV1 lossless) format
        progress_callback: Progress callback function

    Returns:
        True if successful, False otherwise
    """
    # Call the new function with no crop region
    return convert_to_final_format_with_crop(
        input_path=input_path,
        output_path=output_path,
        crop_region=None,
        crop_keyframes=None,
        start_frame=0,
        fps=30.0,
        is_cv_format=is_cv_format,
        progress_callback=progress_callback,
    )


def extract_clip_frames(
    input_path, output_path, start_frame, end_frame, progress_callback=None
):
    """
    Extract a clip using FFmpeg (no calibration).

    Args:
        input_path: Source video path
        output_path: Output clip path
        start_frame: Starting frame number
        end_frame: Ending frame number
        progress_callback: Progress callback function

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get FPS for timestamp calculation
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
            str(input_path),
        ]

        fps = 30.0  # Default fallback
        try:
            fps_output = subprocess.check_output(fps_cmd).decode("utf-8").strip()
            if "/" in fps_output:
                num, den = fps_output.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_output)
        except:
            logger.warning("Could not determine FPS, using 30.0")

        # Calculate timestamps
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        logger.info(f"Extracting frames {start_frame}-{end_frame} ({duration:.2f}s)")

        # Build FFmpeg command for efficient extraction
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-crf",
            "18",  # High quality encoding
            "-preset",
            "fast",
            str(output_path),
        ]

        # Run extraction
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode == 0:
            logger.info(f"Extracted clip successfully: {output_path}")
            if progress_callback:
                progress_callback(1.0)
            return True
        else:
            logger.error(f"FFmpeg extraction failed: {process.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error extracting clip frames: {e}")
        return False


def export_clip_efficient(
    clip, config_manager, multi_crop=False, multi_format=False, progress_callback=None
):
    """
    Main efficient clip export function.

    This is the replacement for the inefficient proxy_service.export_clip.

    Pipeline:
    1. Extract + Calibrate (frame-range only)  <- 760x faster!
    2. Generate crop variations (if requested)
    3. Generate format variations (if requested)

    Args:
        clip: Clip object with source_path, start_frame, end_frame, etc.
        config_manager: ConfigManager instance
        multi_crop: Whether to generate crop variations
        multi_format: Whether to generate format variations
        progress_callback: Progress callback function

    Returns:
        List of output file paths or None if failed
    """
    try:
        logger.info(f"Starting efficient export for clip: {clip.name}")

        # Step 0: Resolve the source path (handling legacy paths)
        from scripts.process_clips import resolve_source_path

        resolved_source_path = resolve_source_path(clip.source_path, config_manager)
        logger.info(f"Resolved source path: {resolved_source_path}")

        # Step 1: Extract + Calibrate (frame-range only)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_calibrated_path = temp_file.name

        if is_calibration_enabled(config_manager):
            camera_type = calibration_service.get_camera_type_from_path(
                resolved_source_path, config_manager
            )
            logger.info(f"Applying calibration for camera type: {camera_type}")

            alpha = config_manager.get_calibration_settings().get("alpha", 0.5)
            success = apply_calibration_to_clip(
                input_path=resolved_source_path,
                output_path=temp_calibrated_path,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                camera_type=camera_type,
                alpha=alpha,
                progress_callback=lambda p: (
                    progress_callback(p * 0.8) if progress_callback else None
                ),
                config_manager=config_manager,
            )
        else:
            logger.info("Calibration disabled, extracting frames only")
            success = extract_clip_frames(
                input_path=resolved_source_path,
                output_path=temp_calibrated_path,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                progress_callback=lambda p: (
                    progress_callback(p * 0.8) if progress_callback else None
                ),
            )

        if not success:
            logger.error("Failed to extract/calibrate clip")
            return None

        # Step 2: Generate crop variations if requested
        if multi_crop:
            logger.info("Generating multi-crop variations: original, wide, full")
            crop_variations = ["original", "wide", "full"]
        else:
            logger.info("Single crop mode")
            crop_variations = ["original"]

        # Step 3: Generate crop variations and formats
        output_paths = []

        # Import multi-crop utility
        from src.utils.multi_crop import get_crop_for_variation
        from src.services import video_service

        # Get video dimensions for crop calculations
        video_dimensions = video_service.get_video_dimensions(resolved_source_path)
        if video_dimensions[0] == 0 or video_dimensions[1] == 0:
            logger.warning(
                f"Invalid video dimensions {video_dimensions}, using default 1920x1080"
            )
            video_dimensions = (1920, 1080)

        # Get original crop region from clip
        original_crop = None
        if hasattr(clip, "crop_region") and clip.crop_region is not None:
            original_crop = clip.crop_region
            logger.info(f"Using clip.crop_region: {original_crop}")
        elif hasattr(clip, "crop_keyframes") and clip.crop_keyframes:
            # Get crop from keyframes at start frame
            original_crop = clip.crop_keyframes.get(str(clip.start_frame))
            if original_crop is None and clip.crop_keyframes:
                # Use first available keyframe
                first_frame = sorted(clip.crop_keyframes.keys())[0]
                original_crop = clip.crop_keyframes[first_frame]
            logger.info(f"Using crop from keyframes: {original_crop}")

        # Determine output formats to generate
        formats_to_generate = ["h264"]
        if multi_format:
            formats_to_generate.append("ffv1")

        # Generate each crop variation and format combination
        total_tasks = len(crop_variations) * len(formats_to_generate)
        task_idx = 0

        # Get FPS from the video source
        try:
            video_info = video_service.get_video_info(resolved_source_path)
            video_fps = video_info.get("fps", 30.0) if video_info else 30.0
        except Exception as e:
            logger.warning(f"Could not get FPS from video, using default: {e}")
            video_fps = 30.0

        logger.info(f"Video FPS: {video_fps}")

        for crop_variation in crop_variations:
            # For different variations, we need to handle keyframes differently
            if crop_variation == "original":
                # Use original keyframes as-is
                variation_keyframes = clip.crop_keyframes
                variation_crop_region = None
            elif crop_variation == "wide" and clip.crop_keyframes:
                # For 'wide' variation, scale each keyframe's crop region
                variation_keyframes = {}
                from src.utils.multi_crop import calculate_wider_crop

                # Process each keyframe to create a wider version
                for frame_num, crop_region in clip.crop_keyframes.items():
                    wider_crop = calculate_wider_crop(
                        crop_region,
                        wide_crop_factor=1.5,
                        frame_dimensions=video_dimensions,
                    )
                    variation_keyframes[frame_num] = wider_crop

                logger.info(
                    f"Created {len(variation_keyframes)} scaled keyframes for 'wide' variation"
                )
                variation_crop_region = None
            elif crop_variation == "full":
                # For 'full' variation, no crop at all
                variation_keyframes = None
                variation_crop_region = None
            else:
                # Fallback to static crop for variations without keyframes
                variation_keyframes = None
                variation_crop_region = get_crop_for_variation(
                    crop_variation,
                    original_crop,
                    video_dimensions,
                    wide_crop_factor=1.5,
                )

            # Log crop info
            if variation_keyframes:
                sample_frame = list(variation_keyframes.keys())[0]
                sample_crop = variation_keyframes[sample_frame]
                x, y, w, h = sample_crop
                percent_w = round((w / video_dimensions[0]) * 100, 1)
                percent_h = round((h / video_dimensions[1]) * 100, 1)
                logger.info(
                    f"Crop '{crop_variation}': Using {len(variation_keyframes)} keyframes (sample: {sample_crop} - {percent_w}% × {percent_h}% of frame)"
                )
            elif variation_crop_region:
                x, y, w, h = variation_crop_region
                percent_w = round((w / video_dimensions[0]) * 100, 1)
                percent_h = round((h / video_dimensions[1]) * 100, 1)
                logger.info(
                    f"Crop '{crop_variation}': {variation_crop_region} ({percent_w}% × {percent_h}% of frame)"
                )
            else:
                logger.info(f"Crop '{crop_variation}': Full frame (no crop)")

            for format_type in formats_to_generate:
                is_cv_format = format_type == "ffv1"

                # Create variation suffix for filename
                variation_suffix = (
                    "" if crop_variation == "original" else f"_{crop_variation}"
                )
                clip_name_with_variation = f"{clip.name}{variation_suffix}"

                logger.info(
                    f"Generating {format_type.upper()} format for '{crop_variation}' variation"
                )

                # Get proper export path
                export_base = config_manager.output_base
                export_dir = Path(export_base) / "03_CLIPPED"

                # Use correct directory and extension for format type
                if is_cv_format:
                    export_dir = export_dir / "ffv1"
                    file_ext = ".mkv"
                else:
                    export_dir = export_dir / "h264"
                    file_ext = ".mp4"

                # Create subdirectories based on source path structure
                source_path = Path(clip.source_path)
                video_dirname = os.path.basename(os.path.dirname(source_path))
                parent_dirname = os.path.basename(
                    os.path.dirname(os.path.dirname(source_path))
                )

                # Create camera type and session directories
                export_dir = export_dir / parent_dirname / video_dirname
                export_dir.mkdir(parents=True, exist_ok=True)

                # Create final export file path with variation suffix
                final_export_path = (
                    export_dir
                    / f"{os.path.splitext(os.path.basename(source_path))[0]}_{clip_name_with_variation}{file_ext}"
                )

                # Apply crop and codec settings
                success = convert_to_final_format_with_crop(
                    input_path=temp_calibrated_path,
                    output_path=final_export_path,
                    crop_region=variation_crop_region,
                    crop_keyframes=variation_keyframes,
                    start_frame=clip.start_frame,
                    fps=video_fps,
                    is_cv_format=is_cv_format,
                    progress_callback=lambda p: (
                        progress_callback(
                            0.8 + (p * 0.2 * (task_idx + 1) / total_tasks)
                        )
                        if progress_callback
                        else None
                    ),
                )

                if success:
                    output_paths.append(str(final_export_path))
                    logger.info(
                        f"✅ {format_type.upper()} '{crop_variation}': {final_export_path}"
                    )
                else:
                    logger.error(
                        f"❌ Failed {format_type.upper()} '{crop_variation}' export"
                    )

                task_idx += 1

        # Cleanup temp file
        try:
            os.unlink(temp_calibrated_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {e}")

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Efficient export completed: {len(output_paths)} files generated")
        return output_paths

    except Exception as e:
        logger.error(f"Error in efficient clip export: {e}")
        # Cleanup temp file if it exists
        try:
            if "temp_calibrated_path" in locals() and os.path.exists(
                temp_calibrated_path
            ):
                os.unlink(temp_calibrated_path)
        except:
            pass
        return None
