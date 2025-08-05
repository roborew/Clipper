"""
Utility module for handling multiple crop variations in clip exports.

This module provides functions to calculate different crop regions
for generating multiple variations of the same clip.
"""

import logging
import os
from services import video_service

logger = logging.getLogger("clipper.multi_crop")


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
        logger.warning("Cannot calculate wider crop from None crop region")
        return None

    x, y, width, height = original_crop
    frame_width, frame_height = frame_dimensions

    # Log detailed calculations for debugging
    logger.info(f"Original crop: x={x}, y={y}, width={width}, height={height}")
    logger.info(f"Frame dimensions: width={frame_width}, height={frame_height}")
    logger.info(f"Applying wide crop factor: {factor}")

    # Check for invalid frame dimensions and use fallback
    if frame_width <= 0 or frame_height <= 0:
        logger.warning(
            f"Invalid frame dimensions ({frame_width}, {frame_height}), estimating from crop position"
        )
        # Estimate minimum frame size based on crop position and size
        estimated_width = max(
            3840, x + width + 100
        )  # Assume at least 4K or crop extent + margin
        estimated_height = max(
            2160, y + height + 100
        )  # Assume at least 4K or crop extent + margin
        frame_width, frame_height = estimated_width, estimated_height
        logger.info(f"Using estimated frame dimensions: {frame_width}x{frame_height}")

    # Calculate maximum allowable dimensions (95% of frame size)
    max_width = int(frame_width * 0.95)
    max_height = int(frame_height * 0.95)

    # Calculate target dimensions
    target_width = int(width * factor)
    target_height = int(height * factor)
    logger.info(
        f"Initial target dimensions: width={target_width}, height={target_height}"
    )

    # If target dimensions would exceed max allowed size, scale down proportionally
    if target_width > max_width or target_height > max_height:
        width_scale = max_width / target_width if target_width > max_width else 1.0
        height_scale = max_height / target_height if target_height > max_height else 1.0
        scale_factor = min(width_scale, height_scale)

        # Apply scaling
        new_width = int(target_width * scale_factor)
        new_height = int(target_height * scale_factor)
        logger.info(
            f"Scaled down to fit within 95% of frame: width={new_width}, height={new_height}"
        )
    else:
        new_width = target_width
        new_height = target_height
        logger.info(
            f"Using exact target dimensions: width={new_width}, height={new_height}"
        )

    # Calculate the ideal centered position
    # The center of the original crop should remain the center of the new crop
    original_center_x = x + width // 2
    original_center_y = y + height // 2

    ideal_x = original_center_x - new_width // 2
    ideal_y = original_center_y - new_height // 2

    logger.info(f"Original crop center: ({original_center_x}, {original_center_y})")
    logger.info(f"Ideal centered position: x={ideal_x}, y={ideal_y}")
    logger.info(
        f"Size difference: width_diff={new_width - width}, height_diff={new_height - height}"
    )

    # Adjust position to stay within frame boundaries while maintaining crop size
    # If pushing against left edge
    if ideal_x < 0:
        new_x = 0
        logger.info("Adjusting x position to 0 (left edge)")
    # If pushing against right edge
    elif ideal_x + new_width > frame_width:
        new_x = frame_width - new_width
        logger.info(f"Adjusting x position to {new_x} (right edge)")
    else:
        new_x = ideal_x

    # Same for vertical position
    if ideal_y < 0:
        new_y = 0
        logger.info("Adjusting y position to 0 (top edge)")
    elif ideal_y + new_height > frame_height:
        new_y = frame_height - new_height
        logger.info(f"Adjusting y position to {new_y} (bottom edge)")
    else:
        new_y = ideal_y

    final_crop = (new_x, new_y, new_width, new_height)
    logger.info(f"Final wide crop: {final_crop}")

    # Verify centering is correct
    new_center_x = new_x + new_width // 2
    new_center_y = new_y + new_height // 2
    logger.info(f"New crop center: ({new_center_x}, {new_center_y})")

    center_diff_x = abs(new_center_x - original_center_x)
    center_diff_y = abs(new_center_y - original_center_y)
    if center_diff_x > 5 or center_diff_y > 5:  # Allow small rounding differences
        logger.warning(
            f"WARNING: Center may have shifted! Diff: ({center_diff_x}, {center_diff_y})"
        )

    # Verify crop is actually different from original
    if new_width == width and new_height == height and new_x == x and new_y == y:
        logger.info("Wide crop is identical to original crop (factor = 1.0)")

    # Verify crop is not full frame
    if new_width == frame_width and new_height == frame_height:
        logger.warning("WARNING: Wide crop dimensions match full frame!")
        # Force crop to be 95% of frame size
        new_width = int(frame_width * 0.95)
        new_height = int(frame_height * 0.95)
        new_x = (frame_width - new_width) // 2
        new_y = (frame_height - new_height) // 2
        final_crop = (new_x, new_y, new_width, new_height)
        logger.info(f"Adjusted to 95% of frame size: {final_crop}")

    return final_crop


def get_crop_for_variation(
    variation, original_crop, frame_dimensions, wide_crop_factor=1.5
):
    """
    Get the appropriate crop region for a specific variation.

    Args:
        variation: The crop variation type ('original', 'wide', or 'full')
        original_crop: The original crop region (x, y, width, height)
        frame_dimensions: Video frame dimensions (width, height)
        wide_crop_factor: Factor for the 'wide' crop variation (1.5 = 50% larger)

    Returns:
        Crop region (x, y, width, height) or None for full frame
    """
    # For 'full' variation, always return None to use full frame
    if variation == "full":
        logger.info("Using full frame (no crop) for 'full' variation")
        return None

    # For 'original' variation, return the original crop - but special handling for None
    if variation == "original":
        if original_crop:
            logger.info(f"Using original crop: {original_crop}")
            x, y, w, h = original_crop
            frame_w, frame_h = frame_dimensions
            # Prevent division by zero
            if frame_w > 0 and frame_h > 0:
                percent_w = round((w / frame_w) * 100, 1)
                percent_h = round((h / frame_h) * 100, 1)
                logger.info(
                    f"Original crop dimensions: {w}x{h} ({percent_w}% × {percent_h}% of frame)"
                )
            else:
                logger.warning(
                    f"Frame dimensions are zero ({frame_w}, {frame_h}), cannot calculate percentages"
                )
                logger.info(f"Original crop dimensions: {w}x{h}")
            return original_crop
        else:
            # If no crop is defined for 'original', use None to export the full frame
            # This ensures the behavior matches the UI where a clip with no crop shows the full frame
            logger.warning(
                "No original crop defined, using full frame for 'original' variation"
            )
            return None

    # For 'wide' variation - needs an original crop to calculate from
    if variation == "wide":
        if original_crop:
            # Calculate a wider crop using the helper function
            wider_crop = calculate_wider_crop(
                original_crop, wide_crop_factor, frame_dimensions
            )
            logger.info(
                f"Calculated wider crop: {wider_crop} (factor: {wide_crop_factor}) from original: {original_crop}"
            )

            # Verify the crop is actually wider (for debugging)
            if wider_crop and original_crop:
                orig_x, orig_y, orig_w, orig_h = original_crop
                wide_x, wide_y, wide_w, wide_h = wider_crop

                # Check if dimensions actually changed
                if wide_w == orig_w and wide_h == orig_h:
                    logger.warning(
                        "WARNING: Wide crop has same dimensions as original crop!"
                    )
                elif wide_w <= orig_w or wide_h <= orig_h:
                    logger.warning(
                        f"WARNING: Wide crop seems smaller in at least one dimension! Original: {orig_w}x{orig_h}, Wide: {wide_w}x{wide_h}"
                    )

                # Check if it's equivalent to full frame
                frame_w, frame_h = frame_dimensions
                if (
                    wide_w == frame_w
                    and wide_h == frame_h
                    and wide_x == 0
                    and wide_y == 0
                ):
                    logger.warning("WARNING: Wide crop is equivalent to full frame!")

            return wider_crop
        else:
            # If trying to create 'wide' variation but no original crop exists,
            # we can't calculate a proper 'wide' crop, so we'll use full frame as well
            logger.warning(
                "Cannot calculate 'wide' variation without original crop. Using full frame."
            )
            return None

    # Unknown variation type
    logger.warning(f"Unknown crop variation: {variation}")
    return original_crop  # Fall back to original crop


def process_clip_with_variations(
    clip,
    source_path,
    config_manager,
    crop_variations="original,wide,full",
    wide_crop_factor=1.5,
    cv_optimized=False,
    both_formats=False,
    gpu_acceleration=False,
    progress_callback=None,
):
    """Generate multiple crop variations of a clip with streamlined logging"""

    # Parse variations
    variations_list = [v.strip() for v in crop_variations.split(",") if v.strip()]
    logger.info(
        f"🎯 Generating {len(variations_list)} variations: {', '.join(variations_list)}"
    )

    try:
        # Get clip metadata and crop keyframes
        clip_start_frame = clip.start_frame
        clip_end_frame = clip.end_frame
        duration_frames = clip_end_frame - clip_start_frame + 1

        # Get source FPS for accurate timing calculation
        source_fps = get_source_video_fps(source_path)
        duration_seconds = duration_frames / source_fps
        logger.info(
            f"🎬 Duration: {duration_frames} frames @ {source_fps} fps = {duration_seconds:.2f}s"
        )

        # Get video dimensions for crop boundary calculations
        frame_dimensions = video_service.get_video_dimensions(source_path)
        if frame_dimensions[0] == 0 or frame_dimensions[1] == 0:
            logger.warning(
                f"Invalid video dimensions {frame_dimensions}, using default 3840x2160"
            )
            frame_dimensions = (3840, 2160)
        else:
            logger.info(
                f"🎬 Video dimensions: {frame_dimensions[0]}x{frame_dimensions[1]}"
            )

        # Get crop keyframes
        crop_keyframes = {}
        if hasattr(clip, "crop_keyframes") and clip.crop_keyframes:
            crop_keyframes = clip.crop_keyframes
            logger.info(f"🎯 Keyframes: {len(crop_keyframes)}")
        else:
            logger.warning("⚠️  No crop keyframes found")
            return False

        # Generate all requested variations
        all_success = True
        results = []

        for variation in variations_list:
            logger.info(f"🎬 Creating {variation} variation")

            # Calculate crop for this variation
            if variation == "original":
                variation_keyframes = crop_keyframes
            elif variation == "wide":
                variation_keyframes = create_wide_crop_keyframes(
                    crop_keyframes, wide_crop_factor, frame_dimensions
                )
            elif variation == "full":
                variation_keyframes = create_full_frame_keyframes()
            else:
                logger.warning(f"⚠️  Unknown variation: {variation}")
                continue

            # Process each format
            for format_type in ["regular", "cv"] if both_formats else ["regular"]:
                is_cv = format_type == "cv"

                success = create_variation_export(
                    clip,
                    source_path,
                    config_manager,
                    variation,
                    variation_keyframes,
                    is_cv,
                    gpu_acceleration,
                    duration_seconds,
                )

                if success:
                    results.append(f"{variation}-{format_type}")
                    logger.info(f"✅ {variation} {format_type}")
                else:
                    logger.error(f"❌ {variation} {format_type}")
                    all_success = False

        if all_success:
            logger.info(f"✅ All variations complete: {', '.join(results)}")
        else:
            logger.warning(f"⚠️  Some variations failed: {', '.join(results)}")

        return all_success

    except Exception as e:
        logger.error(f"❌ Variation processing failed: {e}")
        return False


def create_wide_crop_keyframes(
    original_keyframes, wide_factor, frame_dimensions=(3840, 2160)
):
    """Create wider crop keyframes by scaling the original crops"""
    wide_keyframes = {}
    frame_width, frame_height = frame_dimensions

    for frame_num, crop in original_keyframes.items():
        x, y, width, height = crop

        # Calculate wider dimensions
        new_width = int(width * wide_factor)
        new_height = int(height * wide_factor)

        # Center the wider crop
        new_x = max(0, x - (new_width - width) // 2)
        new_y = max(0, y - (new_height - height) // 2)

        # Ensure we don't exceed frame boundaries (use actual frame dimensions)
        new_x = min(new_x, frame_width - new_width)
        new_y = min(new_y, frame_height - new_height)

        # Additional safety check to ensure coordinates are never negative
        new_x = max(0, new_x)
        new_y = max(0, new_y)

        wide_keyframes[frame_num] = (new_x, new_y, new_width, new_height)

    return wide_keyframes


def create_full_frame_keyframes():
    """Create keyframes for full frame (no crop)"""
    # Return None to indicate no cropping
    return None


def get_source_video_fps(source_path):
    """Get the FPS of the source video file"""
    import subprocess

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

    fps = 25.0  # Default fallback
    try:
        fps_output = subprocess.check_output(fps_cmd).decode("utf-8").strip()
        if fps_output:
            # Parse fraction format (e.g., "30000/1001" or "25/1")
            if "/" in fps_output:
                num, den = fps_output.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_output)
    except Exception as e:
        import logging

        logger = logging.getLogger("clipper.multi_crop")
        logger.warning(f"Could not detect FPS from {source_path}, using 25.0: {e}")

    return fps


def create_variation_export(
    clip,
    source_path,
    config_manager,
    variation,
    variation_keyframes,
    is_cv_format,
    gpu_acceleration,
    duration,
):
    """Export a single crop variation using the correct pipeline"""
    try:
        from src.services import clip_export_service, calibration_service
        import tempfile
        import os

        # Get the source video FPS - critical for preserving timing
        source_fps = get_source_video_fps(source_path)

        # Determine output format and directory
        format_ext = ".mkv" if is_cv_format else ".mp4"
        format_dir = "ffv1" if is_cv_format else "h264"

        # Create export path
        camera_type = source_path.parent.parent.name  # Extract camera from path
        session = source_path.parent.name

        # Use configured clips directory from config.yaml
        export_dir = config_manager.clips_dir / format_dir / camera_type / session

        # Create proper filename with source video name prefix
        source_video_name = (
            source_path.stem
        )  # Get filename without extension (e.g., "C0001")
        variation_suffix = "" if variation == "original" else f"_{variation}"
        export_filename = (
            f"{source_video_name}_{clip.name}{variation_suffix}{format_ext}"
        )
        export_path = export_dir / export_filename

        # Ensure directory exists
        export_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract + Calibrate the clip segment (like export_clip_efficient does)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_calibrated_path = temp_file.name

        try:
            # Check if calibration is enabled and apply it
            if clip_export_service.is_calibration_enabled(config_manager):
                camera_type_for_cal = calibration_service.get_camera_type_from_path(
                    source_path, config_manager
                )
                logger.info(f"🔧 Calibrating {variation}: {camera_type_for_cal}")

                alpha = config_manager.get_calibration_settings().get("alpha", 0.5)
                calibrate_success = clip_export_service.apply_calibration_to_clip(
                    input_path=source_path,
                    output_path=temp_calibrated_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    camera_type=camera_type_for_cal,
                    alpha=alpha,
                    config_manager=config_manager,
                    gpu_acceleration=gpu_acceleration,
                )
            else:
                logger.info(f"🎬 Extracting {variation} (no calibration)")
                calibrate_success = clip_export_service.extract_clip_frames(
                    input_path=source_path,
                    output_path=temp_calibrated_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    gpu_acceleration=gpu_acceleration,
                )

            if not calibrate_success:
                logger.error(f"Failed to extract/calibrate {variation}")
                return False

            # Step 2: Apply crop variation to the calibrated clip
            logger.info(f"🎬 Source FPS: {source_fps} (preserving original timing)")

            if variation_keyframes is None:
                # Full frame - no crop
                success = clip_export_service.convert_to_final_format_with_crop(
                    input_path=temp_calibrated_path,
                    output_path=export_path,
                    crop_region=None,
                    crop_keyframes=None,
                    start_frame=clip.start_frame,  # Keep original frame reference for timing
                    fps=source_fps,  # Use detected source FPS to preserve timing
                    is_cv_format=is_cv_format,
                    gpu_acceleration=gpu_acceleration,
                )
            else:
                # Use keyframes for cropping (keep original frame numbers for timing)
                success = clip_export_service.convert_to_final_format_with_crop(
                    input_path=temp_calibrated_path,
                    output_path=export_path,
                    crop_region=None,
                    crop_keyframes=variation_keyframes,
                    start_frame=clip.start_frame,  # Keep original frame reference for keyframe timing
                    fps=source_fps,  # Use detected source FPS to preserve timing
                    is_cv_format=is_cv_format,
                    gpu_acceleration=gpu_acceleration,
                )

            return success

        finally:
            # Clean up temporary calibrated file
            if os.path.exists(temp_calibrated_path):
                try:
                    os.unlink(temp_calibrated_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to clean up temp file {temp_calibrated_path}: {e}"
                    )

    except Exception as e:
        logger.error(f"Export failed for {variation}: {e}")
        return False
