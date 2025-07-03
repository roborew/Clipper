"""
Utility for processing clips in an automated fashion.
"""

import os
import json
import logging
import time
from pathlib import Path
from services.clip_service import Clip, save_clips, load_clips
from services.config_manager import ConfigManager

logger = logging.getLogger("clipper.processor")


def resolve_source_path(source_path, config_manager):
    """Simple wrapper to resolve source paths"""
    # Import here to avoid circular imports
    import sys
    import os

    script_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from scripts.process_clips import resolve_source_path as main_resolve

    return main_resolve(source_path, config_manager)


def extract_camera_type(source_path):
    """Simple wrapper to extract camera type"""
    # Import here to avoid circular imports
    import sys
    import os

    script_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from scripts.process_clips import extract_camera_type as main_extract

    return main_extract(source_path)


def camera_matches_filter(camera_type, camera_filter):
    """Simple wrapper to check camera filter matches"""
    # Import here to avoid circular imports
    import sys
    import os

    script_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from scripts.process_clips import camera_matches_filter as main_matches

    return main_matches(camera_type, camera_filter)


def scan_for_clips_to_process(config_manager=None):
    """
    Scan the config directory for clips with status "Process"

    Args:
        config_manager: ConfigManager instance, will create one if not provided

    Returns:
        List of tuples with (config_file_path, clip_list_index, clip) for each clip to process
    """
    try:
        # Create config manager if not provided
        if config_manager is None:
            config_manager = ConfigManager()

        # Get the base configs directory
        configs_dir = config_manager.configs_dir

        # List to store clips to process
        clips_to_process = []

        # Recursively scan the configs directory
        for root, _, files in os.walk(configs_dir):
            for file in files:
                if file.endswith("_clips.json"):
                    file_path = Path(root) / file

                    # Load clips from file
                    clips = load_clips(file_path)

                    # Look for clips with "Process" status
                    for i, clip in enumerate(clips):
                        if hasattr(clip, "status") and clip.status == "Process":
                            clips_to_process.append((file_path, i, clip))
                            logger.info(
                                f"Found clip to process: {clip.name} in {file_path}"
                            )

        return clips_to_process

    except Exception as e:
        logger.exception(f"Error scanning for clips to process: {str(e)}")
        return []


def update_clip_status(config_file_path, clip_index, new_status, processed_clip=None):
    """
    Update the status of a clip in a config file

    Args:
        config_file_path: Path to the config file
        clip_index: Index of the clip in the file
        new_status: New status to set ("Draft", "Process", or "Complete")
        processed_clip: Optional processed clip object to copy export_path from

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load clips from file
        clips = load_clips(config_file_path)

        # Check if the index is valid
        if clip_index < 0 or clip_index >= len(clips):
            logger.error(
                f"Invalid clip index: {clip_index}, file has {len(clips)} clips"
            )
            return False

        # Update the status
        clips[clip_index].status = new_status

        # If we have a processed clip, update the export_path too
        if processed_clip and hasattr(processed_clip, "export_path"):
            # Handle both string and list types for export_path
            clips[clip_index].export_path = processed_clip.export_path

            # Log what we're storing
            if isinstance(processed_clip.export_path, list):
                logger.info(
                    f"Updated export path to multiple paths: {processed_clip.export_path}"
                )
            else:
                logger.info(f"Updated export path to: {processed_clip.export_path}")

            # Copy any other export-related attributes
            if hasattr(processed_clip, "export_paths"):
                clips[clip_index].export_paths = processed_clip.export_paths

        clips[clip_index].update()  # Update the modified timestamp

        # Save the updated clips
        success = save_clips(clips, config_file_path)

        if success:
            logger.info(
                f"Updated clip status to {new_status}: {clips[clip_index].name}"
            )

        return success

    except Exception as e:
        logger.exception(f"Error updating clip status: {str(e)}")
        return False


def process_pending_clips(process_function, config_manager=None):
    """
    Find all clips with "Process" status and process them using the provided function

    Args:
        process_function: Function that takes a clip as input and returns True on success
        config_manager: ConfigManager instance, will create one if not provided

    Returns:
        Number of clips successfully processed
    """
    try:
        # Scan for clips to process
        clips_to_process = scan_for_clips_to_process(config_manager)

        # Keep track of successful processes
        success_count = 0

        # Process each clip
        for file_path, clip_index, clip in clips_to_process:
            try:
                logger.info(f"Processing clip: {clip.name}")

                # Call the process function
                result = process_function(clip)

                # Update the status if successful
                if result:
                    update_clip_status(file_path, clip_index, "Complete")
                    success_count += 1
                else:
                    logger.error(f"Processing failed for clip: {clip.name}")
            except Exception as e:
                logger.exception(f"Error processing clip {clip.name}: {str(e)}")

        return success_count

    except Exception as e:
        logger.exception(f"Error in process_pending_clips: {str(e)}")
        return 0


def get_pending_clips(config_manager=None):
    """
    Get a list of all clips with status "Process"

    Args:
        config_manager: ConfigManager instance, will create one if not provided

    Returns:
        List of Clip objects that are pending processing
    """
    try:
        # Get clips to process using scan_for_clips_to_process
        clips_to_process = scan_for_clips_to_process(config_manager)

        # Convert the tuples to just the clip objects
        pending_clips = [clip for _, _, clip in clips_to_process]

        logger.info(f"Found {len(pending_clips)} clips pending processing")
        return pending_clips

    except Exception as e:
        logger.exception(f"Error getting pending clips: {str(e)}")
        return []


def process_clip(
    clip,
    camera_filter=None,
    cv_optimized=False,
    both_formats=False,
    multi_crop=False,
    crop_variations="original,wide,full",
    wide_crop_factor=1.5,
    crop_camera_types=None,
    exclude_crop_camera_types=None,
    gpu_acceleration=False,
    progress_callback=None,
):
    """Process a single clip with streamlined logging"""

    try:
        logger.info(f"🎬 Processing: {clip.name}")

        # Resolve source path
        config_manager = ConfigManager()
        source_path = resolve_source_path(clip.source_path, config_manager)

        if not source_path.exists():
            logger.error(f"❌ Source not found: {source_path}")
            return False

        # Extract camera type for filtering
        camera_type = extract_camera_type(source_path)

        # Check if multi-crop applies to this camera
        should_use_multi_crop = multi_crop
        if multi_crop and crop_camera_types:
            should_use_multi_crop = any(
                camera_matches_filter(camera_type, ct) for ct in crop_camera_types
            )
        elif multi_crop and exclude_crop_camera_types:
            should_use_multi_crop = not any(
                camera_matches_filter(camera_type, ct)
                for ct in exclude_crop_camera_types
            )

        if should_use_multi_crop:
            logger.info(f"📐 Multi-crop enabled")

            success = process_clip_with_variations(
                clip=clip,
                source_path=source_path,
                config_manager=config_manager,
                crop_variations=crop_variations,
                wide_crop_factor=wide_crop_factor,
                cv_optimized=cv_optimized,
                both_formats=both_formats,
                gpu_acceleration=gpu_acceleration,
                progress_callback=progress_callback,
            )
        else:
            logger.info(f"🎬 Single crop")

            # Single crop processing
            success = process_single_crop_clip(
                clip=clip,
                source_path=source_path,
                config_manager=config_manager,
                cv_optimized=cv_optimized,
                both_formats=both_formats,
                gpu_acceleration=gpu_acceleration,
                progress_callback=progress_callback,
            )

        if success:
            logger.info(f"✅ Success: {clip.name}")
            return True
        else:
            logger.error(f"❌ Failed: {clip.name}")
            return False

    except Exception as e:
        logger.error(f"❌ Error processing {clip.name}: {e}")
        return False


def process_single_crop_clip(
    clip,
    source_path,
    config_manager,
    cv_optimized=False,
    both_formats=False,
    gpu_acceleration=False,
    progress_callback=None,
):
    """Process a single clip with one crop (no variations) using the correct pipeline"""
    try:
        from services import clip_export_service, calibration_service
        import tempfile
        import os

        # Get the source video FPS - critical for preserving timing
        from utils.multi_crop import get_source_video_fps

        source_fps = get_source_video_fps(source_path)
        logger.info(f"🎬 Source FPS: {source_fps} (preserving original timing)")

        # Determine formats to export
        formats = []
        if both_formats:
            formats = ["regular", "cv"]
        elif cv_optimized:
            formats = ["cv"]
        else:
            formats = ["regular"]

        # Step 1: Extract + Calibrate the clip segment (following export_clip_efficient pattern)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_calibrated_path = temp_file.name

        try:
            # Check if calibration is enabled and apply it
            if clip_export_service.is_calibration_enabled(config_manager):
                camera_type_for_cal = calibration_service.get_camera_type_from_path(
                    source_path, config_manager
                )
                logger.info(f"🔧 Calibrating: {camera_type_for_cal}")

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
                logger.info(f"🎬 Extracting (no calibration)")
                calibrate_success = clip_export_service.extract_clip_frames(
                    input_path=source_path,
                    output_path=temp_calibrated_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    gpu_acceleration=gpu_acceleration,
                )

            if not calibrate_success:
                logger.error(f"Failed to extract/calibrate clip")
                return False

            # Step 2: Apply crop and convert to final formats
            all_success = True

            for format_type in formats:
                is_cv = format_type == "cv"

                # Create export path
                camera_type = source_path.parent.parent.name
                session = source_path.parent.name

                format_ext = ".mkv" if is_cv else ".mp4"
                format_dir = "ffv1" if is_cv else "h264"

                # Use configured clips directory from config.yaml
                export_dir = (
                    config_manager.clips_dir / format_dir / camera_type / session
                )
                export_dir.mkdir(parents=True, exist_ok=True)

                # Create proper filename with source video name prefix
                source_video_name = (
                    source_path.stem
                )  # Get filename without extension (e.g., "C0001")
                export_filename = f"{source_video_name}_{clip.name}{format_ext}"
                export_path = export_dir / export_filename

                # Apply crop and convert to final format
                if hasattr(clip, "crop_keyframes") and clip.crop_keyframes:
                    # Use keyframes for cropping
                    success = clip_export_service.convert_to_final_format_with_crop(
                        input_path=temp_calibrated_path,
                        output_path=export_path,
                        crop_region=None,
                        crop_keyframes=clip.crop_keyframes,
                        start_frame=clip.start_frame,  # Keep original frame reference for keyframe timing
                        fps=source_fps,  # Use detected source FPS to preserve timing
                        is_cv_format=is_cv,
                        gpu_acceleration=gpu_acceleration,
                    )
                else:
                    # Use static crop region if available
                    crop_region = getattr(clip, "crop_region", None)
                    success = clip_export_service.convert_to_final_format_with_crop(
                        input_path=temp_calibrated_path,
                        output_path=export_path,
                        crop_region=crop_region,
                        crop_keyframes=None,
                        start_frame=clip.start_frame,  # Keep original frame reference for timing
                        fps=source_fps,  # Use detected source FPS to preserve timing
                        is_cv_format=is_cv,
                        gpu_acceleration=gpu_acceleration,
                    )

                if not success:
                    all_success = False
                    logger.error(f"❌ {format_type} format failed")
                else:
                    logger.info(f"✅ {format_type} format")

            return all_success

        finally:
            # Clean up temporary calibrated file
            if os.path.exists(temp_calibrated_path):
                try:
                    os.unlink(temp_calibrated_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

    except Exception as e:
        logger.error(f"Single crop processing failed: {e}")
        return False


def process_clip_with_variations(
    clip,
    source_path,
    config_manager,
    crop_variations,
    wide_crop_factor,
    cv_optimized,
    both_formats,
    gpu_acceleration,
    progress_callback,
):
    """Simple wrapper for multi-crop processing"""
    # Import here to avoid circular imports
    from utils.multi_crop import process_clip_with_variations as multi_crop_func

    return multi_crop_func(
        clip=clip,
        source_path=source_path,
        config_manager=config_manager,
        crop_variations=crop_variations,
        wide_crop_factor=wide_crop_factor,
        cv_optimized=cv_optimized,
        both_formats=both_formats,
        gpu_acceleration=gpu_acceleration,
        progress_callback=progress_callback,
    )


# Example usage:
# def my_processing_function(clip):
#     # Process the clip here (e.g., run FFmpeg to encode it)
#     # ...
#     return True  # Return True if successful
#
# num_processed = process_pending_clips(my_processing_function)
# print(f"Successfully processed {num_processed} clips")
