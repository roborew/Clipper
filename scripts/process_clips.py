#!/usr/bin/env python
"""
Script for automated clip processing.

This script scans for clips with status "Process" and processes them.
It can be run as a scheduled job or a daemon process.
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent.futures
from datetime import datetime

# Add the project root to the path so we can import the app modules
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.utils.clip_processor import (
    process_pending_clips,
    get_pending_clips,
    update_clip_status,
    scan_for_clips_to_process,
)
from src.services.clip_service import save_clips, load_clips
from src.services.config_manager import ConfigManager
from src.services import video_service
from src.services import proxy_service
from src.services import clip_service
from src.services import clip_export_service  # NEW: Efficient clip export
from src.utils.multi_crop import process_clip_with_variations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(script_dir / "clip_processor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("clip_processor")


def extract_camera_type(video_path):
    """
    Extract camera type from the video path

    Args:
        video_path: Path to the video file

    Returns:
        Camera type as string, or "UNKNOWN" if not found
    """
    parts = Path(video_path).parts

    # Common camera brand prefixes
    camera_prefixes = ["SONY", "GP", "GOPRO", "CANON", "NIKON", "CAM"]

    # First, try to find an exact folder name match
    for part in parts:
        # Return the exact folder name that contains a camera prefix
        for prefix in camera_prefixes:
            if prefix in part.upper():
                return part

    # If no exact match, return "UNKNOWN"
    return "UNKNOWN"


def camera_matches_filter(camera_type, camera_filter):
    """
    Check if a camera type matches the filter

    Args:
        camera_type: The camera type extracted from the path
        camera_filter: The camera filter to match against

    Returns:
        True if the camera matches the filter, False otherwise
    """
    if not camera_filter:
        return True

    # Case-insensitive comparison
    camera_type_upper = camera_type.upper()
    camera_filter_upper = camera_filter.upper()

    # Exact match
    if camera_type_upper == camera_filter_upper:
        return True

    # Prefix match (e.g., "SONY" matches "SONY_70")
    if camera_type_upper.startswith(camera_filter_upper):
        return True

    # Substring match (e.g., "GP" matches "GP1")
    if camera_filter_upper in camera_type_upper:
        return True

    return False


def extract_session_folder(video_path):
    """
    Extract session folder from the video path

    Args:
        video_path: Path to the video file

    Returns:
        Session folder name as string, or "UNKNOWN" if not found
    """
    parts = Path(video_path).parts

    # Common camera brand prefixes
    camera_prefixes = ["SONY", "GP", "GOPRO", "CANON", "NIKON", "CAM"]

    # Try to find the session folder (folder after camera type)
    for i, part in enumerate(parts):
        # Check if this part contains a camera prefix
        if any(prefix in part.upper() for prefix in camera_prefixes):
            # Return the next folder if it exists
            if i + 1 < len(parts):
                return parts[i + 1]
            break

    # If no session folder found, try to use the parent folder of the video file
    if len(parts) >= 2:
        return parts[-2]  # Second-to-last part is the parent folder

    return "UNKNOWN"


def calculate_wider_crop(original_crop, factor, frame_dimensions):
    """
    Calculate a wider crop region centered around the original crop,
    with intelligent edge boundary handling.

    Args:
        original_crop: Original crop region (x, y, width, height)
        factor: Multiplier for crop size (1.5 = 50% larger)
        frame_dimensions: (width, height) of the source frame

    Returns:
        New crop region (x, y, width, height)
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
    # If we're pushing against left edge
    if ideal_x < 0:
        new_x = 0
    # If we're pushing against right edge
    elif ideal_x + new_width > frame_width:
        new_x = max(0, frame_width - new_width)
    else:
        new_x = ideal_x

    # Same for vertical position
    if ideal_y < 0:
        new_y = 0
    elif ideal_y + new_height > frame_height:
        new_y = max(0, frame_height - new_height)
    else:
        new_y = ideal_y

    return (new_x, new_y, new_width, new_height)


def resolve_source_path(source_path, config_manager):
    """
    Resolve a source path to an absolute path, handling relative paths and calibration settings

    Args:
        source_path: The source path to resolve
        config_manager: ConfigManager instance

    Returns:
        Resolved absolute path as a Path object
    """
    # Convert to Path object if it's a string
    path = Path(source_path) if isinstance(source_path, str) else source_path

    # If it's already an absolute path, return it
    if path.is_absolute():
        return path

    # Get calibration settings to determine which source folder to use
    calib_settings = config_manager.get_calibration_settings()
    use_calibrated_footage = calib_settings.get("use_calibrated_footage", False)

    # Get folder names from config
    raw_folder = config_manager.config["directories"]["source"]["raw"]
    calibrated_folder = config_manager.config["directories"]["source"]["calibrated"]

    logger.debug(f"Raw folder from config: {raw_folder}")
    logger.debug(f"Calibrated folder from config: {calibrated_folder}")

    # Determine the correct source path based on calibration settings
    str_path = str(path)

    # HANDLE LEGACY HARDCODED PATHS: Check for old "data/source/" prefix
    if str_path.startswith("data/source/"):
        logger.info(f"Detected legacy hardcoded path: {str_path}")
        # Strip the "data/source/" prefix to get the relative part
        legacy_relative = str_path[len("data/source/") :]
        logger.info(f"Extracted legacy relative path: {legacy_relative}")

        # The legacy relative path should start with a folder like "00_RAW" or "RAW"
        # Replace it with the current config and rebuild with proper source base
        if legacy_relative.startswith("00_RAW/") or legacy_relative.startswith("RAW/"):
            # Extract the part after the folder
            if legacy_relative.startswith("00_RAW/"):
                rel_path_after_folder = legacy_relative[len("00_RAW/") :]
            else:  # RAW/
                rel_path_after_folder = legacy_relative[len("RAW/") :]

            # Build the new path using the current config
            if use_calibrated_footage:
                corrected_path = (
                    config_manager.source_calibrated / rel_path_after_folder
                )
            else:
                corrected_path = config_manager.source_raw / rel_path_after_folder

            logger.info(f"Corrected legacy path to: {corrected_path}")
            return corrected_path

        elif legacy_relative.startswith(
            "01_CALIBRATION/"
        ) or legacy_relative.startswith("CALIBRATION/"):
            # Extract the part after the calibration folder
            if legacy_relative.startswith("01_CALIBRATION/"):
                rel_path_after_folder = legacy_relative[len("01_CALIBRATION/") :]
            else:  # CALIBRATION/
                rel_path_after_folder = legacy_relative[len("CALIBRATION/") :]

            # Build the new path using the current config
            if use_calibrated_footage:
                corrected_path = (
                    config_manager.source_calibrated / rel_path_after_folder
                )
            else:
                corrected_path = config_manager.source_raw / rel_path_after_folder

            logger.info(f"Corrected legacy calibration path to: {corrected_path}")
            return corrected_path

    # Handle relative paths that start with the configured source folder names
    if str_path.startswith(raw_folder) or str_path.startswith(calibrated_folder):
        # Determine which subfolder to use based on calibration setting
        if use_calibrated_footage:
            # If using calibrated footage, ensure path uses the calibrated subfolder
            if str_path.startswith(raw_folder):
                rel_path = str_path.replace(raw_folder, calibrated_folder, 1)
                logger.info(f"Adjusted path to use calibrated footage: {rel_path}")
            else:
                rel_path = str_path
        else:
            # If using raw footage, ensure path uses the raw subfolder
            if str_path.startswith(calibrated_folder):
                rel_path = str_path.replace(calibrated_folder, raw_folder, 1)
                logger.info(f"Adjusted path to use raw footage: {rel_path}")
            else:
                rel_path = str_path

        return Path(os.path.join(config_manager.source_base, rel_path))

    # Try to resolve using the config manager's base directories
    try:
        # Try source base first with appropriate subfolder based on calibration setting
        if use_calibrated_footage:
            # Try with calibrated footage path
            calibrated_path = config_manager.source_calibrated / path
            if calibrated_path.exists():
                return calibrated_path
        else:
            # Try with raw footage path
            raw_path = config_manager.source_raw / path
            if raw_path.exists():
                return raw_path

        # Try with direct source base as fallback
        full_path = Path(config_manager.source_base) / path
        if full_path.exists():
            return full_path

        # Then try output base
        full_path = Path(config_manager.output_base) / path
        if full_path.exists():
            return full_path
    except Exception as e:
        logger.warning(f"Error resolving path {path}: {str(e)}")

    # Return the original path if we couldn't resolve it
    return path


def process_clip(
    clip,
    camera_filter=None,
    cv_optimized=False,
    both_formats=False,
    multi_crop=False,
    crop_variations=None,
    wide_crop_factor=1.5,
    gpu_acceleration=False,
):
    """
    Process a single clip - this is where you would implement your custom processing logic

    Args:
        clip: The Clip object to process
        camera_filter: Optional filter to only process clips from specific cameras
        cv_optimized: Whether to optimize for computer vision (higher quality)
        both_formats: Whether to export in both regular and CV-optimized formats
        multi_crop: Whether to generate multiple crop variations
        crop_variations: List of crop variations to generate (original, wide, full)
        wide_crop_factor: Multiplier for the wide crop (1.5 = 50% larger)

    Returns:
        True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Processing clip: {clip.name}")
        logger.info(
            f"DEBUG: Clip details - start_frame: {clip.start_frame}, end_frame: {clip.end_frame}"
        )
        if cv_optimized:
            logger.info("Using CV optimization for export")
        if both_formats:
            logger.info("Exporting in both regular and CV-optimized formats")
        if multi_crop:
            logger.info(f"Generating multiple crop variations: {crop_variations}")
            logger.info(f"Wide crop factor: {wide_crop_factor}")

        # Get the full path to the source video
        config_manager = ConfigManager()
        source_path = resolve_source_path(clip.source_path, config_manager)

        logger.info(f"Resolved source path: {source_path}")
        logger.info(f"DEBUG: Source path exists: {os.path.exists(source_path)}")

        # Check camera type if filter is specified
        camera_type = extract_camera_type(source_path)
        if camera_filter:
            if not camera_matches_filter(camera_type, camera_filter):
                logger.info(
                    f"Skipping clip {clip.name} from camera {camera_type} (doesn't match filter: {camera_filter})"
                )
                return False

        # STEP 1: Only delete previous exports if clip was previously processed
        # Check if clip has existing export paths that need cleaning up
        has_existing_exports = False
        paths_to_delete = []

        # Collect paths from export_path (either list or string)
        if hasattr(clip, "export_path") and clip.export_path:
            if isinstance(clip.export_path, list):
                # Only add non-empty paths
                valid_paths = [p for p in clip.export_path if p]
                if valid_paths:
                    has_existing_exports = True
                    paths_to_delete.extend(valid_paths)
                    logger.info(
                        f"Found {len(valid_paths)} existing paths in export_path array"
                    )
            elif isinstance(clip.export_path, str) and clip.export_path.strip():
                has_existing_exports = True
                paths_to_delete.append(clip.export_path)
                logger.info(f"Found existing path in export_path: {clip.export_path}")

        # Also check export_paths attribute as a backup
        if hasattr(clip, "export_paths") and clip.export_paths:
            additional_paths = [
                p.strip() for p in clip.export_paths.split(",") if p.strip()
            ]
            if additional_paths:
                has_existing_exports = True
                for path in additional_paths:
                    if path and path not in paths_to_delete:
                        paths_to_delete.append(path)
                logger.info(
                    f"Found {len(additional_paths)} paths in export_paths string"
                )

        # Only delete if clip was previously processed (has export paths)
        if has_existing_exports and paths_to_delete:
            logger.info(
                f"Clip was previously processed. Deleting {len(paths_to_delete)} previous export(s) before processing"
            )
            for path in paths_to_delete:
                if os.path.exists(path):
                    try:
                        logger.info(f"Cleaning up previous export: {path}")
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Failed to delete previous export {path}: {e}")
        else:
            if not has_existing_exports:
                logger.info(
                    f"First-time processing for clip {clip.name} - no previous exports to clean up"
                )
            elif not paths_to_delete:
                logger.info(
                    f"Clip {clip.name} has export_path set but no valid paths to delete"
                )

        # STEP 2: Initialize export collections
        # Always initialize export_path as an empty list to collect new paths
        clip.export_path = []
        if hasattr(clip, "export_paths"):
            clip.export_paths = ""

        # Keep a master list of all generated export paths
        all_export_paths = []

        # STEP 3: Determine formats to process
        formats_to_process = []
        if both_formats:
            formats_to_process = [False, True]  # Regular, then CV-optimized
        elif cv_optimized:
            formats_to_process = [True]  # CV-optimized only
        else:
            formats_to_process = [False]  # Regular only

        overall_success = True

        # Define a function to process a single clip variation with a specific format
        def process_single_clip(variation_clip, is_cv_format=False):
            try:
                # Create a descriptive progress bar
                format_label = "CV" if is_cv_format else "H.264"
                pbar = tqdm(
                    total=100,
                    desc=f"Processing {variation_clip.name} ({format_label}) - EFFICIENT",
                    unit="%",
                )

                # Create a progress callback for export
                def progress_callback(progress):
                    # Update progress bar based on percentage (0-1)
                    current_percent = int(progress * 100)
                    # Update to the current percentage, avoiding going backwards
                    if current_percent > pbar.n:
                        pbar.update(current_percent - pbar.n)

                # Use the NEW efficient clip export service instead of proxy_service
                # This processes only the needed frames (59 vs 44,760) - ~760x faster!
                export_paths = clip_export_service.export_clip_efficient(
                    clip=variation_clip,
                    config_manager=config_manager,
                    multi_crop=False,  # Single clip processing
                    multi_format=is_cv_format,  # CV format when requested
                    progress_callback=progress_callback,
                    gpu_acceleration=gpu_acceleration,
                )

                pbar.close()

                if export_paths and len(export_paths) > 0:
                    export_path = export_paths[
                        0
                    ]  # Take the first (and likely only) path
                    logger.info(
                        f"Successfully processed {variation_clip.name} ({format_label}) - EFFICIENT: {export_path}"
                    )
                    return export_path
                else:
                    logger.error(
                        f"Failed to process {variation_clip.name} ({format_label}) - EFFICIENT"
                    )
                    return None
            except Exception as e:
                logger.exception(f"Error processing {variation_clip.name}: {str(e)}")
                return None

        # STEP 4: Process each format and collect ALL export paths
        for is_cv_optimized in formats_to_process:
            format_label = "CV" if is_cv_optimized else "H.264"
            logger.info(f"Processing format: {format_label}")

            # If multi-crop is enabled, use the process_clip_with_variations function
            if multi_crop:
                # Process with variations - this adds paths to clip.export_path
                format_results = process_clip_with_variations(
                    lambda var_clip, **kwargs: process_single_clip(
                        var_clip, is_cv_optimized
                    ),
                    clip,
                    crop_variations,
                    wide_crop_factor,
                    camera_type=camera_type,
                    config_manager=config_manager,
                    multi_crop=multi_crop,
                    source_path=source_path,  # Pass the resolved source path
                )

                # Check if all variations were successful
                if format_results and isinstance(format_results, dict):
                    variation_success = all(format_results.values())
                    if not variation_success:
                        overall_success = False
                        logger.warning(
                            f"Some variations failed for {format_label} format"
                        )

                    # Log all variation results for debugging
                    for variation, success in format_results.items():
                        logger.info(
                            f"Variation {variation} result: {'Success' if success else 'Failed'}"
                        )
                else:
                    # Handle the case where format_results isn't a dict (for backward compatibility)
                    if hasattr(clip, "export_path") and clip.export_path:
                        # We have export paths, so something worked
                        logger.info(
                            f"Got export paths after multi-crop processing: {clip.export_path}"
                        )
                    else:
                        # No export paths suggests failure
                        overall_success = False
                        logger.warning(
                            f"No export paths found after multi-crop processing for {format_label} format"
                        )

                # Log current state of export paths
                if isinstance(clip.export_path, list):
                    logger.info(
                        f"After processing {format_label} format, export_path contains {len(clip.export_path)} paths"
                    )
                    # Collect paths from this format's variations
                    for path in clip.export_path:
                        if path and path not in all_export_paths:
                            all_export_paths.append(path)
                            logger.info(
                                f"Added path from multi_crop to master list: {path}"
                            )
            else:
                # Process a single clip with no variations
                export_path = process_single_clip(clip, is_cv_optimized)
                if export_path:
                    # Add to clip.export_path
                    if isinstance(clip.export_path, list):
                        clip.export_path.append(export_path)
                    else:
                        clip.export_path = [export_path]
                    # Also add to our master list
                    all_export_paths.append(export_path)
                else:
                    overall_success = False

        # Also collect all paths from clip.export_path
        if hasattr(clip, "export_path") and isinstance(clip.export_path, list):
            for path in clip.export_path:
                if path and path not in all_export_paths:
                    all_export_paths.append(path)

        # STEP 5: Final updates to ensure consistent export path format
        if hasattr(clip, "export_path"):
            # Make sure clip.export_path contains ALL collected paths
            if all_export_paths:
                clip.export_path = list(
                    all_export_paths
                )  # Create a new list to avoid reference issues
                logger.info(
                    f"Set final export_path using master list with {len(all_export_paths)} paths"
                )

            if isinstance(clip.export_path, list):
                logger.info(
                    f"Final export_path contains {len(clip.export_path)} paths: {clip.export_path}"
                )

                # Also update export_paths string attribute for compatibility
                if hasattr(clip, "export_paths"):
                    clip.export_paths = ",".join(clip.export_path)
            else:
                # Shouldn't happen but just in case
                logger.warning(
                    f"Unexpected: Final export_path is not a list: {clip.export_path}"
                )
                if isinstance(clip.export_path, str) and clip.export_path:
                    # Convert single string to list
                    clip.export_path = [clip.export_path]
                    if hasattr(clip, "export_paths"):
                        clip.export_paths = clip.export_path

        # Log final export path state
        logger.info(f"Final export_path: {clip.export_path}")

        # FINAL CHECK: Make absolutely sure export_path is not empty if we have successful exports
        if overall_success and (not clip.export_path or len(clip.export_path) == 0):
            logger.error(
                f"CRITICAL ERROR: clip.export_path is empty despite successful processing!"
            )
            # Restore the export paths we collected during processing
            if "all_export_paths" in locals() and all_export_paths:
                logger.info(f"Restoring collected export paths: {all_export_paths}")
                clip.export_path = all_export_paths
                # Update export_paths string attribute for compatibility
                if hasattr(clip, "export_paths"):
                    clip.export_paths = ",".join(clip.export_path)
            else:
                # Try to find generated clips by scanning the filesystem
                logger.info(f"Scanning filesystem for generated clips for {clip.name}")
                found_files = find_generated_clips(
                    clip,
                    config_manager,
                    both_formats=both_formats,
                    multi_crop=multi_crop,
                    crop_variations=crop_variations,
                )

                if found_files:
                    logger.info(
                        f"Found {len(found_files)} generated files on disk for clip {clip.name}"
                    )
                    clip.export_path = found_files

                    # Update export_paths for compatibility
                    if hasattr(clip, "export_paths"):
                        clip.export_paths = ",".join(found_files)
                        logger.info(f"Updated export_paths string: {clip.export_paths}")
                else:
                    logger.error(
                        f"Could not find any generated clips on disk for {clip.name}!"
                    )

        # Last resort check when running with both_formats - make sure we have both mp4 and mkv files
        if (
            both_formats
            and overall_success
            and hasattr(clip, "export_path")
            and isinstance(clip.export_path, list)
        ):
            mp4_files = [p for p in clip.export_path if p.endswith(".mp4")]
            mkv_files = [p for p in clip.export_path if p.endswith(".mkv")]

            # If we have MP4 files but no MKV files (or vice versa), we're missing some paths
            if (mp4_files and not mkv_files) or (mkv_files and not mp4_files):
                logger.warning(
                    f"Missing some format files - MP4: {len(mp4_files)}, MKV: {len(mkv_files)}"
                )

                # Try one more search to find all clips
                found_files = find_generated_clips(
                    clip,
                    config_manager,
                    both_formats=both_formats,
                    multi_crop=multi_crop,
                    crop_variations=crop_variations,
                )

                if len(found_files) > len(clip.export_path):
                    logger.info(
                        f"Found more files on disk ({len(found_files)}) than in export_path ({len(clip.export_path)})"
                    )
                    clip.export_path = found_files

                    # Update export_paths for compatibility
                    if hasattr(clip, "export_paths"):
                        clip.export_paths = ",".join(found_files)
                        logger.info(f"Updated export_paths string: {clip.export_paths}")

        # After successful processing, immediately update the clip's status and save
        if clip.export_path and len(clip.export_path) > 0:
            clip.status = "Complete"
            clip.modified_at = datetime.now().isoformat()

            # Find and update the clip in its config file
            for config_file in config_manager.configs_dir.glob("**/*.json"):
                try:
                    config_clips = clip_service.load_clips(config_file)
                    for i, config_clip in enumerate(config_clips):
                        if (
                            hasattr(clip, "id")
                            and hasattr(config_clip, "id")
                            and clip.id == config_clip.id
                        ):
                            # Update the clip in the config
                            config_clips[i].status = "Complete"
                            config_clips[i].export_path = clip.export_path
                            config_clips[i].modified_at = clip.modified_at

                            # Save immediately
                            save_success = clip_service.save_clips(
                                config_clips, config_file
                            )
                            if save_success:
                                logger.info(
                                    f"Successfully updated and saved clip {clip.name} status to Complete"
                                )
                            else:
                                logger.error(
                                    f"Failed to save updated status for clip {clip.name}"
                                )
                            break
                except Exception as e:
                    logger.warning(
                        f"Error updating clip in config file {config_file}: {e}"
                    )
                    continue

        return True if clip.status == "Complete" else False

    except Exception as e:
        logger.exception(f"Error processing clip {clip.name}: {str(e)}")
        return False


def save_processed_clips(clips, config_manager):
    """
    Save processed clips back to their config files

    Args:
        clips: List of clips that have been processed
        config_manager: ConfigManager instance

    Returns:
        Number of successfully saved clips
    """
    saved_count = 0
    logger.info(f"Saving {len(clips)} processed clips")

    # Get all clip config files
    clip_files = {}
    for config_file in config_manager.configs_dir.glob("**/*.json"):
        try:
            config_clips = clip_service.load_clips(config_file)
            if config_clips:
                clip_files[config_file] = config_clips
        except Exception as e:
            logger.warning(f"Error loading clips from {config_file}: {e}")

    # For each processed clip, find its source config file and update it
    for processed_clip in clips:
        # Log the export path before saving
        logger.info(
            f"DEBUG before saving - Clip {processed_clip.name} export_path: {processed_clip.export_path}"
        )
        if hasattr(processed_clip, "export_path") and processed_clip.export_path:
            # Check if it's list or string
            if isinstance(processed_clip.export_path, list):
                logger.info(
                    f"Export path is a list with {len(processed_clip.export_path)} items"
                )
                # Check if list contains valid items
                if all(os.path.exists(p) for p in processed_clip.export_path):
                    logger.info("All paths in export_path exist on disk")
                else:
                    missing = [
                        p for p in processed_clip.export_path if not os.path.exists(p)
                    ]
                    logger.warning(f"Some paths in export_path don't exist: {missing}")

                    # Filter out missing paths to avoid saving invalid paths
                    processed_clip.export_path = [
                        p for p in processed_clip.export_path if os.path.exists(p)
                    ]
                    logger.info(
                        f"Filtered export_path to only include existing files: {len(processed_clip.export_path)} remain"
                    )
            else:
                logger.info(
                    f"Export path is not a list but a {type(processed_clip.export_path)}"
                )
                # Convert string to list if needed
                if (
                    isinstance(processed_clip.export_path, str)
                    and processed_clip.export_path
                ):
                    processed_clip.export_path = [processed_clip.export_path]
                    logger.info(
                        f"Converted string export_path to list: {processed_clip.export_path}"
                    )

        found = False
        for config_file, config_clips in clip_files.items():
            for i, config_clip in enumerate(config_clips):
                # Match by ID if available
                if (
                    hasattr(processed_clip, "id")
                    and hasattr(config_clip, "id")
                    and processed_clip.id == config_clip.id
                ):
                    logger.info(
                        f"Found clip {processed_clip.name} (ID: {processed_clip.id}) in {config_file}"
                    )

                    # Check state of export_path before updating
                    if hasattr(config_clip, "export_path"):
                        logger.info(
                            f"Before update - config_clip.export_path: {config_clip.export_path}"
                        )

                    # Make a deep copy of the export_path to ensure it's preserved
                    if (
                        hasattr(processed_clip, "export_path")
                        and processed_clip.export_path
                    ):
                        # Keep a reference to the original export_path to debug serialization
                        original_export_path = processed_clip.export_path

                        if isinstance(processed_clip.export_path, list):
                            # Make a copy of the list to avoid reference issues
                            processed_clip.export_path = list(
                                processed_clip.export_path
                            )

                            # Update the export_paths string attribute for compatibility
                            processed_clip.export_paths = ",".join(
                                processed_clip.export_path
                            )
                            logger.info(
                                f"Set export_paths string attribute to: {processed_clip.export_paths}"
                            )

                    # Update the clip
                    config_clips[i].status = processed_clip.status

                    # Copy the export_path correctly
                    if (
                        hasattr(processed_clip, "export_path")
                        and processed_clip.export_path
                    ):
                        config_clips[i].export_path = processed_clip.export_path

                    # Copy the export_paths string if it exists
                    if (
                        hasattr(processed_clip, "export_paths")
                        and processed_clip.export_paths
                    ):
                        config_clips[i].export_paths = processed_clip.export_paths

                    # Make sure the status is correct
                    if processed_clip.status == "Complete":
                        config_clips[i].status = "Complete"
                        logger.info(
                            f"Setting status to Complete for clip {processed_clip.name}"
                        )

                    # Update modified timestamp
                    config_clips[i].update()

                    # Log what we're about to save
                    logger.info(
                        f"Before saving - Updated clip {config_clips[i].name} with status: {config_clips[i].status}, "
                        f"export_path: {config_clips[i].export_path}"
                    )

                    # Save the changes
                    save_success = clip_service.save_clips(config_clips, config_file)

                    if save_success:
                        saved_count += 1
                        logger.info(
                            f"Successfully saved updated clip {processed_clip.name} to {config_file}"
                        )

                        # Double-check the saved file to verify changes were saved
                        try:
                            reloaded_clips = clip_service.load_clips(config_file)
                            if i < len(reloaded_clips):
                                logger.info(
                                    f"Verification - Reloaded clip has status: {reloaded_clips[i].status}, "
                                    f"export_path: {reloaded_clips[i].export_path}"
                                )
                            else:
                                logger.warning(
                                    f"Verification - Could not reload clip at index {i}"
                                )
                        except Exception as e:
                            logger.warning(f"Error verifying saved clip: {e}")
                    else:
                        logger.error(
                            f"Failed to save updated clip {processed_clip.name}"
                        )

                    found = True
                    break

            if found:
                break

        if not found:
            logger.warning(
                f"Could not find configuration file for clip: {processed_clip.name}"
            )

    return saved_count


def verify_export_count(clip, both_formats, multi_crop, crop_variations):
    """
    Verify that the correct number of exported files were created.

    Args:
        clip: The processed clip
        both_formats: Whether both formats were enabled
        multi_crop: Whether multi-crop was enabled
        crop_variations: The crop variations used

    Returns:
        bool: True if the correct number of exports exist, False otherwise
    """
    if not hasattr(clip, "export_path"):
        logger.warning(f"Clip {clip.name} has no export_path attribute")
        return False

    if not isinstance(clip.export_path, list):
        logger.warning(
            f"Clip {clip.name} export_path is not a list: {clip.export_path}"
        )
        return False

    # Calculate expected number of exports
    expected_count = 1  # Default: single export

    if multi_crop:
        # Count the number of variations
        if isinstance(crop_variations, str):
            variation_list = [
                v.strip() for v in crop_variations.split(",") if v.strip()
            ]
            expected_count = len(variation_list)
        elif isinstance(crop_variations, list):
            expected_count = len(crop_variations)

    if both_formats:
        expected_count *= 2  # Double for both formats

    actual_count = len(clip.export_path)

    if actual_count != expected_count:
        logger.warning(
            f"Expected {expected_count} exports for clip {clip.name}, but found {actual_count}. "
            f"Options: both_formats={both_formats}, multi_crop={multi_crop}, variations={crop_variations}"
        )
        # Check if files actually exist on disk
        missing_files = [path for path in clip.export_path if not os.path.exists(path)]
        if missing_files:
            logger.warning(
                f"The following files in export_path don't exist on disk: {missing_files}"
            )
        return False
    else:
        logger.info(
            f"Verified correct number of exports ({actual_count}) for clip {clip.name}"
        )
        # Check if files actually exist on disk
        all_exist = all(os.path.exists(path) for path in clip.export_path)
        if not all_exist:
            missing_files = [
                path for path in clip.export_path if not os.path.exists(path)
            ]
            logger.warning(
                f"The following files in export_path don't exist on disk: {missing_files}"
            )
            return False

        logger.info(f"All exported files exist on disk")
        return True


def find_generated_clips(
    clip, config_manager, both_formats=False, multi_crop=False, crop_variations=None
):
    """
    Scan filesystem for generated clips that match the clip name pattern

    Args:
        clip: The Clip object
        config_manager: ConfigManager instance
        both_formats: Whether both formats were generated
        multi_crop: Whether multiple crop variations were generated
        crop_variations: List of crop variations used

    Returns:
        List of paths to the generated clips found on disk
    """
    try:
        # Figure out potential output directories
        export_base = config_manager.output_base
        h264_dir = Path(export_base) / "03_CLIPPED" / "h264"
        ffv1_dir = Path(export_base) / "03_CLIPPED" / "ffv1"

        # Get source path components
        source_path = Path(clip.source_path)
        video_name = os.path.splitext(source_path.name)[0]

        # Get camera type and session from source path
        camera_type = extract_camera_type(source_path)
        session = extract_session_folder(source_path)

        # Determine which directories to check based on format options
        dirs_to_check = []
        if both_formats:
            dirs_to_check = [h264_dir, ffv1_dir]
            logger.info(f"Checking both format directories: {dirs_to_check}")
        else:
            # Default to h264 unless cv_optimized was explicitly used alone
            dirs_to_check = [h264_dir]
            logger.info(f"Checking single format directory: {dirs_to_check}")

        # Determine potential clip name patterns
        clip_patterns = []

        # Base name without variation suffix
        base_name = clip.name
        for suffix in ["_original", "_wide", "_full"]:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        # If multi_crop, create patterns for each variation
        if multi_crop and crop_variations:
            if isinstance(crop_variations, str):
                variations = [
                    v.strip() for v in crop_variations.split(",") if v.strip()
                ]
            else:
                variations = crop_variations

            logger.info(f"Looking for variations: {variations}")

            # Create pattern for each variation
            for variation in variations:
                if variation == "original":
                    # Original might not have a suffix
                    clip_patterns.append(f"{video_name}_{base_name}")
                    clip_patterns.append(f"{video_name}_{base_name}_original")
                else:
                    clip_patterns.append(f"{video_name}_{base_name}_{variation}")
        else:
            # Just use the clip name as is
            clip_patterns.append(f"{video_name}_{clip.name}")

        logger.info(f"Searching for clip patterns: {clip_patterns}")

        # Find matching files in each directory
        found_files = []

        for dir_path in dirs_to_check:
            # Construct full directory path with camera and session subfolders
            full_dir = dir_path / camera_type / session

            if not full_dir.exists():
                logger.warning(f"Export directory does not exist: {full_dir}")
                continue

            logger.info(f"Checking directory: {full_dir}")

            # Check for each pattern
            for pattern in clip_patterns:
                # Look for files with the pattern and both mp4 and mkv extensions
                for ext in [".mp4", ".mkv"]:
                    # Use EXACT filename match instead of glob pattern to avoid matching other clips
                    # This prevents C0001_clip_1 from matching C0001_clip_10, C0001_clip_11, etc.
                    exact_filename = f"{pattern}{ext}"
                    exact_file_path = full_dir / exact_filename
                    logger.info(f"Searching with exact filename: {exact_filename}")

                    if (
                        exact_file_path.exists()
                        and str(exact_file_path) not in found_files
                    ):
                        found_files.append(str(exact_file_path))
                        logger.info(f"Found generated clip: {exact_file_path}")
                    else:
                        logger.debug(f"File does not exist: {exact_file_path}")

        logger.info(f"Total found files: {len(found_files)}")
        return found_files

    except Exception as e:
        logger.exception(f"Error finding generated clips: {str(e)}")
        return []


def process_batch(
    clips,
    max_workers=1,
    camera_filter=None,
    cv_optimized=False,
    both_formats=False,
    multi_crop=False,
    crop_variations="original,wide,full",
    wide_crop_factor=1.5,
    crop_camera_types=None,
    exclude_crop_camera_types=None,
    gpu_acceleration=False,
):
    logger.info(f"Processing batch of {len(clips)} clips")

    successful = 0
    processed_clips = []
    config_manager = ConfigManager()

    # Create a map of clips to their config files
    clip_sources = {}
    for clip in clips:
        found = False
        for config_file in config_manager.configs_dir.glob("**/*.json"):
            if not found:
                try:
                    config_clips = clip_service.load_clips(config_file)
                    for i, config_clip in enumerate(config_clips):
                        if (
                            hasattr(clip, "id")
                            and hasattr(config_clip, "id")
                            and clip.id == config_clip.id
                        ):
                            clip_sources[clip.id] = (config_file, i)
                            found = True
                            break
                except Exception as e:
                    logger.warning(f"Error loading clips from {config_file}: {e}")

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_clip = {
            executor.submit(
                process_clip,
                clip,
                camera_filter,
                cv_optimized,
                both_formats,
                multi_crop,
                crop_variations,
                wide_crop_factor,
                gpu_acceleration,
            ): clip
            for clip in clips
        }

        with tqdm(
            total=len(clips), desc="Processing clips", position=0, leave=True
        ) as pbar:
            for future in concurrent.futures.as_completed(future_to_clip):
                clip = future_to_clip[future]
                try:
                    success = future.result()
                    if success:
                        successful += 1
                        logger.info(f"Successfully processed clip: {clip.name}")

                        # Verify the clip was properly updated
                        if clip.id in clip_sources:
                            config_file, clip_index = clip_sources[clip.id]
                            # Double-check the saved file
                            try:
                                reloaded_clips = clip_service.load_clips(config_file)
                                if clip_index < len(reloaded_clips):
                                    saved_clip = reloaded_clips[clip_index]
                                    if (
                                        saved_clip.status != "Complete"
                                        or not saved_clip.export_path
                                    ):
                                        logger.warning(
                                            f"Clip {clip.name} status or export_path not properly saved, attempting to fix..."
                                        )
                                        # Try to fix it
                                        reloaded_clips[clip_index].status = "Complete"
                                        reloaded_clips[clip_index].export_path = (
                                            clip.export_path
                                        )
                                        reloaded_clips[clip_index].modified_at = (
                                            datetime.now().isoformat()
                                        )
                                        save_success = clip_service.save_clips(
                                            reloaded_clips, config_file
                                        )
                                        if save_success:
                                            logger.info(
                                                f"Fixed clip {clip.name} status and export_path"
                                            )
                                        else:
                                            logger.error(
                                                f"Failed to fix clip {clip.name} status"
                                            )
                            except Exception as e:
                                logger.warning(
                                    f"Error verifying clip {clip.name} update: {e}"
                                )
                    else:
                        logger.error(f"Failed to process clip: {clip.name}")
                except Exception as e:
                    logger.exception(f"Error processing clip {clip.name}: {str(e)}")
                finally:
                    pbar.update(1)

    return successful


def list_available_cameras(config_manager):
    """
    List all available camera types in the source directories

    Args:
        config_manager: ConfigManager instance

    Returns:
        List of unique camera types
    """
    try:
        # Get all video files
        video_files = config_manager.get_video_files()

        # Extract camera types from all videos
        camera_types = set()
        for video_path in video_files:
            camera_type = extract_camera_type(video_path)
            if camera_type != "UNKNOWN":
                camera_types.add(camera_type)

        return sorted(list(camera_types))
    except Exception as e:
        logger.exception(f"Error listing available cameras: {str(e)}")
        return []


def watch_for_new_footage(
    config_manager, interval=300, ignore_existing=False, camera_filter=None
):
    """
    Watch for new raw footage and automatically generate proxies.

    Args:
        config_manager: ConfigManager instance
        interval: Interval in seconds between checks
        ignore_existing: If True, ignore existing files without proxies
        camera_filter: Optional camera type filter

    Returns:
        None
    """
    try:
        logger.info(f"Starting proxy watch daemon, checking every {interval} seconds")
        if camera_filter:
            logger.info(f"Filtering for camera type: {camera_filter}")

        # If not ignoring existing, process all files without proxies first
        if not ignore_existing:
            logger.info("Processing existing files without proxies...")
            process_all_raw_footage(config_manager, camera_filter)

        # Track processed files to avoid re-processing
        processed_files = set()

        print(f"\n{'='*80}")
        print(f"Proxy Watch Mode Active")
        print(f"Checking for new raw footage every {interval} seconds")
        if camera_filter:
            print(f"Camera filter: {camera_filter}")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*80}\n")

        # Initial scan to populate processed_files
        all_video_files = config_manager.get_video_files()
        for video_path in all_video_files:
            processed_files.add(str(video_path))

        if camera_filter:
            logger.info(
                f"Initial scan found {len(processed_files)} videos, filtering for camera type: {camera_filter}"
            )
        else:
            logger.info(f"Initial scan found {len(processed_files)} videos")

        while True:
            try:
                # Get current video files
                all_video_files = config_manager.get_video_files()

                # Find new files
                new_files = []
                for video_path in all_video_files:
                    video_path_str = str(video_path)
                    if video_path_str not in processed_files:
                        # Apply camera filter if specified
                        if camera_filter:
                            camera_type = extract_camera_type(video_path)
                            if not camera_matches_filter(camera_type, camera_filter):
                                logger.info(
                                    f"Skipping new file {os.path.basename(video_path_str)} from camera {camera_type} (doesn't match filter)"
                                )
                                # Add to processed files to avoid checking again
                                processed_files.add(video_path_str)
                                continue

                        # Check if proxy already exists
                        if not proxy_service.proxy_exists_for_video(
                            video_path, config_manager
                        ):
                            new_files.append(video_path)

                        # Add to processed files to avoid checking again
                        processed_files.add(video_path_str)

                # Process new files if any
                if new_files:
                    logger.info(
                        f"Found {len(new_files)} new raw footage files without proxies"
                    )
                    for video_path in new_files:
                        logger.info(
                            f"Processing new raw footage: {os.path.basename(str(video_path))}"
                        )
                        try:
                            # Create proxy directly
                            result = proxy_service.create_proxy_video(
                                source_path=str(video_path),
                                config_manager=config_manager,
                            )
                            if result:
                                logger.info(
                                    f"Successfully created proxy for {os.path.basename(str(video_path))}"
                                )
                                print(
                                    f"✅ Created proxy for: {os.path.basename(str(video_path))}"
                                )
                            else:
                                logger.error(
                                    f"Failed to create proxy for {os.path.basename(str(video_path))}"
                                )
                                print(
                                    f"❌ Failed to create proxy for: {os.path.basename(str(video_path))}"
                                )
                        except Exception as e:
                            logger.exception(
                                f"Error creating proxy for {os.path.basename(str(video_path))}: {str(e)}"
                            )
                            print(
                                f"❌ Error creating proxy for: {os.path.basename(str(video_path))}"
                            )
                else:
                    logger.info("No new raw footage found")

                # Wait for next check
                time.sleep(interval)

            except Exception as e:
                logger.exception(f"Error in watch daemon loop: {str(e)}")
                # Continue running despite errors
                time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Watch daemon stopped by user")
        print("\nWatch daemon stopped by user")
    except Exception as e:
        logger.exception(f"Error in watch daemon: {str(e)}")
        print(f"\nError in watch daemon: {str(e)}")


def process_all_raw_footage(config_manager, camera_filter=None):
    """
    Process all raw footage files without proxies

    Args:
        config_manager: ConfigManager instance
        camera_filter: Optional camera type filter

    Returns:
        Number of successfully processed files
    """
    # Get all video files
    video_files = config_manager.get_video_files()

    # Filter files without proxies
    files_without_proxies = []
    for video_path in video_files:
        # Apply camera filter if specified
        if camera_filter:
            camera_type = extract_camera_type(video_path)
            if not camera_matches_filter(camera_type, camera_filter):
                logger.info(
                    f"Skipping file {os.path.basename(str(video_path))} from camera {camera_type} (doesn't match filter)"
                )
                continue

        # Check if proxy already exists
        if not proxy_service.proxy_exists_for_video(video_path, config_manager):
            files_without_proxies.append(video_path)

    total_files = len(files_without_proxies)
    logger.info(f"Found {total_files} raw footage files without proxies")

    if total_files == 0:
        logger.info("No raw footage files without proxies found")
        return 0

    # Process all files
    successful = 0
    for i, video_path in enumerate(files_without_proxies):
        logger.info(
            f"Processing raw footage {i+1}/{total_files}: {os.path.basename(str(video_path))}"
        )
        print(f"Processing {i+1}/{total_files}: {os.path.basename(str(video_path))}")

        try:
            # Create proxy directly
            result = proxy_service.create_proxy_video(
                source_path=str(video_path),
                config_manager=config_manager,
            )
            if result:
                logger.info(
                    f"Successfully created proxy for {os.path.basename(str(video_path))}"
                )
                print(f"✅ Created proxy for: {os.path.basename(str(video_path))}")
                successful += 1
            else:
                logger.error(
                    f"Failed to create proxy for {os.path.basename(str(video_path))}"
                )
                print(
                    f"❌ Failed to create proxy for: {os.path.basename(str(video_path))}"
                )
        except Exception as e:
            logger.exception(
                f"Error creating proxy for {os.path.basename(str(video_path))}: {str(e)}"
            )
            print(f"❌ Error creating proxy for: {os.path.basename(str(video_path))}")

    logger.info(f"Processed {successful} of {total_files} raw footage files")
    return successful


def fix_processed_clips(config_manager):
    """
    Fix clips that have already been processed but not marked as Complete.
    This function scans for clips that have export_path but status is not Complete
    and updates them.

    Args:
        config_manager: ConfigManager instance

    Returns:
        Number of fixed clips
    """
    fixed_count = 0
    logger.info("Scanning for clips with export_path but not marked Complete...")

    # Scan all clip config files
    for config_file in config_manager.configs_dir.glob("**/*.json"):
        try:
            clips = clip_service.load_clips(config_file)
            needs_save = False

            for i, clip in enumerate(clips):
                clip_was_modified = False

                # Case 1: Check if clip has export_path but is not marked Complete
                has_export_path = False
                if hasattr(clip, "export_path") and clip.export_path:
                    if isinstance(clip.export_path, list) and len(clip.export_path) > 0:
                        has_export_path = True
                    elif isinstance(clip.export_path, str) and clip.export_path.strip():
                        has_export_path = True

                # Check alternate export_paths attribute also
                if (
                    not has_export_path
                    and hasattr(clip, "export_paths")
                    and clip.export_paths
                ):
                    paths = [
                        p.strip() for p in clip.export_paths.split(",") if p.strip()
                    ]
                    if paths:
                        has_export_path = True
                        # Convert to proper export_path list
                        clip.export_path = paths
                        clip_was_modified = True
                        logger.info(
                            f"Converted export_paths to export_path list for {clip.name}: {paths}"
                        )

                # Case 2: Check if clip is marked Complete but has incomplete export paths
                is_complete = hasattr(clip, "status") and clip.status == "Complete"
                is_incomplete_paths = False

                if (
                    is_complete
                    and hasattr(clip, "export_path")
                    and isinstance(clip.export_path, list)
                ):
                    # Check if we have MP4 files but missing MKV files
                    mp4_files = [p for p in clip.export_path if p.endswith(".mp4")]
                    mkv_files = [p for p in clip.export_path if p.endswith(".mkv")]

                    # If we have MP4 files but no MKV files, we might be missing some paths
                    if mp4_files and not mkv_files:
                        logger.info(
                            f"Clip {clip.name} is marked Complete but has only MP4 files in export_path"
                        )
                        is_incomplete_paths = True

                # Update status if needed
                if (has_export_path and not is_complete) or is_incomplete_paths:
                    logger.info(
                        f"Found clip {clip.name} with issues - has_export_path: {has_export_path}, is_complete: {is_complete}, is_incomplete_paths: {is_incomplete_paths}"
                    )

                    # Try to find all files generated for this clip
                    try:
                        # Check if files exist on disk directly
                        paths_exist = False
                        if hasattr(clip, "export_path") and isinstance(
                            clip.export_path, list
                        ):
                            existing_paths = [
                                p for p in clip.export_path if os.path.exists(p)
                            ]
                            if existing_paths:
                                paths_exist = True

                        # Search for additional files that might not be in export_path
                        found_files = find_generated_clips(
                            clip,
                            config_manager,
                            both_formats=True,  # Assume both formats to be thorough
                            multi_crop=True,  # Assume multi-crop to be thorough
                            crop_variations="original,wide,full",  # Standard variations
                        )

                        if found_files:
                            logger.info(
                                f"Found {len(found_files)} files for clip {clip.name}"
                            )

                            # If we found more files than are in export_path, update export_path
                            if (
                                not hasattr(clip, "export_path")
                                or not isinstance(clip.export_path, list)
                                or len(found_files) > len(clip.export_path)
                            ):
                                logger.info(
                                    f"Updating export_path with {len(found_files)} found files"
                                )
                                clip.export_path = found_files
                                clip_was_modified = True

                                # Update export_paths for compatibility
                                if hasattr(clip, "export_paths"):
                                    clip.export_paths = ",".join(found_files)

                            # If we have files, consider it fixed
                            paths_exist = True

                        if paths_exist:
                            # Update status to Complete
                            if not is_complete:
                                clip.status = "Complete"
                                clip_was_modified = True
                                logger.info(
                                    f"Updated clip {clip.name} status to Complete"
                                )

                            # Update timestamp if we made changes
                            if clip_was_modified:
                                clip.update()  # Update modified timestamp
                                fixed_count += 1
                                needs_save = True
                        else:
                            logger.warning(
                                f"Clip {clip.name} has export_path but files don't exist, not updating status"
                            )
                    except Exception as e:
                        logger.exception(f"Error fixing clip {clip.name}: {str(e)}")

            # Save the config file if any clips were updated
            if needs_save:
                success = clip_service.save_clips(clips, config_file)
                if success:
                    logger.info(f"Saved updated clips to {config_file}")
                else:
                    logger.error(f"Failed to save updated clips to {config_file}")

        except Exception as e:
            logger.exception(f"Error processing config file {config_file}: {str(e)}")

    return fixed_count


def main():
    """
    Main function to process clips
    """
    parser = argparse.ArgumentParser(
        description='Process clips with "Process" status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all clips
  python scripts/process_clips.py
  
  # List available camera types
  python scripts/process_clips.py --list-cameras
  
  # Process clips from a specific camera
  python scripts/process_clips.py --camera SONY_70
  
  # Process clips with CV optimization
  python scripts/process_clips.py --cv-optimized
  
  # Process clips with multiple crop variations
  python scripts/process_clips.py --multi-crop
  
  # Process clips with specific crop variations
  python scripts/process_clips.py --multi-crop --crop-variations "original,wide"
  
  # Process clips with custom wide crop factor (75% larger)
  python scripts/process_clips.py --multi-crop --wide-crop-factor 1.75
  
  # Run as a daemon, checking every 5 minutes
  python scripts/process_clips.py --daemon --interval 300
  
  # Watch for new raw footage and auto-generate proxies
  python scripts/process_clips.py --generate-proxies
  
  # Watch for new raw footage from a specific camera only, ignoring existing files
  python scripts/process_clips.py --generate-proxies --camera GP1 --ignore-existing
  
  # Watch for new footage, checking every 2 minutes
  python scripts/process_clips.py --generate-proxies --watch-interval 120
  
  # Fix clips that were processed but not marked as Complete
  python scripts/process_clips.py --fix-processed
  
Camera Filtering:
  The --camera option supports flexible matching:
  - Exact match: --camera GP1 matches only "GP1"
  - Prefix match: --camera SONY matches "SONY", "SONY_70", "SONY_300", etc.
  - Substring match: --camera GP matches "GP", "GP1", "GP2", etc.
  
  Use --list-cameras to see all available camera types.
  
Watch Mode:
  The --generate-proxies option sets up a daemon that watches for new raw footage and 
  automatically generates proxies when new files are detected. This is useful 
  for automatically processing footage as it's imported.
  
  --ignore-existing: Skip processing existing raw footage without proxies (only process new files)
  --watch-interval: How often to check for new files (in seconds)
  --camera: Filter to only watch for footage from specific cameras
""",
    )
    parser.add_argument("--daemon", action="store_true", help="Run as a daemon process")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Scan interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        help="Only process clips from a specific camera type (e.g., SONY, GP1, SONY_70)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List all available camera types and exit",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of clips to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Maximum clips to process in one batch (0 for unlimited)",
    )
    parser.add_argument(
        "--cv-optimized",
        action="store_true",
        help="Export clips with CV optimization (higher quality)",
    )
    parser.add_argument(
        "--both-formats",
        action="store_true",
        help="Export clips in both regular H.264 and CV-optimized formats",
    )
    parser.add_argument(
        "--multi-crop",
        action="store_true",
        help="Generate multiple crop variations of each clip",
    )
    parser.add_argument(
        "--crop-variations",
        type=str,
        default="original,wide,full",
        help="Comma-separated list of crop variations to generate (original, wide, full)",
    )
    parser.add_argument(
        "--wide-crop-factor",
        type=float,
        default=1.5,
        help="Multiplier for the wide crop variation - higher values = more zoomed out (default: 1.2, 1.5 = 50%% larger than original)",
    )
    parser.add_argument(
        "--generate-proxies",
        action="store_true",
        help="Watch for new raw footage and auto-generate proxies",
    )
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=300,
        help="Interval in seconds to check for new raw footage (default: 300)",
    )
    parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="When watching for new footage, ignore existing files without proxies",
    )
    parser.add_argument(
        "--crop-camera-types",
        type=str,
        help="Comma-separated list of camera types to apply crop variations to",
    )
    parser.add_argument(
        "--exclude-crop-camera-types",
        type=str,
        help="Comma-separated list of camera types to exclude from crop variations",
    )
    parser.add_argument(
        "--fix-processed",
        action="store_true",
        help="Fix clips that were processed but not marked as Complete",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU hardware acceleration for video processing (H.264 only, requires NVIDIA GPU)",
    )
    args = parser.parse_args()

    config_manager = ConfigManager()

    # Handle fixing processed clips
    if args.fix_processed:
        fixed_count = fix_processed_clips(config_manager)
        if fixed_count > 0:
            print(
                f"\nFixed {fixed_count} clips that were processed but not marked as Complete"
            )
        else:
            print("\nNo clips needed fixing")
        return

    # Parse crop camera types
    crop_camera_types = None
    exclude_crop_camera_types = None

    if args.crop_camera_types:
        crop_camera_types = [
            ct.strip() for ct in args.crop_camera_types.split(",") if ct.strip()
        ]
        logger.info(
            f"Applying crop variations only to camera types: {crop_camera_types}"
        )

    if args.exclude_crop_camera_types:
        exclude_crop_camera_types = [
            ct.strip() for ct in args.exclude_crop_camera_types.split(",") if ct.strip()
        ]
        logger.info(
            f"Excluding crop variations for camera types: {exclude_crop_camera_types}"
        )

    # List available cameras if requested
    if args.list_cameras:
        cameras = list_available_cameras(config_manager)
        if cameras:
            print("\nAvailable camera types:")
            for camera in cameras:
                print(f"  - {camera}")
            print("\nUse --camera CAMERA_TYPE to filter by camera type")
        else:
            print(
                "\nNo camera types found. Make sure your videos are organized in camera-specific folders."
            )
        return

    # Handle watch mode
    if args.generate_proxies:
        logger.info("Starting proxy generation mode for automatic proxy creation")
        watch_for_new_footage(
            config_manager,
            interval=args.watch_interval,
            ignore_existing=args.ignore_existing,
            camera_filter=args.camera,
        )
        return

    if args.daemon:
        logger.info(
            f"Starting clip processor daemon, scanning every {args.interval} seconds"
        )
        if args.camera:
            logger.info(f"Filtering for camera type: {args.camera}")
        if args.cv_optimized:
            logger.info("Using CV optimization for exports")
        if args.both_formats:
            logger.info("Exporting in both regular and CV-optimized formats")
        if args.multi_crop:
            logger.info("Generating multiple crop variations")
            logger.info(f"Crop variations: {args.crop_variations}")
            logger.info(
                f"Wide crop factor: {args.wide_crop_factor} (higher values = more zoomed out)"
            )
        if args.gpu:
            logger.info("GPU hardware acceleration enabled (NVENC for H.264, CPU fallback for FFV1)")
        else:
            logger.info("Using CPU-only video processing")

        # Add a heartbeat file to track if daemon is running
        heartbeat_file = script_dir / "daemon_heartbeat.txt"
        last_heartbeat = time.time()

        try:
            iteration = 0
            while True:
                iteration += 1
                current_time = time.time()

                # Update heartbeat file
                with open(heartbeat_file, "w") as f:
                    f.write(f"Last heartbeat: {datetime.now().isoformat()}\n")
                    f.write(f"Iteration: {iteration}\n")
                    f.write(
                        f"Total runtime: {int(current_time - last_heartbeat)} seconds\n"
                    )

                logger.info(f"Starting daemon iteration {iteration}")
                print(f"\n{'='*80}")
                print(f"Daemon Iteration {iteration}")
                print(f"Last heartbeat: {datetime.now().isoformat()}")
                print(f"{'='*80}\n")

                try:
                    # Get all pending clips
                    pending_clips = get_pending_clips(config_manager)
                    logger.info(f"Found {len(pending_clips)} pending clips")

                    # Apply camera filter if specified
                    if args.camera and pending_clips:
                        filtered_clips = []
                        for clip in pending_clips:
                            try:
                                # Resolve the source path
                                source_path = resolve_source_path(
                                    clip.source_path, config_manager
                                )
                                # Extract camera type
                                camera_type = extract_camera_type(source_path)
                                if camera_matches_filter(camera_type, args.camera):
                                    filtered_clips.append(clip)
                            except Exception as e:
                                logger.warning(
                                    f"Error filtering clip {clip.name}: {str(e)}"
                                )
                                # Skip this clip if we can't resolve its path
                                continue
                        logger.info(
                            f"Found {len(filtered_clips)} clips matching camera filter '{args.camera}' out of {len(pending_clips)} total pending clips"
                        )
                        pending_clips = filtered_clips

                    # Process clips in batches if batch size is specified
                    if args.batch_size > 0 and pending_clips:
                        total_processed = 0
                        batch_start_time = time.time()

                        # Store mapping of processed clips to their config files and indices
                        processed_clip_info = []

                        # Before processing, find the file paths and indices for all clips
                        for clip in pending_clips:
                            # Find clip's source config file
                            for config_file in config_manager.configs_dir.glob(
                                "**/*.json"
                            ):
                                try:
                                    config_clips = clip_service.load_clips(config_file)
                                    for i, config_clip in enumerate(config_clips):
                                        # Match by ID if available
                                        if (
                                            hasattr(clip, "id")
                                            and hasattr(config_clip, "id")
                                            and clip.id == config_clip.id
                                        ):
                                            processed_clip_info.append(
                                                (clip.id, config_file, i)
                                            )
                                            logger.info(
                                                f"Found config file for clip {clip.name}: {config_file}"
                                            )
                                            break
                                except Exception as e:
                                    logger.warning(
                                        f"Error loading clips from {config_file}: {e}"
                                    )

                        while pending_clips:
                            batch = pending_clips[: args.batch_size]
                            pending_clips = pending_clips[args.batch_size :]

                            logger.info(
                                f"Processing batch of {len(batch)} clips (remaining: {len(pending_clips)})"
                            )

                            # Keep track of successfully processed clips in this batch
                            successfully_processed = []

                            try:
                                num_processed = process_batch(
                                    batch,
                                    args.max_workers,
                                    args.camera,
                                    args.cv_optimized,
                                    args.both_formats,
                                    args.multi_crop,
                                    args.crop_variations,
                                    args.wide_crop_factor,
                                    crop_camera_types,
                                    exclude_crop_camera_types,
                                    args.gpu,
                                )
                                total_processed += num_processed

                                # For each processed clip, directly update its status using update_clip_status
                                for clip in batch:
                                    if (
                                        hasattr(clip, "status")
                                        and clip.status == "Complete"
                                    ):
                                        successfully_processed.append(clip)
                                        # Find the clip info in our mapping
                                        for (
                                            clip_id,
                                            config_file,
                                            clip_index,
                                        ) in processed_clip_info:
                                            if (
                                                hasattr(clip, "id")
                                                and clip.id == clip_id
                                            ):
                                                logger.info(
                                                    f"Directly updating status for clip {clip.name} in {config_file}"
                                                )
                                                update_success = update_clip_status(
                                                    config_file,
                                                    clip_index,
                                                    "Complete",
                                                    clip,
                                                )
                                                if update_success:
                                                    logger.info(
                                                        f"Successfully updated clip status for {clip.name}"
                                                    )
                                                else:
                                                    logger.error(
                                                        f"Failed to update clip status for {clip.name}"
                                                    )

                                logger.info(
                                    f"Successfully processed {len(successfully_processed)} clips in this batch"
                                )
                            except Exception as e:
                                logger.exception(f"Error processing batch: {str(e)}")
                                # Continue with next batch even if this one failed
                                continue

                        batch_time = time.time() - batch_start_time
                        logger.info(
                            f"Processed {total_processed} clips in {batch_time:.2f} seconds"
                        )
                    elif pending_clips:
                        # Process all clips at once
                        try:
                            num_processed = process_batch(
                                pending_clips,
                                args.max_workers,
                                args.camera,
                                args.cv_optimized,
                                args.both_formats,
                                args.multi_crop,
                                args.crop_variations,
                                args.wide_crop_factor,
                                crop_camera_types,
                                exclude_crop_camera_types,
                                args.gpu,
                            )
                            logger.info(f"Processed {num_processed} clips")
                        except Exception as e:
                            logger.exception(f"Error processing clips: {str(e)}")
                    else:
                        logger.info("No pending clips found")

                except Exception as e:
                    logger.exception(f"Error in daemon iteration: {str(e)}")
                    # Continue running despite errors

                # Calculate sleep time, ensuring we don't sleep too long
                elapsed = time.time() - current_time
                sleep_time = max(0, args.interval - elapsed)
                logger.info(
                    f"Sleeping for {sleep_time:.2f} seconds until next iteration"
                )
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
            print("\nDaemon stopped by user")
        except Exception as e:
            logger.exception(f"Fatal error in daemon: {str(e)}")
            print(f"\nFatal error in daemon: {str(e)}")
        finally:
            # Clean up heartbeat file
            if heartbeat_file.exists():
                heartbeat_file.unlink()
    else:
        logger.info("Running one-time clip processing scan")
        if args.camera:
            logger.info(f"Filtering for camera type: {args.camera}")
        if args.cv_optimized:
            logger.info("Using CV optimization for exports")
        if args.both_formats:
            logger.info("Exporting in both regular and CV-optimized formats")
        if args.gpu:
            logger.info("GPU hardware acceleration enabled (NVENC for H.264, CPU fallback for FFV1)")
        else:
            logger.info("Using CPU-only video processing")

        # Get all pending clips
        pending_clips = get_pending_clips(config_manager)

        # Apply camera filter if specified
        if args.camera and pending_clips:
            filtered_clips = []
            for clip in pending_clips:
                try:
                    # Resolve the source path
                    source_path = resolve_source_path(clip.source_path, config_manager)
                    # Extract camera type
                    camera_type = extract_camera_type(source_path)
                    if camera_matches_filter(camera_type, args.camera):
                        filtered_clips.append(clip)
                except Exception as e:
                    logger.warning(f"Error filtering clip {clip.name}: {str(e)}")
                    # Skip this clip if we can't resolve its path
                    continue
            logger.info(
                f"Found {len(filtered_clips)} clips matching camera filter '{args.camera}' out of {len(pending_clips)} total pending clips"
            )
            pending_clips = filtered_clips

        # Process clips in batches if batch size is specified
        if args.batch_size > 0 and pending_clips:
            total_processed = 0
            while pending_clips:
                batch = pending_clips[: args.batch_size]
                pending_clips = pending_clips[args.batch_size :]

                logger.info(
                    f"Processing batch of {len(batch)} clips (remaining: {len(pending_clips)})"
                )
                num_processed = process_batch(
                    batch,
                    args.max_workers,
                    args.camera,
                    args.cv_optimized,
                    args.both_formats,
                    args.multi_crop,
                    args.crop_variations,
                    args.wide_crop_factor,
                    crop_camera_types,
                    exclude_crop_camera_types,
                    args.gpu,
                )
                total_processed += num_processed

            logger.info(f"Processed {total_processed} clips in batches")
            print(f"\n{'='*80}")
            print(
                f"PROCESSING COMPLETE: Successfully processed {total_processed} clips"
            )
            print(f"{'='*80}\n")
        elif pending_clips:
            # Process all clips at once
            num_processed = process_batch(
                pending_clips,
                args.max_workers,
                args.camera,
                args.cv_optimized,
                args.both_formats,
                args.multi_crop,
                args.crop_variations,
                args.wide_crop_factor,
                crop_camera_types,
                exclude_crop_camera_types,
                args.gpu,
            )
            logger.info(f"Processed {num_processed} clips")
            print(f"\n{'='*80}")
            print(
                f"PROCESSING COMPLETE: Successfully processed {num_processed} of {len(pending_clips)} clips"
            )
            print(f"{'='*80}\n")
        else:
            logger.info("No pending clips found")
            print(
                "\nNo clips marked for processing. Mark clips with 'Process' status in the UI."
            )


if __name__ == "__main__":
    main()
