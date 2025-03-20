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


def resolve_source_path(source_path, config_manager):
    """
    Resolve a source path to an absolute path, handling relative paths and symlinks

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
    raw_folder = config_manager.config["source"]["raw"]
    calibrated_folder = config_manager.config["source"]["calibrated"]

    logger.debug(f"Raw folder from config: {raw_folder}")
    logger.debug(f"Calibrated folder from config: {calibrated_folder}")

    # Determine the correct source path based on calibration settings
    str_path = str(path)

    # Handle source paths that might need to be adjusted based on the calibration toggle
    if "data/source/" in str_path:
        # Extract the relative path after data/source/
        rel_path = str_path.split("data/source/")[-1]

        # Check if path contains either raw or calibrated folder and needs adjustment
        if raw_folder in rel_path and use_calibrated_footage:
            # Change from RAW to CALIBRATED
            rel_path = rel_path.replace(raw_folder, calibrated_folder)
            logger.info(f"Adjusted path to use calibrated footage: {rel_path}")
        elif calibrated_folder in rel_path and not use_calibrated_footage:
            # Change from CALIBRATED to RAW
            rel_path = rel_path.replace(calibrated_folder, raw_folder)
            logger.info(f"Adjusted path to use raw footage: {rel_path}")

        # Join with source base
        return Path(os.path.join(config_manager.source_base, rel_path))

    # Handle relative paths with data/source or data/prept prefixes
    if str_path.startswith("data/source"):
        rel_path = str_path.split("data/source/")[-1]

        # Determine which subfolder to use based on calibration setting
        if use_calibrated_footage:
            # If using calibrated footage, ensure path includes the calibrated subfolder
            if calibrated_folder not in rel_path:
                if raw_folder in rel_path:
                    rel_path = rel_path.replace(raw_folder, calibrated_folder)
                else:
                    rel_path = os.path.join(calibrated_folder, rel_path)
        else:
            # If using raw footage, ensure path includes the raw subfolder
            if raw_folder not in rel_path:
                if calibrated_folder in rel_path:
                    rel_path = rel_path.replace(calibrated_folder, raw_folder)
                else:
                    rel_path = os.path.join(raw_folder, rel_path)

        return Path(os.path.join(config_manager.source_base, rel_path))
    elif str_path.startswith("data/prept"):
        return Path(
            os.path.join(config_manager.output_base, str_path.split("data/prept/")[-1])
        )

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


def process_clip(clip, camera_filter=None, cv_optimized=False):
    """
    Process a single clip - this is where you would implement your custom processing logic

    Args:
        clip: The Clip object to process
        camera_filter: Optional filter to only process clips from specific cameras
        cv_optimized: Whether to optimize for computer vision (higher quality)

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

        # Get the full path to the source video
        config_manager = ConfigManager()
        source_path = resolve_source_path(clip.source_path, config_manager)

        logger.info(f"Resolved source path: {source_path}")
        logger.info(f"DEBUG: Source path exists: {os.path.exists(source_path)}")

        # Get calibration settings
        calib_settings = config_manager.get_calibration_settings()
        use_calibrated_footage = calib_settings.get("use_calibrated_footage", False)
        logger.info(f"Using calibrated footage: {use_calibrated_footage}")

        # Extract camera type and session folder
        camera_type = extract_camera_type(source_path)
        session_folder = extract_session_folder(source_path)

        # Check camera type if filter is specified
        if camera_filter:
            if not camera_matches_filter(camera_type, camera_filter):
                logger.info(
                    f"Skipping clip {clip.name} from camera {camera_type} (doesn't match filter: {camera_filter})"
                )
                return False

        # Get original filename without extension
        original_filename = os.path.splitext(os.path.basename(source_path))[0]

        # Find the next available clip number for this source file
        clip_number = 1
        while True:
            # Generate output path preserving camera and session structure
            output_filename = f"{original_filename}_clip{clip_number}.mp4"
            if cv_optimized:
                output_filename = output_filename.replace(".mp4", "_cv_optimized.mp4")
            output_dir = config_manager.clips_dir / camera_type / session_folder
            output_path = output_dir / output_filename

            if not output_path.exists():
                break
            clip_number += 1

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Exporting clip from {source_path} to {output_path}")
        logger.info(
            f"Calibration will {'be skipped' if use_calibrated_footage else 'be applied'} based on configuration"
        )

        # Create a progress bar
        total_frames = clip.end_frame - clip.start_frame + 1
        pbar = tqdm(total=100, desc=f"Processing {clip.name}", unit="%")

        # Create a progress callback for export
        def progress_callback(progress):
            # Update progress bar based on percentage (0-1)
            current_percent = int(progress * 100)
            # Update to the current percentage, avoiding going backwards
            if current_percent > pbar.n:
                pbar.update(current_percent - pbar.n)

        # Get crop region for current frame - we'll use this as a fallback
        # if there are no keyframes
        static_crop_region = clip.get_crop_region_at_frame(
            clip.start_frame, use_proxy=False  # Use source resolution for export
        )

        logger.info(f"DEBUG: About to call proxy_service.export_clip")
        logger.info(f"DEBUG: Crop keyframes: {clip.crop_keyframes}")
        logger.info(f"DEBUG: Static crop region: {static_crop_region}")

        # Use proxy_service.export_clip which handles keyframes properly
        export_path = proxy_service.export_clip(
            source_path=source_path,
            clip_name=clip.name,
            start_frame=clip.start_frame,
            end_frame=clip.end_frame,
            crop_region=static_crop_region,
            crop_keyframes=clip.crop_keyframes,  # Use original keyframes for export
            output_resolution=clip.output_resolution,
            cv_optimized=cv_optimized,
            config_manager=config_manager,  # Pass config_manager with calibration settings
            progress_callback=progress_callback,  # Add the progress callback
        )

        logger.info(f"DEBUG: proxy_service.export_clip returned: {export_path}")

        # Close the progress bar
        pbar.close()

        if export_path:
            logger.info(f"Successfully processed clip: {clip.name} -> {export_path}")
            # Update clip's export path
            clip.export_path = str(export_path)
            return True
        else:
            logger.error(f"Failed to process clip: {clip.name}")
            return False

    except Exception as e:
        logger.exception(f"Error processing clip {clip.name}: {str(e)}")
        return False


def process_batch(clips, max_workers=4, camera_filter=None, cv_optimized=False):
    """
    Process a batch of clips using a thread pool for parallel processing

    Args:
        clips: List of clips to process
        max_workers: Maximum number of parallel workers
        camera_filter: Optional camera type filter
        cv_optimized: Whether to optimize for computer vision

    Returns:
        Number of successfully processed clips
    """
    logger.info(f"Processing batch of {len(clips)} clips with {max_workers} workers")
    successful = 0

    # Track which clips were successfully processed
    processed_clips = []

    # Print summary to terminal
    print(f"\n{'='*80}")
    print(
        f"Processing {len(clips)} clips with {max_workers} {'worker' if max_workers == 1 else 'workers'}"
    )
    if camera_filter:
        print(f"Camera filter: {camera_filter}")
    if cv_optimized:
        print(f"CV optimization: Enabled")
    print(f"{'='*80}\n")

    # Get the mapping of clips to their config files and indices
    config_manager = ConfigManager()
    clips_to_process = scan_for_clips_to_process(config_manager)

    # Create lookup dictionaries for reliable matching
    # Primary lookup by ID
    clip_lookup_by_id = {}
    # Backup lookup by name and source path
    clip_lookup_by_name_and_path = {}

    for file_path, clip_index, clip_obj in clips_to_process:
        # Use clip ID as primary key
        clip_lookup_by_id[clip_obj.id] = (file_path, clip_index)
        # Use name and source path as backup key
        key = (clip_obj.name, clip_obj.source_path)
        clip_lookup_by_name_and_path[key] = (file_path, clip_index)
        # Debug log the clip details
        logger.info(
            f"DEBUG: Clip to process - ID: {clip_obj.id}, Name: {clip_obj.name}, Status: {clip_obj.status}, Export path: {clip_obj.export_path}"
        )

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all clips for processing
        future_to_clip = {
            executor.submit(process_clip, clip, camera_filter, cv_optimized): clip
            for clip in clips
        }

        # Process results as they complete
        for future in future_to_clip:
            clip = future_to_clip[future]
            try:
                if future.result():
                    successful += 1
                    # Update the clip status to "Complete"
                    clip.status = "Complete"
                    clip.update()
                    logger.info(
                        f"DEBUG: After processing - Clip ID: {clip.id}, Name: {clip.name}, Status: {clip.status}, Export path: {clip.export_path}"
                    )
                    processed_clips.append(clip)

                    # Try to find the clip in the lookup tables
                    file_path = None
                    clip_index = None

                    # First try by ID
                    if clip.id in clip_lookup_by_id:
                        file_path, clip_index = clip_lookup_by_id[clip.id]
                        logger.info(
                            f"DEBUG: Found clip by ID in lookup table - File: {file_path}, Index: {clip_index}"
                        )
                    # Then try by name and path
                    elif (clip.name, clip.source_path) in clip_lookup_by_name_and_path:
                        file_path, clip_index = clip_lookup_by_name_and_path[
                            (clip.name, clip.source_path)
                        ]
                        logger.info(
                            f"DEBUG: Found clip by name and path in lookup table - File: {file_path}, Index: {clip_index}"
                        )
                    else:
                        logger.warning(
                            f"DEBUG: Could not find clip in lookup tables - ID: {clip.id}, Name: {clip.name}"
                        )

                    # Update the clip status in the config file if found
                    if file_path and clip_index is not None:
                        # Read the current clip from the file to check its status
                        current_clips = load_clips(file_path)
                        if clip_index < len(current_clips):
                            logger.info(
                                f"DEBUG: Current clip status in file before update: {current_clips[clip_index].status}, Export path: {current_clips[clip_index].export_path}"
                            )

                        update_success = update_clip_status(
                            file_path, clip_index, "Complete", processed_clip=clip
                        )
                        if update_success:
                            logger.info(
                                f"Updated clip status in config file: {clip.name}"
                            )

                            # Read the clip again to verify the update
                            updated_clips = load_clips(file_path)
                            if clip_index < len(updated_clips):
                                logger.info(
                                    f"DEBUG: Clip status after update: {updated_clips[clip_index].status}, Export path: {updated_clips[clip_index].export_path}"
                                )
                        else:
                            logger.warning(
                                f"Failed to update clip status in config file: {clip.name}"
                            )
                    else:
                        logger.warning(
                            f"Could not find clip in lookup tables: {clip.name}"
                        )
                else:
                    logger.warning(f"Failed to process clip: {clip.name}")
            except Exception as e:
                logger.exception(
                    f"Exception while processing clip {clip.name}: {str(e)}"
                )

    # As a fallback, try to save all processed clips directly
    if processed_clips:
        # Group clips by their source video
        clips_by_source = {}
        for clip in processed_clips:
            source = clip.source_path
            if source not in clips_by_source:
                clips_by_source[source] = []
            clips_by_source[source].append(clip)

        # For each source video, load all clips, update the processed ones, and save
        for source, source_clips in clips_by_source.items():
            try:
                # Resolve the source path
                source_path = resolve_source_path(source, config_manager)

                # Get the clips file path for this source
                clips_file = config_manager.get_clips_file_path(source_path)

                # If we couldn't get a clips file path, try a simpler approach
                if not clips_file:
                    # Use the video filename to create a clips file path
                    video_filename = os.path.basename(str(source))
                    video_name = os.path.splitext(video_filename)[0]
                    clips_file = config_manager.configs_dir / f"{video_name}_clips.json"
                    logger.info(f"Using fallback clips file path: {clips_file}")

                # Load all clips for this source
                all_clips = load_clips(clips_file)

                # Update status for processed clips
                updated = False
                for processed_clip in source_clips:
                    for i, clip in enumerate(all_clips):
                        # Match by ID or by name and time range
                        if clip.id == processed_clip.id or (
                            clip.name == processed_clip.name
                            and clip.start_frame == processed_clip.start_frame
                            and clip.end_frame == processed_clip.end_frame
                        ):
                            # Update status
                            all_clips[i].status = "Complete"
                            all_clips[i].export_path = processed_clip.export_path
                            all_clips[i].update()
                            updated = True
                            logger.info(
                                f"Updated clip status in direct save: {clip.name}"
                            )
                            logger.info(
                                f"DEBUG: Updated export path in direct save: {processed_clip.export_path}"
                            )

                # Save all clips if any were updated
                if updated:
                    save_success = save_clips(all_clips, clips_file)
                    if save_success:
                        logger.info(f"Saved updated clips to {clips_file}")
                    else:
                        logger.warning(f"Failed to save updated clips to {clips_file}")
            except Exception as e:
                logger.exception(f"Error in direct clip save fallback: {str(e)}")

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
  
  # Run as a daemon, checking every 5 minutes
  python scripts/process_clips.py --daemon --interval 300
  
Camera Filtering:
  The --camera option supports flexible matching:
  - Exact match: --camera GP1 matches only "GP1"
  - Prefix match: --camera SONY matches "SONY", "SONY_70", "SONY_300", etc.
  - Substring match: --camera GP matches "GP", "GP1", "GP2", etc.
  
  Use --list-cameras to see all available camera types.
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
        default=4,
        help="Maximum number of clips to process in parallel (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Maximum clips to process in one batch (0 for unlimited)",
    )
    parser.add_argument(
        "--cv-optimized",
        action="store_true",
        help="Export clips with CV optimization (higher quality)",
    )
    args = parser.parse_args()

    config_manager = ConfigManager()

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

    if args.daemon:
        logger.info(
            f"Starting clip processor daemon, scanning every {args.interval} seconds"
        )
        if args.camera:
            logger.info(f"Filtering for camera type: {args.camera}")
        if args.cv_optimized:
            logger.info("Using CV optimization for exports")

        try:
            while True:
                # Get all pending clips
                pending_clips = get_pending_clips(config_manager)

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
                    while pending_clips:
                        batch = pending_clips[: args.batch_size]
                        pending_clips = pending_clips[args.batch_size :]

                        logger.info(
                            f"Processing batch of {len(batch)} clips (remaining: {len(pending_clips)})"
                        )
                        num_processed = process_batch(
                            batch, args.max_workers, args.camera, args.cv_optimized
                        )
                        total_processed += num_processed

                    logger.info(f"Processed {total_processed} clips in batches")
                elif pending_clips:
                    # Process all clips at once
                    num_processed = process_batch(
                        pending_clips, args.max_workers, args.camera, args.cv_optimized
                    )
                    logger.info(f"Processed {num_processed} clips")
                else:
                    logger.info("No pending clips found")

                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
    else:
        logger.info("Running one-time clip processing scan")
        if args.camera:
            logger.info(f"Filtering for camera type: {args.camera}")
        if args.cv_optimized:
            logger.info("Using CV optimization for exports")

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
                    batch, args.max_workers, args.camera, args.cv_optimized
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
                pending_clips, args.max_workers, args.camera, args.cv_optimized
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
