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

# Add the project root to the path so we can import the app modules
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.utils.clip_processor import process_pending_clips, get_pending_clips
from src.services.config_manager import ConfigManager
from src.services import video_service

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

    # Try to find a camera type in the path
    for part in parts:
        # Look for common camera type patterns in the path
        if any(
            cam in part.upper()
            for cam in ["SONY", "GP", "GOPRO", "CANON", "NIKON", "CAM"]
        ):
            return part

    return "UNKNOWN"


def process_clip(clip, camera_filter=None):
    """
    Process a single clip - this is where you would implement your custom processing logic

    Args:
        clip: The Clip object to process
        camera_filter: Optional filter to only process clips from specific cameras

    Returns:
        True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Processing clip: {clip.name}")

        # Get the full path to the source video
        config_manager = ConfigManager()
        source_path = clip.source_path

        # If the path is relative, resolve it
        if not os.path.isabs(source_path):
            if source_path.startswith("data/source"):
                source_path = os.path.join(
                    config_manager.source_base, source_path.split("data/source/")[-1]
                )
            elif source_path.startswith("data/prept"):
                source_path = os.path.join(
                    config_manager.output_base, source_path.split("data/prept/")[-1]
                )

        # Check camera type if filter is specified
        if camera_filter:
            camera_type = extract_camera_type(source_path)
            if camera_type != camera_filter:
                logger.info(
                    f"Skipping clip {clip.name} from camera {camera_type} (doesn't match filter: {camera_filter})"
                )
                return False

        # Generate an output filename
        output_dir = config_manager.clips_dir
        output_filename = f"{clip.name}_processed.mp4"
        output_path = output_dir / output_filename

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Exporting clip from {source_path} to {output_path}")

        # Here you would implement your actual processing logic
        # For example, you might use video_service.export_clip to export the clip
        success = video_service.export_clip(
            source_path=source_path,
            output_path=output_path,
            start_frame=clip.start_frame,
            end_frame=clip.end_frame,
            crop_region=clip.get_crop_region_at_frame(clip.start_frame),
            output_resolution=clip.output_resolution,
        )

        if success:
            logger.info(f"Successfully processed clip: {clip.name}")
            return True
        else:
            logger.error(f"Failed to process clip: {clip.name}")
            return False

    except Exception as e:
        logger.exception(f"Error processing clip {clip.name}: {str(e)}")
        return False


def process_batch(clips, max_workers=4, camera_filter=None):
    """
    Process a batch of clips using a thread pool for parallel processing

    Args:
        clips: List of clips to process
        max_workers: Maximum number of parallel workers
        camera_filter: Optional camera type filter

    Returns:
        Number of successfully processed clips
    """
    logger.info(f"Processing batch of {len(clips)} clips with {max_workers} workers")
    successful = 0

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all clips for processing
        future_to_clip = {
            executor.submit(process_clip, clip, camera_filter): clip for clip in clips
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
                else:
                    logger.warning(f"Failed to process clip: {clip.name}")
            except Exception as e:
                logger.exception(
                    f"Exception while processing clip {clip.name}: {str(e)}"
                )

    return successful


def main():
    """
    Main function to process clips
    """
    parser = argparse.ArgumentParser(description='Process clips with "Process" status')
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
        help="Only process clips from a specific camera type (e.g., SONY, GOPRO)",
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
    args = parser.parse_args()

    config_manager = ConfigManager()

    if args.daemon:
        logger.info(
            f"Starting clip processor daemon, scanning every {args.interval} seconds"
        )
        if args.camera:
            logger.info(f"Filtering for camera type: {args.camera}")

        try:
            while True:
                # Get all pending clips
                pending_clips = get_pending_clips(config_manager)

                # Apply camera filter if specified
                if args.camera and pending_clips:
                    filtered_clips = []
                    for clip in pending_clips:
                        camera_type = extract_camera_type(clip.source_path)
                        if camera_type == args.camera:
                            filtered_clips.append(clip)
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
                        num_processed = process_batch(batch, args.max_workers)
                        total_processed += num_processed

                    logger.info(f"Processed {total_processed} clips in batches")
                elif pending_clips:
                    # Process all clips at once
                    num_processed = process_batch(
                        pending_clips, args.max_workers, args.camera
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

        # Get all pending clips
        pending_clips = get_pending_clips(config_manager)

        # Apply camera filter if specified
        if args.camera and pending_clips:
            filtered_clips = []
            for clip in pending_clips:
                camera_type = extract_camera_type(clip.source_path)
                if camera_type == args.camera:
                    filtered_clips.append(clip)
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
                num_processed = process_batch(batch, args.max_workers)
                total_processed += num_processed

            logger.info(f"Processed {total_processed} clips in batches")
        elif pending_clips:
            # Process all clips at once
            num_processed = process_batch(pending_clips, args.max_workers, args.camera)
            logger.info(f"Processed {num_processed} clips")
        else:
            logger.info("No pending clips found")


if __name__ == "__main__":
    main()
