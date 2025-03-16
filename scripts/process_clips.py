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

# Add the project root to the path so we can import the app modules
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.utils.clip_processor import process_pending_clips
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


def process_clip(clip):
    """
    Process a single clip - this is where you would implement your custom processing logic

    Args:
        clip: The Clip object to process

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
    args = parser.parse_args()

    config_manager = ConfigManager()

    if args.daemon:
        logger.info(
            f"Starting clip processor daemon, scanning every {args.interval} seconds"
        )
        try:
            while True:
                num_processed = process_pending_clips(process_clip, config_manager)
                logger.info(f"Processed {num_processed} clips")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
    else:
        logger.info("Running one-time clip processing scan")
        num_processed = process_pending_clips(process_clip, config_manager)
        logger.info(f"Processed {num_processed} clips")


if __name__ == "__main__":
    main()
