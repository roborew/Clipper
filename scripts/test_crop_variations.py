#!/usr/bin/env python
"""
Test script for crop variations functionality.
This script exports a clip with crop variations to validate the functionality.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from src.services.config_manager import ConfigManager
from src.services import proxy_service
from src.utils import multi_crop
from src.services import clip_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to test crop variations"""
    parser = argparse.ArgumentParser(description="Test crop variations functionality")
    parser.add_argument("--clip-id", type=str, help="ID of clip to export")
    parser.add_argument("--clip-name", type=str, help="Name of clip to export")
    parser.add_argument(
        "--all-clips",
        action="store_true",
        help='Process all clips with "Process" status',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process clips regardless of status",
    )
    parser.add_argument(
        "--specify-crop",
        type=str,
        help="Specify crop region as 'x,y,width,height' (e.g., '100,100,400,300')",
    )
    parser.add_argument(
        "--wide-crop-factor",
        type=float,
        default=1.5,
        help="Multiplier for wide crop (1.5 = 50% larger)",
    )
    parser.add_argument(
        "--variations",
        type=str,
        default="original,wide,full",
        help="Comma-separated crop variations to generate",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available clips instead of processing them",
    )
    args = parser.parse_args()

    config_manager = ConfigManager()

    # Parse manual crop if specified
    manual_crop = None
    if args.specify_crop:
        try:
            crop_parts = [int(x) for x in args.specify_crop.split(",")]
            if len(crop_parts) == 4:
                manual_crop = tuple(crop_parts)
                logger.info(f"Using manually specified crop: {manual_crop}")
            else:
                logger.error("Crop must have 4 values: x,y,width,height")
                return
        except ValueError:
            logger.error("Invalid crop format. Should be 4 integers: x,y,width,height")
            return

    # Get clips
    clip_configs = get_clip_configs(config_manager)

    # List all clips if requested
    if args.list:
        print("\nAvailable clips:")
        for clip_file, clips in clip_configs.items():
            print(f"\nFile: {clip_file}")
            for i, clip in enumerate(clips):
                status_marker = "→" if clip.status == "Process" else " "
                has_crop = (
                    "*"
                    if hasattr(clip, "crop_keyframes") and clip.crop_keyframes
                    else " "
                )
                print(
                    f"  {status_marker} {has_crop} [{i}] {clip.name} (ID: {getattr(clip, 'id', 'N/A')})"
                )
        print("\nLegend: → = 'Process' status, * = Has crop keyframes")
        return

    if not args.clip_id and not args.all_clips and not args.clip_name and not args.list:
        parser.error(
            "Either --clip-id, --clip-name, --all-clips, or --list is required"
        )

    # Find clips to process
    clips_to_process = []

    if args.all_clips:
        # Find all clips with "Process" status
        logger.info("Searching for clips with 'Process' status...")
        for clip_file, clips in clip_configs.items():
            for i, clip in enumerate(clips):
                if args.force or clip.status == "Process":
                    if args.force:
                        logger.info(f"Including clip {clip.name} due to --force flag")
                    clips_to_process.append((clip_file, i, clip))
    elif args.clip_id:
        # Find clip by ID
        logger.info(f"Searching for clip with ID: {args.clip_id}")
        found = False
        for clip_file, clips in clip_configs.items():
            for i, clip in enumerate(clips):
                if getattr(clip, "id", None) == args.clip_id:
                    clips_to_process.append((clip_file, i, clip))
                    found = True
                    break
        if not found:
            logger.error(f"No clip found with ID: {args.clip_id}")
    elif args.clip_name:
        # Find clip by name
        logger.info(f"Searching for clip with name: {args.clip_name}")
        found = False
        for clip_file, clips in clip_configs.items():
            for i, clip in enumerate(clips):
                if clip.name == args.clip_name:
                    clips_to_process.append((clip_file, i, clip))
                    found = True
                    break
        if not found:
            logger.error(f"No clip found with name: {args.clip_name}")

    if not clips_to_process:
        logger.error("No clips found to process")
        return

    logger.info(f"Found {len(clips_to_process)} clips to process")

    # Process each clip
    for clip_file, clip_index, clip in clips_to_process:
        try:
            process_clip_with_variations(
                clip,
                args.variations,
                args.wide_crop_factor,
                config_manager,
                manual_crop,
            )

            # Save the updated clip
            clips = clip_service.load_clips(clip_file)
            clips[clip_index] = clip
            clip_service.save_clips(clips, clip_file)
            logger.info(f"Saved updated clip to {clip_file}")
        except Exception as e:
            logger.error(f"Error processing clip {clip.name}: {e}")
            import traceback

            logger.error(traceback.format_exc())


def get_clip_configs(config_manager):
    """Get all clip configuration files"""
    result = {}
    clips_dir = config_manager.configs_dir
    for config_file in clips_dir.glob("**/*.json"):
        try:
            clips = clip_service.load_clips(config_file)
            if clips:
                result[config_file] = clips
        except Exception as e:
            logger.warning(f"Error loading clips from {config_file}: {e}")
    return result


def process_clip_with_variations(
    clip, variations_str, wide_crop_factor, config_manager, manual_crop=None
):
    """Process a clip with variations"""
    # Resolve the source path
    source_path = resolve_source_path(clip.source_path, config_manager)
    logger.info(f"Processing clip {clip.name} from {source_path}")

    # Check if source path exists
    if not os.path.exists(source_path):
        logger.error(f"Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Get video dimensions
    try:
        frame_dimensions = get_video_dimensions(source_path)
        logger.info(f"Video dimensions: {frame_dimensions}")
    except Exception as e:
        logger.error(f"Error getting video dimensions: {e}")
        frame_dimensions = (1920, 1080)  # Default to 1080p

    # Get the original crop region
    original_crop = manual_crop  # Use manual crop if provided

    # If no manual crop, try to get from clip's keyframes
    if (
        original_crop is None
        and hasattr(clip, "crop_keyframes")
        and clip.crop_keyframes
    ):
        try:
            # Use the first keyframe if available
            first_keyframe = min(int(k) for k in clip.crop_keyframes.keys())
            original_crop = clip.crop_keyframes[str(first_keyframe)]
            logger.info(f"Using crop from first keyframe: {original_crop}")
        except Exception as e:
            logger.error(f"Error getting crop from keyframes: {e}")

    if original_crop is None:
        logger.warning("No crop region available. 'wide' variation will be ignored.")

    # Process each variation
    variations = [v.strip() for v in variations_str.split(",") if v.strip()]
    logger.info(f"Processing variations: {variations}")

    results = {}
    for variation in variations:
        if variation not in ["original", "wide", "full"]:
            logger.warning(f"Skipping unknown variation: {variation}")
            continue

        # Skip wide variation if no crop region is available
        if variation == "wide" and original_crop is None:
            logger.warning(
                "Skipping 'wide' variation because no crop region is available"
            )
            continue

        variation_suffix = "" if variation == "original" else f"_{variation}"
        variation_clip_name = f"{clip.name}{variation_suffix}"
        logger.info(f"Processing variation: {variation} as {variation_clip_name}")

        # Create a copy of the clip
        variation_clip = clip.copy()
        variation_clip.name = variation_clip_name

        # Set crop keyframes and crop region based on variation
        if variation == "original" and not manual_crop:
            # For original, keep the original keyframes unless manual crop is provided
            crop_keyframes = (
                variation_clip.crop_keyframes
                if hasattr(variation_clip, "crop_keyframes")
                else None
            )
            crop_region = original_crop
        else:
            # For other variations or when manual crop is provided, use specific crop
            crop_keyframes = None
            crop_region = multi_crop.get_crop_for_variation(
                variation, original_crop, frame_dimensions, wide_crop_factor
            )

        if crop_region:
            logger.info(f"Using crop for {variation}: {crop_region}")
        else:
            logger.info(f"No crop for {variation} (full frame)")

        try:
            # Export the clip
            export_path = proxy_service.export_clip(
                source_path=source_path,
                clip_name=variation_clip_name,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                crop_region=crop_region,
                crop_keyframes=crop_keyframes,
                config_manager=config_manager,
            )

            if export_path:
                results[variation] = export_path
                logger.info(f"Successfully exported {variation} as {export_path}")
            else:
                logger.error(f"Failed to export {variation}")
        except Exception as e:
            logger.error(f"Error exporting {variation}: {e}")
            # Continue with next variation

    # Update the original clip with all export paths
    if results:
        all_paths = list(results.values())
        if len(all_paths) == 1:
            clip.export_path = all_paths[0]
        else:
            clip.export_path = all_paths

        clip.export_paths = ",".join(all_paths)
        logger.info(f"Updated clip with export paths: {all_paths}")
    else:
        logger.warning("No variations were successfully exported")


def resolve_source_path(rel_path, config_manager):
    """Resolve relative path to absolute path using ConfigManager"""
    from scripts.process_clips import resolve_source_path as main_resolve
    return main_resolve(rel_path, config_manager)


def get_video_dimensions(video_path):
    """Get video dimensions using FFprobe"""
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    dimensions = result.stdout.strip().split(",")
    return (int(dimensions[0]), int(dimensions[1]))


if __name__ == "__main__":
    main()
