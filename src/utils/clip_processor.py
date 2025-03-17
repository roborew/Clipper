"""
Utility for processing clips in an automated fashion.
"""

import os
import json
import logging
from pathlib import Path
from src.services.clip_service import Clip, save_clips, load_clips
from src.services.config_manager import ConfigManager

logger = logging.getLogger("clipper.processor")


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
            clips[clip_index].export_path = processed_clip.export_path
            logger.info(f"Updated export path to: {processed_clip.export_path}")

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


# Example usage:
# def my_processing_function(clip):
#     # Process the clip here (e.g., run FFmpeg to encode it)
#     # ...
#     return True  # Return True if successful
#
# num_processed = process_pending_clips(my_processing_function)
# print(f"Successfully processed {num_processed} clips")
