"""
Utility module for handling multiple crop variations in clip exports.

This module provides functions to calculate different crop regions
for generating multiple variations of the same clip.
"""

import logging
from src.services import video_service

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
    # For 'full' variation, return None to use full frame
    if variation == "full":
        logger.info("Using full frame (no crop) for 'full' variation")
        return None

    # For 'original' variation, return the original crop (which could be None)
    if variation == "original":
        if original_crop:
            logger.info(f"Using original crop: {original_crop}")
            return original_crop
        else:
            logger.warning(
                "No original crop defined, using full frame for 'original' variation"
            )
            return None

    # For 'wide' variation
    if variation == "wide":
        if original_crop:
            # Calculate a wider crop using the helper function
            wider_crop = calculate_wider_crop(
                original_crop, wide_crop_factor, frame_dimensions
            )
            logger.info(
                f"Calculated wider crop: {wider_crop} from original: {original_crop}"
            )
            return wider_crop
        else:
            logger.warning(
                "Cannot calculate 'wide' variation without original crop. Using full frame."
            )
            return None

    # Unknown variation type
    logger.warning(f"Unknown crop variation: {variation}")
    return original_crop  # Fall back to original crop


def process_clip_with_variations(
    process_clip_func,
    clip,
    crop_variations,
    wide_crop_factor=1.5,
    camera_type=None,
    crop_camera_types=None,
    exclude_camera_types=None,
    config_manager=None,
    multi_crop=False,  # Explicitly use multi_crop flag
    **kwargs,
):
    """
    Process a clip with multiple crop variations.

    Args:
        process_clip_func: Function to process a single clip variation
        clip: The clip to process
        crop_variations: List of crop variations to process ('original', 'wide', 'full')
        wide_crop_factor: Factor for the 'wide' crop variation (1.5 = 50% larger)
        camera_type: The camera type for this clip
        crop_camera_types: List of camera types to apply crop variations to
        exclude_camera_types: List of camera types to exclude from crop variations
        config_manager: ConfigManager instance to get config settings
        multi_crop: Whether to force multiple crop variations
        **kwargs: Additional arguments to pass to process_clip_func

    Returns:
        Dictionary mapping variation names to success/failure status
    """
    # Ensure export_path exists as a list - we expect it to be an empty list
    # initialized by the parent process_clip function that calls this
    if not hasattr(clip, "export_path"):
        clip.export_path = []
    elif not isinstance(clip.export_path, list):
        # Convert to list if somehow it's not a list
        if clip.export_path:
            clip.export_path = [clip.export_path] if clip.export_path else []
        else:
            clip.export_path = []

    # Log the initial state of export_path
    logger.info(
        f"Initial export_path state for multi_crop processing: {clip.export_path}"
    )

    # Check configuration if not explicitly provided
    if config_manager and (crop_camera_types is None and exclude_camera_types is None):
        try:
            # Access the export configuration directly from the config property
            export_config = config_manager.config.get("export", {})
            crop_variations_config = export_config.get("crop_variations", {})

            if crop_variations_config.get("enabled", True):
                # Get camera types from config
                crop_camera_types = crop_variations_config.get("camera_types")
                exclude_camera_types = crop_variations_config.get(
                    "exclude_camera_types"
                )
                logger.info(f"Read crop camera types from config: {crop_camera_types}")
                logger.info(
                    f"Read exclude camera types from config: {exclude_camera_types}"
                )
        except Exception as e:
            logger.warning(f"Error reading crop variations config: {e}")

    # Force multi_crop if explicitly requested with variations
    force_multi_crop = kwargs.get("multi_crop", False) or multi_crop

    # Determine if we should apply all variations or just the original
    should_apply_variations = True

    # Log camera type for debugging
    if camera_type:
        logger.info(f"Checking camera type: {camera_type} for variations")
    else:
        logger.warning("No camera type specified for variation filtering")

    # If camera type is specified and we have inclusion/exclusion lists
    if camera_type and not force_multi_crop:
        # Check against inclusion list if provided
        if crop_camera_types:
            camera_type_upper = camera_type.upper() if camera_type else ""
            crop_camera_types_upper = [ct.upper() for ct in crop_camera_types if ct]

            # Check if any part of the camera type matches the inclusion list
            matches_inclusion = False
            for allowed_type in crop_camera_types_upper:
                if allowed_type and allowed_type in camera_type_upper:
                    matches_inclusion = True
                    break

            should_apply_variations = matches_inclusion

            if not should_apply_variations:
                logger.info(
                    f"Camera type {camera_type} not in crop_camera_types list: {crop_camera_types}. Only processing original variation."
                )

        # Check against exclusion list if provided
        if exclude_camera_types and should_apply_variations:
            camera_type_upper = camera_type.upper() if camera_type else ""
            exclude_camera_types_upper = [
                ct.upper() for ct in exclude_camera_types if ct
            ]

            # Check if any part of the camera type matches the exclusion list
            for excluded_type in exclude_camera_types_upper:
                if excluded_type and excluded_type in camera_type_upper:
                    should_apply_variations = False
                    logger.info(
                        f"Camera type {camera_type} matches excluded type {excluded_type}. Only processing original variation."
                    )
                    break

    # If multi_crop is explicitly set, override the camera type filtering
    if force_multi_crop:
        logger.info("Forcing multiple crop variations due to --multi-crop flag")
        should_apply_variations = True

    # Parse crop variations if provided as string
    if isinstance(crop_variations, str):
        variation_list = [v.strip() for v in crop_variations.split(",") if v.strip()]
    elif isinstance(crop_variations, list):
        variation_list = crop_variations
    else:
        variation_list = ["original"]  # Default to original only

    logger.info(f"Crop variations requested: {variation_list}")

    # If we shouldn't apply variations, only use the original
    if not should_apply_variations:
        logger.info("Only processing original variation due to camera type filtering")
        variation_list = ["original"]
    else:
        logger.info(f"Processing all crop variations: {variation_list}")

    # Check if clip has a crop region
    original_crop = clip.get_crop_region_at_frame(clip.start_frame, use_proxy=False)
    if not original_crop and "wide" in variation_list:
        logger.warning(
            f"Clip {clip.name} has no crop region, 'wide' variation may not work properly"
        )

    # Track results and export paths
    results = {}
    all_export_paths = []

    # Get the source video dimensions
    try:
        source_path = kwargs.get("source_path", clip.source_path)
        frame_dimensions = video_service.get_video_dimensions(source_path)
        logger.info(f"Source video dimensions: {frame_dimensions}")
    except Exception as e:
        logger.error(f"Error getting video dimensions: {str(e)}")
        frame_dimensions = (
            1920,
            1080,
        )  # Default to 1080p if dimensions can't be determined

    # Process each variation
    for variation in variation_list:
        if variation not in ["original", "wide", "full"]:
            logger.warning(f"Skipping unknown variation: {variation}")
            continue

        variation_suffix = "" if variation == "original" else f"_{variation}"
        variation_clip_name = f"{clip.name}{variation_suffix}"
        logger.info(f"Processing variation: {variation} as {variation_clip_name}")

        # Create a modified clip for this variation
        variation_clip = clip.copy()
        variation_clip.name = variation_clip_name

        # Only use keyframes for original crop
        if variation != "original":
            variation_clip.crop_keyframes = None
            logger.info(
                f"Cleared crop keyframes for non-original variation: {variation}"
            )

        # Set crop region based on variation
        variation_crop = get_crop_for_variation(
            variation, original_crop, frame_dimensions, wide_crop_factor
        )

        if variation_crop:
            logger.info(f"Using crop for {variation}: {variation_crop}")
        else:
            logger.info(f"No crop for {variation} (full frame)")

        # Store the crop region in the clip
        variation_clip.crop_region = variation_crop

        # Process this variation
        logger.info(f"Sending variation {variation} to processing function")
        success = process_clip_func(variation_clip, **kwargs)
        results[variation] = success
        logger.info(f"Result for {variation}: {'Success' if success else 'Failed'}")

        # Store export path
        if success and hasattr(variation_clip, "export_path"):
            if variation_clip.export_path:
                # Handle list or string export_path from the variation
                if isinstance(variation_clip.export_path, list):
                    for path in variation_clip.export_path:
                        if path and path not in all_export_paths:
                            all_export_paths.append(path)
                            logger.info(f"Added path from {variation}: {path}")
                else:
                    path = variation_clip.export_path
                    if path and path not in all_export_paths:
                        all_export_paths.append(path)
                        logger.info(f"Added path from {variation}: {path}")

    # Simply set the original clip's export_path to all collected paths
    # Don't worry about preserving previous paths - that's handled in process_clip
    if all_export_paths:
        # Set export_path as a list
        clip.export_path = all_export_paths
        logger.info(
            f"Set clip.export_path to {len(all_export_paths)} paths: {all_export_paths}"
        )

        # Update export_paths string attribute for backward compatibility
        if hasattr(clip, "export_paths"):
            clip.export_paths = ",".join(all_export_paths)
            logger.info(f"Updated clip.export_paths to: {clip.export_paths}")
    else:
        logger.warning("No export paths were collected from any variations")

    return results
