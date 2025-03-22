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
        logger.warning("Cannot calculate wider crop from None crop region")
        return None

    x, y, width, height = original_crop
    frame_width, frame_height = frame_dimensions

    # Log detailed calculations for debugging
    logger.info(f"Original crop: x={x}, y={y}, width={width}, height={height}")
    logger.info(f"Frame dimensions: width={frame_width}, height={frame_height}")
    logger.info(f"Applying wide crop factor: {factor}")

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
    ideal_x = x - (new_width - width) // 2
    ideal_y = y - (new_height - height) // 2

    logger.info(f"Ideal centered position: x={ideal_x}, y={ideal_y}")

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
            percent_w = round((w / frame_w) * 100, 1)
            percent_h = round((h / frame_h) * 100, 1)
            logger.info(
                f"Original crop dimensions: {w}x{h} ({percent_w}% × {percent_h}% of frame)"
            )
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
        process_clip_func: Function to call for processing each variation
        clip: Clip object to process
        crop_variations: List of crop variations to generate or comma-separated string
        wide_crop_factor: Factor for the 'wide' crop variation (1.5 = 50% larger)
        camera_type: Optional camera type for filtering
        crop_camera_types: Optional list of camera types to apply variations to
        exclude_camera_types: Optional list of camera types to exclude from variations
        config_manager: Optional ConfigManager instance
        multi_crop: Whether to force multi-crop processing
        **kwargs: Additional arguments to pass to process_clip_func

    Returns:
        Dict of results for each variation and combined export paths
    """
    logger.info(
        f"Initial export_path state for multi_crop processing: {getattr(clip, 'export_path', None)}"
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

    # If multi_crop is forced, always apply variations regardless of camera type
    if force_multi_crop:
        logger.info("Forcing multiple crop variations due to --multi-crop flag")
        should_apply_variations = True
    # Otherwise, only apply variations if camera type matches criteria
    elif camera_type and crop_camera_types:
        # Convert to list if it's a string
        if isinstance(crop_camera_types, str):
            crop_camera_types = [ct.strip() for ct in crop_camera_types.split(",")]

        # Check if the camera type is in the allowed list
        should_apply_variations = camera_type in crop_camera_types
        logger.info(
            f"Camera type {camera_type} {'is' if should_apply_variations else 'is not'} in allowed types {crop_camera_types}"
        )
    # Or if camera type is not in excluded list
    elif camera_type and exclude_camera_types:
        # Convert to list if it's a string
        if isinstance(exclude_camera_types, str):
            exclude_camera_types = [
                ct.strip() for ct in exclude_camera_types.split(",")
            ]

        # Check if camera type is in exclude list
        should_apply_variations = camera_type not in exclude_camera_types
        logger.info(
            f"Camera type {camera_type} {'is not' if should_apply_variations else 'is'} in excluded types {exclude_camera_types}"
        )

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

    # Check if clip has a crop region - IMPORTANT: Some clips may have None as crop_region
    # Try to get it from crop_keyframes if available
    original_crop = None
    if hasattr(clip, "crop_region") and clip.crop_region is not None:
        original_crop = clip.crop_region
        logger.info(f"Using clip.crop_region: {original_crop}")
    else:
        # Try to get crop region from keyframes
        original_crop = clip.get_crop_region_at_frame(clip.start_frame, use_proxy=False)
        logger.info(f"Using crop from keyframes: {original_crop}")

    # If no crop is defined at all, log an important warning
    if original_crop is None:
        logger.warning(f"Clip {clip.name} has no crop region defined!")
        # For 'wide' variation to work properly, we need to create a fake original crop
        # that matches the full frame, so the 'wide' variation can be calculated
        if "wide" in variation_list:
            logger.info(
                "Creating default crop region based on full frame for 'wide' variation"
            )
            # Use a crop that's centered in frame but ~20% smaller than full frame
            frame_width, frame_height = frame_dimensions
            default_width = int(frame_width * 0.8)
            default_height = int(frame_height * 0.8)
            default_x = (frame_width - default_width) // 2
            default_y = (frame_height - default_height) // 2
            original_crop = (default_x, default_y, default_width, default_height)
            logger.info(f"Created default crop region: {original_crop}")

    # Track results and export paths
    results = {}
    all_export_paths = []

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

        # For 'wide' variation, scale each keyframe's crop region if keyframes exist
        if (
            variation == "wide"
            and hasattr(clip, "crop_keyframes")
            and clip.crop_keyframes
        ):
            # Create a copy of the keyframes to modify
            scaled_keyframes = {}

            # Process each keyframe to create a wider version
            for frame_num, crop_region in clip.crop_keyframes.items():
                # Calculate a wider crop for this keyframe
                wider_crop = calculate_wider_crop(
                    crop_region, wide_crop_factor, frame_dimensions
                )

                # Store the wider crop in the new keyframes dictionary
                scaled_keyframes[frame_num] = wider_crop

            # Set the scaled keyframes on the variation clip
            variation_clip.crop_keyframes = scaled_keyframes
            logger.info(
                f"Created {len(scaled_keyframes)} scaled keyframes for 'wide' variation"
            )
        # For 'full' variation or if no crop keyframes, clear keyframes
        elif variation != "original":
            variation_clip.crop_keyframes = None
            logger.info(
                f"Cleared crop keyframes for non-original variation: {variation}"
            )

        # Set static crop region based on variation - for full and fallback for wide if no keyframes
        if (
            variation == "wide"
            and not hasattr(variation_clip, "crop_keyframes")
            or not variation_clip.crop_keyframes
        ):
            variation_crop = get_crop_for_variation(
                variation, original_crop, frame_dimensions, wide_crop_factor
            )
            variation_clip.crop_region = variation_crop
        elif variation == "full":
            variation_clip.crop_region = None

        # Log crop details for clarity
        if (
            variation == "wide"
            and hasattr(variation_clip, "crop_keyframes")
            and variation_clip.crop_keyframes
        ):
            # Log a sample of the scaled keyframes (first one)
            first_frame = sorted(variation_clip.crop_keyframes.keys())[0]
            sample_crop = variation_clip.crop_keyframes[first_frame]
            x, y, w, h = sample_crop
            frame_w, frame_h = frame_dimensions
            percent_w = round((w / frame_w) * 100, 1)
            percent_h = round((h / frame_h) * 100, 1)
            logger.info(
                f"Using scaled keyframes for {variation}. First keyframe ({first_frame}): {sample_crop} ({percent_w}% × {percent_h}% of frame)"
            )
        elif hasattr(variation_clip, "crop_region") and variation_clip.crop_region:
            x, y, w, h = variation_clip.crop_region
            frame_w, frame_h = frame_dimensions
            percent_w = round((w / frame_w) * 100, 1)
            percent_h = round((h / frame_h) * 100, 1)
            logger.info(
                f"Using crop for {variation}: {variation_clip.crop_region} ({percent_w}% × {percent_h}% of frame)"
            )
        else:
            logger.info(f"No crop for {variation} (full frame)")

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

    # If any paths were generated, add them to the original clip
    if all_export_paths:
        logger.info(f"Setting export_path on original clip to: {all_export_paths}")
        clip.export_path = all_export_paths
    else:
        logger.warning("No export paths were collected from any variations")

    return {"results": results, "export_paths": all_export_paths}
