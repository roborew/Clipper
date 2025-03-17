"""
Crop selector component for the Clipper application.
"""

import streamlit as st
import logging
import numpy as np
from src.services import video_service

logger = logging.getLogger("clipper.ui.crop")


def select_crop_region(frame, output_resolution="1080p"):
    """
    Interactive crop region selector (temporarily disabled)

    Args:
        frame: The frame to select crop region from (numpy array)
        output_resolution: Target output resolution

    Returns:
        None (functionality is temporarily disabled)
    """
    st.warning("Crop selection is temporarily disabled.")
    return None


def select_crop_region_direct(frame, current_frame, clip, output_resolution="1080p"):
    """
    Direct crop region selector that updates the main frame display

    Args:
        frame: The frame to select crop region from (numpy array)
        current_frame: Current frame number
        clip: Current clip object
        output_resolution: Target output resolution

    Returns:
        Tuple of (x, y, width, height) or None if cancelled
    """
    try:
        if frame is None:
            st.error("No frame available for crop selection")
            return None

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Calculate default crop dimensions based on output resolution
        target_width, target_height = video_service.calculate_crop_dimensions(
            output_resolution
        )
        aspect_ratio = target_width / target_height

        # Check if there's an existing crop region for this frame
        existing_crop = None
        if clip:
            existing_crop = clip.get_crop_region_at_frame(current_frame, use_proxy=True)

        # Initialize crop region
        if existing_crop:
            # Use existing crop region
            x, y, crop_width, crop_height = existing_crop
        else:
            # Calculate default crop size (centered, maintaining aspect ratio)
            if frame_width / frame_height > aspect_ratio:
                # Frame is wider than target aspect ratio
                crop_height = frame_height
                crop_width = int(crop_height * aspect_ratio)
            else:
                # Frame is taller than target aspect ratio
                crop_width = frame_width
                crop_height = int(crop_width / aspect_ratio)

            # Center the crop region
            x = max(0, (frame_width - crop_width) // 2)
            y = max(0, (frame_height - crop_height) // 2)

        # Initialize session state for crop position if not exists
        if "crop_selection_active" not in st.session_state:
            st.session_state.crop_selection_active = True
            st.session_state.crop_x = x
            st.session_state.crop_y = y
            st.session_state.crop_width = crop_width
            st.session_state.crop_height = crop_height

        # Create UI for crop selection
        st.subheader("Crop Region Selection")

        # Display information about the crop
        st.info(
            f"Adjusting crop region with {output_resolution} aspect ratio ({aspect_ratio:.3f})"
        )

        # Create columns for the position controls
        pos_col1, pos_col2, pos_col3 = st.columns(3)

        with pos_col1:
            # Left/Right buttons
            if st.button("◀️ Left", key="move_left"):
                st.session_state.crop_x = max(0, st.session_state.crop_x - 10)

        with pos_col2:
            # Up/Down buttons
            if st.button("🔼 Up", key="move_up"):
                st.session_state.crop_y = max(0, st.session_state.crop_y - 10)
            if st.button("🔽 Down", key="move_down"):
                st.session_state.crop_y = min(
                    frame_height - st.session_state.crop_height,
                    st.session_state.crop_y + 10,
                )

        with pos_col3:
            # Right button
            if st.button("Right ▶️", key="move_right"):
                st.session_state.crop_x = min(
                    frame_width - st.session_state.crop_width,
                    st.session_state.crop_x + 10,
                )

        # Center button
        if st.button("Center", key="center_crop"):
            st.session_state.crop_x = max(
                0, (frame_width - st.session_state.crop_width) // 2
            )
            st.session_state.crop_y = max(
                0, (frame_height - st.session_state.crop_height) // 2
            )

        # Create columns for the size controls
        size_col1, size_col2 = st.columns(2)

        with size_col1:
            # Smaller button
            if st.button("Smaller", key="smaller_crop"):
                # Reduce size by 10%
                new_width = max(50, int(st.session_state.crop_width * 0.9))
                new_height = int(new_width / aspect_ratio)

                # Adjust position to keep centered
                x_diff = (st.session_state.crop_width - new_width) // 2
                y_diff = (st.session_state.crop_height - new_height) // 2

                st.session_state.crop_x += x_diff
                st.session_state.crop_y += y_diff
                st.session_state.crop_width = new_width
                st.session_state.crop_height = new_height

        with size_col2:
            # Larger button
            if st.button("Larger", key="larger_crop"):
                # Increase size by 10%
                new_width = min(frame_width, int(st.session_state.crop_width * 1.1))
                new_height = int(new_width / aspect_ratio)

                # Ensure it fits within the frame
                if new_height > frame_height:
                    new_height = frame_height
                    new_width = int(new_height * aspect_ratio)

                # Adjust position to keep centered
                x_diff = (st.session_state.crop_width - new_width) // 2
                y_diff = (st.session_state.crop_height - new_height) // 2

                st.session_state.crop_x = max(0, st.session_state.crop_x + x_diff)
                st.session_state.crop_y = max(0, st.session_state.crop_y + y_diff)
                st.session_state.crop_width = new_width
                st.session_state.crop_height = new_height

        # Ensure crop region fits within frame
        st.session_state.crop_x = max(
            0, min(st.session_state.crop_x, frame_width - st.session_state.crop_width)
        )
        st.session_state.crop_y = max(
            0, min(st.session_state.crop_y, frame_height - st.session_state.crop_height)
        )

        # Display crop dimensions
        st.text(f"Position: X={st.session_state.crop_x}, Y={st.session_state.crop_y}")
        st.text(
            f"Size: {st.session_state.crop_width}x{st.session_state.crop_height} pixels"
        )

        # Calculate output dimensions
        out_width, out_height = video_service.calculate_crop_dimensions(
            output_resolution,
            st.session_state.crop_width / st.session_state.crop_height,
        )
        st.text(f"Output: {out_width}x{out_height} pixels ({output_resolution})")

        # Create crop region tuple
        crop_region = (
            st.session_state.crop_x,
            st.session_state.crop_y,
            st.session_state.crop_width,
            st.session_state.crop_height,
        )

        # Confirmation buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Set Crop Position", key="confirm_crop"):
                # Clear the active flag
                st.session_state.crop_selection_active = False
                return crop_region

        with col2:
            if st.button("Cancel", key="cancel_crop"):
                # Clear the active flag
                st.session_state.crop_selection_active = False
                return None

        # Return the current crop region for display
        return crop_region

    except Exception as e:
        logger.exception(f"Error in direct crop selection: {str(e)}")
        st.error(f"Error in direct crop selection: {str(e)}")
        return None
