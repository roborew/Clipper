"""
Simple crop selector component for the Clipper application.
"""

import streamlit as st
import logging
import numpy as np
import cv2
from src.services import video_service

logger = logging.getLogger("clipper.ui.simple_crop")


def select_crop_region(frame, current_frame, clip=None, output_resolution="1080p"):
    """
    Simple crop region selector using Streamlit components

    Args:
        frame: The frame to select crop region from (numpy array)
        current_frame: Current frame number
        clip: Current clip object (optional)
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

        # Calculate target dimensions based on output resolution
        target_width, target_height = video_service.calculate_crop_dimensions(
            output_resolution
        )
        aspect_ratio = target_width / target_height

        # Get proxy settings for scaling calculation
        proxy_settings = st.session_state.config_manager.get_proxy_settings()
        proxy_width = proxy_settings["width"]
        scaling_factor = proxy_width / frame_width
        logger.info(f"Scaling factor: {scaling_factor}")

        # Check if we're editing an existing keyframe from the proxy crop keyframes
        existing_crop = None
        first_time_editing = False

        if clip:
            # If editing an existing keyframe, get its crop values
            if (
                "editing_keyframe" in st.session_state
                and st.session_state.editing_keyframe is not None
            ):
                # Get the crop region from proxy keyframes for the editing frame
                editing_frame = st.session_state.editing_keyframe
                editing_frame_str = str(editing_frame)
                first_time_editing = True

                if editing_frame_str in clip.crop_keyframes_proxy:
                    existing_crop = clip.crop_keyframes_proxy[editing_frame_str]
                    logger.info(
                        f"Editing existing keyframe at frame {editing_frame} with crop {existing_crop}"
                    )
                else:
                    # If we're in editing mode but no keyframe exists, check for interpolated values
                    existing_crop = clip.get_crop_region_at_frame(
                        editing_frame, use_proxy=True
                    )
            else:
                # Not editing, just get current frame crop if available
                existing_crop = clip.get_crop_region_at_frame(
                    current_frame, use_proxy=True
                )

        # Calculate default crop size based on output resolution
        if existing_crop:
            # Use existing crop region from the keyframe we're editing
            crop_width = existing_crop[2]
            crop_height = existing_crop[3]
            x = existing_crop[0]
            y = existing_crop[1]
            logger.info(
                f"Using existing crop region: X={x}, Y={y}, Width={crop_width}, Height={crop_height}"
            )
        else:
            # Calculate the proxy dimensions by scaling down the target dimensions
            crop_width = int(target_width * scaling_factor)
            crop_height = int(target_height * scaling_factor)

            # Make sure the crop isn't larger than the frame
            if crop_width > frame_width:
                crop_width = frame_width
                crop_height = int(crop_width / aspect_ratio)

            # Center the crop region
            x = max(0, (frame_width - crop_width) // 2)
            y = max(0, (frame_height - crop_height) // 2)

            logger.info(
                f"Created new crop region: X={x}, Y={y}, Width={crop_width}, Height={crop_height}"
            )

        # Initialize position in session state if not already set or if first time editing a keyframe
        if "crop_x" not in st.session_state or first_time_editing:
            st.session_state.crop_x = x
        if "crop_y" not in st.session_state or first_time_editing:
            st.session_state.crop_y = y
        if "crop_width" not in st.session_state or first_time_editing:
            st.session_state.crop_width = crop_width
        if "crop_height" not in st.session_state or first_time_editing:
            st.session_state.crop_height = crop_height
        if "move_amount" not in st.session_state:
            st.session_state.move_amount = 10

        # Clear the editing flag once we've initialized the crop values
        if "editing_keyframe" in st.session_state:
            st.session_state.editing_keyframe = None

        # Get current position from session state
        x = st.session_state.crop_x
        y = st.session_state.crop_y
        width = st.session_state.crop_width
        height = st.session_state.crop_height

        # Draw a rectangle on the frame to show the crop region
        overlay_frame = frame.copy()
        cv_x, cv_y = int(x), int(y)
        cv_width, cv_height = int(width), int(height)

        # Draw rectangle on a copy of the frame
        cv2.rectangle(
            overlay_frame,
            (cv_x, cv_y),
            (cv_x + cv_width, cv_y + cv_height),
            (0, 255, 0),
            2,
        )

        # Add semi-transparent overlay
        alpha = 0.3
        mask = np.zeros_like(frame)
        cv2.rectangle(
            mask,
            (cv_x, cv_y),
            (cv_x + cv_width, cv_y + cv_height),
            (0, 255, 0),
            -1,  # Fill the rectangle
        )
        overlay_frame = cv2.addWeighted(overlay_frame, 1, mask, alpha, 0)

        # Display the frame with overlay
        st.image(overlay_frame, width=None)

        # Display current crop position
        st.write(f"Position: X={x}, Y={y}, Width={width}, Height={height}")

        # Create columns for movement controls
        col1, col2, col3 = st.columns(3)

        # Movement amount slider with direct input
        slider_col, input_col = st.columns([3, 1])

        with slider_col:
            st.session_state.move_amount = st.slider(
                "Movement Step Size", 1, 200, st.session_state.move_amount
            )

        with input_col:
            # Function to handle direct step size input
            def handle_step_input():
                # Ensure the input is within valid range
                step_size = max(1, min(st.session_state.step_input, 200))
                # Update the slider
                st.session_state.move_amount = step_size

            # Initialize step_input if not exists
            if "step_input" not in st.session_state:
                st.session_state.step_input = st.session_state.move_amount

            # Direct step size input
            st.number_input(
                "Pixels",
                min_value=1,
                max_value=200,
                value=st.session_state.step_input,
                step=1,
                key="step_input",
                on_change=handle_step_input,
            )

        move_amount = st.session_state.move_amount

        with col1:
            st.write("Move Horizontally")
            left_col, right_col = st.columns(2)
            with left_col:
                if st.button("← Left", key=f"move_left_{current_frame}"):
                    st.session_state.crop_x = max(
                        0, st.session_state.crop_x - move_amount
                    )
                    st.rerun()
            with right_col:
                if st.button("Right →", key=f"move_right_{current_frame}"):
                    st.session_state.crop_x = min(
                        frame_width - width, st.session_state.crop_x + move_amount
                    )
                    st.rerun()

        with col2:
            st.write("Move Vertically")
            up_col, down_col = st.columns(2)
            with up_col:
                if st.button("↑ Up", key=f"move_up_{current_frame}"):
                    st.session_state.crop_y = max(
                        0, st.session_state.crop_y - move_amount
                    )
                    st.rerun()
            with down_col:
                if st.button("Down ↓", key=f"move_down_{current_frame}"):
                    st.session_state.crop_y = min(
                        frame_height - height, st.session_state.crop_y + move_amount
                    )
                    st.rerun()

        with col3:
            st.write("Actions")
            if st.button("Center", key=f"center_{current_frame}"):
                st.session_state.crop_x = max(0, (frame_width - width) // 2)
                st.session_state.crop_y = max(0, (frame_height - height) // 2)
                st.rerun()

            if st.button("Confirm", key=f"confirm_{current_frame}"):
                crop_region = (
                    st.session_state.crop_x,
                    st.session_state.crop_y,
                    st.session_state.crop_width,
                    st.session_state.crop_height,
                )
                st.session_state.crop_selection_active = False
                return crop_region

            if st.button("Cancel", key=f"cancel_{current_frame}"):
                st.session_state.crop_selection_active = False
                st.rerun()
                return None

        return None  # Return None until confirmed

    except Exception as e:
        logger.exception(f"Error in crop selection: {str(e)}")
        st.error(f"Error in crop selection: {str(e)}")
        return None
