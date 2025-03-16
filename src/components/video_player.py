"""
Video player component for the Clipper application.
"""

import streamlit as st
import logging
from pathlib import Path
import os
import time
import datetime
import streamlit.components.v1 as components
import base64
import cv2
import io
from PIL import Image

from src.services import video_service

logger = logging.getLogger("clipper.ui.player")

# Get the absolute path to the static directory
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)


# Function to create animation container with unique ID
def create_animation_container(container_id):
    """Create a container for animation with a unique ID"""
    container_html = f"""
    <div id="{container_id}" 
         style="width: 100%; 
                min-height: 300px; 
                border: 1px solid #ddd; 
                padding: 10px; 
                margin-top: 10px; 
                background-color: #f9f9f9;">
        <div style="text-align: center; padding: 20px;">
            <p>Animation container ready</p>
        </div>
    </div>
    """
    return container_html


def display_video_player(
    video_path,
    current_frame,
    fps,
    total_frames,
    on_frame_change=None,
    crop_region=None,
    config_manager=None,
    clip=None,
):
    """
    Display a video player with controls

    Args:
        video_path: Path to the video file
        current_frame: Current frame number
        fps: Frames per second
        total_frames: Total number of frames
        on_frame_change: Callback function when frame changes
        crop_region: Optional crop region to display
        config_manager: ConfigManager instance
        clip: Current clip object for keyframe interpolation

    Returns:
        The updated current frame number
    """
    try:
        # Check if we should use proxy video
        if "proxy_path" in st.session_state and st.session_state.proxy_path:
            # Use proxy video if available
            proxy_path = st.session_state.proxy_path
            logger.debug(f"Using proxy video for playback: {proxy_path}")
            video_path = proxy_path

        # Initialize session state for navigation
        if "nav_action" not in st.session_state:
            st.session_state.nav_action = None

        # Initialize animation state if not exists
        if "animation_active" not in st.session_state:
            st.session_state.animation_active = False

        if "animation_speed" not in st.session_state:
            st.session_state.animation_speed = 1.0

        # Initialize show_previews toggle if not exists
        if "show_previews" not in st.session_state:
            st.session_state.show_previews = True

        # Initialize frame navigation state
        if "frame_to_navigate" not in st.session_state:
            st.session_state.frame_to_navigate = None

        # Store the current frame in session state for the JavaScript animation
        if "js_current_frame" not in st.session_state:
            st.session_state.js_current_frame = current_frame

        # Check if we need to process a navigation update from previous run
        if st.session_state.frame_to_navigate is not None:
            current_frame = st.session_state.frame_to_navigate
            # Call the frame change callback if provided
            if on_frame_change:
                on_frame_change(current_frame)
            # Reset navigation state
            st.session_state.frame_to_navigate = None

        # Initialize original_frame to avoid scope issues
        original_frame = current_frame

        # Store the start frame in session state for the animation
        st.session_state.animation_start_frame = original_frame

        # Add a toggle to show/hide previews
        st.session_state.show_previews = st.checkbox(
            "Show Video Previews",
            value=st.session_state.show_previews,
            help="Toggle to show or hide the video preview panels",
        )

        # Only display video previews if the toggle is on
        if st.session_state.show_previews:
            # Display the video with HTML components (half size)
            col_video1, col_video2 = st.columns([1, 1])

            with col_video1:
                st.subheader("Full Video Preview")
                # Calculate current time in seconds
                current_time = current_frame / fps if fps > 0 else 0

                # Display the video starting at the current frame
                video_file = open(video_path, "rb")
                video_bytes = video_file.read()
                st.video(video_bytes, start_time=current_time)

            with col_video2:
                # Preview Animation Section
                preview_col1, preview_col2 = st.columns([2, 3])
                with preview_col1:
                    st.subheader("Animation Preview")
                with preview_col2:
                    if "animation_frames" not in st.session_state:
                        st.session_state.animation_frames = []
                    # Create a list of frames for animation
                    if st.button("Generate Preview", key="prepare_frames"):
                        st.session_state.animation_frames = []

                        # Create a container for progress and status
                        progress_container = st.container()

                        # Initialize progress bar and status placeholder
                        progress_bar = progress_container.progress(0)
                        status_placeholder = progress_container.empty()

                        # Get frame range from clip if available, otherwise use current position
                        if clip:
                            start_frame = clip.start_frame
                            end_frame = clip.end_frame
                            total_frames_to_animate = end_frame - start_frame + 1
                            animation_start = start_frame
                        else:
                            # If no clip, animate from current position
                            start_frame = current_frame
                            end_frame = total_frames - 1
                            total_frames_to_animate = end_frame - start_frame + 1
                            animation_start = current_frame

                        # Store the start frame in session state for the animation
                        st.session_state.animation_start_frame = animation_start

                        # For each frame in the range, get the frame with crop region already applied
                        for i in range(total_frames_to_animate):
                            # Calculate the frame index
                            frame_idx = start_frame + i

                            # Get the frame using the same method as the main display
                            frame = video_service.get_frame(video_path, frame_idx)

                            if frame is not None:
                                # Get crop region for current frame
                                frame_crop = None
                                if clip:
                                    frame_crop = clip.get_crop_region_at_frame(
                                        frame_idx,
                                        use_proxy=True,  # Use proxy resolution for UI display
                                    )

                                # Apply crop overlay if specified
                                if frame_crop:
                                    frame = video_service.draw_crop_overlay(
                                        frame, frame_crop
                                    )
                                    logger.debug(
                                        f"Applied crop overlay to clip frame {frame_idx}"
                                    )
                                else:
                                    logger.debug(
                                        f"No crop region for clip frame {frame_idx}"
                                    )

                                # Convert OpenCV image from BGR to RGB for correct colors
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                                # Convert OpenCV image to bytes
                                _, buffer = cv2.imencode(".jpg", frame_rgb)
                                img_str = base64.b64encode(buffer).decode("utf-8")

                                st.session_state.animation_frames.append(img_str)

                            # Update progress
                            progress_bar.progress((i + 1) / total_frames_to_animate)

                        # Update status message in the placeholder
                        status_placeholder.success(
                            f"Prepared {len(st.session_state.animation_frames)} frames for preview"
                        )

                # Add animation container and JavaScript
                if len(st.session_state.get("animation_frames", [])) > 0:
                    # Get the animation start frame from session state
                    animation_start_frame = st.session_state.get(
                        "animation_start_frame", current_frame
                    )

                    # Create the JavaScript animation component
                    animation_html = create_js_animation(
                        "animation-container",
                        st.session_state.animation_frames,
                        animation_start_frame,  # Use the stored start frame
                        animation_start_frame
                        + len(st.session_state.animation_frames)
                        - 1,  # Calculate the end frame
                        1.0,  # Fixed speed
                    )

                    if animation_html:
                        # Use a fixed height to match the video preview
                        components.html(animation_html, height=500, scrolling=False)

        # Display current frame and controls side by side
        st.subheader("Current Frame & Controls")
        frame_col, controls_col = st.columns([3, 2])

        with frame_col:
            # Display current frame as an image for precise frame viewing
            frame = video_service.get_frame(video_path, current_frame)

            if frame is not None:
                # Get crop region for current frame
                frame_crop = crop_region  # Use crop region directly

                # Apply crop overlay if specified
                if frame_crop:
                    frame = video_service.draw_crop_overlay(frame, frame_crop)

                # Display the frame
                st.image(frame, use_container_width=True)

                # Display frame information
                st.caption(
                    f"Frame: {current_frame} / {total_frames-1} | "
                    f"Time: {video_service.format_timecode(current_frame, fps)}"
                )

            else:
                st.error(f"Could not load frame {current_frame}")

        with controls_col:
            # Video controls
            st.subheader("Frame Controls")

            # Navigation action functions
            def go_to_first_frame():
                st.session_state.frame_to_navigate = 0

            def go_to_previous_frame():
                st.session_state.frame_to_navigate = max(0, current_frame - 1)

            def go_to_next_frame():
                st.session_state.frame_to_navigate = min(
                    total_frames - 1, current_frame + 1
                )

            def go_back_10_frames():
                st.session_state.frame_to_navigate = max(0, current_frame - 10)

            def go_forward_10_frames():
                st.session_state.frame_to_navigate = min(
                    total_frames - 1, current_frame + 10
                )

            def go_to_last_frame():
                st.session_state.frame_to_navigate = total_frames - 1

            # Frame navigation buttons
            if st.button("⏮️ First Frame", key="btn_first", on_click=go_to_first_frame):
                pass  # Action handled in on_click

            # Previous/Next frame buttons in a row
            prev_col, next_col = st.columns(2)
            with prev_col:
                if st.button(
                    "⏪ Previous", key="btn_prev", on_click=go_to_previous_frame
                ):
                    pass  # Action handled in on_click
            with next_col:
                if st.button("Next ⏩", key="btn_next", on_click=go_to_next_frame):
                    pass  # Action handled in on_click

            # Jump buttons
            jump_col1, jump_col2 = st.columns(2)
            with jump_col1:
                if st.button(
                    "-10 Frames", key="btn_back10", on_click=go_back_10_frames
                ):
                    pass  # Action handled in on_click
            with jump_col2:
                if st.button(
                    "+10 Frames", key="btn_forward10", on_click=go_forward_10_frames
                ):
                    pass  # Action handled in on_click

            # Last frame button
            if st.button("Last Frame ⏭️", key="btn_last", on_click=go_to_last_frame):
                pass  # Action handled in on_click

            # Frame slider
            def handle_slider_change():
                """Handle slider change event after release"""
                if on_frame_change:
                    on_frame_change(st.session_state.frame_slider)
                # Removed st.rerun() as it's a no-op in callbacks

            new_frame = st.slider(
                "Frame",
                0,
                max(0, total_frames - 1),
                current_frame,
                key="frame_slider",
                on_change=handle_slider_change,
            )
            if new_frame != current_frame:
                current_frame = new_frame
                # Remove the immediate frame change handling here since it's handled in on_change

            # Timecode display and input
            time_col1, time_col2 = st.columns([1, 1])
            with time_col1:
                # Show current timecode
                st.text(
                    f"Timecode: {video_service.format_timecode(current_frame, fps)}"
                )

            with time_col2:
                # Add timecode input
                def go_to_timecode():
                    try:
                        # Parse entered timecode
                        entered_timecode = st.session_state.timecode_input
                        # Convert timecode to frame number
                        target_frame = video_service.parse_timecode_to_frame(
                            entered_timecode, fps
                        )
                        # Make sure it's within valid range
                        target_frame = max(0, min(total_frames - 1, target_frame))
                        # Set frame to navigate
                        st.session_state.frame_to_navigate = target_frame
                    except Exception as e:
                        # If invalid timecode, just log error
                        logger.error(f"Invalid timecode: {e}")
                        st.session_state.error_message = f"Invalid timecode format. Use HH:MM:SS:FF, HH:MM:SS, or MM:SS"

                # Initialize timecode input in session state if not exists
                if "timecode_input" not in st.session_state:
                    st.session_state.timecode_input = video_service.format_timecode(
                        current_frame, fps
                    )

                # Timecode input field with button
                st.text_input(
                    "Go to Timecode",
                    value=video_service.format_timecode(current_frame, fps),
                    key="timecode_input",
                    help="Enter timecode in format HH:MM:SS:FF, HH:MM:SS, or MM:SS",
                    on_change=go_to_timecode,
                )

                # Display error message if exists
                if (
                    "error_message" in st.session_state
                    and st.session_state.error_message
                ):
                    st.error(st.session_state.error_message)
                    # Clear error message after displaying it
                    st.session_state.error_message = ""

        return current_frame

    except Exception as e:
        logger.exception(f"Error displaying video player: {str(e)}")
        st.error(f"Error displaying video player: {str(e)}")
        return current_frame


def display_clip_controls(
    clip=None,
    on_set_start=None,
    on_set_end=None,
):
    """
    Display clip editing controls

    Args:
        clip: Current clip object
        on_set_start: Callback for setting start frame
        on_set_end: Callback for setting end frame

    Returns:
        None
    """
    try:
        st.subheader("Clip Controls")

        # Create a flat layout with 4 columns for all controls
        set_start_col, start_info_col, set_end_col, end_info_col = st.columns(4)

        # Set start frame button and info
        with set_start_col:
            if on_set_start:
                st.button("Set Start Frame", on_click=on_set_start)

        with start_info_col:
            if clip:
                st.text(f"Start: {clip.start_frame}")
                if st.button("Go to", key="goto_start"):
                    # Update current frame to start frame
                    st.session_state.current_frame = clip.start_frame
                    st.rerun()

        # Set end frame button and info
        with set_end_col:
            if on_set_end:
                st.button("Set End Frame", on_click=on_set_end)

        with end_info_col:
            if clip:
                st.text(f"End: {clip.end_frame}")
                if st.button("Go to", key="goto_end"):
                    # Update current frame to end frame
                    st.session_state.current_frame = clip.end_frame
                    st.rerun()

        # Display clip duration if clip exists
        if clip:
            duration_frames = clip.get_duration_frames()
            if (
                duration_frames > 0
                and hasattr(st.session_state, "fps")
                and st.session_state.fps > 0
            ):
                duration_seconds = duration_frames / st.session_state.fps
                st.text(
                    f"Duration: {duration_frames} frames ({video_service.format_duration(duration_seconds)})"
                )
            else:
                st.text(f"Duration: {duration_frames} frames")

    except Exception as e:
        logger.exception(f"Error displaying clip controls: {str(e)}")
        st.error(f"Error displaying clip controls: {str(e)}")


def display_crop_controls(
    on_select_crop=None,
    on_clear_crop=None,
    current_crop=None,
    output_resolution="1080p",
):
    """
    Display crop controls (temporarily disabled)

    Args:
        on_select_crop: Callback for select crop button
        on_clear_crop: Callback for clear crop button
        current_crop: Current crop region (x, y, width, height)
        output_resolution: Output resolution

    Returns:
        None
    """
    st.subheader("Crop Controls")
    st.info("Crop functionality is temporarily disabled.")


def play_clip_preview(
    video_path, start_frame, end_frame, fps, crop_region=None, on_frame_change=None
):
    """
    Play a preview of a clip using HTML5 video player

    Args:
        video_path: Path to the video file
        start_frame: Start frame number
        end_frame: End frame number
        fps: Frames per second
        crop_region: Function to get crop region at a specific frame
        on_frame_change: Callback function when frame changes

    Returns:
        None
    """
    try:
        # If we're in preview mode (using preview_clip_path), don't show another video player
        if (
            "preview_clip_path" in st.session_state
            and st.session_state.preview_clip_path
        ):
            return

        # Validate inputs
        if start_frame > end_frame:
            st.error("Start frame must be before end frame")
            return

        # Check if we should use proxy video
        if "proxy_path" in st.session_state and st.session_state.proxy_path:
            # Use proxy video if available
            proxy_path = st.session_state.proxy_path
            logger.debug(f"Using proxy video for clip preview: {proxy_path}")
            video_path = proxy_path

        # Calculate start and end times in seconds
        start_time = start_frame / fps if fps > 0 else 0
        end_time = end_frame / fps if fps > 0 else 0
        duration = end_time - start_time

        # Calculate total frames in clip
        total_frames = end_frame - start_frame + 1

        # Initialize animation state if not exists
        if "preview_animation_speed" not in st.session_state:
            st.session_state.preview_animation_speed = 1.0

        # Initialize show_previews toggle if not exists
        if "show_previews" not in st.session_state:
            st.session_state.show_previews = True

        # Initialize preview frame navigation state
        if "preview_frame_to_navigate" not in st.session_state:
            st.session_state.preview_frame_to_navigate = None

        # Initialize session state for current preview frame if not exists
        if "preview_current_frame" not in st.session_state:
            st.session_state.preview_current_frame = start_frame

        # Check if we need to process a navigation update from previous run
        if st.session_state.preview_frame_to_navigate is not None:
            st.session_state.preview_current_frame = (
                st.session_state.preview_frame_to_navigate
            )
            # Call the frame change callback if provided
            if on_frame_change:
                on_frame_change(st.session_state.preview_current_frame)
            # Reset navigation state
            st.session_state.preview_frame_to_navigate = None

        # Add a toggle to show/hide previews (same as in display_video_player)
        st.session_state.show_previews = st.checkbox(
            "Show Video Previews",
            value=st.session_state.show_previews,
            help="Toggle to show or hide the video preview panels",
        )

        # Display video information
        st.text(
            f"Clip preview: {total_frames} frames ({video_service.format_duration(duration)})"
        )

        # Only display video previews if the toggle is on
        if st.session_state.show_previews:
            # Create compact layout for video preview
            col_video1, col_video2 = st.columns([1, 1])

            with col_video1:
                # Display the video with HTML components to allow seeking
                video_file = open(video_path, "rb")
                video_bytes = video_file.read()
                st.video(video_bytes, start_time=start_time)

            with col_video2:
                # Display in/out points with go-to buttons
                st.subheader("Clip Points")

                # Define navigation functions
                def go_to_start_frame():
                    st.session_state.preview_frame_to_navigate = start_frame

                def go_to_end_frame():
                    st.session_state.preview_frame_to_navigate = end_frame

                # Use a flat layout for in/out points
                st.text(f"In Point: {start_frame}")
                if st.button(
                    "Go to In Point", key="goto_in_point", on_click=go_to_start_frame
                ):
                    pass  # Action handled in on_click

                st.text(f"Out Point: {end_frame}")
                if st.button(
                    "Go to Out Point", key="goto_out_point", on_click=go_to_end_frame
                ):
                    pass  # Action handled in on_click

        # Display current frame and controls side by side
        frame_col, controls_col = st.columns([3, 2])

        with frame_col:
            # Display the current frame as an image for precise frame viewing
            current_frame = st.session_state.preview_current_frame
            frame = video_service.get_frame(video_path, current_frame)

            if frame is not None:
                # Get crop region for current frame
                frame_crop = None
                if crop_region and callable(crop_region):
                    frame_crop = crop_region(current_frame)

                # Apply crop overlay if specified
                if frame_crop:
                    frame = video_service.draw_crop_overlay(frame, frame_crop)

                # Display the frame
                st.image(frame, use_container_width=True)

                # Display frame information
                st.caption(
                    f"Frame: {current_frame} / {end_frame} | "
                    f"Time: {video_service.format_timecode(current_frame, fps)}"
                )

                # Animation controls - only show if previews are visible
                if st.session_state.show_previews:
                    # Animation controls in a row
                    anim_col1, anim_col2 = st.columns([1, 1])

                    with anim_col1:
                        # Create a list of frames for animation
                        if st.button(
                            "Prepare Animation Frames", key="preview_prepare_frames"
                        ):
                            st.session_state.preview_animation_frames = []
                            progress_bar = st.progress(0)

                            # Use all frames in the clip range
                            total_clip_frames = end_frame - start_frame + 1
                            frame_range = int(
                                total_clip_frames
                            )  # Remove the 100 frame limit

                            # Store the current frame to restore it later
                            original_frame = current_frame

                            # Store the clip start frame in session state for the animation
                            st.session_state.preview_animation_start_frame = start_frame

                            # For each frame in the clip range, get the frame with crop region already applied
                            for i in range(frame_range):
                                # Calculate the exact frame index within the clip range
                                frame_idx = start_frame + i

                                # Ensure we don't go beyond the end frame
                                if frame_idx <= end_frame:
                                    # Get the frame using the same method as the main display
                                    frame = video_service.get_frame(
                                        video_path, frame_idx
                                    )

                                    if frame is not None:
                                        # Get crop region for current frame
                                        frame_crop = None
                                        if crop_region and callable(crop_region):
                                            frame_crop = crop_region(frame_idx)

                                        # Apply crop overlay if specified
                                        if frame_crop:
                                            frame = video_service.draw_crop_overlay(
                                                frame, frame_crop
                                            )
                                            logger.debug(
                                                f"Applied crop overlay to clip frame {frame_idx}"
                                            )
                                        else:
                                            logger.debug(
                                                f"No crop region for clip frame {frame_idx}"
                                            )

                                        # Convert OpenCV image from BGR to RGB for correct colors
                                        frame_rgb = cv2.cvtColor(
                                            frame, cv2.COLOR_BGR2RGB
                                        )

                                        # Convert OpenCV image to bytes
                                        _, buffer = cv2.imencode(".jpg", frame_rgb)
                                        img_str = base64.b64encode(buffer).decode(
                                            "utf-8"
                                        )

                                        st.session_state.preview_animation_frames.append(
                                            img_str
                                        )

                                # Update progress
                                progress_bar.progress((i + 1) / frame_range)

                            st.success(
                                f"Prepared {len(st.session_state.preview_animation_frames)} frames for animation from frame {start_frame} to {start_frame + len(st.session_state.preview_animation_frames) - 1}"
                            )

                            # Restore the original frame
                            current_frame = original_frame

                    with anim_col2:
                        # Animation speed control
                        speed_options = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "5x": 5.0}
                        selected_speed = st.selectbox(
                            "Speed",
                            options=list(speed_options.keys()),
                            index=1,  # Default to 1x
                            key="preview_js_playback_speed_selector",
                        )
                        st.session_state.preview_animation_speed = speed_options[
                            selected_speed
                        ]

                    # Add animation container and JavaScript
                    if len(st.session_state.get("preview_animation_frames", [])) > 0:
                        st.subheader("Animation")

                        # Get the animation start frame from session state
                        preview_animation_start_frame = st.session_state.get(
                            "preview_animation_start_frame", start_frame
                        )

                        # Create the JavaScript animation component
                        animation_html = create_js_animation(
                            "preview-animation-container",
                            st.session_state.preview_animation_frames,
                            preview_animation_start_frame,  # Use the stored start frame
                            end_frame,  # Use the clip end frame
                            st.session_state.preview_animation_speed,
                        )

                        if animation_html:
                            # Use a fixed height to ensure the component is visible
                            components.html(animation_html, height=400, scrolling=True)

        with controls_col:
            # Add frame navigation controls
            st.subheader("Frame Controls")

            # Display current frame information
            st.text(f"Current Frame: {current_frame} / {end_frame}")
            st.text(f"Time: {video_service.format_timecode(current_frame, fps)}")

            # Define frame navigation functions
            def go_to_start_frame():
                st.session_state.preview_frame_to_navigate = start_frame

            def set_in_point():
                if on_frame_change:
                    on_frame_change(st.session_state.preview_current_frame)
                st.session_state.message = (
                    f"In point set at frame {st.session_state.preview_current_frame}"
                )

            def set_out_point():
                if on_frame_change:
                    on_frame_change(st.session_state.preview_current_frame)
                st.session_state.message = (
                    f"Out point set at frame {st.session_state.preview_current_frame}"
                )

            # Frame navigation - use a flat layout
            if st.button(
                "⏮️ Start Frame", key="preview_first", on_click=go_to_start_frame
            ):
                pass  # Action handled in on_click

            if st.button("Set In Point", key="set_in_point", on_click=set_in_point):
                pass  # Action handled in on_click

            # Display success message if one exists
            if "message" in st.session_state and st.session_state.message:
                st.success(st.session_state.message)
                # Clear message after displaying it
                st.session_state.message = ""

            if st.button("Set Out Point", key="set_out_point", on_click=set_out_point):
                pass  # Action handled in on_click

            # Frame slider for precise navigation
            def handle_preview_slider_change():
                """Handle preview slider change event after release"""
                if on_frame_change:
                    on_frame_change(st.session_state.preview_frame_slider)
                # Removed st.rerun() as it's a no-op in callbacks

            new_frame = st.slider(
                "Frame",
                start_frame,
                end_frame,
                st.session_state.preview_current_frame,
                key="preview_frame_slider",
                on_change=handle_preview_slider_change,
            )
            if new_frame != st.session_state.preview_current_frame:
                st.session_state.preview_current_frame = new_frame
                # Remove the immediate frame change handling here since it's handled in on_change

            # Timecode display and input
            preview_time_col1, preview_time_col2 = st.columns([1, 1])
            with preview_time_col1:
                # Show current timecode
                st.text(
                    f"Timecode: {video_service.format_timecode(current_frame, fps)}"
                )

            with preview_time_col2:
                # Add timecode input
                def go_to_preview_timecode():
                    try:
                        # Parse entered timecode
                        entered_timecode = st.session_state.preview_timecode_input
                        # Convert timecode to frame number
                        target_frame = video_service.parse_timecode_to_frame(
                            entered_timecode, fps
                        )
                        # Make sure it's within valid range
                        target_frame = max(start_frame, min(end_frame, target_frame))
                        # Set frame to navigate
                        st.session_state.preview_frame_to_navigate = target_frame
                    except Exception as e:
                        # If invalid timecode, just log error
                        logger.error(f"Invalid timecode: {e}")
                        st.session_state.preview_error_message = f"Invalid timecode format. Use HH:MM:SS:FF, HH:MM:SS, or MM:SS"

                # Initialize timecode input in session state if not exists
                if "preview_timecode_input" not in st.session_state:
                    st.session_state.preview_timecode_input = (
                        video_service.format_timecode(current_frame, fps)
                    )

                # Timecode input field with button
                st.text_input(
                    "Go to Timecode",
                    value=video_service.format_timecode(current_frame, fps),
                    key="preview_timecode_input",
                    help="Enter timecode in format HH:MM:SS:FF, HH:MM:SS, or MM:SS",
                    on_change=go_to_preview_timecode,
                )

                # Display error message if exists
                if (
                    "preview_error_message" in st.session_state
                    and st.session_state.preview_error_message
                ):
                    st.error(st.session_state.preview_error_message)
                    # Clear error message after displaying it
                    st.session_state.preview_error_message = ""

    except Exception as e:
        logger.exception(f"Error playing clip preview: {str(e)}")
        st.error(f"Error playing clip preview: {str(e)}")


def display_keyframe_list(
    keyframes, current_frame, on_select_keyframe=None, on_delete_keyframe=None
):
    """
    Display a list of keyframes with options to select or delete

    Args:
        keyframes: Dictionary of frame numbers to crop regions
        current_frame: Current frame number
        on_select_keyframe: Callback for selecting a keyframe
        on_delete_keyframe: Callback for deleting a keyframe

    Returns:
        None
    """
    try:
        if not keyframes:
            st.info("No keyframes defined")
            return

        st.subheader("Keyframes")

        # Convert string keys to integers and sort
        keyframe_frames = sorted([int(k) for k in keyframes.keys()])

        # Display each keyframe
        for frame in keyframe_frames:
            # Create a container for this keyframe
            with st.container():
                # Use columns for layout - add an extra column for edit button
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    # Highlight current frame
                    if frame == current_frame:
                        st.markdown(f"**Frame {frame}** (current)")
                    else:
                        st.text(f"Frame {frame}")

                with col2:
                    # Go to keyframe button with frame number
                    if on_select_keyframe:
                        if st.button(f"Go to Frame #{frame}", key=f"goto_{frame}"):
                            on_select_keyframe(frame)

                with col3:
                    # Edit keyframe button
                    if st.button(f"Edit", key=f"edit_{frame}"):
                        # Go to this frame
                        if on_select_keyframe:
                            on_select_keyframe(frame)
                        # Set crop selection mode
                        st.session_state.crop_selection_active = True
                        # Store the frame being edited
                        st.session_state.editing_keyframe = frame
                        st.rerun()

                with col4:
                    # Delete keyframe button
                    if on_delete_keyframe:
                        if st.button(f"Delete", key=f"delete_{frame}"):
                            on_delete_keyframe(frame)

    except Exception as e:
        logger.exception(f"Error displaying keyframe list: {str(e)}")
        st.error(f"Error displaying keyframe list: {str(e)}")


# Add a function to create a JavaScript animation component
def create_js_animation(container_id, frames, start_frame=0, end_frame=None, speed=1.0):
    """Create a JavaScript animation component"""
    # Load the JavaScript file
    js_path = os.path.join(STATIC_DIR, "js", "frame_animation.js")

    # Check if the file exists
    if not os.path.exists(js_path):
        st.error(f"JavaScript file not found: {js_path}")
        return None

    # Read the JavaScript file
    with open(js_path, "r") as f:
        js_code = f.read()

    # Create HTML with the JavaScript and initialization code
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ 
                margin: 0; 
                padding: 0; 
                background-color: transparent;
            }}
            #animation-wrapper {{ 
                width: 100%; 
                height: 500px;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: transparent;
            }}
            #{container_id} {{ 
                width: 100%;
                height: 100%;
                background-color: transparent;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }}
            #{container_id} img {{
                max-width: 100%;
                max-height: calc(100% - 40px);
                object-fit: contain;
                margin-bottom: 10px;
            }}
            .animation-controls {{
                width: 100%;
                padding: 10px;
                text-align: center;
                background-color: transparent;
            }}
        </style>
    </head>
    <body>
        <div id="animation-wrapper">
            <div id="{container_id}">
                <div class="animation-controls">
                    <p style="margin: 0; color: #666;">Animation ready - {len(frames)} frames</p>
                </div>
            </div>
        </div>
        
        <script>
        {js_code}
        
        // Initialize animation when the page loads
        window.addEventListener('DOMContentLoaded', function() {{
            console.log("DOM loaded, initializing animation for {container_id}");
            
            // Short delay to ensure DOM is ready
            setTimeout(function() {{
                try {{
                    // Initialize the animation
                    initializeAnimation(
                        '{container_id}', 
                        {frames}, 
                        {start_frame}, 
                        {end_frame if end_frame is not None else 'null'}
                    );
                }} catch (e) {{
                    console.error("Error initializing animation:", e);
                    
                    // Display error in the container
                    const container = document.getElementById('{container_id}');
                    if (container) {{
                        container.innerHTML = '<div style="color: red; padding: 20px;">Error initializing animation: ' + e.message + '</div>';
                    }}
                }}
            }}, 100);
        }});
        </script>
    </body>
    </html>
    """

    return html
