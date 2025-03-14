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

        # Store the current frame in session state for the JavaScript animation
        if "js_current_frame" not in st.session_state:
            st.session_state.js_current_frame = current_frame

        # Initialize original_frame to avoid scope issues
        original_frame = current_frame

        # Store the start frame in session state for the animation
        st.session_state.animation_start_frame = original_frame

        # Display the video with HTML components (half size)
        col_video1, col_video2 = st.columns([1, 1])

        with col_video1:
            st.subheader("Video Preview")
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
                st.subheader("Preview")
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
                            # Get the interpolated crop region for this specific frame
                            frame_crop = None
                            # First try to get crop region from clip keyframes if available
                            if (
                                clip
                                and hasattr(clip, "crop_keyframes")
                                and clip.crop_keyframes
                            ):
                                frame_crop = clip.get_crop_region_at_frame(frame_idx)
                                logger.debug(
                                    f"Frame {frame_idx} interpolated crop region from clip: {frame_crop}"
                                )
                            # Fall back to provided crop_region if no clip keyframes
                            elif crop_region and callable(crop_region):
                                frame_crop = crop_region(frame_idx)
                                logger.debug(
                                    f"Frame {frame_idx} function crop region: {frame_crop}"
                                )
                            elif crop_region:
                                frame_crop = crop_region
                                logger.debug(
                                    f"Frame {frame_idx} static crop region: {frame_crop}"
                                )

                            # Apply the crop overlay to the frame
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
                # Apply crop overlay if specified
                if crop_region:
                    frame = video_service.draw_crop_overlay(frame, crop_region)

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

            # Frame navigation
            if st.button("⏮️ First Frame", key="btn_first"):
                current_frame = 0
                if on_frame_change:
                    on_frame_change(current_frame)
                st.rerun()

            # Previous/Next frame buttons in a row
            prev_col, next_col = st.columns(2)
            with prev_col:
                if st.button("⏪ Previous", key="btn_prev"):
                    current_frame = max(0, current_frame - 1)
                    if on_frame_change:
                        on_frame_change(current_frame)
                    st.rerun()
            with next_col:
                if st.button("Next ⏩", key="btn_next"):
                    current_frame = min(total_frames - 1, current_frame + 1)
                    if on_frame_change:
                        on_frame_change(current_frame)
                    st.rerun()

            # Jump buttons
            jump_col1, jump_col2 = st.columns(2)
            with jump_col1:
                if st.button("-10 Frames", key="btn_back10"):
                    current_frame = max(0, current_frame - 10)
                    if on_frame_change:
                        on_frame_change(current_frame)
                    st.rerun()
            with jump_col2:
                if st.button("+10 Frames", key="btn_forward10"):
                    current_frame = min(total_frames - 1, current_frame + 10)
                    if on_frame_change:
                        on_frame_change(current_frame)
                    st.rerun()

            # Last frame button
            if st.button("Last Frame ⏭️", key="btn_last"):
                current_frame = total_frames - 1
                if on_frame_change:
                    on_frame_change(current_frame)
                st.rerun()

            # Frame slider
            new_frame = st.slider("Frame", 0, max(0, total_frames - 1), current_frame)
            if new_frame != current_frame:
                current_frame = new_frame
                if on_frame_change:
                    on_frame_change(current_frame)
                st.rerun()

            # Timecode display
            st.text(f"Timecode: {video_service.format_timecode(current_frame, fps)}")

        return current_frame

    except Exception as e:
        logger.exception(f"Error displaying video player: {str(e)}")
        st.error(f"Error displaying video player: {str(e)}")
        return current_frame


def display_clip_controls(
    clip=None,
    on_set_start=None,
    on_set_end=None,
    on_play_clip=None,
    on_export_clip=None,
):
    """
    Display clip editing controls

    Args:
        clip: Current clip object
        on_set_start: Callback for setting start frame
        on_set_end: Callback for setting end frame
        on_play_clip: Callback for playing the clip
        on_export_clip: Callback for exporting the clip

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

        # Play and export buttons
        play_col, export_col = st.columns(2)

        with play_col:
            if on_play_clip and clip:
                st.button("▶️ Play Clip", on_click=on_play_clip)

        with export_col:
            if on_export_clip and clip:
                st.button("Export Clip", on_click=on_export_clip)

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

        # Display video information
        st.text(
            f"Clip preview: {total_frames} frames ({video_service.format_duration(duration)})"
        )

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

            # Use a flat layout for in/out points
            st.text(f"In Point: {start_frame}")
            if st.button("Go to In Point", key="goto_in_point"):
                st.session_state.preview_current_frame = start_frame
                if on_frame_change:
                    on_frame_change(start_frame)
                st.rerun()

            st.text(f"Out Point: {end_frame}")
            if st.button("Go to Out Point", key="goto_out_point"):
                st.session_state.preview_current_frame = end_frame
                if on_frame_change:
                    on_frame_change(end_frame)
                st.rerun()

        # Initialize session state for current preview frame if not exists
        if "preview_current_frame" not in st.session_state:
            st.session_state.preview_current_frame = start_frame

        # Display current frame and controls side by side
        frame_col, controls_col = st.columns([3, 2])

        with frame_col:
            # Display the current frame as an image for precise frame viewing
            current_frame = st.session_state.preview_current_frame
            frame = video_service.get_frame(video_path, current_frame)

            if frame is not None:
                # Apply crop overlay if specified
                if crop_region and callable(crop_region):
                    frame_crop = crop_region(current_frame)
                    if frame_crop:
                        frame = video_service.draw_crop_overlay(frame, frame_crop)

                # Display the frame
                st.image(frame, use_container_width=True)

                # Display frame information
                st.caption(
                    f"Frame: {current_frame} / {end_frame} | "
                    f"Time: {video_service.format_timecode(current_frame, fps)}"
                )

                # Animation controls in a row
                anim_col1, anim_col2 = st.columns([1, 1])

                with anim_col1:
                    # Create a list of frames for animation
                    if st.button(
                        "Prepare Animation Frames", key="preview_prepare_frames"
                    ):
                        # Remove redundant explanation
                        # st.info(
                        #     f"""
                        # **Clip Animation System**:
                        # 1. Capturing ONLY frames within the clip range ({start_frame} to {end_frame})
                        # 2. Using the exact same frames you see in the main display
                        # 3. Including all crop regions exactly as they appear
                        # 4. No regeneration of images - using the existing rendered frames
                        # """
                        # )

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
                                frame = video_service.get_frame(video_path, frame_idx)

                                if frame is not None:
                                    # Get the interpolated crop region for this specific frame
                                    frame_crop = None
                                    # First try to get crop region from clip keyframes if available
                                    if (
                                        clip
                                        and hasattr(clip, "crop_keyframes")
                                        and clip.crop_keyframes
                                    ):
                                        frame_crop = clip.get_crop_region_at_frame(
                                            frame_idx
                                        )
                                        logger.debug(
                                            f"Frame {frame_idx} interpolated crop region from clip: {frame_crop}"
                                        )
                                    # Fall back to provided crop_region if no clip keyframes
                                    elif crop_region and callable(crop_region):
                                        frame_crop = crop_region(frame_idx)
                                        logger.debug(
                                            f"Frame {frame_idx} function crop region: {frame_crop}"
                                        )
                                    elif crop_region:
                                        frame_crop = crop_region
                                        logger.debug(
                                            f"Frame {frame_idx} static crop region: {frame_crop}"
                                        )

                                    # Apply the crop overlay to the frame
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

                    # Add a debug button to check container existence
                    if st.button(
                        "Debug: Check Containers", key="preview_debug_containers"
                    ):
                        st.info("Checking for animation containers in the DOM...")
                        debug_html = """
                        <script>
                        setTimeout(function() {
                            const result = window.checkAnimationContainers();
                            const resultElement = document.createElement('div');
                            resultElement.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                            document.body.appendChild(resultElement);
                            
                            // Also log to console
                            console.log('Container check results:', result);
                        }, 500);
                        </script>
                        """
                        components.html(debug_html, height=100)

        with controls_col:
            # Add frame navigation controls
            st.subheader("Frame Controls")

            # Display current frame information
            st.text(f"Current Frame: {current_frame} / {end_frame}")
            st.text(f"Time: {video_service.format_timecode(current_frame, fps)}")

            # Frame navigation - use a flat layout
            if st.button("⏮️ Start Frame", key="preview_first"):
                st.session_state.preview_current_frame = start_frame
                if on_frame_change:
                    on_frame_change(start_frame)
                st.rerun()

            if st.button("Set In Point", key="set_in_point"):
                # Call the frame change callback to set in point
                if on_frame_change:
                    on_frame_change(st.session_state.preview_current_frame)
                st.success(
                    f"In point set at frame {st.session_state.preview_current_frame}"
                )

            if st.button("Set Out Point", key="set_out_point"):
                # Call the frame change callback to set out point
                if on_frame_change:
                    on_frame_change(st.session_state.preview_current_frame)
                st.success(
                    f"Out point set at frame {st.session_state.preview_current_frame}"
                )

            # Frame slider for precise navigation
            new_frame = st.slider(
                "Frame",
                start_frame,
                end_frame,
                st.session_state.preview_current_frame,
                key="preview_frame_slider",
            )
            if new_frame != st.session_state.preview_current_frame:
                st.session_state.preview_current_frame = new_frame
                if on_frame_change:
                    on_frame_change(new_frame)
                st.rerun()

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
