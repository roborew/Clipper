"""
Video player component for the Clipper application.
"""

import streamlit as st
import logging
from pathlib import Path
import os
import time
import datetime

from src.services import video_service

logger = logging.getLogger("clipper.ui.player")


def display_video_player(
    video_path,
    current_frame,
    fps,
    total_frames,
    on_frame_change=None,
    crop_region=None,
    config_manager=None,
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

        # Display the video with HTML components (half size)
        st.subheader("Video Preview")
        col_video1, col_video2 = st.columns([1, 1])

        with col_video1:
            # Calculate current time in seconds
            current_time = current_frame / fps if fps > 0 else 0

            # Display the video starting at the current frame
            video_file = open(video_path, "rb")
            video_bytes = video_file.read()
            st.video(video_bytes, start_time=current_time)

        with col_video2:
            # Empty space to balance layout
            st.write("")

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
            frame = video_service.get_frame(
                video_path, st.session_state.preview_current_frame
            )
            if frame is not None:
                # Apply crop overlay if specified
                if crop_region and callable(crop_region):
                    frame_crop = crop_region(st.session_state.preview_current_frame)
                    if frame_crop:
                        frame = video_service.draw_crop_overlay(frame, frame_crop)

                # Display the frame
                st.image(frame, use_container_width=True)

                # Display frame information
                st.caption(
                    f"Frame: {st.session_state.preview_current_frame} / {end_frame} | "
                    f"Time: {video_service.format_timecode(st.session_state.preview_current_frame, fps)}"
                )
            else:
                st.error(
                    f"Could not load frame {st.session_state.preview_current_frame}"
                )

        with controls_col:
            # Add frame navigation controls
            st.subheader("Frame Controls")

            # Display current frame information
            current_frame = st.session_state.preview_current_frame
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
                # Use columns for layout
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    # Highlight current frame
                    if frame == current_frame:
                        st.markdown(f"**Frame {frame}** (current)")
                    else:
                        st.text(f"Frame {frame}")

                with col2:
                    # Go to keyframe button
                    if on_select_keyframe:
                        if st.button(f"Go to", key=f"goto_{frame}"):
                            on_select_keyframe(frame)

                with col3:
                    # Delete keyframe button
                    if on_delete_keyframe:
                        if st.button(f"Delete", key=f"delete_{frame}"):
                            on_delete_keyframe(frame)

    except Exception as e:
        logger.exception(f"Error displaying keyframe list: {str(e)}")
        st.error(f"Error displaying keyframe list: {str(e)}")
