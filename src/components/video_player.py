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
        # Initialize session state for navigation
        if "nav_action" not in st.session_state:
            st.session_state.nav_action = None

        # Initialize play state if not exists
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False
        if "play_speed" not in st.session_state:
            st.session_state.play_speed = 1.0  # Default playback speed multiplier
        if "last_play_time" not in st.session_state:
            st.session_state.last_play_time = None
        if "auto_advance" not in st.session_state:
            st.session_state.auto_advance = False

        # Process navigation actions from previous render
        if st.session_state.nav_action is not None:
            action = st.session_state.nav_action
            st.session_state.nav_action = None  # Reset the action

            if action == "first":
                current_frame = 0
            elif action == "prev":
                current_frame = max(0, current_frame - 1)
            elif action == "next":
                current_frame = min(total_frames - 1, current_frame + 1)
            elif action == "back10":
                current_frame = max(0, current_frame - 10)
            elif action == "forward10":
                current_frame = min(total_frames - 1, current_frame + 10)
            elif action == "last":
                current_frame = total_frames - 1
            elif action == "play_pause":
                st.session_state.is_playing = not st.session_state.is_playing
                if st.session_state.is_playing:
                    st.session_state.last_play_time = datetime.datetime.now()
                    st.session_state.auto_advance = True
                else:
                    st.session_state.auto_advance = False

            # Call the frame change callback
            if on_frame_change:
                on_frame_change(current_frame)

        # Handle playback if playing
        if st.session_state.is_playing and st.session_state.auto_advance:
            # Calculate how many frames to advance based on elapsed time
            now = datetime.datetime.now()
            if st.session_state.last_play_time:
                elapsed = (now - st.session_state.last_play_time).total_seconds()
                frames_to_advance = int(elapsed * fps * st.session_state.play_speed)

                if frames_to_advance > 0:
                    # Advance frames
                    current_frame = min(
                        total_frames - 1, current_frame + frames_to_advance
                    )
                    st.session_state.last_play_time = now

                    # Call the frame change callback
                    if on_frame_change:
                        on_frame_change(current_frame)

                    # Stop at the end of the video
                    if current_frame >= total_frames - 1:
                        st.session_state.is_playing = False
                        st.session_state.auto_advance = False
            else:
                st.session_state.last_play_time = now

        # Create columns for the player layout
        col1, col2 = st.columns([4, 1])

        with col1:
            # Display the current frame
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

        with col2:
            # Video controls
            st.subheader("Controls")

            # Play/Pause button
            play_col, speed_col = st.columns(2)
            with play_col:
                play_label = "⏸️ Pause" if st.session_state.is_playing else "▶️ Play"
                if st.button(play_label, key="btn_play_pause"):
                    st.session_state.nav_action = "play_pause"
                    st.rerun()

            with speed_col:
                # Playback speed selector
                speed_options = [0.25, 0.5, 1.0, 2.0, 4.0]
                speed_index = (
                    speed_options.index(st.session_state.play_speed)
                    if st.session_state.play_speed in speed_options
                    else 2
                )
                new_speed = st.selectbox(
                    "Speed", speed_options, index=speed_index, key="playback_speed"
                )
                if new_speed != st.session_state.play_speed:
                    st.session_state.play_speed = new_speed
                    if st.session_state.is_playing:
                        st.session_state.last_play_time = datetime.datetime.now()

            # Frame navigation
            if st.button("⏮️ First Frame", key="btn_first"):
                st.session_state.nav_action = "first"
                st.rerun()

            # Previous/Next frame buttons in a row
            prev_col, next_col = st.columns(2)
            with prev_col:
                if st.button("⏪ Previous", key="btn_prev"):
                    st.session_state.nav_action = "prev"
                    st.rerun()
            with next_col:
                if st.button("Next ⏩", key="btn_next"):
                    st.session_state.nav_action = "next"
                    st.rerun()

            # Jump buttons
            jump_col1, jump_col2 = st.columns(2)
            with jump_col1:
                if st.button("-10 Frames", key="btn_back10"):
                    st.session_state.nav_action = "back10"
                    st.rerun()
            with jump_col2:
                if st.button("+10 Frames", key="btn_forward10"):
                    st.session_state.nav_action = "forward10"
                    st.rerun()

            # Last frame button
            if st.button("Last Frame ⏭️", key="btn_last"):
                st.session_state.nav_action = "last"
                st.rerun()

            # Frame slider
            new_frame = st.slider("Frame", 0, max(0, total_frames - 1), current_frame)
            if new_frame != current_frame:
                current_frame = new_frame
                if on_frame_change:
                    on_frame_change(current_frame)

            # Timecode display
            st.text(f"Timecode: {video_service.format_timecode(current_frame, fps)}")

        # Add a small delay and rerun if playing to advance frames
        if st.session_state.is_playing:
            time.sleep(0.1)  # Small delay to prevent UI freezing
            st.rerun()

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

        # Create columns for the clip controls
        col1, col2 = st.columns(2)

        with col1:
            # Set start/end frame buttons
            if on_set_start:
                st.button("Set Start Frame", on_click=on_set_start)

            # Display current start frame if clip exists
            if clip:
                st.text(f"Start: {clip.start_frame}")

        with col2:
            # Set end frame button
            if on_set_end:
                st.button("Set End Frame", on_click=on_set_end)

            # Display current end frame if clip exists
            if clip:
                st.text(f"End: {clip.end_frame}")

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
    Display crop region controls

    Args:
        on_select_crop: Callback for selecting crop region
        on_clear_crop: Callback for clearing crop region
        current_crop: Current crop region (x, y, width, height)
        output_resolution: Output resolution for the crop

    Returns:
        None
    """
    try:
        st.subheader("Crop Controls")

        # Create columns for the crop controls
        col1, col2 = st.columns(2)

        with col1:
            # Select crop button
            if on_select_crop:
                st.button("Select Crop at Current Frame", on_click=on_select_crop)

        with col2:
            # Clear crop button
            if on_clear_crop:
                st.button("Clear Crop Keyframe", on_click=on_clear_crop)

        # Display current crop information
        if current_crop:
            x, y, width, height = current_crop
            st.text(f"Crop: X={x}, Y={y}, Width={width}, Height={height}")

            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            st.text(f"Aspect Ratio: {aspect_ratio:.3f}")

            # Show output dimensions
            out_width, out_height = video_service.calculate_crop_dimensions(
                output_resolution, aspect_ratio
            )
            st.text(f"Output: {out_width}x{out_height} ({output_resolution})")

    except Exception as e:
        logger.exception(f"Error displaying crop controls: {str(e)}")
        st.error(f"Error displaying crop controls: {str(e)}")


def play_clip_preview(
    video_path, start_frame, end_frame, fps, crop_region=None, on_frame_change=None
):
    """
    Play a preview of the clip

    Args:
        video_path: Path to the video file
        start_frame: Starting frame number
        end_frame: Ending frame number
        fps: Frames per second
        crop_region: Optional crop region to display
        on_frame_change: Callback function when frame changes

    Returns:
        None
    """
    try:
        # Create a placeholder for the preview
        preview_placeholder = st.empty()

        # Calculate total frames and duration
        total_frames = end_frame - start_frame + 1
        duration_seconds = total_frames / fps if fps > 0 else 0

        # Display preview information
        st.text(
            f"Playing clip preview: {total_frames} frames ({video_service.format_duration(duration_seconds)})"
        )

        # Create a progress bar
        progress_bar = st.progress(0)

        # Play the clip
        for i in range(total_frames):
            # Calculate current frame
            current_frame = start_frame + i

            # Get the frame
            frame = video_service.get_frame(video_path, current_frame)

            if frame is not None:
                # Apply crop if specified
                if crop_region:
                    # Get the crop region for this frame (could be interpolated)
                    if callable(crop_region):
                        frame_crop = crop_region(current_frame)
                    else:
                        frame_crop = crop_region

                    if frame_crop:
                        # Apply crop overlay
                        frame = video_service.draw_crop_overlay(frame, frame_crop)

                # Display the frame
                preview_placeholder.image(frame)

                # Update progress
                progress = i / max(1, total_frames - 1)
                progress_bar.progress(progress)

                # Call frame change callback if provided
                if on_frame_change:
                    on_frame_change(current_frame)

                # Sleep to maintain playback speed
                time.sleep(1 / fps)
            else:
                st.error(f"Could not load frame {current_frame}")
                break

        # Set progress to 100% when done
        progress_bar.progress(1.0)
        st.success("Clip preview complete")

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
