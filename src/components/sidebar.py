"""
Sidebar component for the Clipper application.
"""

import streamlit as st
import logging
import os
from pathlib import Path

from src.services import proxy_service, clip_service

logger = logging.getLogger("clipper.ui.sidebar")


def display_sidebar(config_manager):
    """
    Display the sidebar with video selection, clip management, and settings

    Args:
        config_manager: ConfigManager instance

    Returns:
        Selected video path or None
    """
    try:
        st.sidebar.title("Clipper")

        # Create tabs for different sidebar sections
        video_tab, clips_tab, settings_tab = st.sidebar.tabs(
            ["Videos", "Clips", "Settings"]
        )

        with video_tab:
            selected_video = display_video_selection(config_manager)

        with clips_tab:
            display_clip_management()

        with settings_tab:
            display_settings(config_manager)

        # Display proxy generation progress if active
        if (
            hasattr(st.session_state, "proxy_generation_active")
            and st.session_state.proxy_generation_active
        ):
            st.sidebar.markdown("---")
            proxy_service.display_proxy_generation_progress()

        # Display logs at the bottom of the sidebar
        if "display_logs" in st.session_state and st.session_state.display_logs:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Logs")
            if "log_messages" in st.session_state:
                for log in st.session_state.log_messages[-10:]:
                    st.sidebar.text(log)

        return selected_video

    except Exception as e:
        logger.exception(f"Error displaying sidebar: {str(e)}")
        st.sidebar.error(f"Error displaying sidebar: {str(e)}")
        return None


def display_video_selection(config_manager):
    """
    Display video selection section in the sidebar

    Args:
        config_manager: ConfigManager instance

    Returns:
        Selected video path or None
    """
    try:
        st.subheader("Video Selection")

        # Get list of videos
        video_files = config_manager.get_video_files()

        if not video_files:
            st.info("No video files found in the configured directories")

            # Show refresh button
            if st.button("Refresh Video List"):
                st.rerun()

            return None

        # Create a selectbox with video filenames (not full paths)
        video_options = [os.path.basename(v) for v in video_files]

        # Add a "None" option at the beginning
        video_options.insert(0, "Select a video...")

        # Get the selected index from session state or default to 0
        selected_index = 0
        if "selected_video_index" in st.session_state:
            selected_index = st.session_state.selected_video_index

        # Display the selectbox
        selected_option = st.selectbox(
            "Select Video", options=video_options, index=selected_index
        )

        # Update the selected index in session state
        st.session_state.selected_video_index = video_options.index(selected_option)

        # Return the full path of the selected video
        if selected_option != "Select a video...":
            selected_index = (
                video_options.index(selected_option) - 1
            )  # Adjust for the "None" option
            selected_video = video_files[selected_index]

            # Display video information
            display_video_info(selected_video, config_manager)

            return selected_video

        return None

    except Exception as e:
        logger.exception(f"Error displaying video selection: {str(e)}")
        st.error(f"Error displaying video selection: {str(e)}")
        return None


def display_video_info(video_path, config_manager):
    """
    Display information about the selected video

    Args:
        video_path: Path to the video file
        config_manager: ConfigManager instance

    Returns:
        None
    """
    try:
        # Check if video info is already in session state
        video_info_key = f"video_info_{video_path}"

        if video_info_key not in st.session_state:
            # Import here to avoid circular imports
            from src.services import video_service

            # Get video information
            video_info = video_service.get_video_info(video_path)

            # Store in session state
            st.session_state[video_info_key] = video_info
        else:
            video_info = st.session_state[video_info_key]

        if video_info:
            # Display video information
            st.text(f"Resolution: {video_info['width']}x{video_info['height']}")
            st.text(f"Duration: {video_info['duration_formatted']}")
            st.text(f"FPS: {video_info['fps']:.2f}")
            st.text(f"Frames: {video_info['total_frames']}")

            # Check if proxy exists
            proxy_exists = proxy_service.proxy_exists_for_video(
                video_path, config_manager
            )

            if proxy_exists:
                st.success("Proxy: Available")

                # Get proxy path
                proxy_path = config_manager.get_proxy_path(Path(video_path))

                # Store proxy path in session state
                st.session_state.proxy_path = str(proxy_path)
            else:
                st.warning("Proxy: Not available")

                # Create proxy button
                if st.button("Create Proxy"):
                    # Create a placeholder for progress
                    progress_placeholder = st.empty()

                    # Create proxy video
                    proxy_path = proxy_service.create_proxy_video(
                        video_path,
                        progress_placeholder=progress_placeholder,
                        config_manager=config_manager,
                    )

                    if proxy_path:
                        st.session_state.proxy_path = proxy_path
                        st.rerun()
        else:
            st.error("Could not get video information")

    except Exception as e:
        logger.exception(f"Error displaying video info: {str(e)}")
        st.error(f"Error displaying video info: {str(e)}")


def display_clip_management():
    """
    Display clip management section in the sidebar

    Returns:
        None
    """
    try:
        st.subheader("Clip Management")

        # Check if clips are initialized
        if "clips" not in st.session_state:
            st.info("No clips available")
            return

        # Get clips from session state
        clips = st.session_state.clips

        if not clips:
            st.info("No clips created yet")
            return

        # Display list of clips
        for i, clip in enumerate(clips):
            # Create a container for this clip
            with st.container():
                # Use expander to save space
                with st.expander(
                    f"{i+1}. {clip.name}",
                    expanded=(i == st.session_state.current_clip_index),
                ):
                    # Display clip information
                    st.text(
                        f"Source: {os.path.basename(clip.source_path) if clip.source_path else 'None'}"
                    )
                    st.text(f"Frames: {clip.start_frame} to {clip.end_frame}")
                    st.text(f"Duration: {clip.get_duration_frames()} frames")
                    st.text(f"Keyframes: {len(clip.crop_keyframes)}")

                    # Select button
                    if st.button("Select", key=f"select_clip_{i}"):
                        st.session_state.current_clip_index = i
                        st.rerun()

                    # Delete button
                    if st.button("Delete", key=f"delete_clip_{i}"):
                        clip_service.delete_clip(i)
                        st.rerun()

        # Save clips button
        if st.session_state.clip_modified:
            if st.button("Save Clips"):
                clip_service.save_session_clips()
                st.rerun()

    except Exception as e:
        logger.exception(f"Error displaying clip management: {str(e)}")
        st.error(f"Error displaying clip management: {str(e)}")


def display_settings(config_manager):
    """
    Display settings section in the sidebar

    Args:
        config_manager: ConfigManager instance

    Returns:
        None
    """
    try:
        st.subheader("Settings")

        # Output resolution setting
        output_resolutions = ["2160p", "1440p", "1080p", "720p", "480p", "360p"]

        # Get current output resolution from session state or default to 1080p
        current_resolution = "1080p"
        if "output_resolution" in st.session_state:
            current_resolution = st.session_state.output_resolution

        # Display the selectbox
        selected_resolution = st.selectbox(
            "Output Resolution",
            options=output_resolutions,
            index=output_resolutions.index(current_resolution),
        )

        # Update the output resolution in session state
        st.session_state.output_resolution = selected_resolution

        # Update current clip if one is selected
        if (
            "current_clip_index" in st.session_state
            and st.session_state.current_clip_index >= 0
        ):
            clip_service.update_current_clip(output_resolution=selected_resolution)

        # Display logs toggle
        display_logs = st.checkbox(
            "Display Logs", value=st.session_state.get("display_logs", False)
        )
        st.session_state.display_logs = display_logs

        # Generate all proxies button
        if st.button("Generate All Proxies"):
            proxy_service.generate_all_proxies(config_manager)

        # Clean up proxies button
        proxy_service.cleanup_proxy_files(config_manager)

    except Exception as e:
        logger.exception(f"Error displaying settings: {str(e)}")
        st.error(f"Error displaying settings: {str(e)}")
