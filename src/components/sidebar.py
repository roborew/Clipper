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
    Display video selection section in the sidebar with a two-level selection:
    1. Select camera type
    2. Select specific video from that camera

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
            if st.button("Refresh Video List", key="refresh_video_list_btn"):
                st.rerun()

            return None

        # Check which videos have proxies
        videos_with_proxies = {}
        for video_path in video_files:
            has_proxy = proxy_service.proxy_exists_for_video(video_path, config_manager)
            videos_with_proxies[str(video_path)] = has_proxy

        # Group videos by camera type
        camera_groups = {}
        for video_path in video_files:
            # Extract camera type from path
            # Assuming path structure like: data/source/CAMERA_TYPE/SESSION/filename.mp4
            parts = Path(video_path).parts

            # Try to find a camera type in the path
            camera_type = "Other"
            for part in parts:
                # Look for common camera type patterns in the path
                if any(
                    cam in part.upper()
                    for cam in ["SONY", "GP", "GOPRO", "CANON", "NIKON", "CAM"]
                ):
                    camera_type = part
                    break

            # Add to camera groups
            if camera_type not in camera_groups:
                camera_groups[camera_type] = []
            camera_groups[camera_type].append(video_path)

        # Add "All" option
        camera_groups["All Videos"] = video_files

        # Sort camera types
        camera_types = sorted(list(camera_groups.keys()))
        # Move "All Videos" to the beginning
        if "All Videos" in camera_types:
            camera_types.remove("All Videos")
            camera_types.insert(0, "All Videos")

        # Initialize session state for camera selection if not exists
        if "selected_camera_type" not in st.session_state:
            st.session_state.selected_camera_type = camera_types[0]

        # Camera type selection
        selected_camera = st.selectbox(
            "Select Camera",
            options=camera_types,
            index=camera_types.index(st.session_state.selected_camera_type),
            key="camera_type_select",
        )

        # Update selected camera in session state
        st.session_state.selected_camera_type = selected_camera

        # Get videos for the selected camera
        filtered_videos = camera_groups[selected_camera]

        # Create a selectbox with video filenames and proxy indicators
        video_options = []
        for v in filtered_videos:
            basename = os.path.basename(v)
            if videos_with_proxies.get(str(v), False):
                # Add green tick for videos with proxies
                video_options.append(f"✅ {basename}")
            else:
                video_options.append(basename)

        # Add a "None" option at the beginning
        video_options.insert(0, "Select a video...")

        # Display a legend for the indicators
        st.caption("✅ = Proxy available")

        # Get the selected index from session state or default to 0
        # We need to handle the case where the camera type changes
        selected_index = 0
        video_select_key = f"video_select_{selected_camera}"

        if video_select_key in st.session_state:
            # Try to use the saved index for this camera type
            selected_index = st.session_state[video_select_key]
            # Make sure the index is valid for the current options
            if selected_index >= len(video_options):
                selected_index = 0

        # Display the video selectbox
        selected_option = st.selectbox(
            "Select Video",
            options=video_options,
            index=selected_index,
            key=f"video_select_{selected_camera}_box",
        )

        # Save the selected index for this camera type
        st.session_state[video_select_key] = video_options.index(selected_option)

        # Return the full path of the selected video
        if selected_option != "Select a video...":
            selected_index = (
                video_options.index(selected_option) - 1
            )  # Adjust for the "None" option
            selected_video = filtered_videos[selected_index]

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
                if st.button("Create Proxy", key="create_proxy_btn"):
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

        # Create New Clip button at the top of the clip management section
        if st.button("Create New Clip", key="create_new_clip_sidebar"):
            # Import here to avoid circular imports
            from src.app import handle_new_clip

            # Create a new clip using the current video
            handle_new_clip()
            st.rerun()

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
            if st.button("Save Clips", key="save_clips_btn_clips_section"):
                clip_service.save_session_clips()
                st.rerun()

    except Exception as e:
        logger.exception(f"Error displaying clip management: {str(e)}")
        st.error(f"Error displaying clip management: {str(e)}")


def display_settings(config_manager):
    """
    Display settings in the sidebar

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
            # Import clip_service to ensure it's available
            from src.services import clip_service

            clip_service.update_current_clip(output_resolution=selected_resolution)

        # Display logs toggle
        display_logs = st.checkbox(
            "Display Logs", value=st.session_state.get("display_logs", False)
        )
        st.session_state.display_logs = display_logs

        # Proxy settings
        st.subheader("Proxy Settings")
        proxy_settings = config_manager.get_proxy_settings()

        # Display current proxy directory
        st.info(f"Proxy directory: {config_manager.proxy_dir}")

        # Add info about directory structure
        if config_manager.config["export"]["preserve_structure"]:
            st.info(
                "Proxies will be stored in a directory structure that mirrors the source videos."
            )
        else:
            st.info("Proxies will be stored in a flat directory structure.")

        # Toggle for proxy creation
        proxy_enabled = st.checkbox(
            "Enable proxy videos", value=proxy_settings["enabled"]
        )

        # Only show these settings if proxy is enabled
        if proxy_enabled:
            proxy_width = st.number_input(
                "Proxy width",
                min_value=320,
                max_value=1920,
                value=proxy_settings["width"],
                help="Width of proxy videos (height will be calculated to maintain aspect ratio)",
            )

            proxy_quality = st.slider(
                "Proxy quality",
                min_value=18,
                max_value=35,
                value=proxy_settings["quality"],
                help="CRF value (18=high quality/larger file, 35=low quality/smaller file)",
            )

            # Update proxy settings in config if changed
            if (
                proxy_enabled != proxy_settings["enabled"]
                or proxy_width != proxy_settings["width"]
                or proxy_quality != proxy_settings["quality"]
            ):
                # Update config in memory
                config_manager.config["proxy"]["enabled"] = proxy_enabled
                config_manager.config["proxy"]["width"] = proxy_width
                config_manager.config["proxy"]["quality"] = proxy_quality

                # Save config to file
                import yaml

                with open(config_manager.config_path, "w") as f:
                    yaml.dump(
                        config_manager.config,
                        f,
                        default_flow_style=False,
                    )

                st.success("Proxy settings updated")

        # Generate all proxies button
        if st.button("Generate All Missing Proxies", key="generate_all_proxies_btn"):
            proxy_service.generate_all_proxies(config_manager)

        # Clean up proxies section
        st.subheader("Maintenance")
        proxy_service.cleanup_proxy_files(config_manager)

        # Configuration file management
        st.subheader("Configuration")

        # Get the current video path
        current_video = st.session_state.get("current_video", None)

        # Get the current clips file path
        clips_file = config_manager.get_clips_file_path(current_video)
        st.info(f"Clips file: {clips_file}")

        col1, col2 = st.columns(2)

        with col1:
            # Save configuration button
            if st.button("Save Clips", key="save_clips_btn"):
                from src.services import clip_service

                success = clip_service.save_session_clips(config_manager)
                if success:
                    st.success("Clips saved successfully")
                else:
                    st.error("Failed to save clips")

        with col2:
            # Load configuration button
            if st.button("Reload Clips", key="reload_clips_btn"):
                from src.services import clip_service

                success = clip_service.initialize_session_clips(config_manager)
                if success:
                    st.success("Clips reloaded successfully")
                    st.rerun()
                else:
                    st.error("Failed to reload clips")

    except Exception as e:
        logger.exception(f"Error displaying settings: {str(e)}")
        st.error(f"Error displaying settings: {str(e)}")
