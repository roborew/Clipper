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

        # Initialize active tab in session state if not present
        if "active_sidebar_tab" not in st.session_state:
            st.session_state.active_sidebar_tab = "Videos"

        # Create tabs for different sidebar sections
        tabs = ["Videos", "Clips", "Settings"]
        video_tab, clips_tab, settings_tab = st.sidebar.tabs(tabs)

        selected_video = None

        with video_tab:
            selected_video = display_video_selection(config_manager)
            if video_tab._active:
                st.session_state.active_sidebar_tab = "Videos"

        with clips_tab:
            if clips_tab._active:
                st.session_state.active_sidebar_tab = "Clips"
                # Check if we have a current video and config file
                current_video = st.session_state.get("current_video", None)
                if current_video:
                    clips_file = config_manager.get_clips_file_path(current_video)
                    # If config doesn't exist, create it
                    if not clips_file.exists():
                        try:
                            os.makedirs(os.path.dirname(clips_file), exist_ok=True)
                            with open(clips_file, "w") as f:
                                f.write("[]")
                            logger.info(f"Created new config file: {clips_file}")
                        except Exception as e:
                            logger.exception(f"Error creating config file: {str(e)}")
                            st.error("Failed to create config file")
            display_clip_management()

        with settings_tab:
            display_settings(config_manager)
            if settings_tab._active:
                st.session_state.active_sidebar_tab = "Settings"

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

        # Check status of configuration files
        config_statuses = {}
        for video_path in video_files:
            clips_file = config_manager.get_clips_file_path(video_path)

            # Default status is None (no config)
            status = None

            if clips_file.exists():
                try:
                    # Load the clips config
                    with open(clips_file, "r") as f:
                        import json

                        clips_data = json.load(f)

                    if not clips_data:
                        # Empty config file
                        status = "empty"
                    else:
                        # Check status of all clips
                        has_draft = False
                        has_process = False
                        all_complete = True

                        for clip_data in clips_data:
                            clip_status = clip_data.get(
                                "status", "Draft"
                            )  # Default to Draft if not specified

                            if clip_status == "Draft":
                                has_draft = True
                                all_complete = False
                            elif clip_status == "Process":
                                has_process = True
                                all_complete = False

                        if has_draft:
                            status = "draft"
                        elif has_process:
                            status = "process"
                        elif all_complete:
                            status = "complete"
                        else:
                            status = "empty"
                except Exception as e:
                    logger.exception(f"Error reading config file: {str(e)}")
                    status = "error"

            config_statuses[str(video_path)] = status

        # Group videos by camera type
        camera_groups = {}
        for video_path in video_files:
            # Extract camera type from path
            # Assuming path structure like: data/source/CAMERA_TYPE/SESSION/filename.mp4
            parts = Path(video_path).parts

            # Try to find a camera type in the path
            camera_type = "Other"
            session_folder = "Unknown"
            for i, part in enumerate(parts):
                # Look for common camera type patterns in the path
                if any(
                    cam in part.upper()
                    for cam in ["SONY", "GP", "GOPRO", "CANON", "NIKON", "CAM"]
                ):
                    camera_type = part
                    # Try to get the session folder (next folder after camera type)
                    if i + 1 < len(parts):
                        session_folder = parts[i + 1]
                    break

            # Add to camera groups
            if camera_type not in camera_groups:
                camera_groups[camera_type] = []
            camera_groups[camera_type].append((video_path, session_folder))

        # Add "All" option with all videos
        camera_groups["All Videos"] = [(v, get_session_folder(v)) for v in video_files]

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

        # Create a selectbox with video filenames, session folders, and proxy indicators
        video_options = []
        for v, session in filtered_videos:
            basename = os.path.basename(v)
            proxy_indicator = "✅ " if videos_with_proxies.get(str(v), False) else ""

            # Add status indicator
            status = config_statuses.get(str(v))
            if status == "draft":
                status_indicator = "🔴 "  # Red for draft
            elif status == "process":
                status_indicator = "🟠 "  # Amber/orange for process
            elif status == "complete":
                status_indicator = "🟢 "  # Green for complete
            else:
                status_indicator = "⚪ "  # Grey for no config or empty

            video_options.append(
                f"{proxy_indicator}{status_indicator}[{session}] {basename}"
            )

        # Add a "None" option at the beginning
        video_options.insert(0, "Select a video...")

        # Display a legend for the indicators
        st.caption("✅ = Proxy available")
        st.caption("⚪ = No config | 🔴 = Draft | 🟠 = Process | 🟢 = Complete")

        # Add a refresh button to check status
        if st.button("🔄 Refresh Status", key="refresh_status_btn"):
            st.rerun()

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
            selected_video = filtered_videos[selected_index][
                0
            ]  # Get just the path, not the session

            # Display video information
            display_video_info(selected_video, config_manager)

            # Add regenerate proxy button at the bottom of video tab
            st.markdown("---")  # Add separator
            if proxy_service.proxy_exists_for_video(selected_video, config_manager):
                if st.button("🔄 Regenerate Proxy", key="regenerate_proxy_bottom"):
                    proxy_path = config_manager.get_proxy_path(Path(selected_video))
                    try:
                        # Delete existing proxy
                        if os.path.exists(proxy_path):
                            os.remove(proxy_path)
                            st.info("Deleted existing proxy, generating new one...")

                            # Store the new proxy path in session state
                            st.session_state.proxy_path = str(proxy_path)

                            # Update proxy path in all clips that use this video
                            if "clips" in st.session_state:
                                for clip in st.session_state.clips:
                                    if clip.source_path == selected_video:
                                        clip.proxy_path = str(proxy_path)
                                        clip.update()  # Mark as modified
                                st.session_state.clip_modified = True

                            st.rerun()
                    except Exception as e:
                        logger.exception(f"Error deleting proxy: {str(e)}")
                        st.error("Failed to delete existing proxy")

            # Load configuration for the selected video
            clips_file = config_manager.get_clips_file_path(selected_video)

            # Always try to initialize clips when a video is selected
            success = clip_service.initialize_session_clips(config_manager)
            if success:
                if clips_file.exists():
                    st.success(f"Loaded existing configuration from {clips_file}")
                else:
                    st.info("Creating new configuration for this video")
            else:
                st.error("Failed to initialize configuration")

            return selected_video

        return None

    except Exception as e:
        logger.exception(f"Error displaying video selection: {str(e)}")
        st.error(f"Error displaying video selection: {str(e)}")
        return None


def get_session_folder(video_path):
    """Helper function to extract session folder from video path"""
    parts = Path(video_path).parts
    for i, part in enumerate(parts):
        if any(
            cam in part.upper()
            for cam in ["SONY", "GP", "GOPRO", "CANON", "NIKON", "CAM"]
        ):
            if i + 1 < len(parts):
                return parts[i + 1]
    return "Unknown"


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
        # Import here to avoid circular imports
        from src.services import video_service

        # First verify the original video file exists and is accessible
        if not os.path.exists(video_path):
            st.error(f"Video file not found: {video_path}")
            logger.error(f"Video file not found: {video_path}")
            return

        if not os.access(video_path, os.R_OK):
            st.error(f"Cannot read video file (check permissions): {video_path}")
            logger.error(f"Cannot read video file (check permissions): {video_path}")
            return

        # Display file path for debugging
        st.caption(f"Loading video: {video_path}")

        # Clear any cached video info
        video_info_keys = [
            key
            for key in st.session_state.keys()
            if key.startswith(("video_info_", "original_video_info_"))
        ]
        for key in video_info_keys:
            del st.session_state[key]

        # Temporarily remove proxy path from session state to get original video info
        proxy_path_backup = st.session_state.pop("proxy_path", None)

        # Get original video info
        original_video_info = video_service.get_video_info(video_path)

        # Restore proxy path if it existed
        if proxy_path_backup:
            st.session_state.proxy_path = proxy_path_backup

        # Verify frame count makes sense with duration
        if original_video_info:
            expected_frames = int(
                original_video_info["duration"] * original_video_info["fps"]
            )
            if (
                abs(expected_frames - original_video_info["total_frames"])
                > original_video_info["fps"]
            ):
                logger.warning(
                    f"Frame count mismatch: Got {original_video_info['total_frames']}, expected ~{expected_frames}"
                )
                # Use the calculated frame count instead
                original_video_info["total_frames"] = expected_frames

        # Check if proxy exists and get proxy info
        proxy_exists = proxy_service.proxy_exists_for_video(video_path, config_manager)
        proxy_path = None
        proxy_video_info = None

        if proxy_exists:
            proxy_path = config_manager.get_proxy_path(Path(video_path))
            st.caption(f"Checking proxy: {proxy_path}")

            # Verify proxy file exists and is readable
            if not os.path.exists(proxy_path):
                st.warning(f"Proxy file marked as existing but not found: {proxy_path}")
                logger.warning(f"Proxy file not found: {proxy_path}")
                proxy_exists = False
            elif not os.access(proxy_path, os.R_OK):
                st.warning(f"Cannot read proxy file (check permissions): {proxy_path}")
                logger.warning(f"Cannot read proxy file: {proxy_path}")
                proxy_exists = False
            else:
                # Get proxy video info
                proxy_video_info = video_service.get_video_info(proxy_path)
                if not proxy_video_info:
                    st.warning(
                        "Could not read proxy video information - will regenerate"
                    )
                    logger.warning(f"Could not read proxy video info: {proxy_path}")
                    proxy_exists = False

        if original_video_info:
            # Display original video information
            st.text(
                f"Original Resolution: {original_video_info['width']}x{original_video_info['height']}"
            )
            st.text(f"Duration: {original_video_info['duration_formatted']}")
            st.text(f"FPS: {original_video_info['fps']:.2f}")
            st.text(
                f"Frames: {original_video_info['total_frames']} ({expected_frames} expected)"
            )

            if proxy_exists and proxy_video_info:
                st.success("✅ Proxy: Available")
                st.text(
                    f"Proxy Resolution: {proxy_video_info['width']}x{proxy_video_info['height']}"
                )
                st.text(f"Proxy Frames: {proxy_video_info['total_frames']}")
                # Store proxy path in session state
                st.session_state.proxy_path = str(proxy_path)

                # Show warning if frame counts don't match
                if (
                    abs(
                        proxy_video_info["total_frames"]
                        - original_video_info["total_frames"]
                    )
                    > 10
                ):
                    st.warning(
                        "⚠️ Proxy may be incomplete - use regenerate button below to create a new one"
                    )
            else:
                st.warning(
                    "Proxy: Not available - Generating proxy for smoother performance..."
                )

                # Create a placeholder for progress
                progress_placeholder = st.empty()

                # Automatically create proxy video
                proxy_path = proxy_service.create_proxy_video(
                    video_path,
                    progress_placeholder=progress_placeholder,
                    config_manager=config_manager,
                )

                if proxy_path:
                    st.session_state.proxy_path = proxy_path
                    # Get and display the new proxy's resolution
                    new_proxy_info = video_service.get_video_info(proxy_path)
                    if new_proxy_info:
                        st.success("✅ Proxy video created successfully!")
                        st.text(
                            f"Proxy Resolution: {new_proxy_info['width']}x{new_proxy_info['height']}"
                        )
                        st.text(f"Proxy Frames: {new_proxy_info['total_frames']}")
                else:
                    st.error(
                        "Failed to create proxy video. Using original video (may be slower)."
                    )
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

        # Get current video and config manager
        current_video = st.session_state.get("current_video", None)
        config_manager = st.session_state.get("config_manager", None)

        if not current_video or not config_manager:
            st.info("Select a video to manage clips")
            return

        # Get clips file path
        clips_file = config_manager.get_clips_file_path(current_video)

        # Show config file status
        st.caption(f"Config: {os.path.basename(clips_file) if clips_file else 'None'}")

        # Create columns for buttons
        col1 = st.columns([1])[0]

        with col1:
            # Save button - only enabled if there are unsaved changes
            save_button = st.button(
                "💾 Save",
                key="save_config_btn",
                help="Save current configuration",
                disabled=not (
                    current_video and st.session_state.get("clip_modified", False)
                ),
            )
            if save_button and current_video:
                success = clip_service.save_session_clips(config_manager)
                if success:
                    st.success("Configuration saved successfully")
                    st.rerun()  # Reload to show updated file contents
                else:
                    st.error("Failed to save configuration")

            # Reload button
            if st.button(
                "🔄 Reload",
                key="reload_config_btn",
                help="Reload configuration from file",
            ):
                if st.session_state.get("clip_modified", False):
                    st.warning(
                        "You have unsaved changes. Save first or they will be lost!"
                    )
                    if st.button("Reload anyway", key="reload_anyway"):
                        success = clip_service.initialize_session_clips(
                            config_manager, force_reload=True
                        )
                        if success:
                            st.success("Configuration reloaded")
                            st.rerun()
                else:
                    success = clip_service.initialize_session_clips(
                        config_manager, force_reload=True
                    )
                    if success:
                        st.success("Configuration reloaded")
                        st.rerun()

        # Create New Clip button
        if st.button("Create New Clip", key="create_new_clip_sidebar"):
            # Import here to avoid circular imports
            from src.app import handle_new_clip

            handle_new_clip()
            st.rerun()

        # Load clips directly from file for display
        clips = clip_service.load_clips_from_file(clips_file)

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
                    expanded=(i == st.session_state.get("current_clip_index", -1)),
                ):
                    # Display clip information
                    st.text(
                        f"Source: {os.path.basename(clip.source_path) if clip.source_path else 'None'}"
                    )
                    st.text(f"Frames: {clip.start_frame} to {clip.end_frame}")
                    st.text(f"Duration: {clip.get_duration_frames()} frames")
                    st.text(f"Keyframes: {len(clip.crop_keyframes)}")

                    # Create a single column for vertical button layout
                    col1 = st.columns([1])[0]

                    with col1:
                        # Select button
                        if st.button("Select", key=f"select_clip_{i}"):
                            # Set current clip index
                            st.session_state.current_clip_index = i

                            # Load clip data into state
                            success = clip_service.load_clip_into_state(clip)
                            if success:
                                st.success(f"Selected clip: {clip.name}")
                            else:
                                st.error("Failed to load clip data")

                            st.rerun()

                        # Delete button
                        if st.button(
                            "Delete",
                            key=f"delete_clip_{i}",
                            type="primary",
                            use_container_width=True,
                        ):
                            # Delete the clip from session state
                            clip_service.delete_clip(i)
                            # Auto-save after deletion
                            success = clip_service.save_session_clips()
                            if success:
                                st.success(f"Deleted clip: {clip.name}")
                            else:
                                st.error(
                                    f"Failed to save after deleting clip: {clip.name}"
                                )
                            st.rerun()

                        # Preview button
                        if st.button("Preview", key=f"preview_clip_{i}"):
                            st.session_state.current_clip_index = i  # Set current clip
                            # Create progress placeholder
                            progress_placeholder = st.empty()
                            progress_placeholder.info("Generating clip preview...")

                            # Get crop region for current frame
                            crop_region = clip.get_crop_region_at_frame(
                                clip.start_frame,
                                use_proxy=True,
                            )

                            # Generate preview
                            from src.services import proxy_service

                            preview_path = proxy_service.create_clip_preview(
                                clip.source_path,
                                clip.name,
                                clip.start_frame,
                                clip.end_frame,
                                crop_region=crop_region,
                                crop_keyframes=clip.crop_keyframes,
                                crop_keyframes_proxy=clip.crop_keyframes_proxy,
                                progress_placeholder=progress_placeholder,
                                config_manager=config_manager,
                            )

                            if preview_path:
                                st.session_state.preview_clip_path = preview_path
                                progress_placeholder.success(
                                    "Preview ready! Switching to main view..."
                                )
                                st.rerun()
                            else:
                                progress_placeholder.error("Failed to generate preview")

                        # Status selector
                        st.markdown("---")
                        status_options = ["Draft", "Process", "Complete"]
                        new_status = st.selectbox(
                            "Status",
                            options=status_options,
                            index=(
                                status_options.index(clip.status)
                                if clip.status in status_options
                                else 0
                            ),
                            key=f"status_selector_{i}",
                        )

                        # Update status if changed
                        if new_status != clip.status:
                            # Update in session state
                            if "clips" in st.session_state and i < len(
                                st.session_state.clips
                            ):
                                st.session_state.clips[i].status = new_status
                                st.session_state.clip_modified = True
                                st.info(
                                    f"Status changed to '{new_status}'. Click Save to apply changes."
                                )

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
        st.info(f"Proxy directory: {config_manager.proxy_base}")

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

    except Exception as e:
        logger.exception(f"Error displaying settings: {str(e)}")
        st.error(f"Error displaying settings: {str(e)}")
