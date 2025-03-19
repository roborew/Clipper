"""
Main application file for the Clipper video editing tool.
"""

import streamlit as st
import logging
import os
from pathlib import Path
import platform
import subprocess

# Import services
from src.services import (
    video_service,
    proxy_service,
    clip_service,
    config_manager,
    video_processor,
)
from src.utils import logging_utils

# Import UI components
from src.components import (
    sidebar,
    video_player,
    crop_selector,
    simple_crop_selector,
)

# Set up logging
logger = logging_utils.setup_logger()


def main():
    """Main application entry point"""
    # Set page config
    title = "Clipper"

    # Get current clip name and modified status
    current_clip = clip_service.get_current_clip()
    if current_clip:
        # Get clip index and total clips
        clip_index = st.session_state.get("current_clip_index", -1)
        total_clips = len(st.session_state.get("clips", []))
        title = f"Clipper - {current_clip.name} ({clip_index + 1}/{total_clips})"
        if st.session_state.get("clip_modified", False):
            title += " *"

    st.set_page_config(
        page_title=title,
        page_icon="✂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Add Streamlit handler for logging
    logging_utils.add_streamlit_handler(logger)

    # Check if we need to rerun
    if st.session_state.get("trigger_rerun", False):
        # Reset the flag
        st.session_state.trigger_rerun = False
        # Rerun the app
        st.rerun()

    # Handle direct frame settings from timecode (new approach)
    if st.session_state.get("set_direct_start", False):
        # Reset the flag
        st.session_state.set_direct_start = False
        # Get the target frame
        target_frame = st.session_state.get("direct_start_frame", 0)
        # Get current clip
        current_clip = clip_service.get_current_clip()
        if current_clip:
            # Set the start frame directly
            current_clip.start_frame = target_frame
            clip_service.update_current_clip()
            # Mark as modified
            st.session_state.clip_modified = True
            # Auto-save the changes
            success = clip_service.save_session_clips()
            if success:
                logger.info(f"Auto-saved after setting start frame to {target_frame}")
                # Clear modified flag after successful save
                st.session_state.clip_modified = False
            else:
                logger.error("Failed to auto-save after setting start frame")
            # Also update the current frame to match
            st.session_state.current_frame = target_frame
            if "clip_frame_slider" in st.session_state:
                st.session_state.clip_frame_slider = target_frame
            # Preserve the timecode input after the rerun
            if "fps" in st.session_state and st.session_state.fps > 0:
                formatted_tc = video_service.format_timecode(
                    target_frame, st.session_state.fps
                )
                st.session_state.video_timecode_input = formatted_tc
            logger.info(f"Directly set start frame to {target_frame}")
            st.success(f"Start frame set to {target_frame}")

    if st.session_state.get("set_direct_end", False):
        # Reset the flag
        st.session_state.set_direct_end = False
        # Get the target frame
        target_frame = st.session_state.get("direct_end_frame", 0)
        # Get current clip
        current_clip = clip_service.get_current_clip()
        if current_clip:
            # Set the end frame directly
            current_clip.end_frame = target_frame
            clip_service.update_current_clip()
            # Mark as modified
            st.session_state.clip_modified = True
            # Auto-save the changes
            success = clip_service.save_session_clips()
            if success:
                logger.info(f"Auto-saved after setting end frame to {target_frame}")
                # Clear modified flag after successful save
                st.session_state.clip_modified = False
            else:
                logger.error("Failed to auto-save after setting end frame")
            # Also update the current frame to match
            st.session_state.current_frame = target_frame
            if "clip_frame_slider" in st.session_state:
                st.session_state.clip_frame_slider = target_frame
            # Preserve the timecode input after the rerun
            if "fps" in st.session_state and st.session_state.fps > 0:
                formatted_tc = video_service.format_timecode(
                    target_frame, st.session_state.fps
                )
                st.session_state.video_timecode_input = formatted_tc
            logger.info(f"Directly set end frame to {target_frame}")
            st.success(f"End frame set to {target_frame}")

    # Check if we need to set start/end frames after a refresh from video timecode
    if st.session_state.get("set_start_after_refresh", False):
        # Reset the flag
        st.session_state.set_start_after_refresh = False
        # Call the handler to set start frame
        handle_set_start()
        logger.info(
            f"Set start frame to {st.session_state.current_frame} after video timecode navigation"
        )

    if st.session_state.get("set_end_after_refresh", False):
        # Reset the flag
        st.session_state.set_end_after_refresh = False
        # Call the handler to set end frame
        handle_set_end()
        logger.info(
            f"Set end frame to {st.session_state.current_frame} after video timecode navigation"
        )

    # Display the sidebar
    selected_video = sidebar.display_sidebar(st.session_state.config_manager)

    # Store the selected video in session state
    if selected_video:
        # Check if the video has changed
        if st.session_state.get("current_video") != selected_video:
            st.session_state.current_video = selected_video
            # Initialize clips for the new video
            clip_service.initialize_session_clips(
                config_manager=st.session_state.config_manager
            )
    else:
        # Clear current video if none is selected
        st.session_state.current_video = None

    # Main content area
    if selected_video:
        # Display the main video player and controls
        display_main_content(selected_video)
    else:
        # Display welcome message
        display_welcome()

    # Display logs if enabled
    if st.session_state.get("display_logs", False):
        logging_utils.display_logs()

    # Process the current video if proxy generation is active
    if (
        st.session_state.proxy_generation_active
        and st.session_state.proxy_current_video
        and st.session_state.proxy_current_index < st.session_state.proxy_total_videos
    ):
        # Create proxy for the current video
        proxy_service.create_proxy_video(st.session_state.proxy_current_video)


def initialize_session_state():
    """Initialize session state variables"""
    try:
        # Import here to avoid circular imports
        from src.services.config_manager import ConfigManager

        # Initialize config manager
        if "config_manager" not in st.session_state:
            config_path = "config.yaml"
            st.session_state.config_manager = ConfigManager(config_path)
            logger.info(f"Initialized ConfigManager with config from {config_path}")

        # Initialize video-related state
        if "current_video" not in st.session_state:
            st.session_state.current_video = None
        if "current_frame" not in st.session_state:
            st.session_state.current_frame = 0
        if "total_frames" not in st.session_state:
            st.session_state.total_frames = 0
        if "fps" not in st.session_state:
            st.session_state.fps = 30.0
        if "proxy_path" not in st.session_state:
            st.session_state.proxy_path = None

        # Initialize clip state
        if "clip_name" not in st.session_state:
            st.session_state.clip_name = ""
        if "output_resolution" not in st.session_state:
            st.session_state.output_resolution = "1080p"
        if "crop_selection_active" not in st.session_state:
            st.session_state.crop_selection_active = False
        if "show_coordinate_grid" not in st.session_state:
            st.session_state.show_coordinate_grid = False
        if "trigger_rerun" not in st.session_state:
            st.session_state.trigger_rerun = False

        # Initialize clip frame slider if not exists
        if "clip_frame_slider" not in st.session_state:
            st.session_state.clip_frame_slider = 0

        # Initialize video timecode variables
        if "video_timecode_input" not in st.session_state:
            st.session_state.video_timecode_input = "00:00:00:00"
        if "set_start_after_refresh" not in st.session_state:
            st.session_state.set_start_after_refresh = False
        if "set_end_after_refresh" not in st.session_state:
            st.session_state.set_end_after_refresh = False

        # Initialize direct frame setting variables
        if "set_direct_start" not in st.session_state:
            st.session_state.set_direct_start = False
        if "set_direct_end" not in st.session_state:
            st.session_state.set_direct_end = False
        if "direct_start_frame" not in st.session_state:
            st.session_state.direct_start_frame = 0
        if "direct_end_frame" not in st.session_state:
            st.session_state.direct_end_frame = 0

        # Initialize clips
        clip_service.initialize_session_clips(
            config_manager=st.session_state.config_manager
        )

        # Initialize other session state variables
        if "current_frame" not in st.session_state:
            st.session_state.current_frame = 0

        if "output_resolution" not in st.session_state:
            st.session_state.output_resolution = "1080p"

        if "proxy_generation_active" not in st.session_state:
            st.session_state.proxy_generation_active = False

        # Add missing proxy-related session state variables
        if "proxy_current_video" not in st.session_state:
            st.session_state.proxy_current_video = None

        if "proxy_current_index" not in st.session_state:
            st.session_state.proxy_current_index = 0

        if "proxy_total_videos" not in st.session_state:
            st.session_state.proxy_total_videos = 0

        if "proxy_videos_to_process" not in st.session_state:
            st.session_state.proxy_videos_to_process = []

        if "proxy_completed_videos" not in st.session_state:
            st.session_state.proxy_completed_videos = []

        if "proxy_failed_videos" not in st.session_state:
            st.session_state.proxy_failed_videos = []

        logger.debug("Session state initialized")

    except Exception as e:
        logger.exception(f"Error initializing session state: {str(e)}")
        st.error(f"Error initializing application: {str(e)}")


def display_welcome():
    """Display welcome message when no video is selected"""
    st.title("Welcome to Clipper")

    st.markdown(
        """
    ## Video Editing Made Simple
    
    Clipper is a tool for creating clips from your videos with precise control over:
    
    - Start and end frames
    - Crop regions with keyframe support
    - Export resolution
    
    ### Getting Started
    
    1. Select a video from the sidebar
    2. Use the player controls to navigate the video
    3. Set start and end frames for your clip
    4. Optionally add crop regions with keyframes
    5. Export your clip
    
    ### Features
    
    - **Proxy Videos**: Automatically creates smaller proxy videos for faster playback
    - **Keyframe Crop Regions**: Set different crop regions at different points in your clip
    - **Multiple Clips**: Create and manage multiple clips from your videos
    """
    )

    # Display stats
    if "clips" in st.session_state and st.session_state.clips:
        st.info(f"You have {len(st.session_state.clips)} clips saved")


def display_main_content(video_path):
    """Display the main content area with video player and controls"""
    try:
        # Get current clip info for title
        current_clip = clip_service.get_current_clip()
        if current_clip:
            st.title(f"Editing: {os.path.basename(video_path)} - {current_clip.name}")

        else:
            st.title(f"Editing: {os.path.basename(video_path)}")

        # Check if we have an exported clip to display first
        if (
            "exported_clip_path" in st.session_state
            and st.session_state.exported_clip_path
        ):
            export_path = st.session_state.exported_clip_path
            logger.info(f"Attempting to display exported clip from: {export_path}")

            st.markdown("### 🎬 EXPORTED CLIP")

            if os.path.exists(export_path):
                logger.info("Export file exists, displaying...")

                # Create columns for export title and close button
                export_header_col1, export_header_col2 = st.columns([5, 1])

                with export_header_col1:
                    # Check if this is a CV-optimized export
                    is_cv_optimized = "_cv_optimized" in os.path.basename(export_path)

                    if is_cv_optimized:
                        st.success("✅ CV-Optimized Export completed successfully!")
                        st.caption(
                            f"Playing: {os.path.basename(export_path)} (Optimized for Computer Vision)"
                        )
                    else:
                        st.success("✅ Export completed successfully!")
                        st.caption(f"Playing: {os.path.basename(export_path)}")

                with export_header_col2:
                    if st.button("❌", key="close_export", help="Close export"):
                        logger.info("Closing export view...")
                        st.session_state.exported_clip_path = None
                        st.rerun()

                # Create a container for the video
                video_container = st.container()
                with video_container:
                    try:
                        with open(export_path, "rb") as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes, start_time=0)
                            logger.info("Successfully displayed exported video")
                    except Exception as e:
                        logger.error(f"Error displaying exported video: {str(e)}")
                        st.error(f"⚠️ Error playing exported clip: {str(e)}")

                # Display clip information if available
                current_clip = clip_service.get_current_clip()
                if current_clip:
                    with st.expander("Export Details", expanded=True):
                        info_col1, info_col2, info_col3, info_col4, info_col5 = (
                            st.columns(5)
                        )
                        with info_col1:
                            st.metric("Start Frame", current_clip.start_frame)
                        with info_col2:
                            st.metric("End Frame", current_clip.end_frame)
                        with info_col3:
                            duration_frames = current_clip.get_duration_frames()
                            if st.session_state.fps > 0:
                                duration_seconds = (
                                    duration_frames / st.session_state.fps
                                )
                                st.metric(
                                    "Duration",
                                    video_service.format_duration(duration_seconds),
                                )
                        with info_col4:
                            st.metric("Resolution", current_clip.output_resolution)
                        with info_col5:
                            if is_cv_optimized:
                                st.metric("Format", "CV-Optimized")
                                # Add tooltip-like explanation of the optimizations
                                st.caption(
                                    "⚙️ FFV1 lossless codec, YUV444p pixel format, Lanczos scaling in MKV container"
                                )
                            else:
                                st.metric("Format", "Standard")

                # Add open folder button
                if st.button("📂 Open in Folder", key="open_export_folder"):
                    # Use platform-specific commands to open the folder
                    folder_path = os.path.dirname(export_path)
                    try:
                        if platform.system() == "Windows":
                            os.startfile(folder_path)
                        elif platform.system() == "Darwin":  # macOS
                            subprocess.run(["open", folder_path], check=True)
                        else:  # Linux
                            subprocess.run(["xdg-open", folder_path], check=True)
                        st.success(f"Opened folder: {folder_path}")
                    except Exception as e:
                        logger.error(f"Error opening folder: {str(e)}")
                        st.error(f"Error opening folder: {str(e)}")

                # Add separator after export section
                st.markdown("---")
            else:
                st.error(f"⚠️ Export file not found: {os.path.basename(export_path)}")
                st.error("Please try clicking the export button again")

        # Check if we have a preview to display - MOVED TO TOP
        if (
            "preview_clip_path" in st.session_state
            and st.session_state.preview_clip_path
        ):
            preview_path = st.session_state.preview_clip_path
            logger.info(f"Attempting to display preview from: {preview_path}")

            st.markdown("### 🎬 CLIP PREVIEW SECTION")

            if os.path.exists(preview_path):
                logger.info("Preview file exists, displaying...")

                # Create columns for preview title and close button
                preview_header_col1, preview_header_col2 = st.columns([5, 1])

                with preview_header_col1:
                    st.success("✨ Preview loaded successfully!")
                    st.caption(f"Playing: {os.path.basename(preview_path)}")

                with preview_header_col2:
                    if st.button("❌", key="close_preview", help="Close preview"):
                        logger.info("Closing preview...")
                        st.session_state.preview_clip_path = None
                        st.rerun()

                # Create a container for the video
                video_container = st.container()
                with video_container:
                    try:
                        with open(preview_path, "rb") as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes, start_time=0)
                            logger.info("Successfully displayed preview video")
                    except Exception as e:
                        logger.error(f"Error displaying preview video: {str(e)}")
                        st.error(f"⚠️ Error playing preview: {str(e)}")
                        st.error("Try clicking the preview button again")

                # Display clip information if available
                current_clip = clip_service.get_current_clip()
                if current_clip:
                    with st.expander("Clip Details", expanded=True):
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.metric("Start Frame", current_clip.start_frame)
                        with info_col2:
                            st.metric("End Frame", current_clip.end_frame)
                        with info_col3:
                            duration_frames = current_clip.get_duration_frames()
                            if st.session_state.fps > 0:
                                duration_seconds = (
                                    duration_frames / st.session_state.fps
                                )
                                st.metric(
                                    "Duration",
                                    video_service.format_duration(duration_seconds),
                                )

                # Add separator after preview section
                st.markdown("---")
            else:
                st.error(f"⚠️ Preview file not found: {os.path.basename(preview_path)}")
                st.error("Please try clicking the preview button again")

        # Get video info
        video_info_key = f"video_info_{video_path}"

        if video_info_key not in st.session_state:
            # Get video information
            video_info = video_service.get_video_info(video_path)

            # Store in session state
            st.session_state[video_info_key] = video_info
        else:
            video_info = st.session_state[video_info_key]

        if not video_info:
            st.error(f"Could not get information for video: {video_path}")
            return

        # Store FPS in session state for use elsewhere
        st.session_state.fps = video_info["fps"]
        st.session_state.total_frames = video_info["total_frames"]

        # Get current clip if one is selected
        current_clip = clip_service.get_current_clip()

        # Initialize editing_keyframe if not exists
        if "editing_keyframe" not in st.session_state:
            st.session_state.editing_keyframe = None

        # Get crop region for current frame if we have a clip
        crop_region = None
        if current_clip:
            crop_region = current_clip.get_crop_region_at_frame(
                st.session_state.current_frame,
                use_proxy=True,  # Use proxy resolution for UI display
            )

        # Get the current frame for crop selection
        if st.session_state.get("crop_selection_active", False):
            frame = video_service.get_frame(video_path, st.session_state.current_frame)
            if frame is not None:
                # Show the crop selector
                new_crop_region = simple_crop_selector.select_crop_region(
                    frame,
                    st.session_state.current_frame,
                    current_clip,
                    st.session_state.output_resolution,
                )
                if new_crop_region is not None:
                    # Update the crop region
                    handle_crop_update(new_crop_region)
                    # Exit crop selection mode
                    st.session_state.crop_selection_active = False
                    st.rerun()

        # Display video player
        st.session_state.current_frame = video_player.display_video_player(
            video_path,  # video_service will automatically use proxy if available
            st.session_state.current_frame,
            st.session_state.fps,
            st.session_state.total_frames,
            on_frame_change=handle_frame_change,
            crop_region=crop_region,  # Just pass the crop region directly
            config_manager=config_manager,
            clip=current_clip,
        )

        # Create columns for the controls
        col1, col2 = st.columns(2)

        with col1:
            # Display clip controls
            video_player.display_clip_controls(
                clip=current_clip,
                on_set_start=handle_set_start,
                on_set_end=handle_set_end,
            )

        with col2:
            # Display crop controls
            display_crop_controls(
                current_clip=current_clip,
                current_frame=st.session_state.current_frame,
                crop_region=crop_region,
            )

    except Exception as e:
        logger.exception(f"Error displaying main content: {str(e)}")
        st.error(f"Error displaying main content: {str(e)}")


# Event handlers
def handle_frame_change(frame_number):
    """Handle frame change event"""
    st.session_state.current_frame = frame_number

    # Also update clip frame slider if current clip exists and frame is within clip bounds
    current_clip = clip_service.get_current_clip()
    if (
        current_clip
        and current_clip.start_frame <= frame_number <= current_clip.end_frame
    ):
        st.session_state.clip_frame_slider = frame_number

    # Update both timecode input fields to show the current frame timecode
    if "fps" in st.session_state and st.session_state.fps > 0:
        formatted_tc = video_service.format_timecode(frame_number, st.session_state.fps)
        # Update main timecode input
        st.session_state.timecode_input = formatted_tc
        # Update video capture timecode input
        st.session_state.video_timecode_input = formatted_tc

    logger.debug(f"Frame changed to {frame_number}")


def handle_set_start():
    """Handle set start frame button click"""
    try:
        # Get current clip
        current_clip = clip_service.get_current_clip()
        if not current_clip:
            st.warning("No clip selected")
            return

        # Update start frame
        current_clip.start_frame = st.session_state.current_frame
        clip_service.update_current_clip()

        # Mark as modified
        st.session_state.clip_modified = True

        # Auto-save the changes
        success = clip_service.save_session_clips()
        if success:
            logger.info(
                f"Auto-saved after setting start frame to {current_clip.start_frame}"
            )
            # Clear modified flag after successful save
            st.session_state.clip_modified = False
        else:
            logger.error("Failed to auto-save after setting start frame")

        # Update clip_frame_slider if it's now out of bounds
        if st.session_state.get("clip_frame_slider", 0) < current_clip.start_frame:
            st.session_state.clip_frame_slider = current_clip.start_frame

        # Show success message
        st.success(f"Start frame set to {current_clip.start_frame}")

    except Exception as e:
        logger.exception(f"Error setting start frame: {str(e)}")
        st.error(f"Error setting start frame: {str(e)}")


def handle_set_end():
    """Handle set end frame button click"""
    try:
        # Get current clip
        current_clip = clip_service.get_current_clip()
        if not current_clip:
            st.warning("No clip selected")
            return

        # Update end frame
        current_clip.end_frame = st.session_state.current_frame
        clip_service.update_current_clip()

        # Mark as modified
        st.session_state.clip_modified = True

        # Auto-save the changes
        success = clip_service.save_session_clips()
        if success:
            logger.info(
                f"Auto-saved after setting end frame to {current_clip.end_frame}"
            )
            # Clear modified flag after successful save
            st.session_state.clip_modified = False
        else:
            logger.error("Failed to auto-save after setting end frame")

        # Update clip_frame_slider if it's now out of bounds
        if st.session_state.get("clip_frame_slider", 0) > current_clip.end_frame:
            st.session_state.clip_frame_slider = current_clip.end_frame

        # Show success message
        st.success(f"End frame set to {current_clip.end_frame}")

    except Exception as e:
        logger.exception(f"Error setting end frame: {str(e)}")
        st.error(f"Error setting end frame: {str(e)}")


def handle_select_keyframe(frame_number):
    """Handle select keyframe button click"""
    try:
        # Update current frame
        st.session_state.current_frame = frame_number

        # Get current clip
        current_clip = clip_service.get_current_clip()

        # Log selection of keyframe
        if current_clip and str(frame_number) in current_clip.crop_keyframes_proxy:
            keyframe_crop = current_clip.crop_keyframes_proxy[str(frame_number)]
            logger.info(
                f"Selected keyframe at frame {frame_number} with crop {keyframe_crop}"
            )
        else:
            logger.debug(f"Selected keyframe at frame {frame_number}")

        # Set a flag to trigger rerun
        st.session_state.trigger_rerun = True

    except Exception as e:
        logger.exception(f"Error selecting keyframe: {str(e)}")
        st.error(f"Error selecting keyframe: {str(e)}")


def handle_delete_keyframe(frame_number):
    """Handle delete keyframe button click"""
    try:
        # Remove keyframe
        success = clip_service.remove_crop_keyframe(frame_number)

        if success:
            logger.info(f"Deleted keyframe at frame {frame_number}")
            st.success(f"Deleted keyframe at frame {frame_number}")
        else:
            logger.warning(f"Failed to delete keyframe at frame {frame_number}")
            st.warning(f"Failed to delete keyframe at frame {frame_number}")

        # Set a flag to trigger rerun
        st.session_state.trigger_rerun = True

    except Exception as e:
        logger.exception(f"Error deleting keyframe: {str(e)}")
        st.error(f"Error deleting keyframe: {str(e)}")


def handle_new_clip(video_path=None):
    """Handle new clip button click"""
    try:
        # If no video_path provided, use the current video from session state
        if video_path is None:
            video_path = st.session_state.get("current_video", None)

        # Check if video_path is None
        if video_path is None:
            st.warning("No video selected. Please select a video first.")
            return None

        # Clear keyframes and reset current clip index
        st.session_state.current_frame = 0

        # Reset clip frame slider to match new clip start frame
        st.session_state.clip_frame_slider = 0

        # Generate clip name
        st.session_state.clip_name = f"clip_{len(st.session_state.clips) + 1}"

        # Create a new clip
        clip = clip_service.add_clip(
            video_path,
            0,
            (
                st.session_state.total_frames - 1
                if "total_frames" in st.session_state
                else 0
            ),
            name=st.session_state.clip_name,
            output_resolution=st.session_state.output_resolution,
        )

        # Auto-save after creating new clip
        success = clip_service.save_session_clips()
        if success:
            st.session_state.last_save_status = {
                "success": True,
                "message": f"Created and saved new clip: {clip.name}",
            }
        else:
            st.session_state.last_save_status = {
                "success": False,
                "message": "Failed to save new clip",
            }

        logger.info(f"Created new clip: {clip.name}")

        # Set a flag to trigger rerun
        st.session_state.trigger_rerun = True

        return clip

    except Exception as e:
        logger.exception(f"Error creating new clip: {str(e)}")
        st.error(f"Error creating new clip: {str(e)}")
        return None


def display_crop_controls(current_clip=None, current_frame=0, crop_region=None):
    """
    Display crop controls

    Args:
        current_clip: Current clip object
        current_frame: Current frame number
        crop_region: Current crop region (x, y, width, height)

    Returns:
        None
    """
    st.subheader("Crop Controls")

    if current_clip:
        # Display current crop information
        if crop_region:
            x, y, width, height = crop_region
            st.text(f"Crop: X={x}, Y={y}, Width={width}, Height={height}")

            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            st.text(f"Aspect Ratio: {aspect_ratio:.3f}")

            # Show output dimensions
            output_resolution = st.session_state.output_resolution
            out_width, out_height = video_service.calculate_crop_dimensions(
                output_resolution, aspect_ratio
            )
            st.text(f"Output: {out_width}x{out_height} ({output_resolution})")

            # Clear crop button
            if st.button("Clear Crop Keyframe", key=f"clear_crop_{current_frame}"):
                handle_clear_crop()

        # Select crop button
        if st.button(
            "Select Crop at Current Frame", key=f"select_crop_{current_frame}"
        ):
            handle_select_crop()

        # Display keyframe list if there are keyframes
        if current_clip.crop_keyframes:
            video_player.display_keyframe_list(
                current_clip.crop_keyframes,
                current_frame,
                on_select_keyframe=handle_select_keyframe,
                on_delete_keyframe=handle_delete_keyframe,
            )
    else:
        st.info("Create or select a clip to enable crop controls")


def handle_select_crop():
    """Handle select crop button click"""
    try:
        # Get current clip
        if (
            "current_clip_index" not in st.session_state
            or st.session_state.current_clip_index < 0
        ):
            st.warning("Please create or select a clip first")
            return

        # Set crop selection mode
        st.session_state.crop_selection_active = True

        logger.info(
            f"Entering crop selection mode at frame {st.session_state.current_frame}"
        )
        # Set a flag to trigger rerun
        st.session_state.trigger_rerun = True

    except Exception as e:
        logger.exception(f"Error selecting crop: {str(e)}")
        st.error(f"Error selecting crop: {str(e)}")


def handle_clear_crop():
    """Handle clear crop button click"""
    try:
        # Get current clip
        current_clip = clip_service.get_current_clip()
        if not current_clip:
            st.warning("No clip selected")
            return

        # Clear crop keyframe for current frame from both keyframe sets
        current_frame = st.session_state.current_frame
        frame_key = str(current_frame)
        if frame_key in current_clip.crop_keyframes:
            del current_clip.crop_keyframes[frame_key]
        if frame_key in current_clip.crop_keyframes_proxy:
            del current_clip.crop_keyframes_proxy[frame_key]

        # Update the clip
        clip_service.update_current_clip()

        # Mark as modified
        st.session_state.clip_modified = True

        # Auto-save the changes
        success = clip_service.save_session_clips()
        if success:
            st.session_state.last_save_status = {
                "success": True,
                "message": f"Cleared and saved crop region at frame {current_frame}",
            }
            # Clear modified flag after successful save
            st.session_state.clip_modified = False
        else:
            st.session_state.last_save_status = {
                "success": False,
                "message": "Failed to save after clearing crop region",
            }

        logger.info(f"Cleared crop region at frame {current_frame}")

        # Force a rerun to update the UI
        st.session_state.trigger_rerun = True

    except Exception as e:
        logger.exception(f"Error clearing crop region: {str(e)}")
        st.error(f"Error clearing crop region: {str(e)}")


def handle_crop_update(crop_region):
    """Handle crop region update"""
    try:
        # Get current clip
        current_clip = clip_service.get_current_clip()
        if not current_clip:
            st.warning("No clip selected")
            return

        # Add crop keyframe using clip_service which handles proper scaling
        current_frame = st.session_state.current_frame
        success = clip_service.add_crop_keyframe(current_frame, crop_region)

        if success:
            # Mark as modified
            st.session_state.clip_modified = True

            # Auto-save the changes
            save_success = clip_service.save_session_clips()
            if save_success:
                logger.info(
                    f"Auto-saved after setting crop region at frame {current_frame}"
                )
                st.session_state.last_save_status = {
                    "success": True,
                    "message": f"Set and saved crop region at frame {current_frame}",
                }
                # Clear modified flag after successful save
                st.session_state.clip_modified = False
            else:
                logger.error("Failed to auto-save after setting crop region")
                st.session_state.last_save_status = {
                    "success": False,
                    "message": "Failed to save crop region changes",
                }
        else:
            st.session_state.last_save_status = {
                "success": False,
                "message": "Failed to set crop region",
            }

        logger.info(f"Updated crop region at frame {current_frame}")

        # Force a rerun to update the UI
        st.session_state.trigger_rerun = True

    except Exception as e:
        logger.exception(f"Error updating crop region: {str(e)}")
        st.error(f"Error updating crop region: {str(e)}")


if __name__ == "__main__":
    main()
