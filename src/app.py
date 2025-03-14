"""
Main application file for the Clipper video editing tool.
"""

import streamlit as st
import logging
import os
from pathlib import Path

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
    st.set_page_config(
        page_title="Clipper",
        page_icon="🎬",
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

    # Display the sidebar
    selected_video = sidebar.display_sidebar(st.session_state.config_manager)

    # Store the selected video in session state
    if selected_video:
        # Check if the video has changed
        if st.session_state.get("current_video") != selected_video:
            st.session_state.current_video = selected_video
            # Initialize clips for the new video
            clip_service.initialize_session_clips(st.session_state.config_manager)
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
            st.session_state.config_manager = ConfigManager()
            logger.info("Initialized config manager")

        # Initialize current_video if not set
        if "current_video" not in st.session_state:
            st.session_state.current_video = None

        # Initialize clips
        clip_service.initialize_session_clips(st.session_state.config_manager)

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

        if "crop_selection_active" not in st.session_state:
            st.session_state.crop_selection_active = False

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

        # Title with video name
        st.title(f"Editing: {os.path.basename(video_path)}")

        # If crop selection is active, display the crop selector
        if st.session_state.crop_selection_active:
            # Get the current frame
            frame = video_service.get_frame(
                video_path,  # video_service will automatically use proxy if available
                st.session_state.current_frame,
            )

            # Display crop selector
            crop_region = simple_crop_selector.select_crop_region(
                frame,
                st.session_state.current_frame,
                current_clip,
                st.session_state.output_resolution,
            )

            # If crop region is confirmed, add it as a keyframe
            if crop_region is not None:
                if current_clip:
                    try:
                        clip_service.add_crop_keyframe(
                            st.session_state.current_frame, crop_region
                        )
                        st.success(
                            f"Added crop keyframe at frame {st.session_state.current_frame}"
                        )
                        # Set crop selection to inactive and rerun
                        st.session_state.crop_selection_active = False
                        st.rerun()
                    except Exception as e:
                        logger.exception(f"Error adding crop keyframe: {str(e)}")
                        st.error(f"Error adding crop keyframe: {str(e)}")
        else:
            # Get crop region for current frame if available
            crop_region = None
            if current_clip and current_clip.crop_keyframes:
                crop_region = current_clip.get_crop_region_at_frame(
                    st.session_state.current_frame
                )

            # Display video player
            st.session_state.current_frame = video_player.display_video_player(
                video_path,  # video_service will automatically use proxy if available
                st.session_state.current_frame,
                st.session_state.fps,
                st.session_state.total_frames,
                on_frame_change=handle_frame_change,
                crop_region=crop_region,
                config_manager=config_manager,
            )

            # Create columns for the controls
            col1, col2 = st.columns(2)

            with col1:
                # Display clip controls
                video_player.display_clip_controls(
                    clip=current_clip,
                    on_set_start=handle_set_start,
                    on_set_end=handle_set_end,
                    on_play_clip=handle_play_clip,
                    on_export_clip=handle_export_clip,
                )

            with col2:
                # Display crop controls
                display_crop_controls(
                    current_clip=current_clip,
                    current_frame=st.session_state.current_frame,
                    crop_region=crop_region,
                )

            # Display keyframe list if clip has keyframes
            if current_clip and current_clip.crop_keyframes:
                video_player.display_keyframe_list(
                    current_clip.crop_keyframes,
                    st.session_state.current_frame,
                    on_select_keyframe=handle_select_keyframe,
                    on_delete_keyframe=handle_delete_keyframe,
                )

    except Exception as e:
        logger.exception(f"Error displaying main content: {str(e)}")
        st.error(f"Error displaying video: {str(e)}")


# Event handlers
def handle_frame_change(frame_number):
    """Handle frame change event"""
    st.session_state.current_frame = frame_number
    logger.debug(f"Frame changed to {frame_number}")


def handle_set_start():
    """Handle set start frame button click"""
    try:
        # Get current clip
        if (
            "current_clip_index" not in st.session_state
            or st.session_state.current_clip_index < 0
        ):
            # Create a new clip if none is selected
            handle_new_clip(
                st.session_state.proxy_path
                if "proxy_path" in st.session_state
                else None
            )

        # Update start frame
        clip_service.update_current_clip(start_frame=st.session_state.current_frame)

        logger.info(f"Set start frame to {st.session_state.current_frame}")
        st.success(f"Start frame set to {st.session_state.current_frame}")

        # Set a flag to trigger rerun
        st.session_state.trigger_rerun = True

    except Exception as e:
        logger.exception(f"Error setting start frame: {str(e)}")
        st.error(f"Error setting start frame: {str(e)}")


def handle_set_end():
    """Handle set end frame button click"""
    try:
        # Get current clip
        if (
            "current_clip_index" not in st.session_state
            or st.session_state.current_clip_index < 0
        ):
            # Create a new clip if none is selected
            handle_new_clip(
                st.session_state.proxy_path
                if "proxy_path" in st.session_state
                else None
            )

        # Update end frame
        clip_service.update_current_clip(end_frame=st.session_state.current_frame)

        logger.info(f"Set end frame to {st.session_state.current_frame}")
        st.success(f"End frame set to {st.session_state.current_frame}")

        # Set a flag to trigger rerun
        st.session_state.trigger_rerun = True

    except Exception as e:
        logger.exception(f"Error setting end frame: {str(e)}")
        st.error(f"Error setting end frame: {str(e)}")


def handle_play_clip():
    """Handle play clip button click"""
    try:
        # Get current clip
        current_clip = clip_service.get_current_clip()

        if not current_clip:
            st.warning("No clip selected")
            return

        # Get video path - video_service will automatically use proxy if available
        video_path = current_clip.source_path

        # Play the clip
        video_player.play_clip_preview(
            video_path,
            current_clip.start_frame,
            current_clip.end_frame,
            st.session_state.fps,
            crop_region=current_clip.get_crop_region_at_frame,
            on_frame_change=handle_frame_change,
        )

    except Exception as e:
        logger.exception(f"Error playing clip: {str(e)}")
        st.error(f"Error playing clip: {str(e)}")


def handle_export_clip():
    """Handle export clip button click"""
    try:
        # Get current clip
        current_clip = clip_service.get_current_clip()

        if not current_clip:
            st.warning("No clip selected")
            return

        # Create export directory if it doesn't exist
        export_dir = st.session_state.config_manager.get_export_dir()
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate export filename
        clip_name = current_clip.name.replace(" ", "_")
        export_path = export_dir / f"{clip_name}.mp4"

        # Check if file already exists
        if export_path.exists():
            # Add timestamp to filename
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"{clip_name}_{timestamp}.mp4"

        # Show export progress
        progress_placeholder = st.empty()
        progress_placeholder.info(f"Exporting clip to {export_path}...")

        # Get crop region function
        crop_region_func = None
        if current_clip.crop_keyframes:
            crop_region_func = current_clip.get_crop_region_at_frame

        # Export the clip
        success = video_service.export_clip(
            current_clip.source_path,
            export_path,
            current_clip.start_frame,
            current_clip.end_frame,
            crop_region=(
                crop_region_func(current_clip.start_frame) if crop_region_func else None
            ),
            output_resolution=current_clip.output_resolution,
            config_manager=st.session_state.config_manager,
        )

        if success:
            # Update clip with export path
            current_clip.export_path = str(export_path)
            clip_service.update_current_clip()

            # Save clips
            clip_service.save_session_clips()

            progress_placeholder.success(f"Clip exported to {export_path}")
            logger.info(f"Clip exported to {export_path}")
        else:
            progress_placeholder.error("Failed to export clip")
            logger.error("Failed to export clip")

    except Exception as e:
        logger.exception(f"Error exporting clip: {str(e)}")
        st.error(f"Error exporting clip: {str(e)}")


def handle_select_keyframe(frame_number):
    """Handle select keyframe button click"""
    try:
        # Update current frame
        st.session_state.current_frame = frame_number

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

        logger.info(f"Created new clip: {clip.name}")
        st.success(f"Created new clip: {clip.name}")

        # Save clips to file
        clip_service.save_session_clips()

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
                handle_clear_crop(current_frame)

        # Select crop button
        if st.button(
            "Select Crop at Current Frame", key=f"select_crop_{current_frame}"
        ):
            handle_select_crop()
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


def handle_clear_crop(frame_number=None):
    """Handle clear crop button click"""
    try:
        # Get current clip
        current_clip = clip_service.get_current_clip()

        if not current_clip:
            st.warning("No clip selected")
            return

        # Use current frame if not specified
        if frame_number is None:
            frame_number = st.session_state.current_frame

        # Remove keyframe at current frame
        success = clip_service.remove_crop_keyframe(frame_number)

        if success:
            logger.info(f"Removed crop keyframe at frame {frame_number}")
            st.success(f"Removed crop keyframe at frame {frame_number}")
        else:
            logger.warning(f"No keyframe at frame {frame_number}")
            st.warning(f"No keyframe at frame {frame_number}")

        # Set a flag to trigger rerun
        st.session_state.trigger_rerun = True

    except Exception as e:
        logger.exception(f"Error clearing crop: {str(e)}")
        st.error(f"Error clearing crop: {str(e)}")


if __name__ == "__main__":
    main()
