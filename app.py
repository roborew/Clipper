import streamlit as st
import os
import time
from pathlib import Path
from video_processor import VideoProcessor, save_config, load_config

# Set page configuration
st.set_page_config(
    page_title="Clipper - Video Clipping Tool", page_icon="🎬", layout="wide"
)

# Define directory structure
data_dir = Path("data")
source_dir = data_dir / "source"  # Symlink to raw/calibrated footage
prept_dir = data_dir / "prept"  # Symlink to processed clips
clips_dir = prept_dir / "03_clipped"  # Directory for processed clips
config_dir = clips_dir / "_configs"  # Store configs within the clipped folder

# Create necessary directories if they don't exist
clips_dir.mkdir(exist_ok=True, parents=True)
config_dir.mkdir(exist_ok=True, parents=True)

# Initialize session state variables
if "video_processor" not in st.session_state:
    st.session_state.video_processor = VideoProcessor()
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "clips" not in st.session_state:
    st.session_state.clips = []
if "current_clip_index" not in st.session_state:
    st.session_state.current_clip_index = -1
if "config_file" not in st.session_state:
    st.session_state.config_file = None
if "crop_start" not in st.session_state:
    st.session_state.crop_start = None
if "crop_end" not in st.session_state:
    st.session_state.crop_end = None
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "output_resolution" not in st.session_state:
    st.session_state.output_resolution = "1080p"

# Resolution presets
RESOLUTION_PRESETS = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
    "240p": (426, 240),
}


def load_video_file(video_path):
    """Load a video file and initialize video properties"""
    if video_path:
        success, message = st.session_state.video_processor.load_video(video_path)

        if success:
            st.session_state.video_path = video_path
            st.session_state.current_frame = 0

            # Generate default config filename based on video path structure
            video_path_obj = Path(video_path)
            relative_path = video_path_obj.relative_to(source_dir)
            # Preserve directory structure in config name
            config_name = str(relative_path).replace("/", "_").replace("\\", "_")
            base_name = os.path.splitext(config_name)[0]
            st.session_state.config_file = str(config_dir / f"{base_name}.json")

            # Try to load existing config if it exists
            load_config_file(st.session_state.config_file)

            return True
        else:
            st.error(message)
            return False
    return False


def load_config_file(config_path):
    """Load clip configurations from a JSON file"""
    success, message, video_path, clips = load_config(config_path)

    if success:
        # Only load clips if the video path matches
        if video_path == st.session_state.video_path:
            st.session_state.clips = clips
            st.success(f"Loaded {len(clips)} clips from configuration")
        else:
            st.warning(
                "Config file exists but is for a different video. Starting with empty clips."
            )
            st.session_state.clips = []
    else:
        st.error(message)
        st.session_state.clips = []


def save_config_file(config_path):
    """Save clip configurations to a JSON file"""
    if st.session_state.video_path:
        success, message = save_config(
            config_path, st.session_state.video_path, st.session_state.clips
        )
        if success:
            st.success(message)
        else:
            st.error(message)
    else:
        st.error("No video loaded")


def add_or_update_clip():
    """Add a new clip or update an existing one"""
    if st.session_state.in_point is None or st.session_state.out_point is None:
        st.error("Please set both in and out points")
        return

    if st.session_state.crop_start is None:
        st.error("Please set crop region")
        return

    clip_data = {
        "name": st.session_state.clip_name,
        "in_point": st.session_state.in_point,
        "out_point": st.session_state.out_point,
        "crop_start": st.session_state.crop_start,
        "crop_end": st.session_state.crop_end
        or st.session_state.crop_start,  # Use crop_start if crop_end not set
        "output_resolution": st.session_state.output_resolution,
    }

    if (
        st.session_state.current_clip_index >= 0
        and st.session_state.current_clip_index < len(st.session_state.clips)
    ):
        # Update existing clip
        st.session_state.clips[st.session_state.current_clip_index] = clip_data
        st.success(f"Updated clip: {clip_data['name']}")
    else:
        # Add new clip
        st.session_state.clips.append(clip_data)
        st.success(f"Added new clip: {clip_data['name']}")

    # Save configuration
    save_config_file(st.session_state.config_file)


def delete_clip(index):
    """Delete a clip from the list"""
    if 0 <= index < len(st.session_state.clips):
        deleted_clip = st.session_state.clips.pop(index)
        st.success(f"Deleted clip: {deleted_clip['name']}")
        save_config_file(st.session_state.config_file)

        # Reset current clip index if needed
        if st.session_state.current_clip_index >= len(st.session_state.clips):
            st.session_state.current_clip_index = -1


def export_clips(output_directory):
    """Export all clips based on the configuration"""
    if not st.session_state.clips:
        st.error("No clips to export")
        return

    if not st.session_state.video_path:
        st.error("No video loaded")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    def overall_progress_callback(progress):
        progress_bar.progress(progress)

    def clip_progress_callback(progress):
        status_text.text(f"Processing clip... {int(progress * 100)}%")

    status_text.text("Starting export...")

    results = st.session_state.video_processor.export_clips(
        st.session_state.clips,
        output_directory,
        progress_callback=overall_progress_callback,
        clip_progress_callback=clip_progress_callback,
    )

    # Display results
    for success, message in results:
        if success:
            st.success(message)
        else:
            st.error(message)

    status_text.text("Export complete!")
    progress_bar.progress(1.0)


# Main application UI
st.title("Clipper - Video Clipping Tool")

# Sidebar for controls
with st.sidebar:
    st.header("Video Selection")

    # Video file selection - now specifically from source directory
    if source_dir.exists() and source_dir.is_dir():
        video_files = [
            f.relative_to(source_dir)
            for f in source_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
        ]
        if video_files:
            selected_video = st.selectbox("Select Video", video_files)
            video_path = str(source_dir / selected_video)

            if st.button("Load Video"):
                load_video_file(video_path)
        else:
            st.warning(f"No video files found in {source_dir}")
    else:
        st.error(
            f"Source directory {source_dir} does not exist or is not properly linked"
        )

    st.divider()

    # Configuration file controls
    st.header("Configuration")
    if st.session_state.config_file:
        config_display = Path(st.session_state.config_file).relative_to(config_dir)
    else:
        config_display = ""
    config_path = st.text_input("Config File Path", value=config_display)

    if config_path:
        full_config_path = config_dir / config_path
    else:
        full_config_path = st.session_state.config_file or ""

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Config"):
            if os.path.exists(full_config_path):
                st.session_state.config_file = str(full_config_path)
                load_config_file(full_config_path)
            else:
                st.error("Config file does not exist")

    with col2:
        if st.button("Save Config"):
            save_config_file(full_config_path)

    st.divider()

    # Export controls
    st.header("Export")
    # Allow selecting subdirectory within clips directory for organization
    output_subdir = st.text_input("Output Subdirectory (optional)", value="")
    output_dir = clips_dir  # Changed from prept_dir to clips_dir
    if output_subdir:
        output_dir = output_dir / output_subdir

    if st.button("Export All Clips"):
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        export_clips(str(output_dir))

# Main content area
if st.session_state.video_path is not None:
    # Video player controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("⏮️ Previous Frame"):
            st.session_state.current_frame = max(0, st.session_state.current_frame - 1)

    with col2:
        play_pause = "⏸️ Pause" if st.session_state.is_playing else "▶️ Play"
        if st.button(play_pause):
            st.session_state.is_playing = not st.session_state.is_playing

    with col3:
        if st.button("⏭️ Next Frame"):
            st.session_state.current_frame = min(
                st.session_state.video_processor.total_frames - 1,
                st.session_state.current_frame + 1,
            )

    with col4:
        st.write(
            f"Frame: {st.session_state.current_frame}/{st.session_state.video_processor.total_frames}"
        )

    # Frame slider
    st.session_state.current_frame = st.slider(
        "Frame Position",
        min_value=0,
        max_value=max(0, st.session_state.video_processor.total_frames - 1),
        value=st.session_state.current_frame,
    )

    # Display current frame
    current_frame = st.session_state.video_processor.get_frame(
        st.session_state.current_frame
    )
    if current_frame is not None:
        frame_height, frame_width = current_frame.shape[:2]

        # Display frame with annotations
        frame_container = st.empty()
        frame_container.image(
            current_frame,
            caption=f"Frame {st.session_state.current_frame}",
            use_column_width=True,
        )

        # Clip editing controls
        st.subheader("Clip Controls")

        col1, col2 = st.columns(2)

        with col1:
            # In/Out point controls
            if "in_point" not in st.session_state:
                st.session_state.in_point = None

            if "out_point" not in st.session_state:
                st.session_state.out_point = None

            in_out_col1, in_out_col2 = st.columns(2)

            with in_out_col1:
                if st.button("Set In Point"):
                    st.session_state.in_point = st.session_state.current_frame

            with in_out_col2:
                if st.button("Set Out Point"):
                    st.session_state.out_point = st.session_state.current_frame

            st.write(f"In Point: {st.session_state.in_point}")
            st.write(f"Out Point: {st.session_state.out_point}")

        with col2:
            # Crop region controls
            st.write("Crop Region (x, y, width, height)")

            crop_col1, crop_col2 = st.columns(2)

            with crop_col1:
                if st.button("Set Start Crop"):
                    # Default to full frame if not set
                    st.session_state.crop_start = [0, 0, frame_width, frame_height]

            with crop_col2:
                if st.button("Set End Crop"):
                    # Default to start crop if not set
                    if st.session_state.crop_start:
                        st.session_state.crop_end = st.session_state.crop_start.copy()
                    else:
                        st.session_state.crop_end = [0, 0, frame_width, frame_height]

            # Crop region input fields
            if st.session_state.crop_start:
                crop_x = st.slider(
                    "Crop X", 0, frame_width - 10, st.session_state.crop_start[0]
                )
                crop_y = st.slider(
                    "Crop Y", 0, frame_height - 10, st.session_state.crop_start[1]
                )
                crop_w = st.slider(
                    "Crop Width",
                    10,
                    frame_width - crop_x,
                    st.session_state.crop_start[2],
                )
                crop_h = st.slider(
                    "Crop Height",
                    10,
                    frame_height - crop_y,
                    st.session_state.crop_start[3],
                )

                st.session_state.crop_start = [crop_x, crop_y, crop_w, crop_h]

                if st.checkbox("Same end crop as start"):
                    st.session_state.crop_end = None
                elif st.session_state.crop_end:
                    st.write("End Crop (if different from start):")
                    end_crop_x = st.slider(
                        "End Crop X", 0, frame_width - 10, st.session_state.crop_end[0]
                    )
                    end_crop_y = st.slider(
                        "End Crop Y", 0, frame_height - 10, st.session_state.crop_end[1]
                    )
                    end_crop_w = st.slider(
                        "End Crop Width",
                        10,
                        frame_width - end_crop_x,
                        st.session_state.crop_end[2],
                    )
                    end_crop_h = st.slider(
                        "End Crop Height",
                        10,
                        frame_height - end_crop_y,
                        st.session_state.crop_end[3],
                    )

                    st.session_state.crop_end = [
                        end_crop_x,
                        end_crop_y,
                        end_crop_w,
                        end_crop_h,
                    ]

        # Output resolution selection
        st.subheader("Output Settings")
        st.session_state.output_resolution = st.selectbox(
            "Output Resolution",
            list(RESOLUTION_PRESETS.keys()),
            index=list(RESOLUTION_PRESETS.keys()).index(
                st.session_state.output_resolution
            ),
        )

        # Clip name and save controls
        if "clip_name" not in st.session_state:
            st.session_state.clip_name = f"clip_{len(st.session_state.clips) + 1}"

        st.session_state.clip_name = st.text_input(
            "Clip Name", value=st.session_state.clip_name
        )

        save_col1, save_col2 = st.columns(2)

        with save_col1:
            if st.button("Add/Update Clip"):
                add_or_update_clip()

        with save_col2:
            if st.button("Clear Clip Data"):
                st.session_state.in_point = None
                st.session_state.out_point = None
                st.session_state.crop_start = None
                st.session_state.crop_end = None
                st.session_state.current_clip_index = -1
                st.session_state.clip_name = f"clip_{len(st.session_state.clips) + 1}"

        # List of saved clips
        st.subheader("Saved Clips")

        if st.session_state.clips:
            for i, clip in enumerate(st.session_state.clips):
                with st.expander(
                    f"{i+1}. {clip['name']} ({clip['in_point']} - {clip['out_point']})"
                ):
                    st.write(f"In Point: {clip['in_point']}")
                    st.write(f"Out Point: {clip['out_point']}")
                    st.write(f"Crop Start: {clip['crop_start']}")
                    st.write(
                        f"Crop End: {clip['crop_end'] if clip['crop_end'] else 'Same as start'}"
                    )
                    st.write(f"Output Resolution: {clip['output_resolution']}")

                    clip_col1, clip_col2 = st.columns(2)

                    with clip_col1:
                        if st.button(f"Edit Clip #{i+1}"):
                            st.session_state.current_clip_index = i
                            st.session_state.in_point = clip["in_point"]
                            st.session_state.out_point = clip["out_point"]
                            st.session_state.crop_start = clip["crop_start"]
                            st.session_state.crop_end = clip["crop_end"]
                            st.session_state.clip_name = clip["name"]
                            st.session_state.output_resolution = clip[
                                "output_resolution"
                            ]

                            # Jump to in point
                            st.session_state.current_frame = clip["in_point"]

                    with clip_col2:
                        if st.button(f"Delete Clip #{i+1}"):
                            delete_clip(i)
        else:
            st.info("No clips saved yet")

        # Auto-play functionality
        if st.session_state.is_playing:
            time.sleep(1 / st.session_state.video_processor.fps)
            st.session_state.current_frame += 1
            if (
                st.session_state.current_frame
                >= st.session_state.video_processor.total_frames
            ):
                st.session_state.current_frame = 0
                st.session_state.is_playing = False
            st.experimental_rerun()
else:
    st.info("Please load a video file from the sidebar to get started.")

# Instructions
with st.expander("How to Use"):
    st.markdown(
        """
    ## Instructions
    
    1. **Load a Video**: Select a video file from the source directory and click "Load Video".
    2. **Navigate the Video**: Use the play/pause button, next/previous frame buttons, or the slider to navigate through the video.
    3. **Create a Clip**:
        - Set the in-point (start) and out-point (end) of your clip
        - Define the crop region by setting start and end crop coordinates
        - Choose an output resolution
        - Give your clip a name and click "Add/Update Clip"
    4. **Manage Clips**:
        - View all saved clips in the "Saved Clips" section
        - Edit or delete existing clips as needed
    5. **Export**:
        - Optionally specify an output subdirectory within the 03_clipped folder
        - Click "Export All Clips" to process and save all clips
    
    The application saves your clip configurations to JSON files in the 03_clipped/_configs directory, 
    keeping all clip-related data together in the clipped stage of processing.
    
    Directory Structure:
    - `data/source/`: Source videos (raw and calibrated footage)
    - `data/prept/03_clipped/`: Processed clips output
    - `data/prept/`: Processed clips output
    - `data/prept/configs/`: Clip configurations
    """
    )

# Footer
st.markdown("---")
st.markdown("Clipper - Video Clipping Tool | Created with Streamlit and OpenCV")
