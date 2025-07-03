"""
Clip management services for the Clipper application.
"""

import os
import json
import uuid
from pathlib import Path
import streamlit as st
import logging
from datetime import datetime
from . import video_service

logger = logging.getLogger("clipper.clip")


class Clip:
    """
    Class representing a video clip with start/end frames and optional crop region
    """

    def __init__(
        self,
        name=None,
        source_path=None,
        proxy_path=None,
        start_frame=0,
        end_frame=0,
        crop_keyframes=None,
        crop_keyframes_proxy=None,
        output_resolution="1080p",
        status="Draft",
    ):
        """
        Initialize a new clip

        Args:
            name: Name of the clip
            source_path: Path to the source video (will be stored as relative path)
            proxy_path: Path to the proxy video for previews (will be stored as relative path)
            start_frame: Starting frame number
            end_frame: Ending frame number
            crop_keyframes: Dictionary of frame numbers to crop regions (x, y, width, height)
            crop_keyframes_proxy: Dictionary of frame numbers to crop regions for proxy video
            output_resolution: Target resolution for export
            status: Processing status of the clip (Draft, Process, or Complete)
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Convert source path to relative using ConfigManager
        if source_path:
            source_path = str(source_path)
            # Get ConfigManager from session state or create a new one
            try:
                import streamlit as st

                config_manager = st.session_state.get("config_manager")
                if not config_manager:
                    from src.services.config_manager import ConfigManager

                    config_manager = ConfigManager()

                # Convert absolute path to relative based on actual config
                source_path_obj = Path(source_path)
                if source_path_obj.is_absolute():
                    # Try to make it relative to source directories
                    try:
                        relative_path = source_path_obj.relative_to(
                            config_manager.source_raw
                        )
                        source_path = str(
                            Path(config_manager.config["directories"]["source"]["raw"])
                            / relative_path
                        )
                    except ValueError:
                        try:
                            relative_path = source_path_obj.relative_to(
                                config_manager.source_calibrated
                            )
                            source_path = str(
                                Path(
                                    config_manager.config["directories"]["source"][
                                        "calibrated"
                                    ]
                                )
                                / relative_path
                            )
                        except ValueError:
                            try:
                                relative_path = source_path_obj.relative_to(
                                    config_manager.source_base
                                )
                                source_path = str(relative_path)
                            except ValueError:
                                # If we can't make it relative, keep as is
                                pass
            except Exception as e:
                logger.warning(f"Could not use ConfigManager for path conversion: {e}")
                # Keep original path if ConfigManager fails
                pass
        self.source_path = source_path

        # Convert proxy path to relative using ConfigManager
        if proxy_path:
            proxy_path = str(proxy_path)
            try:
                import streamlit as st

                config_manager = st.session_state.get("config_manager")
                if not config_manager:
                    from src.services.config_manager import ConfigManager

                    config_manager = ConfigManager()

                # Convert absolute path to relative based on actual config
                proxy_path_obj = Path(proxy_path)
                if proxy_path_obj.is_absolute():
                    # Try to make it relative to proxy directories
                    try:
                        relative_path = proxy_path_obj.relative_to(
                            config_manager.proxy_base
                        )
                        proxy_path = str(relative_path)
                    except ValueError:
                        # If we can't make it relative, keep as is
                        pass
            except Exception as e:
                logger.warning(
                    f"Could not use ConfigManager for proxy path conversion: {e}"
                )
                # Keep original path if ConfigManager fails
                pass
        self.proxy_path = proxy_path

        self.start_frame = start_frame
        self.end_frame = end_frame
        self.crop_keyframes = crop_keyframes or {}
        self.crop_keyframes_proxy = crop_keyframes_proxy or {}
        self.output_resolution = output_resolution
        self.export_path = None
        self.created_at = datetime.now().isoformat()
        self.modified_at = self.created_at
        self.status = status  # Default to "Draft" for new clips

    def to_dict(self):
        """Convert clip to dictionary for serialization"""
        # Handle export_path which could be a string or a list of paths
        if isinstance(self.export_path, list):
            # If it's a list, keep it as a list in the JSON
            export_path_value = self.export_path
        else:
            # If it's a string or None, convert as before
            export_path_value = str(self.export_path) if self.export_path else None

        return {
            "id": self.id,
            "name": self.name,
            "source_path": self.source_path,  # Already stored as relative path
            "proxy_path": self.proxy_path,  # Already stored as relative path
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "crop_keyframes": self.crop_keyframes,
            "crop_keyframes_proxy": self.crop_keyframes_proxy,
            "output_resolution": self.output_resolution,
            "export_path": export_path_value,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data):
        """Create clip from dictionary"""
        clip = cls()
        clip.id = data.get("id", str(uuid.uuid4()))
        clip.name = data.get("name", "Unnamed Clip")

        # Handle paths when loading from dict - keep stored paths as they are
        # since they should already be in the correct relative format
        clip.source_path = data.get("source_path")
        clip.proxy_path = data.get("proxy_path")

        clip.start_frame = data.get("start_frame", 0)
        clip.end_frame = data.get("end_frame", 0)
        clip.crop_keyframes = data.get("crop_keyframes", {})
        # Initialize crop_keyframes_proxy with the same values as crop_keyframes if not present
        clip.crop_keyframes_proxy = data.get(
            "crop_keyframes_proxy", clip.crop_keyframes.copy()
        )
        clip.output_resolution = data.get("output_resolution", "1080p")

        # Handle export_path which could be a string or a list in the input data
        export_path = data.get("export_path")
        # If it's a list, keep it as a list
        if isinstance(export_path, list):
            clip.export_path = export_path
        else:
            # Otherwise, store it as is (string or None)
            clip.export_path = export_path

        clip.created_at = data.get("created_at", datetime.now().isoformat())
        clip.modified_at = data.get("modified_at", clip.created_at)
        clip.status = data.get("status", "Draft")  # Default to "Draft" if not specified
        return clip

    def update(self):
        """Mark the clip as modified"""
        self.modified_at = datetime.now().isoformat()

    def copy(self):
        """Create a copy of this clip for variations"""
        new_clip = Clip(
            name=self.name,
            source_path=self.source_path,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            output_resolution=self.output_resolution,
            status=self.status,
            crop_keyframes=self.crop_keyframes.copy() if self.crop_keyframes else None,
        )

        # Set the id to match the original clip
        new_clip.id = self.id

        # Handle optional attributes
        if hasattr(self, "export_path"):
            new_clip.export_path = self.export_path

        if hasattr(self, "crop_region"):
            new_clip.crop_region = self.crop_region

        return new_clip

    def get_duration_frames(self):
        """Get the duration of the clip in frames"""
        return max(0, self.end_frame - self.start_frame + 1)

    def get_crop_region_at_frame(self, frame_number, use_proxy=True):
        """
        Get the crop region at a specific frame by interpolating between keyframes

        Args:
            frame_number: Frame number to get crop region for
            use_proxy: Whether to return proxy resolution crop region (True) or source resolution (False)

        Returns:
            Tuple of (x, y, width, height) or None if no crop keyframes exist
        """
        # Choose which keyframes to use based on use_proxy flag
        keyframes = self.crop_keyframes_proxy if use_proxy else self.crop_keyframes

        if not keyframes:
            return None

        # Convert string keys to integers
        keyframes = {int(k): v for k, v in keyframes.items()}

        # If the exact frame has a keyframe, return it
        if frame_number in keyframes:
            return keyframes[frame_number]

        # Find the nearest keyframes before and after the target frame
        prev_frame = None
        next_frame = None

        for kf in sorted(keyframes.keys()):
            if kf <= frame_number:
                prev_frame = kf
            elif kf > frame_number:
                next_frame = kf
                break

        # If no previous keyframe, use the first one
        if prev_frame is None and keyframes:
            prev_frame = min(keyframes.keys())

        # If no next keyframe, use the last one
        if next_frame is None and keyframes:
            next_frame = max(keyframes.keys())

        # If only one keyframe exists, use it
        if prev_frame is not None and next_frame is None:
            return keyframes[prev_frame]
        elif prev_frame is None and next_frame is not None:
            return keyframes[next_frame]

        # Interpolate between the two nearest keyframes
        if prev_frame is not None and next_frame is not None:
            prev_crop = keyframes[prev_frame]
            next_crop = keyframes[next_frame]

            # Calculate interpolation factor
            total_frames = next_frame - prev_frame
            if total_frames == 0:
                factor = 0
            else:
                factor = (frame_number - prev_frame) / total_frames

            # Interpolate each component
            x = int(prev_crop[0] + factor * (next_crop[0] - prev_crop[0]))
            y = int(prev_crop[1] + factor * (next_crop[1] - prev_crop[1]))
            width = int(prev_crop[2] + factor * (next_crop[2] - prev_crop[2]))
            height = int(prev_crop[3] + factor * (next_crop[3] - prev_crop[3]))

            return (x, y, width, height)

        return None


def save_clips(clips, save_path):
    """
    Save clips to a JSON file

    Args:
        clips: List of Clip objects
        save_path: Path to save the clips

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert clips to dictionaries
        clips_data = []
        for clip in clips:
            # Perform additional checks on export_path before saving
            if hasattr(clip, "export_path"):
                logger.info(
                    f"Before to_dict - clip {clip.name} export_path: {clip.export_path}"
                )

                # If export_path is empty but files exist on disk, try to reconstruct it
                if (
                    (
                        not clip.export_path
                        or (
                            isinstance(clip.export_path, list)
                            and len(clip.export_path) == 0
                        )
                    )
                    and hasattr(clip, "export_paths")
                    and clip.export_paths
                ):
                    # Try to recreate from export_paths string
                    logger.info(
                        f"Attempting to recreate empty export_path from export_paths: {clip.export_paths}"
                    )
                    paths = [
                        p.strip() for p in clip.export_paths.split(",") if p.strip()
                    ]
                    if paths:
                        clip.export_path = paths
                        logger.info(
                            f"Reconstructed export_path from export_paths: {clip.export_path}"
                        )

            # Now convert to dict for JSON serialization
            clip_dict = clip.to_dict()
            logger.info(
                f"After to_dict - clip {clip.name} export_path in dict: {clip_dict.get('export_path')}"
            )

            # If export_path is still empty in dict but should have values based on clip status, log a warning
            if clip.status == "Complete" and (
                not clip_dict.get("export_path")
                or (
                    isinstance(clip_dict.get("export_path"), list)
                    and len(clip_dict.get("export_path")) == 0
                )
            ):
                logger.warning(
                    f"WARNING: Clip {clip.name} has status 'Complete' but empty export_path in dict"
                )

            clips_data.append(clip_dict)

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Write to file
        with open(save_path, "w") as f:
            json.dump(clips_data, f, indent=2)

        logger.info(f"Saved {len(clips)} clips to {save_path}")
        return True

    except Exception as e:
        logger.exception(f"Error saving clips to {save_path}: {str(e)}")
        return False


def load_clips(load_path):
    """
    Load clips from a JSON file

    Args:
        load_path: Path to load the clips from

    Returns:
        List of Clip objects or empty list if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(load_path):
            logger.info(f"Clips file not found: {load_path}")
            return []

        # Read from file
        with open(load_path, "r") as f:
            clips_data = json.load(f)

        # Convert dictionaries to Clip objects
        clips = [Clip.from_dict(data) for data in clips_data]

        logger.info(f"Loaded {len(clips)} clips from {load_path}")
        return clips

    except Exception as e:
        logger.exception(f"Error loading clips from {load_path}: {str(e)}")
        return []


def initialize_session_clips(config_manager=None, force_reload=False):
    """
    Initialize clips in the session state

    Args:
        config_manager: ConfigManager instance
        force_reload: Whether to force reload clips even if already loaded

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get config manager from session state if not provided
        if not config_manager:
            config_manager = st.session_state.config_manager

        # Get current video path if available
        current_video = st.session_state.get("current_video", None)

        # Get clips file path based on current video
        clips_file = config_manager.get_clips_file_path(current_video)

        # Initialize session state variables if they don't exist
        if "clips" not in st.session_state or force_reload:
            # Load clips from file
            clips = load_clips(clips_file)
            st.session_state.clips = clips
            st.session_state.current_clip_index = 0 if clips else -1
            st.session_state.clip_name = ""
            st.session_state.clip_modified = False
            # Store the clips file path for this session
            st.session_state.current_clips_file = clips_file

            logger.info(
                f"Initialized session with {len(clips)} clips from {clips_file}"
            )
        elif (
            current_video
            and st.session_state.get("current_clips_file", None) != clips_file
        ):
            # If video changed, load clips for the new video
            clips = load_clips(clips_file)
            st.session_state.clips = clips
            st.session_state.current_clip_index = 0 if clips else -1
            st.session_state.clip_name = ""
            st.session_state.clip_modified = False
            # Update the clips file path for this session
            st.session_state.current_clips_file = clips_file

            logger.info(f"Loaded {len(clips)} clips for new video from {clips_file}")

        return True

    except Exception as e:
        logger.exception(f"Error initializing session clips: {str(e)}")
        return False


def save_session_clips(config_manager=None):
    """
    Save clips from session state to file

    Args:
        config_manager: ConfigManager instance

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get config manager from session state if not provided
        if not config_manager:
            config_manager = st.session_state.config_manager

        # Get current video path if available
        current_video = st.session_state.get("current_video", None)

        # Get clips file path based on current video
        clips_file = config_manager.get_clips_file_path(current_video)

        # Save clips to file
        if "clips" in st.session_state:
            success = save_clips(st.session_state.clips, clips_file)
            if success:
                st.session_state.clip_modified = False
                # Update the clips file path for this session
                st.session_state.current_clips_file = clips_file
                logger.info(
                    f"Saved {len(st.session_state.clips)} clips to {clips_file}"
                )
            return success

        return False

    except Exception as e:
        logger.exception(f"Error saving session clips: {str(e)}")
        return False


def add_clip(source_path, start_frame, end_frame, name=None, output_resolution="1080p"):
    """
    Add a new clip to the session

    Args:
        source_path: Path to the source video
        start_frame: Starting frame number
        end_frame: Ending frame number
        name: Name of the clip (optional)
        output_resolution: Target resolution for export

    Returns:
        The new clip object
    """
    try:
        # Get config manager from session state
        config_manager = st.session_state.get("config_manager", None)
        if not config_manager:
            logger.error("No config manager found in session state")
            return None

        # Convert source path to relative path using ConfigManager
        source_path_obj = Path(source_path)

        # If it's an absolute path, try to make it relative to the source directories
        if source_path_obj.is_absolute():
            try:
                relative_path = source_path_obj.relative_to(config_manager.source_raw)
                source_path = str(
                    Path(config_manager.config["directories"]["source"]["raw"])
                    / relative_path
                )
            except ValueError:
                try:
                    relative_path = source_path_obj.relative_to(
                        config_manager.source_calibrated
                    )
                    source_path = str(
                        Path(
                            config_manager.config["directories"]["source"]["calibrated"]
                        )
                        / relative_path
                    )
                except ValueError:
                    try:
                        relative_path = source_path_obj.relative_to(
                            config_manager.source_base
                        )
                        source_path = str(relative_path)
                    except ValueError:
                        # Keep as absolute path if we can't make it relative
                        source_path = str(source_path_obj)
        else:
            # Already relative, keep as is
            source_path = str(source_path_obj)

        # Get proxy path from config manager
        proxy_path = config_manager.get_proxy_path(source_path_obj, is_clip=False)

        # Convert proxy path to relative if possible
        if proxy_path and Path(proxy_path).is_absolute():
            try:
                relative_proxy = Path(proxy_path).relative_to(config_manager.proxy_base)
                proxy_path = str(relative_proxy)
            except ValueError:
                # Keep as is if we can't make it relative
                proxy_path = str(proxy_path)

        if os.path.exists(proxy_path):
            logger.info(f"Using relative proxy path: {proxy_path}")
        else:
            logger.info(f"Proxy path not found: {proxy_path}")
            proxy_path = None

        # Create a new clip with relative paths
        clip = Clip(
            name=name,
            source_path=source_path,
            proxy_path=proxy_path,
            start_frame=start_frame,
            end_frame=end_frame,
            output_resolution=output_resolution,
        )

        # Add to session state
        if "clips" not in st.session_state:
            st.session_state.clips = []

        st.session_state.clips.append(clip)
        st.session_state.current_clip_index = len(st.session_state.clips) - 1
        st.session_state.clip_modified = True

        logger.info(
            f"Added new clip: {clip.name} ({clip.start_frame} to {clip.end_frame}) with relative paths"
        )

        return clip

    except Exception as e:
        logger.exception(f"Error adding clip: {str(e)}")
        return None


def update_current_clip(
    start_frame=None,
    end_frame=None,
    name=None,
    crop_keyframes=None,
    crop_keyframes_proxy=None,
    output_resolution=None,
):
    """
    Update the current clip in the session

    Args:
        start_frame: New starting frame number (optional)
        end_frame: New ending frame number (optional)
        name: New name for the clip (optional)
        crop_keyframes: New crop keyframes dictionary (optional)
        crop_keyframes_proxy: New crop keyframes dictionary for proxy video (optional)
        output_resolution: New output resolution (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if there's a current clip
        if (
            "current_clip_index" not in st.session_state
            or st.session_state.current_clip_index < 0
        ):
            logger.warning("No current clip to update")
            return False

        # Get the current clip
        clip_index = st.session_state.current_clip_index
        if clip_index >= len(st.session_state.clips):
            logger.warning(f"Invalid clip index: {clip_index}")
            return False

        clip = st.session_state.clips[clip_index]

        # Update clip properties if provided
        if start_frame is not None:
            clip.start_frame = start_frame
        if end_frame is not None:
            clip.end_frame = end_frame
        if name is not None:
            clip.name = name
        if crop_keyframes is not None:
            clip.crop_keyframes = crop_keyframes
        if crop_keyframes_proxy is not None:
            clip.crop_keyframes_proxy = crop_keyframes_proxy
        if output_resolution is not None:
            clip.output_resolution = output_resolution

        # Update modification timestamp
        clip.update()

        # Mark as modified
        st.session_state.clip_modified = True

        logger.info(
            f"Updated clip: {clip.name} ({clip.start_frame} to {clip.end_frame})"
        )

        return True

    except Exception as e:
        logger.exception(f"Error updating clip: {str(e)}")
        return False


def delete_clip(clip_index=None):
    """
    Delete a clip from the session

    Args:
        clip_index: Index of the clip to delete (uses current clip if None)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use current clip index if not specified
        if clip_index is None:
            if (
                "current_clip_index" not in st.session_state
                or st.session_state.current_clip_index < 0
            ):
                logger.warning("No current clip to delete")
                return False
            clip_index = st.session_state.current_clip_index

        # Check if index is valid
        if clip_index >= len(st.session_state.clips):
            logger.warning(f"Invalid clip index: {clip_index}")
            return False

        # Get clip info for logging
        clip = st.session_state.clips[clip_index]

        # Remove the clip
        st.session_state.clips.pop(clip_index)

        # Update current clip index
        if st.session_state.current_clip_index >= len(st.session_state.clips):
            st.session_state.current_clip_index = len(st.session_state.clips) - 1

        # Mark as modified
        st.session_state.clip_modified = True

        logger.info(f"Deleted clip: {clip.name}")

        return True

    except Exception as e:
        logger.exception(f"Error deleting clip: {str(e)}")
        return False


def add_crop_keyframe(frame_number, crop_region, clip_index=None):
    """
    Add a crop keyframe to a clip and save changes

    Args:
        frame_number: Frame number for the keyframe
        crop_region: Tuple of (x, y, width, height) from the proxy resolution
        clip_index: Index of the clip (uses current clip if None)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use current clip index if not specified
        if clip_index is None:
            if (
                "current_clip_index" not in st.session_state
                or st.session_state.current_clip_index < 0
            ):
                logger.warning("No current clip to add keyframe to")
                return False
            clip_index = st.session_state.current_clip_index

        # Check if index is valid
        if clip_index >= len(st.session_state.clips):
            logger.warning(f"Invalid clip index: {clip_index}")
            return False

        # Get the clip
        clip = st.session_state.clips[clip_index]

        # Get video info for both source and proxy
        video_path = clip.source_path or clip.proxy_path
        if not video_path:
            logger.error("No valid video path found")
            return False

        # Get video info
        video_info = video_service.get_video_info(video_path)
        if not video_info:
            logger.error("Could not get video info")
            return False

        # Get target dimensions based on output resolution
        target_width, target_height = video_service.calculate_crop_dimensions(
            clip.output_resolution
        )

        # Get proxy settings for scaling calculation
        proxy_settings = st.session_state.config_manager.get_proxy_settings()
        proxy_width = proxy_settings["width"]
        proxy_height = int(proxy_width * (video_info["height"] / video_info["width"]))

        # Calculate scaling factor between source and proxy
        scaling_factor = proxy_width / video_info["width"]
        logger.info(f"Scaling factor: {scaling_factor}")

        # Get the crop region values from proxy resolution
        x, y, width, height = crop_region

        # Calculate source dimensions by scaling up from proxy
        source_x = int(x / scaling_factor)
        source_y = int(y / scaling_factor)
        source_width = target_width  # Use target width from output resolution
        source_height = target_height  # Use target height from output resolution

        # Calculate proxy dimensions by scaling down from source
        proxy_x = x  # Keep original x from proxy resolution
        proxy_y = y  # Keep original y from proxy resolution
        proxy_width = int(
            target_width * scaling_factor
        )  # Scale target width to proxy resolution
        proxy_height = int(
            target_height * scaling_factor
        )  # Scale target height to proxy resolution

        logger.info(f"Proxy dimensions: {proxy_width}x{proxy_height}")
        logger.info(f"Source dimensions: {source_width}x{source_height}")

        # Create crop regions for both resolutions
        proxy_crop = (proxy_x, proxy_y, proxy_width, proxy_height)
        source_crop = (source_x, source_y, source_width, source_height)

        logger.info(f"Proxy crop: {proxy_crop}")
        logger.info(f"Source crop: {source_crop}")

        # Add the keyframes - proxy resolution in proxy_keyframes, source resolution in crop_keyframes
        clip.crop_keyframes_proxy[str(frame_number)] = proxy_crop
        clip.crop_keyframes[str(frame_number)] = source_crop

        # Update modification timestamp
        clip.update()

        # Mark as modified
        st.session_state.clip_modified = True

        # Auto-save the changes
        success = save_session_clips()
        return success

    except Exception as e:
        logger.exception(f"Error adding crop keyframe: {str(e)}")
        return False


def remove_crop_keyframe(frame_number, clip_index=None):
    """
    Remove a crop keyframe from a clip

    Args:
        frame_number: Frame number for the keyframe to remove
        clip_index: Index of the clip (uses current clip if None)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use current clip index if not specified
        if clip_index is None:
            if (
                "current_clip_index" not in st.session_state
                or st.session_state.current_clip_index < 0
            ):
                logger.warning("No current clip to remove keyframe from")
                return False
            clip_index = st.session_state.current_clip_index

        # Check if index is valid
        if clip_index >= len(st.session_state.clips):
            logger.warning(f"Invalid clip index: {clip_index}")
            return False

        # Get the clip
        clip = st.session_state.clips[clip_index]

        # Convert frame number to string for dictionary key
        frame_key = str(frame_number)

        # Check if keyframe exists in either set
        keyframe_exists = False
        if frame_key in clip.crop_keyframes:
            del clip.crop_keyframes[frame_key]
            keyframe_exists = True
        if frame_key in clip.crop_keyframes_proxy:
            del clip.crop_keyframes_proxy[frame_key]
            keyframe_exists = True

        if not keyframe_exists:
            logger.warning(f"No keyframe at frame {frame_number} in clip {clip.name}")
            return False

        # Update modification timestamp
        clip.update()

        # Mark as modified
        st.session_state.clip_modified = True

        logger.info(
            f"Removed crop keyframe at frame {frame_number} from clip {clip.name}"
        )

        return True

    except Exception as e:
        logger.exception(f"Error removing crop keyframe: {str(e)}")
        return False


def get_current_clip():
    """
    Get the current clip from the session

    Returns:
        The current Clip object or None if no clip is selected
    """
    try:
        # Check if there's a current clip
        if (
            "current_clip_index" not in st.session_state
            or st.session_state.current_clip_index < 0
        ):
            return None

        # Get the current clip
        clip_index = st.session_state.current_clip_index
        if clip_index >= len(st.session_state.clips):
            return None

        return st.session_state.clips[clip_index]

    except Exception as e:
        logger.exception(f"Error getting current clip: {str(e)}")
        return None


def load_clips_from_file(clips_file):
    """
    Load clips directly from a JSON file without using session state.
    Used for displaying clips in the sidebar.

    Args:
        clips_file: Path to the clips JSON file

    Returns:
        List of Clip objects or empty list if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(clips_file):
            logger.info(f"Clips file not found: {clips_file}")
            return []

        # Read from file
        with open(clips_file, "r") as f:
            clips_data = json.load(f)

        # Convert dictionaries to Clip objects
        clips = [Clip.from_dict(data) for data in clips_data]

        logger.info(f"Loaded {len(clips)} clips directly from {clips_file}")
        return clips

    except Exception as e:
        logger.exception(f"Error loading clips from {clips_file}: {str(e)}")
        return []


def load_clip_into_state(clip):
    """
    Load clip data into session state when a clip is selected.
    This ensures all UI components reflect the selected clip's data.

    Args:
        clip: The Clip object to load into state
    """
    try:
        # Set current frame and slider to clip's start frame if not within clip bounds
        current_frame = st.session_state.get("current_frame", 0)
        if not (clip.start_frame <= current_frame <= clip.end_frame):
            st.session_state.current_frame = clip.start_frame
            st.session_state.clip_frame_slider = clip.start_frame

        # Load other clip properties into state
        st.session_state.clip_name = clip.name
        st.session_state.output_resolution = clip.output_resolution

        # Reset modification flag since we're loading fresh data
        st.session_state.clip_modified = False

        logger.info(
            f"Loaded clip data into state: {clip.name} (frames {clip.start_frame} to {clip.end_frame})"
        )
        return True

    except Exception as e:
        logger.exception(f"Error loading clip into state: {str(e)}")
        return False
