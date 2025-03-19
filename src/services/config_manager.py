"""
Configuration management services for the Clipper application.
"""

import yaml
from pathlib import Path
from typing import List, Optional
import os
import logging

logger = logging.getLogger("clipper.config")


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

        # Initialize paths
        self.source_base = Path(self.config["directories"]["source"]["base"])
        self.source_calibrated = (
            self.source_base / self.config["directories"]["source"]["calibrated"]
        )

        self.output_base = Path(self.config["directories"]["output"]["base"])
        self.clips_dir = (
            self.output_base / self.config["directories"]["output"]["clips"]
        )
        self.configs_dir = (
            self.clips_dir / self.config["directories"]["output"]["configs"]
        )

        # Initialize proxy directories
        self.proxy_base = Path(self.config["directories"]["proxy"]["base"])
        self.proxy_raw_dir = (
            self.proxy_base / self.config["directories"]["proxy"]["raw"]
        )
        self.proxy_clipped_dir = (
            self.proxy_base / self.config["directories"]["proxy"]["clipped"]
        )

        # Create necessary directories
        if self.config["export"]["create_missing_dirs"]:
            try:
                # Check if output_base exists and is a directory
                if self.output_base.exists() and not self.output_base.is_dir():
                    logger.error(
                        f"Path exists but is not a directory: {self.output_base}"
                    )
                    logger.error("Please check your config and fix this path issue")
                else:
                    # Try to create each directory with better error handling
                    self._create_directory_safely(self.configs_dir, "config directory")
                    self._create_directory_safely(
                        self.proxy_raw_dir, "proxy raw directory"
                    )
                    self._create_directory_safely(
                        self.proxy_clipped_dir, "proxy clipped directory"
                    )
            except Exception as e:
                logger.error(f"Error creating application directories: {str(e)}")
                logger.error(
                    "Application will continue but some features may not work correctly"
                )

        logger.info(f"ConfigManager initialized with config from {config_path}")

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Successfully loaded config from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Return default configuration"""
        return {
            "directories": {
                "source": {
                    "base": "data",
                    "calibrated": "",
                },
                "output": {
                    "base": ".",
                    "clips": "exports",
                    "configs": "_configs",
                },
                "proxy": {
                    "base": "proxy_videos",
                    "raw": "RAW",
                    "clipped": "CLIPPED",
                },
            },
            "patterns": {"video_extensions": [".mp4", ".avi", ".mov", ".mkv"]},
            "export": {"preserve_structure": True, "create_missing_dirs": True},
            "proxy": {
                "enabled": True,
                "width": 960,
                "quality": 28,
                "audio_bitrate": "128k",
            },
        }

    def get_video_files(self) -> List[Path]:
        """Get all video files from the configured video directories"""
        video_files = []
        extensions = self.config["patterns"]["video_extensions"]

        # If source_calibrated is empty, use source_base directly
        search_dir = (
            self.source_calibrated if self.source_calibrated.name else self.source_base
        )

        if not search_dir.exists():
            logger.warning(f"Video directory does not exist: {search_dir}")
            return []

        # Walk through all subdirectories
        for root, _, files in os.walk(search_dir):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                if any(
                    file_path.name.lower().endswith(ext.lower()) for ext in extensions
                ):
                    video_files.append(file_path)

        logger.info(f"Found {len(video_files)} video files")
        return sorted(video_files)

    def get_relative_source_path(self, file_path: Path) -> Optional[Path]:
        """Get path relative to source directory"""
        search_dir = (
            self.source_calibrated if self.source_calibrated.name else self.source_base
        )
        try:
            return file_path.relative_to(search_dir)
        except ValueError:
            logger.warning(f"File {file_path} is not within the source directory")
            return None

    def get_config_path(self, video_path: Path) -> Path:
        """Generate config file path for a video"""
        relative_path = self.get_relative_source_path(video_path)
        if relative_path is None:
            logger.warning(f"Using filename only for config path: {video_path.name}")
            # Fall back to using just the filename
            base_name = video_path.stem
        else:
            # Convert path to filename-safe format
            config_name = str(relative_path).replace("/", "_").replace("\\", "_")
            base_name = Path(config_name).stem

        return self.configs_dir / f"{base_name}.json"

    def get_output_path(
        self, video_path: Path, clip_name: str, codec_type=None, *args, **kwargs
    ) -> Path:
        """Generate output path for a clip

        Args:
            video_path: Path to the source video
            clip_name: Name of the clip
            codec_type: Type of codec (h264 or ffv1), defaults to h264 if not specified
            *args: Additional positional arguments (for backward compatibility)
            **kwargs: Additional keyword arguments (for backward compatibility)

        Returns:
            Path object representing the output path
        """
        # Handle legacy calls that might pass more arguments
        if args or kwargs:
            logger.warning(
                "get_output_path() called with extra arguments that will be ignored"
            )

        # Set default codec type if none provided
        if codec_type is None:
            codec_type = "h264"

        relative_path = self.get_relative_source_path(video_path)

        # Determine file extension based on codec type
        extension = ".mkv" if codec_type.lower() == "ffv1" else ".mp4"

        # Create codec subdirectory path
        clips_with_codec = self.clips_dir / codec_type.lower()

        if relative_path is None:
            # Fall back to using just the filename
            base_name = video_path.stem
            output_path = clips_with_codec / f"{base_name}_{clip_name}{extension}"
            # Ensure the directory exists
            clips_with_codec.mkdir(parents=True, exist_ok=True)
            return output_path

        if self.config["export"]["preserve_structure"]:
            # Preserve folder structure
            output_dir = clips_with_codec / relative_path.parent
            if self.config["export"]["create_missing_dirs"]:
                output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / f"{Path(relative_path).stem}_{clip_name}{extension}"
        else:
            # Save directly in clips directory
            # Ensure the directory exists
            clips_with_codec.mkdir(parents=True, exist_ok=True)
            return (
                clips_with_codec / f"{Path(relative_path).stem}_{clip_name}{extension}"
            )

    def get_proxy_path(self, video_path: Path, is_clip: bool = False) -> Path:
        """Generate proxy video path for a source video or clip

        Args:
            video_path: Path to the source video
            is_clip: Whether this is a clip preview (True) or raw video proxy (False)

        Returns:
            Path to the proxy video
        """
        relative_path = self.get_relative_source_path(video_path)

        # Determine base proxy directory based on whether this is a clip or raw video
        proxy_base = self.proxy_clipped_dir if is_clip else self.proxy_raw_dir

        if relative_path is None:
            # Fall back to using just the filename
            base_name = video_path.stem
            proxy_path = proxy_base / f"{base_name}_proxy.mp4"
            # Ensure the proxy directory exists
            proxy_base.mkdir(parents=True, exist_ok=True)
            return proxy_path

        # Preserve folder structure for proxies if configured
        if self.config["export"]["preserve_structure"]:
            # Create proxy directory with same structure as source
            proxy_dir = proxy_base / relative_path.parent
            # Always ensure the directory exists
            proxy_dir.mkdir(parents=True, exist_ok=True)
            return proxy_dir / f"{Path(relative_path).stem}_proxy.mp4"
        else:
            # Save directly in proxy directory (flat structure)
            # Ensure the proxy directory exists
            proxy_base.mkdir(parents=True, exist_ok=True)
            return proxy_base / f"{Path(relative_path).stem}_proxy.mp4"

    def get_clip_preview_path(self, video_path: Path, clip_name: str) -> Path:
        """Generate path for a clip preview

        Args:
            video_path: Path to the source video
            clip_name: Name of the clip

        Returns:
            Path to the clip preview video
        """
        # First check if the provided path is a proxy video
        if str(video_path).startswith(str(self.proxy_raw_dir)):
            try:
                # Extract the relative path from the proxy path
                relative_path = Path(
                    str(video_path).replace(str(self.proxy_raw_dir), "").lstrip("/")
                )
                # Remove the _proxy.mp4 suffix if present
                if relative_path.stem.endswith("_proxy"):
                    relative_path = relative_path.with_stem(relative_path.stem[:-6])
            except Exception as e:
                logger.warning(f"Could not extract relative path from proxy path: {e}")
                relative_path = None
        else:
            # Get relative path from source directory
            relative_path = self.get_relative_source_path(video_path)

        if relative_path is None:
            # Fall back to using just the filename
            base_name = video_path.stem
            if base_name.endswith("_proxy"):
                base_name = base_name[:-6]
            preview_path = (
                self.proxy_clipped_dir / f"{base_name}_{clip_name}_preview.mp4"
            )
            self.proxy_clipped_dir.mkdir(parents=True, exist_ok=True)
            return preview_path

        if self.config["export"]["preserve_structure"]:
            # Create preview directory with same structure as source
            preview_dir = self.proxy_clipped_dir / relative_path.parent
            preview_dir.mkdir(parents=True, exist_ok=True)
            return preview_dir / f"{Path(relative_path).stem}_{clip_name}_preview.mp4"
        else:
            # Save directly in clipped directory
            self.proxy_clipped_dir.mkdir(parents=True, exist_ok=True)
            return (
                self.proxy_clipped_dir
                / f"{Path(relative_path).stem}_{clip_name}_preview.mp4"
            )

    def is_proxy_enabled(self) -> bool:
        """Check if proxy video creation is enabled"""
        return self.config["proxy"]["enabled"]

    def get_proxy_settings(self) -> dict:
        """Get proxy video settings"""
        return self.config["proxy"]

    def get_video_extensions(self) -> List[str]:
        """Get list of supported video extensions"""
        return self.config["patterns"]["video_extensions"]

    def get_clips_file_path(self, video_path=None) -> Path:
        """
        Get path to the clips file for a specific video

        Args:
            video_path: Path to the video file (optional)

        Returns:
            Path to the clips file
        """
        if video_path is None:
            # If no video path is provided, return a temporary clips file
            return self.configs_dir / "temp_clips.json"

        # Get the relative path to preserve camera folder structure
        relative_path = self.get_relative_source_path(video_path)

        if relative_path is None:
            # Fall back to using just the filename
            base_name = Path(video_path).stem
            return self.configs_dir / f"{base_name}_clips.json"

        # Create directory structure matching the source
        if self.config["export"]["preserve_structure"]:
            # Preserve camera folder structure
            config_dir = self.configs_dir / relative_path.parent
            if self.config["export"]["create_missing_dirs"]:
                config_dir.mkdir(parents=True, exist_ok=True)
            return config_dir / f"{Path(relative_path).stem}_clips.json"
        else:
            # Save in configs directory with path-based name
            config_name = str(relative_path).replace("/", "_").replace("\\", "_")
            return self.configs_dir / f"{config_name}_clips.json"

    def _create_directory_safely(self, directory_path, directory_name):
        """Create a directory with proper error handling"""
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            logger.debug(
                f"Successfully created or verified {directory_name}: {directory_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to create {directory_name} at {directory_path}: {str(e)}"
            )
            logger.error(f"Please check permissions or create this directory manually")
            # Continue without raising, but log the issue
