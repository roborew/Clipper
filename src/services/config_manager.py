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

        # Initialize proxy directory
        self.proxy_dir = Path(self.config["directories"]["proxy"]["base"])

        # Create necessary directories
        if self.config["export"]["create_missing_dirs"]:
            self.configs_dir.mkdir(parents=True, exist_ok=True)
            self.proxy_dir.mkdir(parents=True, exist_ok=True)

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

    def get_output_path(self, video_path: Path, clip_name: str) -> Path:
        """Generate output path for a clip"""
        relative_path = self.get_relative_source_path(video_path)

        if relative_path is None:
            # Fall back to using just the filename
            base_name = video_path.stem
            return self.clips_dir / f"{base_name}_{clip_name}.mp4"

        if self.config["export"]["preserve_structure"]:
            # Preserve folder structure
            output_dir = self.clips_dir / relative_path.parent
            if self.config["export"]["create_missing_dirs"]:
                output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / f"{Path(relative_path).stem}_{clip_name}.mp4"
        else:
            # Save directly in clips directory
            return self.clips_dir / f"{Path(relative_path).stem}_{clip_name}.mp4"

    def get_proxy_path(self, video_path: Path) -> Path:
        """Generate proxy video path for a source video"""
        relative_path = self.get_relative_source_path(video_path)

        if relative_path is None:
            # Fall back to using just the filename
            base_name = video_path.stem
            proxy_path = self.proxy_dir / f"{base_name}_proxy.mp4"
            # Ensure the proxy directory exists
            self.proxy_dir.mkdir(parents=True, exist_ok=True)
            return proxy_path

        # Preserve folder structure for proxies if configured
        if self.config["export"]["preserve_structure"]:
            # Create proxy directory with same structure as source
            proxy_dir = self.proxy_dir / relative_path.parent
            # Always ensure the directory exists
            proxy_dir.mkdir(parents=True, exist_ok=True)
            return proxy_dir / f"{Path(relative_path).stem}_proxy.mp4"
        else:
            # Save directly in proxy directory (flat structure)
            # Ensure the proxy directory exists
            self.proxy_dir.mkdir(parents=True, exist_ok=True)
            return self.proxy_dir / f"{Path(relative_path).stem}_proxy.mp4"

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
