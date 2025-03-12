import yaml
from pathlib import Path
from typing import List, Optional
import os


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

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Return default configuration"""
        return {
            "directories": {
                "source": {
                    "base": "data/source",
                    "calibrated": "02_CALIBRATED_FOOTAGE",
                },
                "output": {
                    "base": "data/prept",
                    "clips": "03_CLIPPED",
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
        """Get all video files from the calibrated footage directory"""
        if not self.source_calibrated.exists():
            return []

        video_files = []
        extensions = self.config["patterns"]["video_extensions"]

        # Walk through all subdirectories
        for root, _, files in os.walk(self.source_calibrated):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                if any(
                    file_path.name.lower().endswith(ext.lower()) for ext in extensions
                ):
                    video_files.append(file_path)

        return sorted(video_files)

    def get_relative_source_path(self, file_path: Path) -> Optional[Path]:
        """Get path relative to source calibrated directory"""
        try:
            return file_path.relative_to(self.source_calibrated)
        except ValueError:
            return None

    def get_config_path(self, video_path: Path) -> Path:
        """Generate config file path for a video"""
        relative_path = self.get_relative_source_path(video_path)
        if relative_path is None:
            raise ValueError(
                "Video path must be within the calibrated footage directory"
            )

        # Convert path to filename-safe format
        config_name = str(relative_path).replace("/", "_").replace("\\", "_")
        base_name = Path(config_name).stem
        return self.configs_dir / f"{base_name}.json"

    def get_output_path(self, video_path: Path, clip_name: str) -> Path:
        """Generate output path for a clip"""
        relative_path = self.get_relative_source_path(video_path)
        if relative_path is None:
            raise ValueError(
                "Video path must be within the calibrated footage directory"
            )

        if self.config["export"]["preserve_structure"]:
            # Preserve session/camera folder structure
            output_dir = self.clips_dir / relative_path.parent
            if self.config["export"]["create_missing_dirs"]:
                output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / f"{Path(relative_path).stem}_{clip_name}.mp4"
        else:
            # Save directly in clips directory
            return self.clips_dir / f"{Path(relative_path).stem}_{clip_name}.mp4"

    def get_proxy_path(self, video_path: Path) -> Path:
        """Generate proxy video path for a source video"""
        # Create a clean filename for the proxy
        base_name = os.path.splitext(os.path.basename(str(video_path)))[0]
        return self.proxy_dir / f"{base_name}_proxy.mp4"

    def is_proxy_enabled(self) -> bool:
        """Check if proxy video creation is enabled"""
        return self.config["proxy"]["enabled"]

    def get_proxy_settings(self) -> dict:
        """Get proxy video settings"""
        return self.config["proxy"]

    def get_video_extensions(self) -> List[str]:
        """Get list of supported video extensions"""
        return self.config["patterns"]["video_extensions"]
