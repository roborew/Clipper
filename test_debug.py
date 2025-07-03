#!/usr/bin/env python3
"""Debug script to test processing pipeline step by step"""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.config_manager import ConfigManager
from src.services import clip_service
from src.utils.multi_crop import get_source_video_fps


def test_fps_detection():
    """Test FPS detection on source video"""
    print("=== Testing FPS Detection ===")
    source_path = Path(
        "data/SD_01_SURF_FOOTAGE_SOURCE/00_RAW/SONY_300/SESSION_080325/C0001.MP4"
    )

    if not source_path.exists():
        print(f"❌ Source video not found: {source_path}")
        return False

    fps = get_source_video_fps(source_path)
    print(f"✅ Detected FPS: {fps}")
    return True


def test_config():
    """Test config loading"""
    print("=== Testing Config ===")
    try:
        config_manager = ConfigManager()
        print(f"✅ Config loaded: {config_manager.clips_dir}")
        return config_manager
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return None


def test_clip_loading():
    """Test loading clips"""
    print("=== Testing Clip Loading ===")
    try:
        config_manager = ConfigManager()
        clips = clip_service.get_all_pending_clips(config_manager)

        # Find a SONY_300 clip
        sony_clip = None
        for clip in clips:
            if hasattr(clip, "source_path") and "SONY_300" in str(clip.source_path):
                sony_clip = clip
                break

        if sony_clip:
            print(f"✅ Found SONY_300 clip: {sony_clip.name}")
            print(f"   Start frame: {sony_clip.start_frame}")
            print(f"   End frame: {sony_clip.end_frame}")
            print(f"   Source path: {sony_clip.source_path}")
            return sony_clip
        else:
            print("❌ No SONY_300 clips found")
            return None
    except Exception as e:
        print(f"❌ Clip loading failed: {e}")
        return None


if __name__ == "__main__":
    print("🔧 Debug Testing")

    # Test 1: FPS detection
    test_fps_detection()

    # Test 2: Config loading
    config_manager = test_config()
    if not config_manager:
        exit(1)

    # Test 3: Clip loading
    clip = test_clip_loading()
    if not clip:
        exit(1)

    print("✅ All basic tests passed!")
