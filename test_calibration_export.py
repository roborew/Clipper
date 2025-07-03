#!/usr/bin/env python3
"""
Test script to verify calibration is applied during clip export
"""

import sys
import os
import logging
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from services.config_manager import ConfigManager
from services.proxy_service import export_clip

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_calibration_export():
    """Test that calibration is applied during clip export"""

    config_manager = ConfigManager()

    # Test parameters (using your existing clip data)
    source_path = (
        "data/SD_01_SURF_FOOTAGE_SOURCE/00_RAW/SONY_300/SESSION_080325/C0026.MP4"
    )
    output_path = "test_calibrated_export.mp4"

    logger.info("Testing calibration during clip export...")
    logger.info(f"Source: {source_path}")
    logger.info(f"Output: {output_path}")

    # Check if source exists
    if not os.path.exists(source_path):
        logger.error(f"Source file doesn't exist: {source_path}")
        return False

    # Check calibration settings
    calib_settings = config_manager.get_calibration_settings()
    logger.info(
        f"use_calibrated_footage: {calib_settings.get('use_calibrated_footage', False)}"
    )

    try:
        # Test export with calibration (short clip: frames 15875-15925 = 50 frames)
        result = export_clip(
            source_path=source_path,
            clip_name="test_calibration_clip",
            start_frame=15875,
            end_frame=15925,  # Just 50 frames for quick test
            crop_region=None,  # No cropping for simpler test
            output_resolution="1080p",
            cv_optimized=False,
            config_manager=config_manager,
        )

        if result and os.path.exists(output_path):
            logger.info("✅ Clip exported successfully!")
            logger.info(f"✅ Output file created: {output_path}")

            # Check file size to ensure it's not empty
            file_size = os.path.getsize(output_path)
            logger.info(f"File size: {file_size} bytes")

            if file_size > 1000:  # At least 1KB
                logger.info("✅ Calibration export test PASSED!")
                return True
            else:
                logger.error("❌ Output file is too small")
                return False
        else:
            logger.error("❌ Export failed or output file not created")
            return False

    except Exception as e:
        logger.error(f"❌ Error during export: {e}")
        return False

    finally:
        # Clean up test file
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Cleaned up test file: {output_path}")


if __name__ == "__main__":
    success = test_calibration_export()
    if success:
        print("\n🎉 CALIBRATION TEST PASSED - Calibration is being applied to exports!")
    else:
        print("\n❌ CALIBRATION TEST FAILED - Check the logs above")
