#!/usr/bin/env python3
"""Test script for region selection in State Discovery."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import cv2
import numpy as np

from qontinui.discovery import PixelStabilityAnalyzer
from qontinui.discovery.models import AnalysisConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_screenshots():
    """Create simple test screenshots with UI-like elements."""
    screenshots = []

    for i in range(3):
        # Create blank image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 50  # Dark gray background

        # Add stable elements in different regions
        # Top-left region (100x100)
        cv2.rectangle(img, (50, 50), (150, 150), (100, 150, 200), -1)
        cv2.putText(img, "TL", (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Bottom-right region (100x100)
        cv2.rectangle(img, (600, 400), (700, 500), (100, 200, 150), -1)
        cv2.putText(img, "BR", (625, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Center region - moving element
        x_offset = i * 30
        cv2.circle(img, (400 + x_offset, 300), 30, (255, 255, 0), -1)

        screenshots.append(img)

    return screenshots


def test_full_screen_analysis():
    """Test analysis with full screen."""
    logger.info("=" * 60)
    logger.info("Testing FULL SCREEN analysis")
    logger.info("=" * 60)

    config = AnalysisConfig()
    config.use_component_merging = True
    config.merge_gap = 8
    config.min_component_pixels = 50
    config.variance_threshold = 10.0

    analyzer = PixelStabilityAnalyzer(config)
    screenshots = create_test_screenshots()

    logger.info(f"Analyzing {len(screenshots)} full screenshots (800x600)")

    import time

    start = time.time()
    result = analyzer.analyze_screenshots(screenshots)
    elapsed = time.time() - start

    logger.info("Results:")
    logger.info(f"  Time: {elapsed:.2f} seconds")
    logger.info(f"  State images found: {len(result.state_images)}")
    logger.info(f"  States found: {len(result.states)}")

    # Show found regions
    for i, si in enumerate(result.state_images[:5]):
        logger.info(f"  StateImage {i}: pos=({si.x},{si.y}), size=({si.x2-si.x}x{si.y2-si.y})")

    return result, elapsed


def test_region_analysis():
    """Test analysis with selected region (top-left corner)."""
    logger.info("=" * 60)
    logger.info("Testing REGION analysis (top-left 200x200)")
    logger.info("=" * 60)

    config = AnalysisConfig()
    config.use_component_merging = True
    config.merge_gap = 8
    config.min_component_pixels = 50
    config.variance_threshold = 10.0

    analyzer = PixelStabilityAnalyzer(config)
    screenshots = create_test_screenshots()

    # Crop to top-left region (200x200)
    region = {"x": 0, "y": 0, "width": 200, "height": 200}
    cropped_screenshots = []
    for screenshot in screenshots:
        cropped = screenshot[
            region["y"] : region["y"] + region["height"],
            region["x"] : region["x"] + region["width"],
        ]
        cropped_screenshots.append(cropped)

    logger.info(f"Analyzing {len(cropped_screenshots)} cropped screenshots (200x200)")

    import time

    start = time.time()
    result = analyzer.analyze_screenshots(cropped_screenshots)
    elapsed = time.time() - start

    logger.info("Results:")
    logger.info(f"  Time: {elapsed:.2f} seconds")
    logger.info(f"  State images found: {len(result.state_images)}")
    logger.info(f"  States found: {len(result.states)}")

    # Adjust positions back to full screen coordinates
    for si in result.state_images:
        si.x += region["x"]
        si.y += region["y"]
        si.x2 += region["x"]
        si.y2 += region["y"]

    # Show found regions (with adjusted coordinates)
    for i, si in enumerate(result.state_images[:5]):
        logger.info(f"  StateImage {i}: pos=({si.x},{si.y}), size=({si.x2-si.x}x{si.y2-si.y})")

    return result, elapsed


def compare_results():
    """Compare full screen vs region analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("REGION SELECTION COMPARISON TEST")
    logger.info("=" * 60 + "\n")

    # Test full screen
    result_full, time_full = test_full_screen_analysis()

    print("\n")

    # Test region
    result_region, time_region = test_region_analysis()

    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    logger.info(
        f"Time improvement: {time_full:.2f}s -> {time_region:.2f}s "
        f"({(time_full/time_region):.1f}x faster)"
    )

    logger.info(
        f"Processing area: 800x600 (480,000 pixels) -> 200x200 (40,000 pixels) "
        f"({(40000/480000)*100:.1f}% of original)"
    )

    logger.info(
        f"Regions found: Full={len(result_full.state_images)}, "
        f"Region={len(result_region.state_images)}"
    )


if __name__ == "__main__":
    compare_results()
