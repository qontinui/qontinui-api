#!/usr/bin/env python3
"""Test script for component merging in State Discovery."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time

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

        # Add some stable UI elements
        # Button 1 (stable across all)
        cv2.rectangle(img, (100, 100), (250, 140), (100, 150, 200), -1)
        cv2.putText(img, "Button 1", (110, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Button 2 (stable across all)
        cv2.rectangle(img, (300, 100), (450, 140), (100, 200, 150), -1)
        cv2.putText(img, "Button 2", (310, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Icon (stable)
        cv2.rectangle(img, (500, 100), (564, 164), (200, 100, 100), -1)

        # Add some varying elements
        # Moving element
        x_offset = i * 50
        cv2.circle(img, (200 + x_offset, 300), 30, (255, 255, 0), -1)

        # Changing text
        cv2.putText(
            img, f"Frame {i+1}", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        screenshots.append(img)

    return screenshots


def test_with_merging():
    """Test analysis with component merging enabled."""
    logger.info("=" * 60)
    logger.info("Testing WITH component merging")
    logger.info("=" * 60)

    # Create config with merging enabled
    config = AnalysisConfig()
    config.use_component_merging = True
    config.merge_gap = 8
    config.min_component_pixels = 50
    config.variance_threshold = 10.0

    analyzer = PixelStabilityAnalyzer(config)
    screenshots = create_test_screenshots()

    logger.info(f"Analyzing {len(screenshots)} test screenshots")

    start = time.time()
    try:
        result = analyzer.analyze_screenshots(screenshots)
        elapsed = time.time() - start

        logger.info("\nResults WITH merging:")
        logger.info(f"  Time: {elapsed:.2f} seconds")
        logger.info(f"  State images found: {len(result.state_images)}")
        logger.info(f"  States found: {len(result.states)}")

        # Show details of found regions
        for i, si in enumerate(result.state_images[:10]):  # Show first 10
            logger.info(
                f"  StateImage {i}: pos=({si.x},{si.y}), size=({si.x2-si.x}x{si.y2-si.y}), "
                f"screenshots={len(si.screenshot_ids)}"
            )

        return result, elapsed

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return None, 0


def test_without_merging():
    """Test analysis without component merging (original method)."""
    logger.info("=" * 60)
    logger.info("Testing WITHOUT component merging")
    logger.info("=" * 60)

    # Create config with merging disabled
    config = AnalysisConfig()
    config.use_component_merging = False
    config.variance_threshold = 10.0

    analyzer = PixelStabilityAnalyzer(config)
    screenshots = create_test_screenshots()

    logger.info(f"Analyzing {len(screenshots)} test screenshots")

    start = time.time()
    try:
        # Add timeout protection
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Analysis took too long")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout

        try:
            result = analyzer.analyze_screenshots(screenshots)
            elapsed = time.time() - start
            signal.alarm(0)  # Cancel alarm
        except TimeoutError:
            logger.error("Analysis timed out after 10 seconds")
            return None, 10

        logger.info("\nResults WITHOUT merging:")
        logger.info(f"  Time: {elapsed:.2f} seconds")
        logger.info(f"  State images found: {len(result.state_images)}")
        logger.info(f"  States found: {len(result.states)}")

        # Show details of found regions
        for i, si in enumerate(result.state_images[:10]):  # Show first 10
            logger.info(
                f"  StateImage {i}: pos=({si.x},{si.y}), size=({si.x2-si.x}x{si.y2-si.y}), "
                f"screenshots={len(si.screenshot_ids)}"
            )

        return result, elapsed

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return None, 0


def compare_methods():
    """Compare performance with and without merging."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPONENT MERGING COMPARISON TEST")
    logger.info("=" * 60 + "\n")

    # Test with merging
    result_merged, time_merged = test_with_merging()

    print("\n")

    # Test without merging
    result_original, time_original = test_without_merging()

    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    if result_merged and result_original:
        logger.info(
            f"Time improvement: {time_original:.2f}s -> {time_merged:.2f}s "
            f"({(time_original/time_merged):.1f}x faster)"
        )
        logger.info(
            f"Region reduction: {len(result_original.state_images)} -> "
            f"{len(result_merged.state_images)} regions "
            f"({(1 - len(result_merged.state_images)/len(result_original.state_images))*100:.1f}% reduction)"
        )
    elif result_merged and not result_original:
        logger.info("Original method failed/timed out, merged method succeeded!")
        logger.info(
            f"Merged method found {len(result_merged.state_images)} regions in {time_merged:.2f}s"
        )
    else:
        logger.error("Both methods failed")


def test_merge_parameters():
    """Test different merge parameters to find optimal values."""
    from qontinui.src.qontinui.discovery.pixel_analysis.merge_components import (
        test_merge_parameters,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TESTING MERGE PARAMETERS")
    logger.info("=" * 60 + "\n")

    # Create a test stability map
    screenshots = create_test_screenshots()
    config = AnalysisConfig()
    analyzer = PixelStabilityAnalyzer(config)

    # Create stability map
    stability_map = analyzer.create_stability_map(screenshots)

    # Test different parameters
    results = test_merge_parameters(
        stability_map, gap_values=[5, 8, 10, 15], min_pixel_values=[30, 50, 100]
    )

    # Find best parameters
    best = min(
        results.items(),
        key=lambda x: x[1]["num_regions"] if x[1]["num_regions"] > 0 else float("inf"),
    )
    logger.info(f"\nBest parameters: {best[0]}")
    logger.info(f"  Result: {best[1]}")


if __name__ == "__main__":
    # Run comparison
    compare_methods()

    # Test parameters
    print("\n")
    test_merge_parameters()
