#!/usr/bin/env python3
"""Debug script for testing 125x125 region analysis."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import traceback

import cv2
import numpy as np

from qontinui.discovery import PixelStabilityAnalyzer
from qontinui.discovery.models import AnalysisConfig

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TimingAnalyzer:
    """Helper class to track timing of different operations."""

    def __init__(self):
        self.timings = []
        self.start_time = None

    def start(self, operation):
        self.start_time = time.time()
        logger.info(f"TIMING: Starting {operation}")

    def end(self, operation):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.timings.append((operation, elapsed))
            logger.info(f"TIMING: {operation} took {elapsed:.3f} seconds")
            self.start_time = None

    def summary(self):
        logger.info("=" * 60)
        logger.info("TIMING SUMMARY:")
        logger.info("=" * 60)
        total = sum(t[1] for t in self.timings)
        for op, elapsed in self.timings:
            pct = (elapsed / total * 100) if total > 0 else 0
            logger.info(f"  {op}: {elapsed:.3f}s ({pct:.1f}%)")
        logger.info(f"  TOTAL: {total:.3f}s")


def create_test_screenshots_125x125():
    """Create simple test screenshots of size 125x125."""
    screenshots = []

    for i in range(3):
        # Create small 125x125 image
        img = np.ones((125, 125, 3), dtype=np.uint8) * 50  # Dark gray background

        # Add a small stable element (20x20 button)
        cv2.rectangle(img, (20, 20), (60, 40), (100, 150, 200), -1)
        cv2.putText(img, "BTN", (25, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add a moving element
        x_offset = i * 10
        cv2.circle(img, (70 + x_offset, 70), 8, (255, 255, 0), -1)

        screenshots.append(img)

    return screenshots


def debug_analysis_steps():
    """Debug each step of the analysis process."""
    timer = TimingAnalyzer()

    logger.info("=" * 60)
    logger.info("DEBUGGING 125x125 REGION ANALYSIS")
    logger.info("=" * 60)

    # Create test data
    timer.start("Creating test screenshots")
    screenshots = create_test_screenshots_125x125()
    timer.end("Creating test screenshots")

    logger.info(f"Created {len(screenshots)} screenshots of size {screenshots[0].shape}")

    # Create analyzer with debugging enabled
    config = AnalysisConfig()
    config.use_component_merging = True
    config.merge_gap = 8
    config.min_component_pixels = 10  # Lower threshold for small image
    config.variance_threshold = 10.0

    logger.info(
        f"Config: merge={config.use_component_merging}, gap={config.merge_gap}, min_pixels={config.min_component_pixels}"
    )

    analyzer = PixelStabilityAnalyzer(config)

    try:
        # Step 1: Create stability map
        timer.start("create_stability_map")
        stability_map = analyzer.create_stability_map(screenshots)
        timer.end("create_stability_map")

        stable_pixels = np.sum(stability_map)
        total_pixels = stability_map.size
        logger.info(
            f"Stability map: {stable_pixels}/{total_pixels} stable pixels ({stable_pixels/total_pixels*100:.1f}%)"
        )

        # Step 2: Extract stable regions (THIS IS USUALLY THE BOTTLENECK)
        timer.start("extract_stable_regions")

        # Add more detailed logging inside
        from qontinui.discovery.pixel_analysis.merge_components import merge_nearby_components

        # Test component merging directly
        timer.start("merge_nearby_components")
        merged_regions = merge_nearby_components(
            stability_map,
            max_gap=8,
            min_pixels=10,
            min_region_size=(10, 10),
            max_region_size=(100, 100),
        )
        timer.end("merge_nearby_components")

        logger.info(f"Found {len(merged_regions)} merged regions")

        # Continue with regular extraction
        stable_regions = analyzer.extract_stable_regions(stability_map, screenshots[0])
        timer.end("extract_stable_regions")

        logger.info(f"Extracted {len(stable_regions)} stable regions")

        # Step 3: Create state images
        timer.start("create_state_images")
        state_images = analyzer.create_state_images(stable_regions, screenshots)
        timer.end("create_state_images")

        logger.info(f"Created {len(state_images)} state images")

        # Step 4: Group into states
        timer.start("group_by_cooccurrence")
        if config.enable_cooccurrence_analysis:
            states = analyzer.group_by_cooccurrence(state_images, screenshots)
        else:
            states = []
        timer.end("group_by_cooccurrence")

        logger.info(f"Grouped into {len(states)} states")

        # Show timing summary
        timer.summary()

        return True

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        timer.summary()
        return False


def test_grid_vs_components():
    """Compare grid-based vs component-based extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING GRID VS COMPONENT EXTRACTION")
    logger.info("=" * 60)

    screenshots = create_test_screenshots_125x125()

    # Test 1: Grid-based (old method)
    logger.info("\n--- GRID-BASED METHOD ---")
    config1 = AnalysisConfig()
    config1.use_component_merging = False  # This should use grid

    analyzer1 = PixelStabilityAnalyzer(config1)

    start = time.time()
    try:
        # Just test the extraction step
        stability_map = analyzer1.create_stability_map(screenshots)
        regions = analyzer1.extract_stable_regions(stability_map, screenshots[0])
        elapsed1 = time.time() - start
        logger.info(f"Grid method: {len(regions)} regions in {elapsed1:.3f}s")
    except Exception as e:
        logger.error(f"Grid method failed: {e}")
        elapsed1 = time.time() - start

    # Test 2: Component-based (new method)
    logger.info("\n--- COMPONENT-BASED METHOD ---")
    config2 = AnalysisConfig()
    config2.use_component_merging = True
    config2.merge_gap = 8
    config2.min_component_pixels = 10

    analyzer2 = PixelStabilityAnalyzer(config2)

    start = time.time()
    try:
        stability_map = analyzer2.create_stability_map(screenshots)
        regions = analyzer2.extract_stable_regions(stability_map, screenshots[0])
        elapsed2 = time.time() - start
        logger.info(f"Component method: {len(regions)} regions in {elapsed2:.3f}s")
    except Exception as e:
        logger.error(f"Component method failed: {e}")
        elapsed2 = time.time() - start

    # Compare
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON:")
    logger.info(f"Grid method: {elapsed1:.3f}s")
    logger.info(f"Component method: {elapsed2:.3f}s")
    if elapsed1 > 0 and elapsed2 > 0:
        logger.info(f"Speed difference: {elapsed1/elapsed2:.1f}x")


def test_with_timeout():
    """Test with a timeout to see where it hangs."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Analysis timed out!")

    logger.info("\n" + "=" * 60)
    logger.info("TESTING WITH 5 SECOND TIMEOUT")
    logger.info("=" * 60)

    screenshots = create_test_screenshots_125x125()
    config = AnalysisConfig()
    config.use_component_merging = True
    analyzer = PixelStabilityAnalyzer(config)

    # Set 5 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

    try:
        result = analyzer.analyze_screenshots(screenshots)
        signal.alarm(0)  # Cancel alarm
        logger.info(f"SUCCESS: Analysis completed with {len(result.state_images)} regions")
    except TimeoutError:
        logger.error("TIMEOUT: Analysis took more than 5 seconds!")
        # Try to identify where it hung
        logger.error("Check the last operation logged before timeout")
    except Exception as e:
        signal.alarm(0)
        logger.error(f"ERROR: {e}")


if __name__ == "__main__":
    # Run all tests
    logger.info("Starting debug tests for 125x125 region analysis...")

    # Test 1: Debug each step
    logger.info("\n\nTEST 1: Debug analysis steps")
    debug_analysis_steps()

    # Test 2: Compare methods
    logger.info("\n\nTEST 2: Compare grid vs component methods")
    test_grid_vs_components()

    # Test 3: Test with timeout
    logger.info("\n\nTEST 3: Test with timeout")
    test_with_timeout()

    logger.info("\n\nAll tests completed!")
