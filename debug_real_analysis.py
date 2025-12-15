#!/usr/bin/env python3
"""Debug script for real-world analysis issues."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import traceback

import cv2
import numpy as np
import psutil

from qontinui.discovery import PixelStabilityAnalyzer
from qontinui.discovery.models import AnalysisConfig

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")


def create_realistic_screenshots(size=(125, 125), num_screenshots=7):
    """Create screenshots with realistic variation patterns."""
    screenshots = []

    for i in range(num_screenshots):
        # Create base image with slight variation
        base_color = 50 + i * 2  # Slight color variation
        img = np.ones((size[0], size[1], 3), dtype=np.uint8) * base_color

        # Add stable UI elements (these should be detected)
        # Button 1 - always in same position
        cv2.rectangle(img, (10, 10), (40, 25), (100, 150, 200), -1)

        # Button 2 - always in same position
        cv2.rectangle(img, (50, 10), (80, 25), (150, 100, 200), -1)

        # Add varying elements (these should NOT be stable)
        # Moving circle
        x_pos = 30 + i * 5
        y_pos = 50 + i * 3
        cv2.circle(img, (x_pos, y_pos), 10, (255, 255, 0), -1)

        # Random noise in bottom area
        noise_area = img[90:120, 10:110].copy()
        noise = np.random.randint(0, 30, noise_area.shape, dtype=np.uint8)
        img[90:120, 10:110] = np.clip(noise_area + noise, 0, 255).astype(np.uint8)

        # Changing text
        cv2.putText(img, f"{i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        screenshots.append(img)

    return screenshots


def analyze_with_monitoring(screenshots, config):
    """Analyze with detailed monitoring."""
    analyzer = PixelStabilityAnalyzer(config)

    logger.info("=" * 60)
    logger.info(f"Starting analysis of {len(screenshots)} screenshots")
    logger.info(f"Screenshot size: {screenshots[0].shape}")
    logger.info(f"Config: variance_threshold={config.variance_threshold}")
    log_memory_usage()

    start = time.time()

    try:
        result = analyzer.analyze_screenshots(screenshots)
        elapsed = time.time() - start

        logger.info(f"Analysis completed in {elapsed:.3f}s")
        logger.info(f"Found {len(result.state_images)} state images")
        logger.info(f"Found {len(result.states)} states")
        log_memory_usage()

        return result, elapsed

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Analysis failed after {elapsed:.3f}s: {e}")
        logger.error(traceback.format_exc())
        log_memory_usage()
        return None, elapsed


def test_different_sizes():
    """Test with different region sizes."""
    sizes = [
        (50, 50),
        (100, 100),
        (125, 125),
        (150, 150),
        (200, 200),
    ]

    config = AnalysisConfig()
    config.variance_threshold = 10.0

    results = []

    for size in sizes:
        logger.info("\n" + "=" * 60)
        logger.info(f"Testing {size[0]}x{size[1]} region")

        screenshots = create_realistic_screenshots(size=size, num_screenshots=7)

        # Calculate expected grid
        grid_size = 50
        step_size = 25
        y_steps = max(1, (size[0] - grid_size) // step_size + 1)
        x_steps = max(1, (size[1] - grid_size) // step_size + 1)
        expected_regions = y_steps * x_steps

        logger.info(f"Expected grid: {y_steps}x{x_steps} = {expected_regions} regions")

        result, elapsed = analyze_with_monitoring(screenshots, config)

        results.append(
            {
                "size": size,
                "time": elapsed,
                "success": result is not None,
                "regions": len(result.state_images) if result else 0,
                "expected_regions": expected_regions,
            }
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for r in results:
        status = "✓" if r["success"] else "✗"
        logger.info(
            f"{status} {r['size'][0]}x{r['size'][1]}: {r['time']:.3f}s, "
            f"{r['regions']}/{r['expected_regions']} regions"
        )


def test_variance_threshold():
    """Test how variance threshold affects performance."""
    thresholds = [1.0, 5.0, 10.0, 20.0, 50.0]

    screenshots = create_realistic_screenshots(size=(125, 125), num_screenshots=7)

    logger.info("\n" + "=" * 60)
    logger.info("TESTING VARIANCE THRESHOLDS")
    logger.info("=" * 60)

    for threshold in thresholds:
        config = AnalysisConfig()
        config.variance_threshold = threshold

        logger.info(f"\nTesting threshold={threshold}")
        result, elapsed = analyze_with_monitoring(screenshots, config)

        if result:
            logger.info(f"  Time: {elapsed:.3f}s")
            logger.info(f"  State images: {len(result.state_images)}")
            logger.info(f"  States: {len(result.states)}")


def test_your_actual_case():
    """Test case matching user's scenario."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING USER'S SCENARIO: 125x125, 7 screenshots")
    logger.info("=" * 60)

    # Create 7 screenshots with high variation (worst case)
    screenshots = []
    for _i in range(7):
        # Each screenshot is quite different
        img = np.random.randint(0, 255, (125, 125, 3), dtype=np.uint8)
        # Add some stable elements
        cv2.rectangle(img, (20, 20), (60, 40), (100, 150, 200), -1)
        screenshots.append(img)

    config = AnalysisConfig()
    config.variance_threshold = 10.0

    # Test with timeout
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Analysis timed out after 10 seconds!")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout

    try:
        result, elapsed = analyze_with_monitoring(screenshots, config)
        signal.alarm(0)  # Cancel alarm

        if result:
            logger.info(f"SUCCESS in {elapsed:.3f}s")
        else:
            logger.info(f"FAILED after {elapsed:.3f}s")

    except TimeoutError as e:
        logger.error(str(e))
        logger.error("This matches your timeout issue!")
        logger.error("Check the last logged slow region above")


if __name__ == "__main__":
    # Check if required module is available
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not available, memory monitoring disabled")
        psutil = None

    # Run tests
    logger.info("Starting comprehensive debug tests...")

    # Test 1: Different sizes
    test_different_sizes()

    # Test 2: Different variance thresholds
    test_variance_threshold()

    # Test 3: User's specific case
    test_your_actual_case()

    logger.info("\n\nAll tests completed!")
    logger.info("\nRECOMMENDATIONS:")
    logger.info("1. If analysis is slow, try increasing variance_threshold (e.g., 20-50)")
    logger.info("2. Check if your screenshots have high noise/variation")
    logger.info("3. Consider preprocessing to reduce noise before analysis")
