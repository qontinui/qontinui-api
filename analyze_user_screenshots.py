#!/usr/bin/env python3
"""Helper script for users to analyze their screenshots for variation."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path

import cv2
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_screenshot_variation(screenshots):
    """Analyze variation between screenshots."""
    if len(screenshots) < 2:
        logger.error("Need at least 2 screenshots")
        return

    logger.info(f"Analyzing {len(screenshots)} screenshots")
    logger.info(f"Screenshot size: {screenshots[0].shape}")

    # Calculate pixel-wise variance across screenshots
    stack = np.stack(screenshots, axis=0)
    variance_map = np.var(stack, axis=0).mean(axis=2)  # Average across color channels

    # Statistics
    mean_variance = np.mean(variance_map)
    max_variance = np.max(variance_map)
    min_variance = np.min(variance_map)

    # Count stable pixels (low variance)
    thresholds = [1, 5, 10, 20, 50]
    for threshold in thresholds:
        stable_pixels = np.sum(variance_map < threshold)
        total_pixels = variance_map.size
        percentage = (stable_pixels / total_pixels) * 100
        logger.info(
            f"Variance < {threshold:3d}: {stable_pixels:8d}/{total_pixels} pixels ({percentage:5.1f}%)"
        )

    logger.info("\nStatistics:")
    logger.info(f"  Mean variance: {mean_variance:.2f}")
    logger.info(f"  Min variance:  {min_variance:.2f}")
    logger.info(f"  Max variance:  {max_variance:.2f}")

    # Create visualization
    vis_variance = (
        (variance_map / max_variance * 255).astype(np.uint8)
        if max_variance > 0
        else variance_map.astype(np.uint8)
    )
    vis_variance_colored = cv2.applyColorMap(vis_variance, cv2.COLORMAP_JET)

    # Save visualization
    cv2.imwrite("variance_map.png", vis_variance_colored)
    logger.info("\nSaved variance visualization to variance_map.png")
    logger.info("Blue = stable, Red = high variation")

    # Check for common issues
    logger.info("\nDIAGNOSTICS:")

    if mean_variance > 50:
        logger.warning("⚠️ Very high variation detected! Screenshots may be too different.")
        logger.warning("   This will cause analysis to find few or no stable regions.")
    elif mean_variance > 20:
        logger.info("⚠️ Moderate variation. Analysis should work but may be slower.")
    else:
        logger.info("✓ Low variation. Analysis should work well.")

    # Check for specific problem areas
    high_var_regions = []
    region_size = 50
    for y in range(0, screenshots[0].shape[0] - region_size, region_size):
        for x in range(0, screenshots[0].shape[1] - region_size, region_size):
            region_var = np.mean(variance_map[y : y + region_size, x : x + region_size])
            if region_var > 100:
                high_var_regions.append((x, y, region_var))

    if high_var_regions:
        logger.warning(f"\n⚠️ Found {len(high_var_regions)} regions with very high variation:")
        for x, y, var in high_var_regions[:5]:  # Show first 5
            logger.warning(f"   Region at ({x},{y}): variance={var:.1f}")

    return variance_map


def load_screenshots(directory=None):
    """Load screenshots from directory or use test data."""
    if directory:
        path = Path(directory)
        if not path.exists():
            logger.error(f"Directory {directory} does not exist")
            return []

        # Load image files
        extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        image_files = []
        for ext in extensions:
            image_files.extend(path.glob(f"*{ext}"))

        if not image_files:
            logger.error(f"No image files found in {directory}")
            return []

        logger.info(f"Found {len(image_files)} image files")

        screenshots = []
        for img_file in sorted(image_files)[:10]:  # Limit to 10 for testing
            img = cv2.imread(str(img_file))
            if img is not None:
                screenshots.append(img)
                logger.info(f"  Loaded {img_file.name}: {img.shape}")

        return screenshots
    else:
        # Create test data
        logger.info("No directory specified, creating test data...")
        screenshots = []
        for i in range(7):
            # Create image with varying noise level
            img = np.ones((125, 125, 3), dtype=np.uint8) * 50
            noise = np.random.randint(-20 * i, 20 * i, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            screenshots.append(img)
        return screenshots


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze screenshot variation")
    parser.add_argument("directory", nargs="?", help="Directory containing screenshots")
    parser.add_argument("--region", help="Analyze specific region (format: x,y,width,height)")
    args = parser.parse_args()

    # Load screenshots
    screenshots = load_screenshots(args.directory)
    if not screenshots:
        return

    # Crop to region if specified
    if args.region:
        try:
            x, y, w, h = map(int, args.region.split(","))
            logger.info(f"Cropping to region: ({x},{y}) {w}x{h}")
            cropped = []
            for img in screenshots:
                cropped.append(img[y : y + h, x : x + w])
            screenshots = cropped
        except Exception as e:
            logger.error(f"Invalid region format: {e}")
            return

    # Analyze
    analyze_screenshot_variation(screenshots)

    # Quick stability test
    logger.info("\n" + "=" * 60)
    logger.info("QUICK STABILITY TEST")
    logger.info("=" * 60)

    from qontinui.discovery import PixelStabilityAnalyzer
    from qontinui.discovery.models import AnalysisConfig

    config = AnalysisConfig()
    config.variance_threshold = 10.0

    analyzer = PixelStabilityAnalyzer(config)

    import time

    start = time.time()
    try:
        result = analyzer.analyze_screenshots(screenshots)
        elapsed = time.time() - start
        logger.info(f"✓ Analysis completed in {elapsed:.3f}s")
        logger.info(f"  Found {len(result.state_images)} state images")
        logger.info(f"  Found {len(result.states)} states")
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"✗ Analysis failed after {elapsed:.3f}s: {e}")


if __name__ == "__main__":
    main()
