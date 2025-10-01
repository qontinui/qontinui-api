#!/usr/bin/env python3
"""Direct test of analysis without async complexity."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qontinui.discovery import PixelStabilityAnalyzer
from qontinui.discovery.models import AnalysisConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load screenshots once at startup
SCREENSHOTS = []


def load_screenshots():
    """Load the test screenshots."""
    global SCREENSHOTS
    for i in range(230, 237):
        path = f"/home/jspinak/qontinui_parent_directory/examples/screenshots/screen{i}.png"
        img = cv2.imread(path)
        if img is not None:
            SCREENSHOTS.append(img)
            logger.info(f"Loaded {path}: {img.shape}")


load_screenshots()


@app.get("/test/analyze-region")
def test_analyze_region(x: int = 1789, y: int = 771, width: int = 125, height: int = 125):
    """Test analysis of a specific region."""

    if not SCREENSHOTS:
        return {"error": "No screenshots loaded"}

    # Crop screenshots
    cropped = []
    for img in SCREENSHOTS:
        # Ensure bounds
        x2 = min(x + width, img.shape[1])
        y2 = min(y + height, img.shape[0])
        cropped_img = img[y:y2, x:x2]
        cropped.append(cropped_img)

    logger.info(f"Analyzing region ({x},{y}) {width}x{height}")
    logger.info(f"Cropped shape: {cropped[0].shape}")

    # Create config
    config = AnalysisConfig()
    config.variance_threshold = 10.0

    # Run analysis
    analyzer = PixelStabilityAnalyzer(config)

    start = time.time()
    try:
        result = analyzer.analyze_screenshots(cropped)
        elapsed = time.time() - start

        return {
            "success": True,
            "time": elapsed,
            "region": {"x": x, "y": y, "width": width, "height": height},
            "cropped_shape": cropped[0].shape,
            "state_images": len(result.state_images),
            "states": len(result.states),
            "message": f"Analysis completed in {elapsed:.3f}s",
        }
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Analysis failed: {e}")
        return {
            "success": False,
            "time": elapsed,
            "error": str(e),
            "region": {"x": x, "y": y, "width": width, "height": height},
        }


@app.get("/test/analyze-full")
def test_analyze_full():
    """Test full screenshot analysis."""

    if not SCREENSHOTS:
        return {"error": "No screenshots loaded"}

    logger.info(f"Analyzing full screenshots: {SCREENSHOTS[0].shape}")

    # Create config
    config = AnalysisConfig()
    config.variance_threshold = 10.0

    # Run analysis
    analyzer = PixelStabilityAnalyzer(config)

    start = time.time()
    try:
        result = analyzer.analyze_screenshots(SCREENSHOTS)
        elapsed = time.time() - start

        return {
            "success": True,
            "time": elapsed,
            "screenshot_shape": SCREENSHOTS[0].shape,
            "state_images": len(result.state_images),
            "states": len(result.states),
            "message": f"Full analysis completed in {elapsed:.3f}s",
        }
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Analysis failed: {e}")
        return {"success": False, "time": elapsed, "error": str(e)}


if __name__ == "__main__":
    logger.info("Starting test server on port 8001...")
    logger.info(f"Loaded {len(SCREENSHOTS)} screenshots")
    uvicorn.run(app, host="0.0.0.0", port=8001)
