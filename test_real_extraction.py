#!/usr/bin/env python3
"""
Test pattern extraction with real screenshots
"""
import base64

import requests

# Configuration
API_URL = "http://localhost:8000/api/masked-patterns/extract-masked"
SCREENSHOT_FILES = [
    "/home/jspinak/qontinui_parent_directory/examples/screenshots/screen233.png",
    "/home/jspinak/qontinui_parent_directory/examples/screenshots/screen234.png",
    "/home/jspinak/qontinui_parent_directory/examples/screenshots/screen235.png",
    "/home/jspinak/qontinui_parent_directory/examples/screenshots/screen236.png",
]
REGION = {"x": 316, "y": 182, "width": 55, "height": 51}


def load_and_encode_image(filepath):
    """Load image and convert to base64"""
    with open(filepath, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode("utf-8")


def main():
    print("Loading screenshots...")
    screenshots = []
    regions = []

    for filepath in SCREENSHOT_FILES:
        try:
            base64_img = load_and_encode_image(filepath)
            screenshots.append(base64_img)
            regions.append(REGION)
            print(f"✓ Loaded: {filepath}")
        except Exception as e:
            print(f"✗ Failed to load {filepath}: {e}")
            return

    print(f"\nPreparing request with {len(screenshots)} screenshots...")
    print(f"Region: x={REGION['x']}, y={REGION['y']}, w={REGION['width']}, h={REGION['height']}")

    # Prepare request
    request_body = {
        "state_image_id": "test",
        "pattern_name": "Pattern_RealScreenshots",
        "config": {
            "similarityThreshold": 0.85,
            "minActivePixels": 100,
            "colorAveraging": "weighted",
            "morphologicalOps": {"enabled": True, "erosionSize": 1, "dilationSize": 2},
        },
        "screenshots": screenshots,
        "regions": regions,
    }

    print("\nSending extraction request...")
    try:
        response = requests.post(API_URL, json=request_body, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print("\n✓ Extraction successful!")
            print("\nPattern Details:")
            print(f"  ID: {result['id']}")
            print(f"  Name: {result['name']}")
            print(f"  Size: {result['width']}x{result['height']}")
            print(f"  Active Pixels: {result['activePixels']} / {result['totalPixels']}")
            print(f"  Mask Density: {result['maskDensity']:.2%}")
            print(
                f"  Confidence: avg={result['avgConfidence']:.3f}, min={result['minConfidence']:.3f}, max={result['maxConfidence']:.3f}"
            )

            # Get detailed pattern with images
            details_url = f"http://localhost:8000/api/masked-patterns/{result['id']}"
            details_response = requests.get(details_url)
            if details_response.status_code == 200:
                details = details_response.json()
                print("\n✓ Pattern images retrieved successfully")
                print(
                    f"  - Pattern image: {'present' if details.get('pattern_image') else 'missing'}"
                )
                print(
                    f"  - Confidence map: {'present' if details.get('confidence_image') else 'missing'}"
                )
                print(f"  - Mask image: {'present' if details.get('mask_image') else 'missing'}")
            else:
                print(f"\n✗ Failed to get pattern details: {details_response.status_code}")

        else:
            print(f"\n✗ Extraction failed with status {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.Timeout:
        print("\n✗ Request timed out after 60 seconds")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
