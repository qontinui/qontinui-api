#!/usr/bin/env python3
"""Test script for semantic API to verify real CV implementation."""

import base64
import io

import requests
from PIL import Image, ImageDraw


def create_test_image():
    """Create a simple test image with UI elements."""
    # Create a blank image
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Draw header
    draw.rectangle([0, 0, width, 80], fill="#2196F3")
    draw.text((20, 25), "Test Application", fill="white")

    # Draw button
    draw.rectangle([50, 150, 200, 190], fill="#4CAF50")
    draw.text((100, 165), "Click Me", fill="white")

    # Draw input field
    draw.rectangle([50, 250, 400, 290], outline="#9E9E9E", width=2)
    draw.text((60, 265), "Enter text here...", fill="#757575")

    # Draw image placeholder
    draw.rectangle([500, 150, 700, 350], fill="#E0E0E0")
    draw.text((550, 240), "Image", fill="#757575")

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"


def test_semantic_api():
    """Test the semantic processing endpoint."""
    print("Testing Semantic API...")

    # Create test image
    test_image = create_test_image()

    # Prepare request
    url = "http://localhost:8000/api/semantic/process"
    payload = {
        "image": test_image,
        "strategy": "hybrid",
        "options": {"enable_ocr": True, "min_confidence": 0.5, "description_model": "clip"},
    }

    try:
        # Send request
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success! Processing took {result['processing_time_ms']:.2f}ms")
            print(f"Found {result['scene']['object_count']} objects:")

            for obj in result["scene"]["objects"]:
                print(f"\n  - Type: {obj['type']}")
                print(f"    Description: {obj['description']}")
                print(f"    Confidence: {obj['confidence']:.2f}")
                print(
                    f"    Bounds: ({obj['bounding_box']['x']}, {obj['bounding_box']['y']}, "
                    f"{obj['bounding_box']['width']}x{obj['bounding_box']['height']})"
                )
                if obj.get("ocr_text"):
                    print(f"    OCR Text: {obj['ocr_text']}")
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"\n❌ Request failed: {e}")


if __name__ == "__main__":
    test_semantic_api()
