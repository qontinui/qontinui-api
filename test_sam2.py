#!/usr/bin/env python3
"""Test SAM2 strategy with mask generation."""

import base64
import io

import requests
from PIL import Image, ImageDraw


def create_test_ui_image():
    """Create a test UI image with clear segmentable elements."""
    # Create a blank image with white background
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color="#F5F5F5")
    draw = ImageDraw.Draw(img)

    # Draw header with clear boundaries
    draw.rectangle([0, 0, width, 80], fill="#2196F3", outline="#1976D2", width=2)
    draw.text((30, 30), "Application Header", fill="white")

    # Draw navigation bar
    draw.rectangle([0, 80, width, 120], fill="#37474F", outline="#263238", width=1)
    nav_items = ["Home", "Products", "Services", "Contact"]
    x_pos = 20
    for item in nav_items:
        draw.rectangle([x_pos, 90, x_pos + 100, 110], fill="#546E7A")
        draw.text((x_pos + 20, 94), item, fill="white")
        x_pos += 120

    # Draw sidebar
    draw.rectangle([0, 120, 200, height], fill="#ECEFF1", outline="#CFD8DC", width=1)
    draw.text((20, 140), "Sidebar", fill="#37474F")

    # Draw main content area buttons
    draw.rectangle([250, 160, 450, 210], fill="#4CAF50", outline="#388E3C", width=2)
    draw.text((320, 175), "Primary Button", fill="white")

    draw.rectangle([250, 230, 450, 280], fill="#FF5722", outline="#D84315", width=2)
    draw.text((320, 245), "Secondary Button", fill="white")

    # Draw input fields
    draw.rectangle([250, 320, 550, 360], fill="white", outline="#9E9E9E", width=2)
    draw.text((260, 335), "Input Field 1", fill="#757575")

    draw.rectangle([250, 380, 550, 420], fill="white", outline="#9E9E9E", width=2)
    draw.text((260, 395), "Input Field 2", fill="#757575")

    # Draw card/panel
    draw.rectangle([600, 160, 760, 400], fill="white", outline="#E0E0E0", width=1)
    draw.text((620, 180), "Card Content", fill="#424242")
    draw.rectangle([620, 210, 740, 250], fill="#E3F2FD")
    draw.text((640, 225), "Nested Element", fill="#1976D2")

    # Draw footer
    draw.rectangle([0, height - 60, width, height], fill="#37474F", outline="#263238", width=1)
    draw.text((width // 2 - 50, height - 40), "Footer Text", fill="white")

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"


def test_sam2_strategy():
    """Test the SAM2 strategy specifically."""
    print("Testing SAM2 Strategy with Mask Generation...")

    # Create test image
    test_image = create_test_ui_image()

    # Test different strategies
    strategies = ["sam2", "ocr", "hybrid"]

    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Testing strategy: {strategy}")
        print("=" * 50)

        # Prepare request
        url = "http://localhost:8000/api/semantic/process"
        payload = {
            "image": test_image,
            "strategy": strategy,
            "options": {"enable_ocr": True, "min_confidence": 0.5, "description_model": "clip"},
        }

        try:
            # Send request
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Processing took {result['processing_time_ms']:.2f}ms")
                print(f"Found {result['scene']['object_count']} objects:")

                # Count objects with masks
                objects_with_masks = 0
                for obj in result["scene"]["objects"]:
                    has_mask = obj.get("pixel_mask") is not None
                    if has_mask:
                        objects_with_masks += 1
                        # Check mask is valid base64
                        try:
                            mask_data = base64.b64decode(obj["pixel_mask"])
                            print(f"  ✓ {obj['type']}: Has valid mask ({len(mask_data)} bytes)")
                        except Exception:
                            print(f"  ✗ {obj['type']}: Invalid mask encoding")
                    else:
                        print(f"  - {obj['type']}: No mask")

                print(
                    f"\nSummary: {objects_with_masks}/{result['scene']['object_count']} objects have masks"
                )

                # Additional details for SAM2
                if strategy == "sam2" and objects_with_masks == 0:
                    print("⚠️  WARNING: SAM2 strategy should generate masks but none were found!")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    test_sam2_strategy()
