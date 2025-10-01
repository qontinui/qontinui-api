#!/usr/bin/env python3
"""
Test script to verify process execution API endpoints
"""

import base64
from typing import Any

import requests

API_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def create_test_process() -> dict[str, Any]:
    """Create a simple test process"""
    return {
        "id": "test-process-1",
        "name": "Test Process",
        "categoryId": "test-category",
        "description": "A test process for API verification",
        "actions": [
            {
                "id": "action-1",
                "type": "find",
                "name": "Find test image",
                "description": "Find a test pattern",
                "parameters": {"imageId": "test-image-1", "similarity": 0.8},
            },
            {
                "id": "action-2",
                "type": "click",
                "name": "Click found location",
                "description": "Click on the found pattern",
                "parameters": {"targetId": "action-1"},
            },
        ],
    }


def create_test_state() -> dict[str, Any]:
    """Create a simple test state"""
    return {
        "id": "test-state-1",
        "name": "Test State",
        "description": "A test state",
        "imageIds": ["test-image-1"],
    }


def create_dummy_screenshot() -> str:
    """Create a tiny base64 encoded image for testing"""
    # Create a 10x10 white image
    import io

    from PIL import Image

    img = Image.new("RGB", (10, 10), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_data}"


def test_process_execution():
    """Test the process execution endpoint"""
    try:
        process = create_test_process()
        state = create_test_state()
        screenshot = create_dummy_screenshot()

        request_data = {
            "process": process,
            "screenshots": [screenshot],
            "states": [state],
            "categories": [{"id": "test-category", "name": "Test Category", "description": ""}],
            "mode": "hybrid",
            "similarity": 0.8,
        }

        print("\n✓ Sending process execution request...")
        response = requests.post(
            f"{API_URL}/process/execute",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            print("✓ Process execution initiated successfully")
            print(f"  Session ID: {result.get('session_id')}")
            print(f"  Process: {result.get('process_name')}")
            print(f"  Status: {result.get('status')}")
            print(f"  Actions: {result.get('current_action')}/{result.get('total_actions')}")
            return result.get("session_id")
        else:
            print(f"✗ Process execution failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Process execution error: {e}")
        return None


def test_process_status(session_id: str):
    """Test the process status endpoint"""
    if not session_id:
        print("✗ No session ID to test status")
        return False

    try:
        response = requests.get(f"{API_URL}/process/status/{session_id}")
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Process status retrieved")
            print(f"  Total actions: {result.get('total_actions')}")
            print(f"  Successful: {result.get('successful_actions')}")
            print(f"  Success rate: {result.get('success_rate')}%")
            return True
        else:
            print(f"✗ Process status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Process status error: {e}")
        return False


def test_process_complete(session_id: str):
    """Test the process complete endpoint"""
    if not session_id:
        print("✗ No session ID to complete")
        return False

    try:
        response = requests.post(f"{API_URL}/process/complete/{session_id}")
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Process completed")
            print(f"  Final status: {result.get('status')}")
            print(f"  Success rate: {result.get('success_rate')}%")
            return True
        else:
            print(f"✗ Process complete failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Process complete error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Qontinui API Process Endpoints")
    print("=" * 50)

    # Test health check
    if not test_health_check():
        print("\n❌ API is not responding. Please ensure it's running on port 8000")
        return

    # Test process execution
    session_id = test_process_execution()

    if session_id:
        # Test process status
        test_process_status(session_id)

        # Test process complete
        test_process_complete(session_id)

    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
