#!/usr/bin/env python3
"""
Test script to verify workflow execution API endpoints
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


def create_test_workflow() -> dict[str, Any]:
    """Create a simple test workflow"""
    return {
        "id": "test-workflow-1",
        "name": "Test Workflow",
        "version": "1.0.0",
        "format": "graph",
        "category": "test-category",
        "description": "A test workflow for API verification",
        "actions": [
            {
                "id": "action-1",
                "type": "FIND",
                "config": {"imageId": "test-image-1", "similarity": 0.8},
                "position": [100, 100],
            },
            {
                "id": "action-2",
                "type": "CLICK",
                "config": {"targetId": "action-1"},
                "position": [400, 100],
            },
        ],
        "connections": {},
        "metadata": {
            "viewMode": "sequential",
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T00:00:00Z",
        },
        "tags": ["test"],
    }


def create_test_state() -> dict[str, Any]:
    """Create a simple test state"""
    return {
        "id": "test-state-1",
        "name": "Test State",
        "description": "A test state",
        "is_initial": True,
        "state_images": [],
        "outgoing_transitions": [],
        "incoming_transitions": [],
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


def test_workflow_execution():
    """Test the workflow execution endpoint"""
    try:
        workflow = create_test_workflow()
        state = create_test_state()
        screenshot = create_dummy_screenshot()

        request_data = {
            "workflow": workflow,
            "screenshots": [screenshot],
            "states": [state],
            "categories": [{"id": "test-category", "name": "Test Category", "description": ""}],
            "mode": "hybrid",
            "similarity": 0.8,
        }

        print("\n✓ Sending workflow execution request...")
        response = requests.post(
            f"{API_URL}/workflow/execute",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            print("✓ Workflow execution initiated successfully")
            print(f"  Session ID: {result.get('session_id')}")
            print(f"  Workflow: {result.get('workflow_name')}")
            print(f"  Status: {result.get('status')}")
            print(f"  Actions: {result.get('current_action')}/{result.get('total_actions')}")
            return result.get("session_id")
        else:
            print(f"✗ Workflow execution failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Workflow execution error: {e}")
        return None


def test_workflow_status(session_id: str):
    """Test the workflow status endpoint"""
    if not session_id:
        print("✗ No session ID to test status")
        return False

    try:
        response = requests.get(f"{API_URL}/workflow/status/{session_id}")
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Workflow status retrieved")
            print(f"  Total actions: {result.get('total_actions')}")
            print(f"  Successful: {result.get('successful_actions')}")
            print(f"  Success rate: {result.get('success_rate')}%")
            return True
        else:
            print(f"✗ Workflow status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Workflow status error: {e}")
        return False


def test_workflow_complete(session_id: str):
    """Test the workflow complete endpoint"""
    if not session_id:
        print("✗ No session ID to complete")
        return False

    try:
        response = requests.post(f"{API_URL}/workflow/complete/{session_id}")
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Workflow completed")
            print(f"  Final status: {result.get('status')}")
            print(f"  Success rate: {result.get('success_rate')}%")
            return True
        else:
            print(f"✗ Workflow complete failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Workflow complete error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Qontinui API Workflow Endpoints")
    print("=" * 50)

    # Test health check
    if not test_health_check():
        print("\n❌ API is not responding. Please ensure it's running on port 8000")
        return

    # Test workflow execution
    session_id = test_workflow_execution()

    if session_id:
        # Test workflow status
        test_workflow_status(session_id)

        # Test workflow complete
        test_workflow_complete(session_id)

    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
