"""
Qontinui API Service
Exposes real qontinui library operations for web-based testing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import io
import json
from PIL import Image as PILImage
import numpy as np

# Import actual qontinui library
from qontinui.find import Find
from qontinui.actions import FindOptions
from qontinui.model.element import Image, Pattern, Region, Location
from qontinui.model.state import State, StateImage, StateRegion, StateLocation
from qontinui.model.search_regions import SearchRegions

app = FastAPI(title="Qontinui API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class FindRequest(BaseModel):
    screenshot: str  # base64 encoded
    template: str    # base64 encoded
    similarity: float = 0.8
    search_region: Optional[Dict[str, int]] = None  # {x, y, width, height}
    find_all: bool = False

class StateDetectionRequest(BaseModel):
    screenshot: str  # base64 encoded
    states: List[Dict[str, Any]]  # State definitions
    similarity: float = 0.8

class MockExecutionRequest(BaseModel):
    screenshots: List[str]  # base64 encoded screenshots
    states: List[Dict[str, Any]]
    starting_screenshot_index: int = 0

class MockSession(BaseModel):
    session_id: str
    current_screenshot_index: int = 0
    active_states: List[str] = []
    execution_history: List[Dict[str, Any]] = []
    initial_states: List[str] = []  # States marked as initial
    action_snapshots: List[Dict[str, Any]] = []  # Pre-recorded snapshots for integration testing

class MatchResponse(BaseModel):
    found: bool
    region: Optional[Dict[str, int]]  # {x, y, width, height}
    score: float
    center: Optional[Dict[str, int]]  # {x, y}

class MatchesResponse(BaseModel):
    found: bool
    matches: List[MatchResponse]
    best_match: Optional[MatchResponse]

# Utility functions
def base64_to_pil_image(base64_string: str) -> PILImage.Image:
    """Convert base64 string to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_bytes = base64.b64decode(base64_string)
    return PILImage.open(io.BytesIO(image_bytes))

def base64_to_qontinui_image(base64_string: str, name: str = "image") -> Image:
    """Convert base64 to Qontinui Image using the actual Image class methods"""
    pil_image = base64_to_pil_image(base64_string)
    return Image.from_pil(pil_image, name=name)

def region_to_dict(region: Region) -> Dict[str, int]:
    """Convert Qontinui Region to dict"""
    return {
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height
    }

def match_to_response(match) -> MatchResponse:
    """Convert Qontinui Match to response"""
    if hasattr(match, 'region'):
        region = match.region
    elif hasattr(match, 'get_region'):
        region = match.get_region()
    else:
        # Handle case where match might be different structure
        region = None

    if region:
        return MatchResponse(
            found=True,
            region=region_to_dict(region),
            score=match.score if hasattr(match, 'score') else 0.0,
            center={"x": region.x + region.width // 2,
                    "y": region.y + region.height // 2}
        )
    return MatchResponse(found=False, region=None, score=0.0, center=None)

# === VISION ENDPOINTS - Using Real Qontinui Find ===

@app.post("/find", response_model=MatchResponse)
async def find_image(request: FindRequest):
    """
    Use qontinui's actual Find operation to find template in screenshot.
    This is the REAL pattern matching that qontinui uses in production.
    """
    try:
        # Convert base64 to Qontinui Images
        screenshot_img = base64_to_qontinui_image(request.screenshot, "screenshot")
        template_img = base64_to_qontinui_image(request.template, "template")

        # Create Find with the template image
        find = Find(template_img)

        # Set similarity
        find.similarity(request.similarity)

        # Set search region if provided
        if request.search_region:
            search_region = Region(
                request.search_region["x"],
                request.search_region["y"],
                request.search_region["width"],
                request.search_region["height"]
            )
            find.search_region(search_region)

        # Set the screenshot to search in
        find.screenshot(screenshot_img)

        # Execute real find operation
        match = find.find()

        if match and match.exists():
            return match_to_response(match)

        # No match found
        return MatchResponse(found=False, region=None, score=0.0, center=None)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find_all", response_model=MatchesResponse)
async def find_all_images(request: FindRequest):
    """
    Use qontinui's find_all to find all template instances.
    Returns all matches with their scores and regions.
    """
    try:
        # Convert base64 to Qontinui Images
        screenshot_img = base64_to_qontinui_image(request.screenshot, "screenshot")
        template_img = base64_to_qontinui_image(request.template, "template")

        # Create Find with the template image
        find = Find(template_img)

        # Set similarity
        find.similarity(request.similarity)

        # Set search region if provided
        if request.search_region:
            search_region = Region(
                request.search_region["x"],
                request.search_region["y"],
                request.search_region["width"],
                request.search_region["height"]
            )
            find.search_region(search_region)

        # Set the screenshot to search in
        find.screenshot(screenshot_img)

        # Execute real find_all - using find_all_matches() method
        matches = find.find_all_matches()

        if matches and matches.size() > 0:
            match_responses = [match_to_response(m) for m in matches.to_list()]
            return MatchesResponse(
                found=True,
                matches=match_responses,
                best_match=match_to_response(matches.best) if hasattr(matches, 'best') else match_responses[0]
            )

        return MatchesResponse(found=False, matches=[], best_match=None)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_states")
async def detect_states(request: StateDetectionRequest):
    """
    Detect which states are present in a screenshot using real qontinui.
    For now, this is a simplified version that checks for state images.
    """
    try:
        screenshot_img = base64_to_qontinui_image(request.screenshot, "screenshot")

        detected_states = []

        for state_dict in request.states:
            # For each state, check if we can find any of its identifying images
            state_found = False
            found_images = []
            found_regions = []

            # Get state images from the state dict
            state_images = state_dict.get("stateImages", [])
            if not state_images and "identifyingImages" in state_dict:
                # Fall back to legacy identifyingImages
                state_images = state_dict.get("identifyingImages", [])

            for state_image in state_images:
                # Get image data - could be base64 or reference
                image_data = state_image.get("image") or state_image.get("imageData")
                if not image_data:
                    continue

                try:
                    # Convert to qontinui Image
                    template_img = base64_to_qontinui_image(image_data, state_image.get("name", "state_image"))

                    # Create Find and search
                    find = Find(template_img)
                    find.similarity(request.similarity)
                    find.screenshot(screenshot_img)

                    # Execute find
                    match = find.find()

                    if match and match.exists():
                        state_found = True
                        match_response = match_to_response(match)
                        found_images.append({
                            "image_id": state_image.get("id", ""),
                            "image_name": state_image.get("name", ""),
                            "match": match_response
                        })
                        if match_response.region:
                            found_regions.append(match_response.region)

                except Exception as img_error:
                    # Skip this image if there's an error
                    continue

            if state_found:
                detected_states.append({
                    "state_id": state_dict.get("id", ""),
                    "state_name": state_dict.get("name", ""),
                    "found": True,
                    "found_images": found_images,
                    "regions": found_regions,
                    "confidence": max([img["match"].score for img in found_images]) if found_images else 0.0
                })

        return {
            "screenshot_analyzed": True,
            "detected_states": detected_states,
            "total_states_found": len(detected_states)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_location")
async def validate_location(
    screenshot: str,
    location_x: int,
    location_y: int,
    reference_image: Optional[str] = None
):
    """
    Validate if a location is accessible in a screenshot.
    If reference_image provided, finds it first and calculates relative position.
    """
    try:
        pil_screenshot = base64_to_pil_image(screenshot)
        screenshot_width, screenshot_height = pil_screenshot.size

        if reference_image:
            # Find reference image to calculate relative position
            screenshot_img = base64_to_qontinui_image(screenshot, "screenshot")
            template_img = base64_to_qontinui_image(reference_image, "reference")

            find = Find(template_img)
            find.screenshot(screenshot_img)
            match = find.find()

            if match and match.exists():
                # Get the match region
                if hasattr(match, 'region'):
                    region = match.region
                elif hasattr(match, 'get_region'):
                    region = match.get_region()
                else:
                    region = None

                if region:
                    # Calculate actual location relative to match
                    actual_x = region.x + location_x
                    actual_y = region.y + location_y

                    return {
                        "valid": True,
                        "reference_found": True,
                        "reference_region": region_to_dict(region),
                        "calculated_location": {"x": actual_x, "y": actual_y},
                        "within_bounds": (0 <= actual_x < screenshot_width and
                                        0 <= actual_y < screenshot_height)
                    }

            return {
                "valid": False,
                "reference_found": False,
                "error": "Reference image not found in screenshot"
            }
        else:
            # Validate absolute location
            return {
                "valid": True,
                "absolute_location": {"x": location_x, "y": location_y},
                "within_bounds": (0 <= location_x < screenshot_width and
                                0 <= location_y < screenshot_height)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === MOCK EXECUTION ENDPOINTS ===

import uuid
from typing import Dict

# Store mock sessions in memory (in production, use Redis or similar)
mock_sessions: Dict[str, MockSession] = {}

@app.post("/mock/create_session")
async def create_mock_session(request: MockExecutionRequest):
    """Create a new mock execution session with state activation"""
    session_id = str(uuid.uuid4())

    # Identify initial states (those with initial=True or first state)
    initial_states = []
    for state in request.states:
        if state.get("initial", False):
            initial_states.append(state.get("id"))

    # If no states marked as initial, use first state
    if not initial_states and request.states:
        initial_states = [request.states[0].get("id")]

    session = MockSession(
        session_id=session_id,
        current_screenshot_index=request.starting_screenshot_index,
        active_states=initial_states,  # Start with initial states active
        execution_history=[],
        initial_states=initial_states
    )

    # Store screenshots in session (in production, use proper storage)
    session.screenshots = request.screenshots  # type: ignore
    session.states = request.states  # type: ignore

    mock_sessions[session_id] = session

    # Immediately detect states in first screenshot
    await _detect_and_activate_states(session)

    return {
        "session_id": session_id,
        "status": "created",
        "total_screenshots": len(request.screenshots),
        "total_states": len(request.states),
        "initial_states": initial_states,
        "active_states": session.active_states
    }

async def _detect_and_activate_states(session: MockSession):
    """Internal function to detect states and activate them"""
    if session.current_screenshot_index >= len(session.screenshots):  # type: ignore
        return

    current_screenshot = session.screenshots[session.current_screenshot_index]  # type: ignore

    # Detect states using real qontinui
    detection_request = StateDetectionRequest(
        screenshot=current_screenshot,
        states=session.states,  # type: ignore
        similarity=0.8
    )

    detection_result = await detect_states(detection_request)

    # Activate detected states (StateImage found = state activated)
    for detected in detection_result["detected_states"]:
        state_id = detected["state_id"]
        if state_id not in session.active_states:
            session.active_states.append(state_id)

@app.post("/mock/detect_current_states")
async def detect_current_states(session_id: str, similarity: float = 0.8):
    """Detect states in the current screenshot of the session"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = mock_sessions[session_id]

    # Get current screenshot
    if session.current_screenshot_index >= len(session.screenshots):  # type: ignore
        return {
            "error": "No more screenshots",
            "detected_states": []
        }

    current_screenshot = session.screenshots[session.current_screenshot_index]  # type: ignore

    # Detect states using real qontinui
    detection_request = StateDetectionRequest(
        screenshot=current_screenshot,
        states=session.states,  # type: ignore
        similarity=similarity
    )

    detection_result = await detect_states(detection_request)

    # Update active states
    session.active_states = [ds["state_id"] for ds in detection_result["detected_states"]]

    # Add to execution history
    session.execution_history.append({
        "type": "state_detection",
        "screenshot_index": session.current_screenshot_index,
        "detected_states": session.active_states,
        "timestamp": str(uuid.uuid4())  # Would be actual timestamp
    })

    return {
        "session_id": session_id,
        "current_screenshot_index": session.current_screenshot_index,
        "active_states": session.active_states,
        "detection_result": detection_result
    }

@app.post("/mock/next_screenshot")
async def next_screenshot(session_id: str):
    """Move to the next screenshot in the session"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = mock_sessions[session_id]

    if session.current_screenshot_index < len(session.screenshots) - 1:  # type: ignore
        session.current_screenshot_index += 1

        # Add to history
        session.execution_history.append({
            "type": "navigation",
            "action": "next_screenshot",
            "new_index": session.current_screenshot_index
        })

        return {
            "session_id": session_id,
            "current_screenshot_index": session.current_screenshot_index,
            "has_next": session.current_screenshot_index < len(session.screenshots) - 1,  # type: ignore
            "has_previous": session.current_screenshot_index > 0
        }
    else:
        return {
            "error": "No more screenshots",
            "current_screenshot_index": session.current_screenshot_index
        }

@app.post("/mock/previous_screenshot")
async def previous_screenshot(session_id: str):
    """Move to the previous screenshot in the session"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = mock_sessions[session_id]

    if session.current_screenshot_index > 0:
        session.current_screenshot_index -= 1

        # Add to history
        session.execution_history.append({
            "type": "navigation",
            "action": "previous_screenshot",
            "new_index": session.current_screenshot_index
        })

        return {
            "session_id": session_id,
            "current_screenshot_index": session.current_screenshot_index,
            "has_next": True,
            "has_previous": session.current_screenshot_index > 0
        }
    else:
        return {
            "error": "Already at first screenshot",
            "current_screenshot_index": session.current_screenshot_index
        }

@app.get("/mock/session_status/{session_id}")
async def get_session_status(session_id: str):
    """Get the current status of a mock session"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = mock_sessions[session_id]

    return {
        "session_id": session_id,
        "current_screenshot_index": session.current_screenshot_index,
        "total_screenshots": len(session.screenshots),  # type: ignore
        "active_states": session.active_states,
        "total_states": len(session.states),  # type: ignore
        "history_length": len(session.execution_history)
    }

@app.get("/mock/execution_history/{session_id}")
async def get_execution_history(session_id: str):
    """Get the execution history of a mock session"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = mock_sessions[session_id]

    return {
        "session_id": session_id,
        "history": session.execution_history
    }

@app.post("/mock/add_snapshots/{session_id}")
async def add_action_snapshots(session_id: str, snapshots: List[Dict[str, Any]]):
    """Add action snapshots for integration testing"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = mock_sessions[session_id]
    session.action_snapshots.extend(snapshots)

    return {
        "session_id": session_id,
        "total_snapshots": len(session.action_snapshots)
    }

@app.post("/mock/execute_action/{session_id}")
async def execute_mock_action(
    session_id: str,
    action_type: str,
    action_config: Optional[Dict[str, Any]] = None
):
    """Execute an action using snapshots (integration testing)"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = mock_sessions[session_id]

    # Find matching snapshot for current state and action
    matching_snapshot = _find_matching_snapshot(
        session.action_snapshots,
        action_type,
        session.active_states
    )

    if not matching_snapshot:
        # No snapshot - action fails
        return {
            "success": False,
            "message": f"No snapshot for {action_type} in states {session.active_states}",
            "active_states": session.active_states
        }

    # Execute based on snapshot
    result = {
        "success": matching_snapshot.get("actionSuccess", False),
        "matches": matching_snapshot.get("matches", []),
        "active_states": session.active_states,
        "snapshot_used": matching_snapshot
    }

    # Handle screenshot transition
    next_screenshot_id = matching_snapshot.get("nextScreenshotId")
    if next_screenshot_id:
        # Find screenshot index by ID
        for i, screenshot in enumerate(session.screenshots):  # type: ignore
            # Would need to decode and check ID, simplified here
            if i != session.current_screenshot_index:  # Simple transition
                session.current_screenshot_index = i
                await _detect_and_activate_states(session)
                break

    # Record in history
    session.execution_history.append({
        "type": "action_execution",
        "action_type": action_type,
        "snapshot_id": matching_snapshot.get("id"),
        "success": result["success"],
        "timestamp": str(uuid.uuid4())
    })

    return result

def _find_matching_snapshot(
    snapshots: List[Dict[str, Any]],
    action_type: str,
    active_states: List[str]
) -> Optional[Dict[str, Any]]:
    """Find matching snapshot using Qontinui's logic"""
    import random

    # Filter by action type
    candidates = [s for s in snapshots if s.get("actionType") == action_type]

    if not candidates:
        return None

    # Prefer exact state matches
    exact_matches = [
        s for s in candidates
        if s.get("stateId") in active_states
    ]

    if exact_matches:
        return random.choice(exact_matches)

    # Try overlapping active states
    overlap_matches = [
        s for s in candidates
        if any(state in active_states for state in s.get("activeStates", []))
    ]

    if overlap_matches:
        return random.choice(overlap_matches)

    # Fall back to any snapshot
    return random.choice(candidates) if candidates else None

@app.delete("/mock/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a mock session"""
    if session_id not in mock_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del mock_sessions[session_id]

    return {
        "session_id": session_id,
        "status": "deleted"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "qontinui-api",
        "version": "1.0.0",
        "qontinui_available": True,
        "active_sessions": len(mock_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)