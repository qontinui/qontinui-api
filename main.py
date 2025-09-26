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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "qontinui-api",
        "version": "1.0.0",
        "qontinui_available": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)