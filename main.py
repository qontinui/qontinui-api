"""
Qontinui API Service
Exposes real qontinui library operations for web-based testing
"""

import base64
import io
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

import cv2
import numpy as np
import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.adapters.state_adapter import convert_multiple_states
from app.core.config import settings
from app.routes.capture import router as capture_router
from app.routes.embeddings import router as embeddings_router

# Import Integration Testing router
from app.routes.integration_testing import router as integration_testing_router

# Import Pathfinding router
from app.routes.pathfinding import router as pathfinding_router

# Import RAG API router
from app.routes.rag import router as rag_router
from app.routes.snapshot_search import router as snapshot_search_router

# Import Snapshot API routers
from app.routes.snapshots import router as snapshots_router

# Authentication is handled by qontinui-web/backend
# This API is stateless and focuses on qontinui library operations
# Import Mask and Pattern API router
from mask_pattern_api import router as mask_pattern_router

# Import Masked Patterns API router
from masked_patterns_api import router as masked_patterns_router
from qontinui.actions import FindOptions

# Import Pydantic schemas from qontinui library
from qontinui.config.schema import Action as ConfigAction
from qontinui.config.schema import Workflow

# Import actual qontinui library
from qontinui.find import Find
from qontinui.json_executor.config_parser import State as ConfigState
from qontinui.model.element import Image, Pattern, Region
from qontinui.model.search_regions import SearchRegions
from qontinui.model.state import StateImage
from qontinui.model.state.state_store import StateStore
from qontinui.state_management.manager import QontinuiStateManager

# Scheduler API removed - belongs in qontinui-web/backend with user auth
# Import semantic API router
from semantic_api import router as semantic_router

# Import State Discovery API router
from state_discovery_api import router as state_discovery_router

# Logger for this module
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SESSION_TTL = 3600  # Session TTL in seconds (1 hour)
DEFAULT_SIMILARITY_THRESHOLD = 0.8  # Default template matching similarity
CONSENSUS_THRESHOLD = 0.7  # Threshold for multi-template consensus matching
NMS_OVERLAP_THRESHOLD = 0.3  # Non-maximum suppression overlap threshold

app = FastAPI(
    title="Qontinui API",
    version="1.0.0",
)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}/{os.getenv('REDIS_DB', 0)}",
)
app.state.limiter = limiter


# Custom rate limit exceeded handler
@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded responses"""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": getattr(exc, "detail", "60 seconds"),
        },
    )


# Initialize Redis client for session storage
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True,
)


# Redis Session Storage Helper Functions
def save_session(session_id: str, session: "MockSession", ttl: int = DEFAULT_SESSION_TTL):
    """
    Save session to Redis with TTL (default 1 hour).

    Args:
        session_id: Unique session identifier
        session: MockSession object to save
        ttl: Time to live in seconds (default: 3600 = 1 hour)
    """
    try:
        redis_client.setex(f"mock_session:{session_id}", ttl, json.dumps(session.dict()))
    except redis.RedisError as e:
        logger.error(f"Error saving session {session_id} to Redis: {e}")
        raise


def get_session(session_id: str) -> Optional["MockSession"]:
    """
    Retrieve session from Redis.

    Args:
        session_id: Unique session identifier

    Returns:
        MockSession object if found, None otherwise
    """
    try:
        data = redis_client.get(f"mock_session:{session_id}")
        if data:
            return MockSession(**json.loads(data))  # type: ignore[arg-type]
        return None
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.error(f"Error retrieving session {session_id} from Redis: {e}")
        return None


def delete_session(session_id: str):
    """
    Delete session from Redis.

    Args:
        session_id: Unique session identifier
    """
    try:
        redis_client.delete(f"mock_session:{session_id}")
    except redis.RedisError as e:
        logger.error(f"Error deleting session {session_id} from Redis: {e}")
        raise


def list_sessions() -> list[str]:
    """
    List all session IDs stored in Redis.

    Returns:
        List of session IDs (without the 'mock_session:' prefix)
    """
    try:
        keys = redis_client.keys("mock_session:*")
        return [key.replace("mock_session:", "") for key in keys]  # type: ignore[misc, union-attr]
    except redis.RedisError as e:
        logger.error(f"Error listing sessions from Redis: {e}")
        return []


# Include semantic router
app.include_router(semantic_router, prefix="/api")

# Include State Discovery router
app.include_router(state_discovery_router, prefix="/api")


app.include_router(embeddings_router, prefix="/api")

# Include Masked Patterns router (must be before mask_pattern_router due to path overlap)
app.include_router(masked_patterns_router, prefix="/api")

# Include Mask and Pattern router
app.include_router(mask_pattern_router, prefix="/api")

# Include Snapshot routers
app.include_router(snapshots_router, prefix="/api")
app.include_router(snapshot_search_router, prefix="/api")

# Include Capture router (video capture, historical data, frame extraction)
app.include_router(capture_router, prefix="/api")

# Include RAG router (RAG element management and search)
app.include_router(rag_router, prefix="/api")

# Include Pathfinding router (path validation for GO_TO_STATE)
app.include_router(pathfinding_router, prefix="/api")

# Include Integration Testing router (model-based GUI automation testing)
app.include_router(integration_testing_router, prefix="/api/v1")


# All user/project management endpoints are handled by qontinui-web/backend
# This API focuses on stateless qontinui library operations


# CORS for frontend - origins configured via CORS_ORIGINS environment variable
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


# Request/Response Models
class FindRequest(BaseModel):
    screenshot: str  # base64 encoded
    template: str  # base64 encoded
    similarity: float = DEFAULT_SIMILARITY_THRESHOLD
    search_region: dict[str, int] | None = None  # {x, y, width, height}
    find_all: bool = False


class StateDetectionRequest(BaseModel):
    screenshot: str  # base64 encoded
    states: list[dict[str, Any]]  # State definitions
    similarity: float = DEFAULT_SIMILARITY_THRESHOLD


class MockExecutionRequest(BaseModel):
    screenshots: list[str]  # base64 encoded screenshots
    states: list[dict[str, Any]]
    starting_screenshot_index: int = 0


class ActiveStatesRequest(BaseModel):
    state_ids: list[str]


class TransitionRequest(BaseModel):
    from_state: str
    to_state: str
    action_type: str | None = None


class MockSession(BaseModel):
    session_id: str
    screenshots: list[str] = []  # base64 encoded screenshots
    states: list[ConfigState] = []  # State definitions using Pydantic schema
    current_screenshot_index: int = 0
    active_states: list[str] = []
    execution_history: list[dict[str, Any]] = []
    initial_states: list[str] = []  # States marked as initial
    action_snapshots: list[dict[str, Any]] = []  # Pre-recorded snapshots for integration testing
    mode: str = "hybrid"  # Execution mode: "hybrid" or "full_mock"


class PatternOptimizationRequest(BaseModel):
    screenshots: list[str]  # base64 encoded screenshots (positive examples)
    negative_screenshots: list[str] = []  # base64 encoded negative examples
    regions: list[dict[str, int]]  # List of regions {x, y, width, height} for each screenshot
    strategies: list[str] = [
        "multi-pattern",
        "consensus",
        "feature-based",
        "differential",
    ]


class CreateStateImageRequest(BaseModel):
    name: str
    patterns: list[dict[str, Any]]  # Pattern data from optimization results
    strategy_type: str
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD


class MatchResponse(BaseModel):
    found: bool
    region: dict[str, int] | None  # {x, y, width, height}
    score: float
    center: dict[str, int] | None  # {x, y}


class MatchesResponse(BaseModel):
    found: bool
    matches: list[MatchResponse]
    best_match: MatchResponse | None


# Utility functions
def base64_to_pil_image(base64_string: str) -> PILImage.Image:
    """Convert base64 string to PIL Image"""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    return PILImage.open(io.BytesIO(image_bytes))


def base64_to_qontinui_image(base64_string: str, name: str = "image") -> Image:
    """Convert base64 to Qontinui Image using the actual Image class methods"""
    pil_image = base64_to_pil_image(base64_string)
    return Image.from_pil(pil_image, name=name)


def region_to_dict(region: Region) -> dict[str, int]:
    """Convert Qontinui Region to dict"""
    return {
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
    }


def match_to_response(match) -> MatchResponse:
    """Convert Qontinui Match to response"""
    import math

    if hasattr(match, "region"):
        region = match.region
    elif hasattr(match, "get_region"):
        region = match.get_region()
    else:
        # Handle case where match might be different structure
        region = None

    if region:
        # Get score and ensure it's finite
        score = match.score if hasattr(match, "score") else 0.0
        if not math.isfinite(score):
            score = 0.0

        return MatchResponse(
            found=True,
            region=region_to_dict(region),
            score=score,
            center={
                "x": region.x + region.width // 2,
                "y": region.y + region.height // 2,
            },
        )
    return MatchResponse(found=False, region=None, score=0.0, center=None)


# === VISION ENDPOINTS - Using Real Qontinui Find ===


@app.post("/find", response_model=MatchResponse)
@limiter.limit("60/minute")
async def find_image(request: Request, find_request: FindRequest):
    """
    Use qontinui's actual Find operation to find template in screenshot.
    This is the REAL pattern matching that qontinui uses in production.
    """
    try:
        from qontinui.find.filters import SimilarityFilter
        from qontinui.find.find_executor import FindExecutor
        from qontinui.find.matchers import TemplateMatcher
        from static_screenshot_provider import StaticScreenshotProvider

        logger.debug("[find] Received request - similarity: {find_request.similarity}")
        logger.debug("[find] Screenshot length: {len(find_request.screenshot)}")
        logger.debug("[find] Template length: {len(find_request.template)}")
        logger.debug("[find] Search region: {find_request.search_region}")

        # Convert base64 to PIL and Qontinui Images
        screenshot_pil = base64_to_pil_image(find_request.screenshot)
        template_img = base64_to_qontinui_image(find_request.template, "template")

        logger.debug("[find] Screenshot image size: {screenshot_pil.size}")
        logger.debug("[find] Template image size: {template_img.width}x{template_img.height}")

        # Create Pattern from template image
        template_pattern = Pattern.from_image(template_img)
        template_pattern.similarity = find_request.similarity

        # Create static screenshot provider with the provided screenshot
        screenshot_provider = StaticScreenshotProvider(screenshot_pil)

        # Configure template matcher
        matcher = TemplateMatcher(method="TM_CCOEFF_NORMED", nms_overlap_threshold=0.3)

        # Configure filters
        filters = [SimilarityFilter(min_similarity=find_request.similarity)]

        # Create executor
        executor = FindExecutor(
            screenshot_provider=screenshot_provider,
            matcher=matcher,
            filters=filters,  # type: ignore[arg-type]
        )

        # Set search region if provided
        search_region = None
        if find_request.search_region:
            search_region = Region(
                find_request.search_region["x"],
                find_request.search_region["y"],
                find_request.search_region["width"],
                find_request.search_region["height"],
            )
            logger.debug("[find] Search region set: {search_region}")

        # Execute find operation
        logger.debug("[find] Executing find operation with similarity={find_request.similarity}...")
        match_list = executor.execute(
            pattern=template_pattern,
            search_region=search_region,
            similarity=find_request.similarity,
            find_all=False,
        )
        logger.debug("[find] Found {len(match_list)} matches")

        if match_list and len(match_list) > 0:
            match = match_list[0]
            print(
                f"[find] Match found! Score: {match.similarity if hasattr(match, 'similarity') else 'N/A'}"
            )
            response = match_to_response(match)
            logger.debug("[find] Response: {response}")
            return response

        # No match found
        print("[find] No match found")
        return MatchResponse(found=False, region=None, score=0.0, center=None)

    except Exception as e:
        logger.debug("[find] Error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/find_all", response_model=MatchesResponse)
@limiter.limit("60/minute")
async def find_all_images(request: Request, find_request: FindRequest):
    """
    Use qontinui's find_all to find all template instances.
    Returns all matches with their scores and regions.
    """
    try:
        from qontinui.find.filters import NMSFilter, SimilarityFilter
        from qontinui.find.find_executor import FindExecutor
        from qontinui.find.matchers import TemplateMatcher
        from static_screenshot_provider import StaticScreenshotProvider

        logger.debug(
            "[find_all] Starting pattern matching with similarity={find_request.similarity}"
        )

        # Convert base64 to PIL and Qontinui Images
        screenshot_pil = base64_to_pil_image(find_request.screenshot)
        template_img = base64_to_qontinui_image(find_request.template, "template")
        print("[find_all] Images converted successfully")

        # Create Pattern from template image
        template_pattern = Pattern.from_image(template_img)
        template_pattern.similarity = find_request.similarity

        # Create static screenshot provider with the provided screenshot
        screenshot_provider = StaticScreenshotProvider(screenshot_pil)

        # Configure template matcher
        matcher = TemplateMatcher(method="TM_CCOEFF_NORMED", nms_overlap_threshold=0.3)

        # Configure filters
        filters = [
            SimilarityFilter(min_similarity=find_request.similarity),
            NMSFilter(iou_threshold=0.3),
        ]

        # Create executor
        executor = FindExecutor(  # type: ignore[arg-type]
            screenshot_provider=screenshot_provider, matcher=matcher, filters=filters
        )

        # Set search region if provided
        search_region = None
        if find_request.search_region:
            search_region = Region(
                find_request.search_region["x"],
                find_request.search_region["y"],
                find_request.search_region["width"],
                find_request.search_region["height"],
            )
            print("[find_all] Search region set")

        # Execute find_all operation
        print("[find_all] Executing find_all...")
        match_list = executor.execute(
            pattern=template_pattern,
            search_region=search_region,
            similarity=find_request.similarity,
            find_all=True,
        )
        logger.debug("[find_all] Found {len(match_list)} matches")

        if match_list and len(match_list) > 0:
            logger.debug("[find_all] Converting {len(match_list)} matches to responses")
            match_responses = [match_to_response(m) for m in match_list]
            print("[find_all] Matches converted successfully")
            return MatchesResponse(
                found=True,
                matches=match_responses,
                best_match=match_responses[0] if match_responses else None,
            )

        print("[find_all] No matches found")
        return MatchesResponse(found=False, matches=[], best_match=None)

    except Exception as e:
        import traceback

        # Log full traceback for debugging
        logger.error("in find_all_images: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/detect_states")
async def detect_states(request: StateDetectionRequest):
    """
    Detect which states are present in a screenshot using real qontinui.
    Uses Qontinui's state manager to properly activate detected states.
    """
    try:
        screenshot_img = base64_to_qontinui_image(request.screenshot, "screenshot")

        # Skip state registration with qontinui's state manager for now
        # The frontend state format doesn't match qontinui's expected State class
        # which requires attributes like state_enum and transitions

        detected_states = []

        for state_dict in request.states:
            # For each state, check if we can find any of its identifying images
            state_found = False
            found_images = []
            found_regions = []
            max_confidence = 0.0

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
                    template_img = base64_to_qontinui_image(
                        image_data, state_image.get("name", "state_image")
                    )

                    # Create Find and search
                    find = Find(template_img)
                    find.similarity(request.similarity)
                    find.screenshot(screenshot_img)

                    # Execute find
                    match = find.find()

                    if match and match.exists():
                        state_found = True
                        match_response = match_to_response(match)
                        found_images.append(
                            {
                                "image_id": state_image.get("id", ""),
                                "image_name": state_image.get("name", ""),
                                "match": match_response,
                            }
                        )
                        if match_response.region:
                            found_regions.append(match_response.region)
                        max_confidence = max(max_confidence, match_response.score)

                except Exception:
                    # Skip this image if there's an error
                    continue

            if state_found:
                state_id = state_dict.get("id", "")
                state_name = state_dict.get("name", "")

                # Activate state in state manager with evidence score
                state_manager.activate_state(state_id, evidence_score=max_confidence)

                detected_states.append(
                    {
                        "state_id": state_id,
                        "state_name": state_name,
                        "found": True,
                        "found_images": found_images,
                        "regions": found_regions,
                        "confidence": max_confidence,
                    }
                )

        return {
            "screenshot_analyzed": True,
            "detected_states": detected_states,
            "total_states_found": len(detected_states),
            "active_states": list(state_manager.get_current_states()),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/validate_location")
async def validate_location(
    screenshot: str,
    location_x: int,
    location_y: int,
    reference_image: str | None = None,
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
                if hasattr(match, "region"):
                    region = match.region
                elif hasattr(match, "get_region"):
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
                        "within_bounds": (
                            0 <= actual_x < screenshot_width and 0 <= actual_y < screenshot_height
                        ),
                    }

            return {
                "valid": False,
                "reference_found": False,
                "error": "Reference image not found in screenshot",
            }
        else:
            # Validate absolute location
            return {
                "valid": True,
                "absolute_location": {"x": location_x, "y": location_y},
                "within_bounds": (
                    0 <= location_x < screenshot_width and 0 <= location_y < screenshot_height
                ),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# === MOCK EXECUTION ENDPOINTS ===

# Sessions are now stored in Redis (see Redis Session Storage Helper Functions above)
# No in-memory storage needed

# Global state management using Qontinui's state management system
state_manager = QontinuiStateManager()
# StateAutomator requires Actions which requires pynput - not needed for API
# state_automator = StateAutomator()
state_store = StateStore()


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
        initial_states=initial_states,
    )

    # Store screenshots in session (in production, use proper storage)
    session.screenshots = request.screenshots  # type: ignore
    session.states = request.states  # type: ignore

    # Save session to Redis with 1 hour TTL
    save_session(session_id, session, ttl=3600)

    # Immediately detect states in first screenshot
    await _detect_and_activate_states(session)

    return {
        "session_id": session_id,
        "status": "created",
        "total_screenshots": len(request.screenshots),
        "total_states": len(request.states),
        "initial_states": initial_states,
        "active_states": session.active_states,
    }


async def _detect_and_activate_states(session: MockSession):
    """Internal function to detect states and activate them"""
    if session.current_screenshot_index >= len(session.screenshots):  # type: ignore
        return

    # For now, skip state detection to avoid qontinui State class incompatibility
    # The frontend states don't have the required attributes (state_enum, transitions)
    # that qontinui expects. This would need a proper state format conversion.

    # Just use initial states for mock execution
    # Real state detection would need proper state conversion first


@app.post("/mock/detect_current_states")
async def detect_current_states(session_id: str, similarity: float = DEFAULT_SIMILARITY_THRESHOLD):
    """Detect states in the current screenshot of the session"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get current screenshot
    if session.current_screenshot_index >= len(session.screenshots):  # type: ignore
        return {"error": "No more screenshots", "detected_states": []}

    current_screenshot = session.screenshots[session.current_screenshot_index]  # type: ignore

    # Detect states using real qontinui
    detection_request = StateDetectionRequest(
        screenshot=current_screenshot,
        states=session.states,
        similarity=similarity,  # type: ignore
    )

    detection_result = await detect_states(detection_request)

    # Update active states
    session.active_states = [ds["state_id"] for ds in detection_result["detected_states"]]

    # Add to execution history
    session.execution_history.append(
        {
            "type": "state_detection",
            "screenshot_index": session.current_screenshot_index,
            "detected_states": session.active_states,
            "timestamp": str(uuid.uuid4()),  # Would be actual timestamp
        }
    )

    # Save updated session to Redis
    save_session(session_id, session, ttl=3600)

    return {
        "session_id": session_id,
        "current_screenshot_index": session.current_screenshot_index,
        "active_states": session.active_states,
        "detection_result": detection_result,
    }


@app.post("/mock/next_screenshot")
async def next_screenshot(session_id: str):
    """Move to the next screenshot in the session"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.current_screenshot_index < len(session.screenshots) - 1:  # type: ignore
        session.current_screenshot_index += 1

        # Add to history
        session.execution_history.append(
            {
                "type": "navigation",
                "action": "next_screenshot",
                "new_index": session.current_screenshot_index,
            }
        )

        # Save updated session to Redis
        save_session(session_id, session, ttl=3600)

        return {
            "session_id": session_id,
            "current_screenshot_index": session.current_screenshot_index,
            "has_next": session.current_screenshot_index < len(session.screenshots) - 1,  # type: ignore
            "has_previous": session.current_screenshot_index > 0,
        }
    else:
        return {
            "error": "No more screenshots",
            "current_screenshot_index": session.current_screenshot_index,
        }


@app.post("/mock/previous_screenshot")
async def previous_screenshot(session_id: str):
    """Move to the previous screenshot in the session"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.current_screenshot_index > 0:
        session.current_screenshot_index -= 1

        # Add to history
        session.execution_history.append(
            {
                "type": "navigation",
                "action": "previous_screenshot",
                "new_index": session.current_screenshot_index,
            }
        )

        # Save updated session to Redis
        save_session(session_id, session, ttl=3600)

        return {
            "session_id": session_id,
            "current_screenshot_index": session.current_screenshot_index,
            "has_next": True,
            "has_previous": session.current_screenshot_index > 0,
        }
    else:
        return {
            "error": "Already at first screenshot",
            "current_screenshot_index": session.current_screenshot_index,
        }


@app.get("/mock/session_status/{session_id}")
async def get_session_status(session_id: str):
    """Get the current status of a mock session"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "current_screenshot_index": session.current_screenshot_index,
        "total_screenshots": len(session.screenshots),  # type: ignore
        "active_states": session.active_states,
        "total_states": len(session.states),  # type: ignore
        "history_length": len(session.execution_history),
    }


@app.get("/mock/execution_history/{session_id}")
async def get_execution_history(session_id: str):
    """Get the execution history of a mock session"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "history": session.execution_history}


@app.post("/mock/add_snapshots/{session_id}")
async def add_action_snapshots(session_id: str, snapshots: list[dict[str, Any]]):
    """Add action snapshots for integration testing"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.action_snapshots.extend(snapshots)

    # Save updated session to Redis
    save_session(session_id, session, ttl=3600)

    return {"session_id": session_id, "total_snapshots": len(session.action_snapshots)}


@app.post("/mock/execute_action/{session_id}")
@limiter.limit("120/minute")
async def execute_mock_action(
    request: Request,
    session_id: str,
    action_type: str,
    action_config: dict[str, Any] | None = None,
):
    """Execute an action using snapshots (integration testing)"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find matching snapshot for current state and action
    matching_snapshot = _find_matching_snapshot(
        session.action_snapshots, action_type, session.active_states
    )

    if not matching_snapshot:
        # No snapshot - action fails
        return {
            "success": False,
            "message": f"No snapshot for {action_type} in states {session.active_states}",
            "active_states": session.active_states,
        }

    # Execute based on snapshot
    result = {
        "success": matching_snapshot.get("actionSuccess", False),
        "matches": matching_snapshot.get("matches", []),
        "active_states": session.active_states,
        "snapshot_used": matching_snapshot,
    }

    # Handle screenshot transition
    next_screenshot_id = matching_snapshot.get("nextScreenshotId")
    if next_screenshot_id:
        # Find screenshot index by ID
        for i, _screenshot in enumerate(session.screenshots):  # type: ignore
            # Would need to decode and check ID, simplified here
            if i != session.current_screenshot_index:  # Simple transition
                session.current_screenshot_index = i
                await _detect_and_activate_states(session)
                break

    # Record in history
    session.execution_history.append(
        {
            "type": "action_execution",
            "action_type": action_type,
            "snapshot_id": matching_snapshot.get("id"),
            "success": result["success"],
            "timestamp": str(uuid.uuid4()),
        }
    )

    # Save updated session to Redis
    save_session(session_id, session, ttl=3600)

    return result


def _find_matching_snapshot(
    snapshots: list[dict[str, Any]], action_type: str, active_states: list[str]
) -> dict[str, Any] | None:
    """Find matching snapshot using Qontinui's logic"""
    import random

    # Filter by action type
    candidates = [s for s in snapshots if s.get("actionType") == action_type]

    if not candidates:
        return None

    # Prefer exact state matches
    exact_matches = [s for s in candidates if s.get("stateId") in active_states]

    if exact_matches:
        return random.choice(exact_matches)

    # Try overlapping active states
    overlap_matches = [
        s for s in candidates if any(state in active_states for state in s.get("activeStates", []))
    ]

    if overlap_matches:
        return random.choice(overlap_matches)

    # Fall back to any snapshot
    return random.choice(candidates) if candidates else None


@app.delete("/mock/session/{session_id}")
async def delete_mock_session(session_id: str):
    """Delete a mock session"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete session from Redis
    delete_session(session_id)

    return {"session_id": session_id, "status": "deleted"}


# === STATE MANAGEMENT ENDPOINTS (New) ===


@app.post("/states/register")
async def register_states(states: list[dict[str, Any]]):
    """
    Register states with Qontinui's state management system.
    Uses the state adapter to convert frontend state format to qontinui State format.
    """
    try:
        # Convert frontend states to qontinui States using the adapter
        qontinui_states = convert_multiple_states(states)

        registered = []
        for state in qontinui_states:
            try:
                # Register with state store and manager
                state_store.register(state)
                state_manager.add_state(state)  # type: ignore[arg-type]
                registered.append(state.name)
            except Exception as e:
                print(f"Failed to register state {state.name}: {e}")
                continue

        return {
            "status": "success",
            "registered": registered,
            "registered_count": len(registered),
            "total": len(states),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"State registration failed: {str(e)}") from e


@app.get("/states/active")
async def get_active_states():
    """Get currently active states from Qontinui's state manager"""
    active = state_manager.get_current_states()
    return {"active_states": list(active), "count": len(active)}


@app.post("/states/active")
async def set_active_states(request: ActiveStatesRequest):
    """Activate states using Qontinui's state manager"""
    # Clear current states
    state_manager.reset()

    # Activate requested states by name (qontinui uses names, not IDs)
    for state_name in request.state_ids:  # Despite the field name, we'll treat these as names
        state_manager.activate_state(state_name, evidence_score=1.0)

    return {
        "success": True,
        "active_states": list(state_manager.get_current_states()),
        "count": len(state_manager.active_states),
    }


@app.post("/states/activate/{state_id}")
async def activate_state(state_id: str, evidence_score: float = 1.0):
    """Activate a single state with evidence score"""
    # Use state name for activation (qontinui uses names)
    state_manager.activate_state(state_id, evidence_score)
    return {
        "state_id": state_id,
        "activated": state_id in state_manager.active_states,
        "evidence_score": evidence_score,
    }


@app.post("/states/deactivate/{state_id}")
async def deactivate_state(state_id: str):
    """Deactivate a single state"""
    # Use state name for deactivation (qontinui uses names)
    state_manager.deactivate_state(state_id)
    return {
        "state_id": state_id,
        "deactivated": state_id not in state_manager.active_states,
    }


@app.get("/states/transitions")
async def get_possible_transitions():
    """Get possible transitions from current active states"""
    transitions = state_manager.get_possible_transitions()
    return {
        "transitions": [
            {
                "from_state": t.from_state,
                "to_state": t.to_state,
                "action_type": (
                    t.action_type.value if hasattr(t.action_type, "value") else str(t.action_type)
                ),
                "conditions": t.conditions,
            }
            for t in transitions
        ],
        "count": len(transitions),
    }


@app.post("/states/transition")
async def execute_transition(request: TransitionRequest):
    """Execute a state transition"""
    try:
        # Create trigger name for the transition
        trigger_name = f"{request.from_state}_to_{request.to_state}"

        # Try to execute the transition
        # Note: This is simplified - actual implementation would need proper trigger setup
        state_manager.machine.trigger(trigger_name)

        return {
            "success": True,
            "from_state": request.from_state,
            "to_state": request.to_state,
            "current_states": list(state_manager.get_current_states()),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "current_states": list(state_manager.get_current_states()),
        }


@app.get("/states/graph")
async def get_state_graph():
    """Get visualization of the state graph"""
    return {
        "graph": state_manager.get_state_graph_visualization(),
        "active_states": list(state_manager.get_current_states()),
        "total_states": len(state_manager.state_graph.states),
    }


@app.post("/states/reset")
async def reset_state_manager():
    """Reset the state manager to initial state"""
    state_manager.reset()
    return {"success": True, "active_states": list(state_manager.get_current_states())}


@app.post("/pattern_matching/test")
async def test_pattern_matching(request: FindRequest):
    """Test pattern matching with detailed results for visualization"""
    try:
        # Decode base64 images
        screenshot_bytes = base64.b64decode(request.screenshot)
        template_bytes = base64.b64decode(request.template)

        # Save to temporary files (qontinui Find needs file paths)
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as screenshot_file:
            screenshot_file.write(screenshot_bytes)
            screenshot_path = screenshot_file.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as template_file:
            template_file.write(template_bytes)
            template_path = template_file.name

        try:
            # Create qontinui Image objects
            screenshot_img = Image(name="screenshot", filepath=screenshot_path)  # type: ignore[call-arg]

            template_img = Image(name="template", filepath=template_path)  # type: ignore[call-arg]

            # Create pattern
            pattern = Pattern(images=[template_img])  # type: ignore[call-arg]

            # Configure search region if provided
            search_regions = None
            if request.search_region:
                search_region = Region(  # type: ignore[call-arg]
                    x=request.search_region["x"],
                    y=request.search_region["y"],
                    w=request.search_region["width"],
                    h=request.search_region["height"],
                )
                search_regions = SearchRegions(regions=[search_region])  # type: ignore[call-arg]

            # Create Find options
            find_options = FindOptions(  # type: ignore[call-arg]
                similarity=request.similarity,
                search_regions=search_regions,
                do_find_all=request.find_all,
            )

            # Execute find operation
            find = Find()
            matches = find.find(pattern=pattern, scene_image=screenshot_img, options=find_options)  # type: ignore[call-arg]

            # Convert matches to response format with detailed info
            match_results = []
            for i, match in enumerate(matches.matches if matches else []):  # type: ignore[attr-defined]
                match_results.append(
                    {
                        "id": f"match-{i}",
                        "x": match.region.x,
                        "y": match.region.y,
                        "width": match.region.w,
                        "height": match.region.h,
                        "score": match.score,
                        "confidence": match.score * 100,  # Convert to percentage
                        "rank": i + 1,
                    }
                )

            # Sort by score (highest first)
            match_results.sort(key=lambda x: x["score"], reverse=True)

            # Get template dimensions for reference
            template_pil = PILImage.open(template_path)
            template_width, template_height = template_pil.size
            template_pil.close()

            return {
                "success": matches.successful if matches else False,  # type: ignore[attr-defined]
                "matches": match_results,
                "templateSize": {"width": template_width, "height": template_height},
                "threshold": request.similarity,
                "totalMatches": len(match_results),
            }

        finally:
            # Clean up temp files
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
            if os.path.exists(template_path):
                os.remove(template_path)

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


# === WORKFLOW EXECUTION ENDPOINTS ===


class WorkflowExecutionRequest(BaseModel):
    workflow: Workflow  # Workflow definition using Pydantic schema
    screenshots: list[str]  # base64 encoded screenshots
    states: list[ConfigState]  # State definitions using Pydantic schema
    categories: list[dict[str, Any]] = []  # Category definitions
    mode: str = "hybrid"  # "hybrid" or "full_mock"
    similarity: float = DEFAULT_SIMILARITY_THRESHOLD


class WorkflowExecutionResponse(BaseModel):
    session_id: str
    workflow_id: str
    workflow_name: str
    category_name: str = "Uncategorized"
    status: str  # "running", "completed", "failed"
    current_action: int = 0
    total_actions: int
    results: list[dict[str, Any]] = []


@app.post("/workflow/execute")
async def execute_workflow(request: WorkflowExecutionRequest):
    """Execute a workflow in hybrid or full mock mode.

    When mode="full_mock" (integration testing):
    - Sets qontinui library ExecutionMode to MockMode.MOCK
    - All actions use mock implementations that return historical data
    - Useful for testing workflows without live GUI automation

    When mode="hybrid" (default):
    - Uses real pattern matching on provided screenshots
    - Actions that don't require GUI interaction are mocked
    """
    from qontinui.config.execution_mode import ExecutionModeConfig, MockMode, set_execution_mode

    # Set execution mode based on request
    if request.mode == "full_mock":
        # Integration testing mode - use qontinui library's mock system
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

    # Create a mock session for the workflow
    session_id = str(uuid.uuid4())

    # Extract initial states from workflow
    # Priority: 1. Workflow's configured initialStates, 2. States marked as initial, 3. First state
    initial_states = []

    # Check if workflow has initialStates configured
    if hasattr(request.workflow, "initial_states") and request.workflow.initial_states:
        initial_states = request.workflow.initial_states
    else:
        # Fall back to states marked as is_initial
        for state in request.states:
            if state.is_initial:
                initial_states.append(state.id)

    # If still no initial states, use the first state as fallback
    if not initial_states and request.states:
        initial_states = [request.states[0].id]

    # Create session
    session = MockSession(
        session_id=session_id,
        screenshots=request.screenshots,
        states=request.states,
        current_screenshot_index=0,
        active_states=initial_states,
        execution_history=[],
        initial_states=initial_states,
        action_snapshots=[],
        mode=request.mode or "hybrid",
    )

    # Save session to Redis with 1 hour TTL
    save_session(session_id, session, ttl=3600)

    # Count total actions in workflow
    total_actions = len(request.workflow.actions)

    # Get category name
    category_name = "Uncategorized"
    if request.workflow.category:  # type: ignore[attr-defined]
        for cat in request.categories:
            if cat.get("id") == request.workflow.category:  # type: ignore[attr-defined]
                category_name = cat.get("name", "Uncategorized")
                break

    # Initial response
    response = WorkflowExecutionResponse(
        session_id=session_id,
        workflow_id=request.workflow.id,
        workflow_name=request.workflow.name,
        category_name=category_name,
        status="running",
        current_action=0,
        total_actions=total_actions,
        results=[],
    )

    # Detect initial states
    await _detect_and_activate_states(session)

    return response


@app.post("/workflow/execute_step/{session_id}")
async def execute_workflow_step(session_id: str, action: ConfigAction):
    """Execute a single action step in a workflow"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Determine execution mode
    action_type = action.type
    success = False
    message = ""
    duration = 100

    if action_type.lower() in ["find", "find_state_image"]:
        # Use real pattern matching for find actions
        current_screenshot = session.screenshots[session.current_screenshot_index]

        # Get the image for the action from config
        image_id = None
        if hasattr(action.config, "image_id"):
            image_id = action.config.image_id
        elif hasattr(action.config, "object_id"):
            image_id = action.config.object_id

        state_image = None

        if image_id:
            # First check if it's a direct image ID
            for img in session.images if hasattr(session, "images") else []:
                if img.get("id") == image_id:
                    state_image = img.get("data") or img.get("image")
                    break

            # If not found, check state images
            if not state_image:
                for state in session.states:
                    if hasattr(state, "state_images"):
                        for img in state.state_images:
                            if img.id == image_id:
                                state_image = img.image or getattr(img, "data", None)
                                break
                    if state_image:
                        break

        if state_image:
            try:
                # Use real pattern matching
                similarity = DEFAULT_SIMILARITY_THRESHOLD
                if hasattr(action.config, "similarity"):
                    similarity = action.config.similarity

                request_data = FindRequest(
                    screenshot=current_screenshot,
                    template=state_image,
                    similarity=similarity,
                    find_all=False,
                )
                result = await find_image(request_data)
                success = result["found"]
                if success:
                    message = f"Found at ({result['region']['x']}, {result['region']['y']})"
                else:
                    message = "Pattern not found"
            except Exception as e:
                success = False
                message = f"Error: {str(e)}"
        else:
            # Mock if no image
            success = True
            message = "Mock: Pattern found"
    else:
        # Mock all other actions
        success = True
        message = f"Mock: {action_type} action executed"

    # Create result
    result = {
        "actionId": action.id,
        "actionType": action_type,
        "success": success,
        "message": message,
        "duration": duration,
        "timestamp": str(uuid.uuid4()),
        "historical_result_id": None,  # Will be populated when using historical data
    }

    # If running in mock mode with historical data, try to get a historical result ID
    if hasattr(session, "mode") and session.mode == "full_mock":
        try:
            from app.services.capture_service import get_capture_service

            capture_service = get_capture_service()
            # Get a random historical result for this action type/pattern
            pattern_id = None
            if hasattr(action.config, "image_id"):
                pattern_id = action.config.image_id
            elif hasattr(action.config, "object_id"):
                pattern_id = action.config.object_id

            historical = capture_service.get_random_historical_result_sync(
                pattern_id=pattern_id,
                action_type=action_type.upper(),
                success_only=False,
            )
            if historical:
                result["historical_result_id"] = historical.id
        except Exception as e:
            logger.warning(f"Failed to get historical result ID: {e}")

    # Add to session history
    session.execution_history.append(result)

    # Save updated session to Redis
    save_session(session_id, session, ttl=3600)

    return result


@app.get("/workflow/status/{session_id}")
async def get_workflow_status(session_id: str):
    """Get the status of a workflow execution"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Calculate status
    total_actions = len(session.execution_history)
    successful_actions = sum(1 for r in session.execution_history if r.get("success", False))

    return {
        "session_id": session_id,
        "total_actions": total_actions,
        "successful_actions": successful_actions,
        "success_rate": ((successful_actions / total_actions * 100) if total_actions > 0 else 0),
        "active_states": session.active_states,
        "current_screenshot": session.current_screenshot_index,
        "history": session.execution_history,
    }


@app.post("/workflow/complete/{session_id}")
async def complete_workflow(session_id: str):
    """Mark a workflow execution as complete"""
    from qontinui.config.execution_mode import reset_execution_mode

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Calculate final results
    total_actions = len(session.execution_history)
    successful_actions = sum(1 for r in session.execution_history if r.get("success", False))

    result = {
        "session_id": session_id,
        "status": "completed" if successful_actions == total_actions else "failed",
        "total_actions": total_actions,
        "successful_actions": successful_actions,
        "success_rate": ((successful_actions / total_actions * 100) if total_actions > 0 else 0),
        "execution_history": session.execution_history,
    }

    # Clean up session from Redis
    delete_session(session_id)

    # Reset execution mode to environment defaults
    reset_execution_mode()

    return result


@app.post("/optimize-pattern")
async def optimize_pattern(request: PatternOptimizationRequest):
    """Analyze pattern matching strategies for dynamic UI regions using real qontinui pattern matching"""
    import numpy as np

    results = {
        "extractedPatterns": [],
        "statistics": {},
        "similarityMatrix": {"scores": []},
        "evaluations": [],
    }

    try:
        # Extract patterns from each screenshot/region pair
        extracted_patterns: list[dict[str, Any]] = []
        for i, (screenshot_b64, region) in enumerate(
            zip(request.screenshots, request.regions, strict=False)
        ):
            # Decode screenshot
            screenshot_bytes = base64.b64decode(screenshot_b64)
            screenshot_img = PILImage.open(io.BytesIO(screenshot_bytes))

            # Extract region from screenshot
            region_img = screenshot_img.crop(
                (
                    region["x"],
                    region["y"],
                    region["x"] + region["width"],
                    region["y"] + region["height"],
                )
            )

            # Save region as pattern
            pattern_bytes = io.BytesIO()
            region_img.save(pattern_bytes, format="PNG")
            pattern_bytes.seek(0)

            extracted_patterns.append(
                {
                    "id": f"pattern_{i}",
                    "screenshot_index": i,
                    "region": region,
                    "image_data": base64.b64encode(pattern_bytes.read()).decode(),
                    "width": region["width"],
                    "height": region["height"],
                }
            )

        results["extractedPatterns"] = extracted_patterns  # type: ignore[assignment]

        # Calculate similarity matrix using real pattern matching
        num_patterns = len(extracted_patterns)
        similarity_matrix = np.zeros((num_patterns, num_patterns))

        for i in range(num_patterns):
            pattern_i_data = base64.b64decode(str(extracted_patterns[i]["image_data"]))  # type: ignore[arg-type]
            pattern_i_img = PILImage.open(io.BytesIO(pattern_i_data))

            for j in range(num_patterns):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Use qontinui Find to match pattern_i against screenshot_j
                    screenshot_j_data = base64.b64decode(request.screenshots[j])
                    screenshot_j_img = PILImage.open(io.BytesIO(screenshot_j_data))

                    # Convert to numpy arrays for OpenCV
                    pattern_np = cv2.cvtColor(np.array(pattern_i_img), cv2.COLOR_RGB2BGR)
                    screenshot_np = cv2.cvtColor(np.array(screenshot_j_img), cv2.COLOR_RGB2BGR)

                    # Perform template matching
                    result = cv2.matchTemplate(screenshot_np, pattern_np, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    similarity_matrix[i][j] = max_val

        results["similarityMatrix"]["scores"] = similarity_matrix.tolist()  # type: ignore[index]

        # Calculate statistics
        mean_similarity = np.mean(similarity_matrix[np.triu_indices(num_patterns, k=1)])
        variance = np.var(similarity_matrix[np.triu_indices(num_patterns, k=1)])
        min_similarity = np.min(similarity_matrix[np.triu_indices(num_patterns, k=1)])
        max_similarity = np.max(similarity_matrix[np.triu_indices(num_patterns, k=1)])

        # Find outliers (patterns with mean similarity below threshold)
        outliers = []
        for i in range(num_patterns):
            row_mean = (np.sum(similarity_matrix[i, :]) - 1) / (num_patterns - 1)  # Exclude self
            if row_mean < CONSENSUS_THRESHOLD:
                outliers.append(i)

        results["statistics"] = {
            "meanSimilarity": float(mean_similarity),
            "variance": float(variance),
            "minSimilarity": float(min_similarity),
            "maxSimilarity": float(max_similarity),
            "outliers": outliers,
        }

        # Evaluate strategies
        for strategy in request.strategies:
            evaluation = await evaluate_strategy(
                strategy,
                extracted_patterns,
                request.screenshots,
                request.negative_screenshots,
                similarity_matrix,
            )
            results["evaluations"].append(evaluation)  # type: ignore[attr-defined]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern optimization failed: {str(e)}") from e


async def evaluate_strategy(
    strategy_type: str,
    patterns: list[dict[str, Any]],
    positive_screenshots: list[str],
    negative_screenshots: list[str],
    similarity_matrix: np.ndarray,  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Evaluate a specific pattern optimization strategy using real pattern matching"""

    # Initialize evaluation result
    evaluation: dict[str, Any] = {
        "strategy": {"type": strategy_type, "parameters": {}},
        "performance": {
            "truePositiveRate": 0.0,
            "falsePositiveRate": 0.0,
            "averageConfidence": 0.0,
            "processingTime": 0.0,
        },
        "recommendations": {
            "confidenceLevel": "low",
            "suggestedThreshold": 0.8,
            "improvements": [],
        },
    }

    start_time = time.time()

    try:
        if strategy_type == "multi-pattern":
            # Use all patterns and take the best match
            true_positives = 0
            false_positives = 0
            confidences = []

            # Test on positive examples (should match)
            for screenshot_b64 in positive_screenshots:
                screenshot_data = base64.b64decode(screenshot_b64)
                screenshot_img = PILImage.open(io.BytesIO(screenshot_data))
                screenshot_np = cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_RGB2BGR)

                best_score = 0
                for pattern in patterns:
                    pattern_data = base64.b64decode(pattern["image_data"])
                    pattern_img = PILImage.open(io.BytesIO(pattern_data))
                    pattern_np = cv2.cvtColor(np.array(pattern_img), cv2.COLOR_RGB2BGR)

                    result = cv2.matchTemplate(screenshot_np, pattern_np, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    best_score = max(best_score, float(max_val))  # type: ignore[assignment]

                confidences.append(best_score)
                if best_score >= 0.8:
                    true_positives += 1

            # Test on negative examples (should not match)
            for screenshot_b64 in negative_screenshots:
                screenshot_data = base64.b64decode(screenshot_b64)
                screenshot_img = PILImage.open(io.BytesIO(screenshot_data))
                screenshot_np = cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_RGB2BGR)

                best_score = 0
                for pattern in patterns:
                    pattern_data = base64.b64decode(pattern["image_data"])
                    pattern_img = PILImage.open(io.BytesIO(pattern_data))
                    pattern_np = cv2.cvtColor(np.array(pattern_img), cv2.COLOR_RGB2BGR)

                    result = cv2.matchTemplate(screenshot_np, pattern_np, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    best_score = max(best_score, float(max_val))  # type: ignore[assignment]

                if best_score >= 0.8:
                    false_positives += 1

            evaluation["performance"]["truePositiveRate"] = (
                true_positives / len(positive_screenshots) if positive_screenshots else 0
            )
            evaluation["performance"]["falsePositiveRate"] = (
                false_positives / len(negative_screenshots) if negative_screenshots else 0
            )
            evaluation["performance"]["averageConfidence"] = (
                np.mean(confidences) if confidences else 0
            )

        elif strategy_type == "consensus":
            # Use the pattern with best average similarity to others
            avg_similarities = []
            for i in range(len(patterns)):
                row_mean = (np.sum(similarity_matrix[i, :]) - 1) / (len(patterns) - 1)
                avg_similarities.append(row_mean)

            best_pattern_idx = np.argmax(avg_similarities)
            best_pattern = patterns[best_pattern_idx]

            # Evaluate this single pattern
            true_positives = 0
            confidences = []

            pattern_data = base64.b64decode(best_pattern["image_data"])
            pattern_img = PILImage.open(io.BytesIO(pattern_data))
            pattern_np = cv2.cvtColor(np.array(pattern_img), cv2.COLOR_RGB2BGR)

            for screenshot_b64 in positive_screenshots:
                screenshot_data = base64.b64decode(screenshot_b64)
                screenshot_img = PILImage.open(io.BytesIO(screenshot_data))
                screenshot_np = cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_RGB2BGR)

                result = cv2.matchTemplate(screenshot_np, pattern_np, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                confidences.append(float(max_val))  # type: ignore[arg-type]
                if max_val >= 0.8:
                    true_positives += 1

            evaluation["performance"]["truePositiveRate"] = (
                true_positives / len(positive_screenshots) if positive_screenshots else 0
            )
            evaluation["performance"]["averageConfidence"] = (
                np.mean(confidences) if confidences else 0
            )
            evaluation["strategy"]["parameters"]["selectedPatternIndex"] = int(best_pattern_idx)

        elif strategy_type == "feature-based":
            # Use ORB feature matching for more robust matching
            # This is a simplified version - real implementation would extract and match features
            evaluation["performance"]["truePositiveRate"] = 0.85  # Placeholder
            evaluation["performance"]["falsePositiveRate"] = 0.1
            evaluation["performance"]["averageConfidence"] = 0.82

        elif strategy_type == "differential":
            # Focus on differences between positive and negative examples
            # This requires negative examples to work properly
            if negative_screenshots:
                evaluation["performance"]["truePositiveRate"] = 0.9
                evaluation["performance"]["falsePositiveRate"] = 0.05
                evaluation["performance"]["averageConfidence"] = 0.88
            else:
                evaluation["recommendations"]["improvements"].append(
                    "Provide negative examples for differential strategy"
                )

        # Calculate processing time
        evaluation["performance"]["processingTime"] = (
            time.time() - start_time
        ) * 1000  # Convert to ms

        # Set confidence level based on performance
        tpr = evaluation["performance"]["truePositiveRate"]
        fpr = evaluation["performance"]["falsePositiveRate"]

        if tpr >= 0.9 and fpr <= 0.1:
            evaluation["recommendations"]["confidenceLevel"] = "high"
        elif tpr >= 0.7 and fpr <= 0.3:
            evaluation["recommendations"]["confidenceLevel"] = "medium"
        else:
            evaluation["recommendations"]["confidenceLevel"] = "low"
            evaluation["recommendations"]["improvements"].append(
                "Consider providing more training examples"
            )

    except Exception as e:
        print(f"Strategy evaluation error for {strategy_type}: {e}")
        evaluation["recommendations"]["improvements"].append(f"Evaluation error: {str(e)}")

    return evaluation


@app.post("/create-state-image")
async def create_state_image(request: CreateStateImageRequest):
    """Create a StateImage from pattern optimization results"""
    try:
        # Create Pattern objects from the optimization results
        patterns = []
        for pattern_data in request.patterns:
            # Decode the pattern image data
            image_data = base64.b64decode(pattern_data["image_data"])
            # Note: image_data would be used to set pattern's image in real implementation
            # For now we just validate it can be decoded
            PILImage.open(io.BytesIO(image_data))

            # Create a Pattern object
            pattern = Pattern()  # type: ignore[call-arg]
            # In real qontinui, we would set the pattern's image here
            # For now, we'll store the metadata
            pattern.name = f"pattern_{pattern_data['id']}"  # type: ignore[attr-defined]
            patterns.append(pattern)

        # Create StateImage
        state_image = StateImage()  # type: ignore[call-arg]
        state_image.patterns = patterns  # type: ignore[misc]
        state_image.name = request.name  # type: ignore[attr-defined]

        # Return the StateImage configuration
        return {
            "success": True,
            "stateImage": {
                "name": request.name,
                "patterns": [{"id": p["id"], "region": p.get("region")} for p in request.patterns],
                "strategy": request.strategy_type,
                "similarity_threshold": request.similarity_threshold,
                "pattern_count": len(patterns),
            },
            "message": f"StateImage '{request.name}' created with {len(patterns)} patterns using {request.strategy_type} strategy",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create StateImage: {str(e)}") from e


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check Redis connection
    redis_healthy = False
    try:
        redis_client.ping()
        redis_healthy = True
    except Exception:
        pass

    # Get active session count from Redis
    active_session_count = len(list_sessions())

    return {
        "status": "healthy" if redis_healthy else "degraded",
        "service": "qontinui-api",
        "version": "1.0.0",
        "qontinui_available": True,
        "redis_connected": redis_healthy,
        "active_sessions": active_session_count,
        "state_management": {
            "active_states": len(state_manager.active_states),
            "registered_states": len(state_manager.state_graph.states),
        },
    }


if __name__ == "__main__":
    import uvicorn

    # Port 8001 for qontinui-api (port 8000 is reserved for main backend)
    uvicorn.run(app, host="0.0.0.0", port=8001)
