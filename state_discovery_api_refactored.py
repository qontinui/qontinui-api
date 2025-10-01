"""State Discovery API - Refactored with Single Responsibility Principle.

This module coordinates between the different state discovery components.
Single Responsibility: API endpoint handling and request/response coordination.
"""

import hashlib
import io
import json
import logging
import os

# Import State Discovery components from qontinui
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from PIL import Image as PILImage
from pydantic import BaseModel

from .state_discovery.analysis_handler import task_runner
from .state_discovery.upload_storage import storage

# Import our refactored modules
from .state_discovery.websocket_manager import manager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qontinui.discovery.models import AnalysisConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/state-discovery", tags=["State Discovery"])

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=4)


# Request/Response Models
class UploadResponse(BaseModel):
    upload_id: str
    project_id: str
    screenshots: list[dict[str, Any]]
    total_size: int
    count: int


class RegionBounds(BaseModel):
    x: int
    y: int
    width: int
    height: int


class AnalysisConfigRequest(BaseModel):
    min_region_size: list[int] = [20, 20]
    max_region_size: list[int] = [500, 500]
    color_tolerance: int = 5
    stability_threshold: float = 0.98
    variance_threshold: float = 10.0
    min_screenshots_present: int = 2
    processing_mode: str = "full"
    enable_rectangle_decomposition: bool = True
    enable_cooccurrence_analysis: bool = True
    similarity_threshold: float = 0.95
    region: RegionBounds | None = None


class AnalysisRequest(BaseModel):
    upload_id: str
    config: AnalysisConfigRequest


# API Endpoints


@router.post("/upload", response_model=UploadResponse)
async def upload_screenshots(files: list[UploadFile] = File(...), project_id: str = "default"):
    """Upload screenshots for analysis.

    Single Responsibility: Handle HTTP upload request and convert to internal format.
    """
    print(f"[API] Uploading {len(files)} files for project {project_id}")
    logger.info(f"Uploading {len(files)} files for project {project_id}")

    screenshots = []
    metadata = []
    total_size = 0

    for i, file in enumerate(files):
        try:
            # Read and decode image
            content = await file.read()
            total_size += len(content)

            # Validate file type
            if not file.filename or not any(
                file.filename.lower().endswith(ext)
                for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
            ):
                logger.warning(f"Skipping non-image file: {file.filename}")
                continue

            # Convert to numpy array
            img = PILImage.open(io.BytesIO(content))
            img_array = np.array(img)
            screenshots.append(img_array)

            # Calculate hash for the image
            pixel_hash = hashlib.sha256(content).hexdigest()[:16]

            # Create metadata
            metadata.append(
                {
                    "id": f"screenshot_{i:03d}",
                    "name": file.filename,
                    "size": len(content),
                    "width": img.width,
                    "height": img.height,
                    "pixel_hash": pixel_hash,
                }
            )

            print(f"[API] Processed {file.filename}: {img.width}x{img.height}")

        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {e}")
            continue

    if not screenshots:
        raise HTTPException(status_code=400, detail="No valid images found in upload")

    # Store upload using our storage module
    upload_id = storage.store_upload(screenshots, metadata)

    # Update timestamp
    upload = storage.get_upload(upload_id)
    if upload:
        upload.timestamp = datetime.now().timestamp()

    print(f"[API] Upload complete: {upload_id} with {len(screenshots)} screenshots")
    logger.info(f"Upload complete: {upload_id} with {len(screenshots)} screenshots")

    return UploadResponse(
        upload_id=upload_id,
        project_id=project_id,
        screenshots=metadata,
        total_size=total_size,
        count=len(screenshots),
    )


@router.post("/analyze")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start pixel differential analysis on uploaded screenshots.

    Single Responsibility: Handle analysis request and delegate to background task.
    """
    print(f"[API] Starting analysis for upload {request.upload_id}")
    logger.info(f"Starting analysis for upload {request.upload_id}")

    # Get upload from storage
    upload = storage.get_upload(request.upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Create analysis ID
    analysis_id = f"analysis_{uuid.uuid4().hex[:12]}"

    # Convert request config to internal config
    config = AnalysisConfig(
        min_region_size=tuple(request.config.min_region_size),
        max_region_size=tuple(request.config.max_region_size),
        color_tolerance=request.config.color_tolerance,
        stability_threshold=request.config.stability_threshold,
        variance_threshold=request.config.variance_threshold,
        min_screenshots_present=request.config.min_screenshots_present,
        processing_mode=request.config.processing_mode,
        enable_rectangle_decomposition=request.config.enable_rectangle_decomposition,
        enable_cooccurrence_analysis=request.config.enable_cooccurrence_analysis,
    )

    # Convert region bounds if provided
    region = None
    if request.config.region:
        r = request.config.region
        region = (r.x, r.y, r.width, r.height)
        print(f"[API] Using region: ({r.x}, {r.y}) {r.width}x{r.height}")

    # Submit analysis to background task runner
    print(f"[API] Submitting analysis {analysis_id} to background task runner")
    logger.info(f"Submitting analysis {analysis_id} to background task runner")

    # Use thread pool executor instead of FastAPI background tasks
    # This ensures the task runs in a separate thread with its own event loop
    _future = executor.submit(task_runner.run_analysis_sync, analysis_id, upload, config, region)

    print(f"[API] Analysis {analysis_id} submitted successfully")
    logger.info(f"Analysis {analysis_id} submitted successfully")

    return {
        "analysis_id": analysis_id,
        "status": "queued",
        "estimated_time_seconds": 30,
        "websocket_url": f"ws://localhost:8000/api/state-discovery/ws/{analysis_id}",
    }


@router.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """WebSocket for real-time analysis updates.

    Single Responsibility: Handle WebSocket connection lifecycle.
    """
    print(f"[API-WS] WebSocket connection request for {analysis_id}")
    await manager.connect(analysis_id, websocket)

    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()

            # Handle client messages
            try:
                message = json.loads(data)
                print(f"[API-WS] Received message from {analysis_id}: {message.get('type')}")

                if message.get("type") == "ping":
                    # Respond to ping
                    await manager.send_update(analysis_id, {"type": "pong"})
                elif message.get("type") == "cancel":
                    # TODO: Implement analysis cancellation
                    print(f"[API-WS] Cancel requested for {analysis_id}")
                    pass

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from {analysis_id}: {data}")

    except WebSocketDisconnect:
        print(f"[API-WS] WebSocket disconnected for {analysis_id}")
        manager.disconnect(analysis_id)


@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get analysis status.

    Single Responsibility: Return current analysis status.
    """
    # Check if analysis is in active connections (still running)
    if analysis_id in manager.active_connections:
        return {
            "analysis_id": analysis_id,
            "status": "processing",
            "progress": {"percentage": 50, "stage": "analyzing"},
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    # TODO: Check if analysis is complete in storage
    return {
        "analysis_id": analysis_id,
        "status": "complete",
        "progress": {"percentage": 100, "stage": "complete"},
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


@router.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Get complete analysis results.

    Single Responsibility: Retrieve and return analysis results.
    """
    # TODO: Retrieve from results storage
    # For now, return mock data
    return {
        "analysis_id": analysis_id,
        "states": [],
        "state_images": [],
        "statistics": {"total_states": 0, "total_state_images": 0, "analysis_time": 0},
    }


@router.post("/cancel/{analysis_id}")
async def cancel_analysis(analysis_id: str):
    """Cancel running analysis.

    Single Responsibility: Handle analysis cancellation request.
    """
    # TODO: Implement actual cancellation
    print(f"[API] Cancellation requested for {analysis_id}")
    logger.info(f"Cancellation requested for {analysis_id}")

    return {
        "analysis_id": analysis_id,
        "status": "cancelled",
        "message": "Analysis cancelled by user",
    }


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "uploads_count": len(storage.uploads),
        "active_websockets": len(manager.active_connections),
    }
