"""
State Discovery API for automated state and StateImage detection.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os

# Import State Discovery components from qontinui
import sys
import uuid
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
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qontinui.discovery import (
    AnalysisResult,
    PixelStabilityMatrixAnalyzer,
)
from qontinui.discovery.deletion_manager import DeletionManager
from qontinui.discovery.models import AnalysisConfig, DeleteOptions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer | np.floating):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Create router
router = APIRouter(prefix="/state-discovery", tags=["State Discovery"])


# In-memory storage for demo (replace with database in production)
class StateDiscoveryStore:
    def __init__(self):
        self.uploads = {}  # upload_id -> screenshots
        self.analyses = {}  # analysis_id -> results
        self.state_structures = {}  # project_id -> state structure
        self.websocket_connections = {}  # analysis_id -> websocket
        self.deletion_manager = DeletionManager()
        self.project_screenshots = {}  # project_id -> {screenshot_id: screenshot_data}
        self.screenshot_hashes = {}  # hash -> screenshot_id

    def store_upload(self, upload_id: str, screenshots: list[np.ndarray], metadata: dict):
        self.uploads[upload_id] = {
            "screenshots": screenshots,
            "metadata": metadata,
            "timestamp": datetime.now(),
        }

    def get_upload(self, upload_id: str):
        return self.uploads.get(upload_id)

    def store_analysis(self, analysis_id: str, result: AnalysisResult):
        self.analyses[analysis_id] = result

    def get_analysis(self, analysis_id: str):
        return self.analyses.get(analysis_id)

    def store_project_screenshot(self, project_id: str, screenshot_data: dict):
        """Store a screenshot in the project with duplicate detection."""
        if project_id not in self.project_screenshots:
            self.project_screenshots[project_id] = {}

        # Calculate hash for duplicate detection
        img_bytes = screenshot_data["image_bytes"]
        hash_value = hashlib.sha256(img_bytes).hexdigest()

        # Check if this hash already exists
        if hash_value in self.screenshot_hashes:
            return None, hash_value  # Return None to indicate duplicate

        # Generate unique ID
        screenshot_id = f"ps_{uuid.uuid4().hex[:8]}"

        # Store screenshot
        screenshot_data["id"] = screenshot_id
        screenshot_data["hash"] = hash_value
        screenshot_data["created_at"] = datetime.now().isoformat()

        self.project_screenshots[project_id][screenshot_id] = screenshot_data
        self.screenshot_hashes[hash_value] = screenshot_id

        return screenshot_id, hash_value

    def get_project_screenshots(self, project_id: str):
        """Get all screenshots for a project."""
        return self.project_screenshots.get(project_id, {})


# Global store instance
store = StateDiscoveryStore()


# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, analysis_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[analysis_id] = websocket

    def disconnect(self, analysis_id: str):
        if analysis_id in self.active_connections:
            del self.active_connections[analysis_id]

    async def send_update(self, analysis_id: str, message: dict):
        if analysis_id in self.active_connections:
            try:
                # Use custom encoder to handle numpy types
                json_str = json.dumps(message, cls=NumpyEncoder)
                await self.active_connections[analysis_id].send_text(json_str)
            except Exception as e:
                logger.error(f"Error sending update: {e}")


manager = ConnectionManager()


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


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: str
    progress: dict[str, Any] | None = None
    started_at: str
    updated_at: str


class StateImageUpdateRequest(BaseModel):
    name: str | None = None
    x: int | None = None
    y: int | None = None
    x2: int | None = None
    y2: int | None = None
    tags: list[str] | None = None


class BulkDeleteRequest(BaseModel):
    ids: list[str]
    options: dict[str, Any] | None = None


class SaveStructureRequest(BaseModel):
    project_id: str
    analysis_id: str
    name: str
    description: str | None = None


class ProjectScreenshot(BaseModel):
    id: str
    name: str
    hash: str
    size: int
    created_at: str
    thumbnail_url: str | None = None


class ProjectScreenshotResponse(BaseModel):
    screenshots: list[ProjectScreenshot]
    count: int


class SaveScreenshotResponse(BaseModel):
    saved: list[dict[str, Any]]
    duplicates: list[dict[str, Any]]
    total_saved: int
    total_duplicates: int


# Endpoints


@router.post("/upload", response_model=UploadResponse)
async def upload_screenshots(files: list[UploadFile] = File(...), project_id: str = "default"):
    """Upload screenshots for analysis."""
    upload_id = f"upload_{uuid.uuid4().hex[:12]}"
    screenshots = []
    screenshot_metadata = []
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

            # Create metadata
            screenshot_metadata.append(
                {
                    "id": f"screenshot_{i:03d}",
                    "filename": file.filename,
                    "size": len(content),
                    "dimensions": {"width": img.width, "height": img.height},
                }
            )
        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {e}")
            continue

    if not screenshots:
        raise HTTPException(status_code=400, detail="No valid images found in upload")

    # Store upload
    store.store_upload(
        upload_id, screenshots, {"project_id": project_id, "files": screenshot_metadata}
    )

    return UploadResponse(
        upload_id=upload_id,
        project_id=project_id,
        screenshots=screenshot_metadata,
        total_size=total_size,
        count=len(screenshots),
    )


@router.post("/analyze")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start pixel differential analysis on uploaded screenshots."""
    # Get upload
    upload = store.get_upload(request.upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Create analysis ID
    analysis_id = f"analysis_{uuid.uuid4().hex[:12]}"

    # Get screenshots
    screenshots = upload["screenshots"]

    # Crop screenshots if region is specified
    if request.config.region:
        region = request.config.region
        cropped_screenshots = []

        logger.info(f"Cropping to region: ({region.x},{region.y}) {region.width}x{region.height}")

        for i, screenshot in enumerate(screenshots):
            original_shape = screenshot.shape
            # Ensure bounds are within image
            x = max(0, min(region.x, screenshot.shape[1]))
            y = max(0, min(region.y, screenshot.shape[0]))
            x2 = max(0, min(region.x + region.width, screenshot.shape[1]))
            y2 = max(0, min(region.y + region.height, screenshot.shape[0]))

            # Crop the screenshot
            cropped = screenshot[y:y2, x:x2]
            cropped_screenshots.append(cropped)

            logger.info(f"Screenshot {i}: {original_shape} -> {cropped.shape}")

        screenshots = cropped_screenshots
        logger.info(
            f"Cropped {len(screenshots)} screenshots, new size: {screenshots[0].shape if screenshots else 'N/A'}"
        )

    # Convert config
    # Note: similarity_threshold from UI is converted to color_tolerance in frontend
    # The color_tolerance is what controls pixel matching tolerance in the analyzer
    config = AnalysisConfig(
        min_region_size=tuple(request.config.min_region_size),
        max_region_size=tuple(request.config.max_region_size),
        color_tolerance=request.config.color_tolerance,  # Derived from similarity_threshold in UI
        stability_threshold=request.config.stability_threshold,
        variance_threshold=request.config.variance_threshold,
        min_screenshots_present=request.config.min_screenshots_present,
        processing_mode=request.config.processing_mode,
        enable_rectangle_decomposition=request.config.enable_rectangle_decomposition,
        enable_cooccurrence_analysis=request.config.enable_cooccurrence_analysis,
    )
    # Store similarity_threshold for masked pattern analysis
    if hasattr(config, "similarity_threshold") and hasattr(request.config, "similarity_threshold"):
        config.similarity_threshold = request.config.similarity_threshold

    # Start analysis in background
    print(f"[DEBUG] Adding background task for analysis {analysis_id}")
    print(f"[DEBUG] Screenshots count: {len(screenshots)}, Config: {config}")
    logger.info(f"Adding background task for analysis {analysis_id}")
    logger.info(f"Screenshots count: {len(screenshots)}, Config: {config}")

    background_tasks.add_task(
        run_analysis_sync,
        analysis_id,
        screenshots,
        config,
        region_offset=(
            (request.config.region.x, request.config.region.y) if request.config.region else None
        ),
    )

    print(f"[DEBUG] Background task added for {analysis_id}")
    logger.info(f"Background task added for {analysis_id}")

    return {
        "analysis_id": analysis_id,
        "status": "queued",
        "estimated_time_seconds": 30,
        "websocket_url": f"ws://localhost:8000/api/state-discovery/ws/{analysis_id}",
    }


def run_analysis_sync(
    analysis_id: str,
    screenshots: list[np.ndarray],
    config: AnalysisConfig,
    region_offset: tuple | None = None,
):
    """Run analysis in background (sync wrapper for background tasks)."""
    print(f"[BACKGROUND TASK] Starting run_analysis_sync for {analysis_id}")
    print(f"[BACKGROUND TASK] Screenshots: {len(screenshots)}, Region offset: {region_offset}")
    logger.info(f"[BACKGROUND TASK] Starting run_analysis_sync for {analysis_id}")
    logger.info(
        f"[BACKGROUND TASK] Screenshots: {len(screenshots)}, Region offset: {region_offset}"
    )

    # Create a new event loop for this thread

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        print(f"[BACKGROUND TASK] Running async analysis for {analysis_id}")
        logger.info(f"[BACKGROUND TASK] Running async analysis for {analysis_id}")
        # Run the async function
        loop.run_until_complete(run_analysis(analysis_id, screenshots, config, region_offset))
        print(f"[BACKGROUND TASK] Analysis complete for {analysis_id}")
        logger.info(f"[BACKGROUND TASK] Analysis complete for {analysis_id}")
    except Exception as e:
        print(f"[BACKGROUND TASK] Error in run_analysis_sync: {e}")
        logger.error(f"[BACKGROUND TASK] Error in run_analysis_sync: {e}")
        import traceback

        print(traceback.format_exc())
        logger.error(traceback.format_exc())
    finally:
        # Clean up the loop
        loop.close()
        print(f"[BACKGROUND TASK] Closed event loop for {analysis_id}")
        logger.info(f"[BACKGROUND TASK] Closed event loop for {analysis_id}")


async def run_analysis(
    analysis_id: str,
    screenshots: list[np.ndarray],
    config: AnalysisConfig,
    region_offset: tuple | None = None,
):
    """Run analysis in background with progress updates."""
    try:
        # Create analyzer - use Matrix analyzer for non-rectangular shape discovery
        # The Matrix analyzer finds actual shapes, not just rectangular regions
        analyzer = PixelStabilityMatrixAnalyzer(config)
        logger.info("Using Pixel Stability Matrix analyzer for non-rectangular shape discovery")

        # Progress callback
        async def progress_callback(progress_data):
            await manager.send_update(analysis_id, {"type": "progress", "data": progress_data})

        # Run analysis with progress updates
        # Note: This is a simplified version. In production, you'd need
        # to make the analyzer async or run in thread pool
        await manager.send_update(
            analysis_id,
            {
                "type": "progress",
                "data": {
                    "stage": "pixel_analysis",
                    "percentage": 0,
                    "message": "Starting analysis...",
                },
            },
        )

        # Simulate analysis steps with progress
        import time

        analysis_start = time.time()

        logger.info(f"Starting analysis for {len(screenshots)} screenshots")
        logger.info(f"Screenshot shapes: {[s.shape for s in screenshots[:3]]}...")  # Show first 3
        if region_offset:
            logger.info(f"Using region offset: ({region_offset[0]}, {region_offset[1]})")

        # Simply run the analysis without signal alarms (they don't work well in threads)
        try:
            result = analyzer.analyze_screenshots(screenshots)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

        analysis_time = time.time() - analysis_start
        logger.info(
            f"Analysis complete in {analysis_time:.3f}s: {len(result.states)} states, {len(result.state_images)} state images"
        )

        # Adjust state image positions if region was cropped
        if region_offset:
            for state_image in result.state_images:
                state_image.x += region_offset[0]
                state_image.y += region_offset[1]
                state_image.x2 += region_offset[0]
                state_image.y2 += region_offset[1]

        # Send discovered StateImages
        for state_image in result.state_images:
            await manager.send_update(
                analysis_id, {"type": "state_image_found", "data": state_image.to_dict()}
            )

        # Store results
        store.store_analysis(analysis_id, result)

        # Send completion
        await manager.send_update(
            analysis_id,
            {
                "type": "complete",
                "data": {
                    "analysis_id": analysis_id,
                    "states": [s.to_dict() for s in result.states],
                    "state_images": [si.to_dict() for si in result.state_images],
                    "statistics": result.statistics,
                },
            },
        )

    except Exception as e:
        import traceback

        logger.error(f"Analysis failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        await manager.send_update(
            analysis_id, {"type": "error", "data": {"code": "ANALYSIS_FAILED", "message": str(e)}}
        )


@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get analysis status."""
    result = store.get_analysis(analysis_id)

    if result:
        return {
            "analysis_id": analysis_id,
            "status": "complete",
            "progress": {"percentage": 100, "stage": "complete"},
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    # Check if analysis exists but not complete
    if analysis_id in store.analyses:
        return {
            "analysis_id": analysis_id,
            "status": "processing",
            "progress": {"percentage": 50, "stage": "processing"},
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    raise HTTPException(status_code=404, detail="Analysis not found")


@router.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Get complete analysis results."""
    result = store.get_analysis(analysis_id)

    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Return the result dictionary directly
    result_dict = result.to_dict()
    result_dict["analysis_id"] = analysis_id
    return result_dict


@router.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """WebSocket for real-time analysis updates."""
    await manager.connect(analysis_id, websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()

            # Handle client messages
            message = json.loads(data)
            if message.get("type") == "cancel":
                # Cancel analysis
                pass
            elif message.get("type") == "pause":
                # Pause analysis
                pass

    except WebSocketDisconnect:
        manager.disconnect(analysis_id)


# StateImage Management


@router.get("/state-image/{state_image_id}/delete-impact")
async def check_deletion_impact(state_image_id: str):
    """Check the impact of deleting a StateImage."""
    try:
        impact = store.deletion_manager.analyze_deletion_impact(state_image_id)
        return impact.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/state-image/{state_image_id}")
async def delete_state_image(state_image_id: str, cascade: bool = False, force: bool = False):
    """Delete a StateImage."""
    options = DeleteOptions(cascade=cascade, force=force)

    try:
        result = store.deletion_manager.delete_state_image(state_image_id, options)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/state-image/{state_image_id}")
async def update_state_image(state_image_id: str, request: StateImageUpdateRequest):
    """Update StateImage properties."""
    # In production, this would update the database
    # For now, return mock response
    return {
        "id": state_image_id,
        "name": request.name,
        "x": request.x,
        "y": request.y,
        "x2": request.x2,
        "y2": request.y2,
        "updated_at": datetime.now().isoformat(),
    }


@router.delete("/state-images/batch")
async def delete_bulk_state_images(request: BulkDeleteRequest):
    """Delete multiple StateImages."""
    options = DeleteOptions(**request.options) if request.options else DeleteOptions()

    try:
        result = store.deletion_manager.delete_bulk_state_images(request.ids, options)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/state-image/merge")
async def merge_state_images(
    source_ids: list[str], target_name: str, merge_strategy: str = "union"
):
    """Merge multiple StateImages."""
    # Mock implementation
    return {
        "id": f"si_merged_{uuid.uuid4().hex[:8]}",
        "name": target_name,
        "x": 100,
        "y": 100,
        "x2": 200,
        "y2": 200,
        "source_count": len(source_ids),
    }


# State Structure Management


@router.post("/save-structure")
async def save_state_structure(request: SaveStructureRequest):
    """Save analyzed state structure."""
    structure_id = f"struct_{uuid.uuid4().hex[:12]}"

    # Get analysis results
    result = store.get_analysis(request.analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Store structure
    store.state_structures[request.project_id] = {
        "id": structure_id,
        "name": request.name,
        "description": request.description,
        "states": request.states or [s.to_dict() for s in result.states],
        "state_images": request.state_images or [si.to_dict() for si in result.state_images],
        "created_at": datetime.now().isoformat(),
    }

    return {
        "structure_id": structure_id,
        "project_id": request.project_id,
        "name": request.name,
        "state_count": len(result.states),
        "state_image_count": len(result.state_images),
        "created_at": datetime.now().isoformat(),
    }


@router.get("/export/{structure_id}")
async def export_state_structure(
    structure_id: str, format: str = "json", include_images: bool = False
):
    """Export state structure."""
    # Find structure
    structure = None
    for _proj_id, struct in store.state_structures.items():
        if struct["id"] == structure_id:
            structure = struct
            break

    if not structure:
        raise HTTPException(status_code=404, detail="Structure not found")

    return {"version": "1.0", "exported_at": datetime.now().isoformat(), "structure": structure}


# Utility endpoints


@router.get("/thumbnails/{screenshot_id}")
async def get_thumbnail(screenshot_id: str):
    """Get thumbnail for a screenshot."""
    # In production, this would return actual thumbnail
    # For now, return placeholder
    return JSONResponse({"thumbnail_url": f"data:image/png;base64,placeholder_{screenshot_id}"})


@router.post("/cancel/{analysis_id}")
async def cancel_analysis(analysis_id: str):
    """Cancel running analysis."""
    # In production, this would actually cancel the analysis
    return {
        "analysis_id": analysis_id,
        "status": "cancelled",
        "message": "Analysis cancelled by user",
    }


# Project Screenshot Management Endpoints


@router.post("/project/{project_id}/screenshots", response_model=SaveScreenshotResponse)
async def save_project_screenshots(project_id: str, files: list[UploadFile] = File(...)):
    """Save screenshots to project with duplicate detection."""
    saved = []
    duplicates = []

    for file in files:
        try:
            # Read file contents
            contents = await file.read()

            # Store screenshot data
            screenshot_data = {
                "name": file.filename,
                "size": len(contents),
                "image_bytes": contents,
                "content_type": file.content_type,
            }

            # Try to store (will detect duplicates)
            screenshot_id, hash_value = store.store_project_screenshot(project_id, screenshot_data)

            if screenshot_id is None:
                # Duplicate found
                duplicates.append(
                    {
                        "name": file.filename,
                        "hash": hash_value,
                        "reason": "Already exists in project",
                    }
                )
            else:
                # Successfully saved
                saved.append(
                    {
                        "id": screenshot_id,
                        "name": file.filename,
                        "hash": hash_value,
                        "size": len(contents),
                    }
                )

        except Exception as e:
            logger.error(f"Error saving screenshot {file.filename}: {e}")
            duplicates.append({"name": file.filename, "reason": str(e)})

    return SaveScreenshotResponse(
        saved=saved, duplicates=duplicates, total_saved=len(saved), total_duplicates=len(duplicates)
    )


@router.get("/project/{project_id}/screenshots", response_model=ProjectScreenshotResponse)
async def get_project_screenshots(project_id: str):
    """Get all screenshots for a project."""
    screenshots_dict = store.get_project_screenshots(project_id)

    screenshots = []
    for screenshot_id, screenshot_data in screenshots_dict.items():
        # Don't include raw image bytes in response
        screenshots.append(
            ProjectScreenshot(
                id=screenshot_data["id"],
                name=screenshot_data["name"],
                hash=screenshot_data["hash"],
                size=screenshot_data["size"],
                created_at=screenshot_data["created_at"],
                thumbnail_url=f"/api/state-discovery/project/{project_id}/screenshots/{screenshot_id}/thumbnail",
            )
        )

    # Sort by created_at (newest first)
    screenshots.sort(key=lambda x: x.created_at, reverse=True)

    return ProjectScreenshotResponse(screenshots=screenshots, count=len(screenshots))


@router.get("/project/{project_id}/screenshots/{screenshot_id}/thumbnail")
async def get_project_screenshot_thumbnail(project_id: str, screenshot_id: str):
    """Get thumbnail for a project screenshot."""
    screenshots = store.get_project_screenshots(project_id)

    if screenshot_id not in screenshots:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    screenshot_data = screenshots[screenshot_id]
    img_bytes = screenshot_data["image_bytes"]

    # Create thumbnail (resize to max 200px width)
    try:
        # Convert bytes to PIL Image
        img = PILImage.open(io.BytesIO(img_bytes))

        # Calculate thumbnail size
        max_width = 200
        aspect_ratio = img.height / img.width
        thumbnail_width = min(img.width, max_width)
        thumbnail_height = int(thumbnail_width * aspect_ratio)

        # Resize image
        thumbnail = img.resize((thumbnail_width, thumbnail_height), PILImage.Resampling.LANCZOS)

        # Convert to base64
        buffer = io.BytesIO()
        thumbnail.save(buffer, format="PNG")
        thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"thumbnail_url": f"data:image/png;base64,{thumbnail_base64}"}
    except Exception as e:
        logger.error(f"Error creating thumbnail: {e}")
        raise HTTPException(status_code=500, detail="Failed to create thumbnail") from e


@router.get("/project/{project_id}/screenshots/{screenshot_id}")
async def get_project_screenshot(project_id: str, screenshot_id: str):
    """Get full screenshot data."""
    screenshots = store.get_project_screenshots(project_id)

    if screenshot_id not in screenshots:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    screenshot_data = screenshots[screenshot_id]
    img_bytes = screenshot_data["image_bytes"]

    # Convert to base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return {
        "id": screenshot_data["id"],
        "name": screenshot_data["name"],
        "hash": screenshot_data["hash"],
        "size": screenshot_data["size"],
        "created_at": screenshot_data["created_at"],
        "image_data": f"data:image/png;base64,{img_base64}",
    }


# Mask endpoints for StateImages


@router.post("/state-image/{state_image_id}/generate-mask")
async def generate_state_image_mask(
    state_image_id: str, mask_type: str = "full", threshold: float = 0.95
):
    """Generate a mask for a StateImage."""
    # Import mask generator

    # For now, return a mock response
    # In production, this would fetch the StateImage and generate a mask
    return {
        "state_image_id": state_image_id,
        "mask_type": mask_type,
        "mask_density": 0.85,
        "active_pixels": 15000,
        "total_pixels": 20000,
        "message": "Mask generated successfully",
    }


@router.get("/state-image/{state_image_id}/mask")
async def get_state_image_mask(state_image_id: str):
    """Get the mask for a StateImage."""
    # In production, this would fetch the actual mask
    # For now, return a placeholder
    return {
        "state_image_id": state_image_id,
        "has_mask": True,
        "mask_density": 0.85,
        "mask_type": "stability",
        "mask_data": "base64_encoded_mask_placeholder",
    }


@router.post("/state-image/{state_image_id}/optimize-mask")
async def optimize_state_image_mask(
    state_image_id: str,
    positive_sample_ids: list[str],
    negative_sample_ids: list[str] | None = None,
    method: str = "stability",
):
    """Optimize a StateImage's mask based on samples."""
    return {
        "state_image_id": state_image_id,
        "optimization_method": method,
        "positive_samples": len(positive_sample_ids),
        "negative_samples": len(negative_sample_ids) if negative_sample_ids else 0,
        "new_mask_density": 0.75,
        "improvement": 0.05,
        "message": "Mask optimized successfully",
    }


@router.post("/state-image/batch-generate-masks")
async def batch_generate_masks(state_image_ids: list[str], mask_type: str = "adaptive"):
    """Generate masks for multiple StateImages."""
    results = []
    for sid in state_image_ids:
        results.append(
            {
                "state_image_id": sid,
                "status": "success",
                "mask_density": 0.8 + (hash(sid) % 20) / 100,  # Mock variation
            }
        )

    return {
        "total": len(state_image_ids),
        "successful": len(results),
        "failed": 0,
        "results": results,
    }


@router.post("/pattern/from-state-image")
async def create_pattern_from_state_image(
    state_image_id: str, pattern_name: str, similarity_threshold: float = 0.95
):
    """Create a MaskedPattern from a StateImage."""
    # This would integrate with the pattern API
    return {
        "pattern_id": f"pattern_{state_image_id}",
        "name": pattern_name,
        "source_state_image": state_image_id,
        "similarity_threshold": similarity_threshold,
        "message": "Pattern created successfully",
    }  # Force reload Mon Sep 29 16:55:25 CEST 2025
