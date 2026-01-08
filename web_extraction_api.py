"""
Web Extraction API endpoints for State Machine data collection.

This module provides REST API endpoints for:
- Starting web extraction jobs
- Monitoring job progress
- Retrieving extracted elements
- Building State Machine states from extractions
"""

import base64
import logging
import uuid
from enum import Enum
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web-extraction", tags=["web-extraction"])

# In-memory job storage (use Redis in production)
extraction_jobs: dict[str, dict[str, Any]] = {}


class RiskLevel(str, Enum):
    """Maximum risk level for auto-clicking."""

    safe = "safe"
    caution = "caution"
    # dangerous and blocked are never auto-clicked


class ExtractionStatus(str, Enum):
    """Status of an extraction job."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ExtractionRequest(BaseModel):
    """Request to start web extraction."""

    url: str = Field(..., description="Starting URL to extract from")
    max_depth: int = Field(default=2, ge=0, le=5, description="How many clicks deep to explore")
    max_elements_per_page: int = Field(default=50, ge=1, le=200)

    # Safety settings
    max_risk_level: RiskLevel = Field(
        default=RiskLevel.safe, description="Maximum risk level to auto-click"
    )
    dry_run: bool = Field(
        default=True,  # Default to dry run for safety!
        description="If true, identify elements without clicking",
    )

    # Custom safety rules
    additional_blocked_keywords: list[str] = Field(default_factory=list)
    additional_safe_keywords: list[str] = Field(default_factory=list)
    blocked_selectors: list[str] = Field(default_factory=list)

    # Verification
    verify_extractions: bool = Field(
        default=True, description="Verify extracted images are detectable"
    )
    verification_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class SinglePageExtractionRequest(BaseModel):
    """Request for single-page extraction (no navigation)."""

    url: str = Field(..., description="URL to extract from")
    max_elements: int = Field(default=100, ge=1, le=500)
    verify_extractions: bool = Field(default=True)
    verification_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    additional_blocked_keywords: list[str] = Field(default_factory=list)


class ExtractionJobResponse(BaseModel):
    """Response for extraction job status."""

    job_id: str
    status: ExtractionStatus
    progress: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class ExtractedElementResponse(BaseModel):
    """Response for a single extracted element."""

    element_id: str
    selector: str
    tag_name: str
    text: str | None
    aria_label: str | None
    bounding_box: dict[str, int]
    risk_level: str
    risk_reason: str
    was_clicked: bool
    is_verified: bool
    match_confidence: float
    screenshot_base64: str | None = None


@router.post("/start", response_model=ExtractionJobResponse)
async def start_extraction(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """
    Start a web extraction job.

    Returns immediately with a job_id. Poll /status/{job_id} for progress.

    Safety: By default, dry_run=True means elements are identified but not clicked.
    Set dry_run=False only when you're confident in the safety configuration.
    """
    job_id = str(uuid.uuid4())

    extraction_jobs[job_id] = {
        "status": ExtractionStatus.pending,
        "request": request.model_dump(),
        "progress": {"stage": "initializing", "percent": 0},
        "result": None,
        "error": None,
    }

    # Run extraction in background
    background_tasks.add_task(_run_extraction, job_id, request)

    return ExtractionJobResponse(
        job_id=job_id,
        status=ExtractionStatus.pending,
        progress={"stage": "queued", "percent": 0},
    )


@router.post("/single-page", response_model=ExtractionJobResponse)
async def start_single_page_extraction(
    request: SinglePageExtractionRequest, background_tasks: BackgroundTasks
):
    """
    Start a single-page extraction (no navigation/clicking).

    This is faster and safer - only extracts elements from one page.
    """
    job_id = str(uuid.uuid4())

    extraction_jobs[job_id] = {
        "status": ExtractionStatus.pending,
        "request": request.model_dump(),
        "progress": {"stage": "initializing", "percent": 0},
        "result": None,
        "error": None,
        "single_page": True,
    }

    # Run extraction in background
    background_tasks.add_task(_run_single_page_extraction, job_id, request)

    return ExtractionJobResponse(
        job_id=job_id,
        status=ExtractionStatus.pending,
        progress={"stage": "queued", "percent": 0},
    )


@router.get("/status/{job_id}", response_model=ExtractionJobResponse)
async def get_extraction_status(job_id: str):
    """Get the status of an extraction job."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = extraction_jobs[job_id]
    return ExtractionJobResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"],
        error=job["error"],
    )


@router.get("/results/{job_id}/elements")
async def get_extracted_elements(
    job_id: str,
    verified_only: bool = False,
    min_confidence: float = 0.0,
    include_screenshots: bool = False,
    limit: int = 100,
    offset: int = 0,
):
    """Get extracted elements from a completed job."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = extraction_jobs[job_id]
    if job["status"] != ExtractionStatus.completed:
        raise HTTPException(status_code=400, detail="Job not yet completed")

    elements = job["result"]["clickables"]

    # Apply filters
    if verified_only:
        elements = [e for e in elements if e.get("is_verified", False)]

    if min_confidence > 0:
        elements = [e for e in elements if e.get("match_confidence", 0) >= min_confidence]

    # Paginate
    total = len(elements)
    elements = elements[offset : offset + limit]

    # Optionally include screenshots
    if not include_screenshots:
        for elem in elements:
            elem.pop("screenshot_base64", None)

    return {
        "elements": elements,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/results/{job_id}/skipped")
async def get_skipped_elements(job_id: str):
    """Get elements that were skipped due to safety rules."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = extraction_jobs[job_id]
    if job["status"] != ExtractionStatus.completed:
        raise HTTPException(status_code=400, detail="Job not yet completed")

    return {"skipped": job["result"]["skipped_dangerous"]}


@router.get("/results/{job_id}/metrics")
async def get_extraction_metrics(job_id: str):
    """Get extraction metrics and statistics."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = extraction_jobs[job_id]
    if job["status"] != ExtractionStatus.completed:
        raise HTTPException(status_code=400, detail="Job not yet completed")

    return {"metrics": job["result"]["metrics"]}


@router.post("/results/{job_id}/build-states")
async def build_state_machine(
    job_id: str,
    state_name_prefix: str = "Page",
    verified_only: bool = True,
    min_confidence: float = 0.8,
):
    """Convert extracted elements into State Machine states."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = extraction_jobs[job_id]
    if job["status"] != ExtractionStatus.completed:
        raise HTTPException(status_code=400, detail="Job not yet completed")

    clickables = job["result"]["clickables"]

    # Filter by verification status
    if verified_only:
        clickables = [c for c in clickables if c.get("is_verified", False)]

    if min_confidence > 0:
        clickables = [c for c in clickables if c.get("match_confidence", 0) >= min_confidence]

    # Group by page URL (if available)
    pages: dict[str, list[dict]] = {}
    for clickable in clickables:
        # Use a default page key if no URL info
        page_key = "default"
        pages.setdefault(page_key, []).append(clickable)

    # Build states
    states = []
    for i, (_page_key, page_clickables) in enumerate(pages.items()):
        state = {
            "id": f"state_{uuid.uuid4().hex[:8]}",
            "name": f"{state_name_prefix}_{i + 1}",
            "description": f"Auto-discovered state with {len(page_clickables)} elements",
            "stateImages": [],
            "stateLocations": [],
        }

        for clickable in page_clickables:
            # Create StateImage entry
            state_image = {
                "id": f"img_{clickable['element_id']}",
                "name": clickable.get("text")
                or clickable.get("aria_label")
                or clickable["selector"][:50],
                "searchRegion": clickable["bounding_box"],
                "similarity": clickable.get("match_confidence", 0.85),
            }
            state["stateImages"].append(state_image)

            # Create StateLocation entry (click point)
            bbox = clickable["bounding_box"]
            state_location = {
                "id": f"loc_{clickable['element_id']}",
                "name": f"click_{clickable['element_id']}",
                "x": bbox["x"] + bbox["width"] // 2,
                "y": bbox["y"] + bbox["height"] // 2,
            }
            state["stateLocations"].append(state_location)

        states.append(state)

    return {
        "states": states,
        "total_states": len(states),
        "total_elements": len(clickables),
        "verification_stats": {
            "verified": len([c for c in clickables if c.get("is_verified")]),
            "unverified": len([c for c in clickables if not c.get("is_verified")]),
        },
    }


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del extraction_jobs[job_id]
    return {"status": "deleted", "job_id": job_id}


@router.get("/jobs")
async def list_jobs(status: ExtractionStatus | None = None, limit: int = 50):
    """List extraction jobs."""
    jobs = []
    for job_id, job in extraction_jobs.items():
        if status and job["status"] != status:
            continue
        jobs.append(
            {
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "url": job["request"].get("url"),
            }
        )
        if len(jobs) >= limit:
            break

    return {"jobs": jobs, "total": len(jobs)}


async def _run_extraction(job_id: str, request: ExtractionRequest) -> None:
    """Background task to run full extraction."""
    try:
        extraction_jobs[job_id]["status"] = ExtractionStatus.running

        # Import here to avoid import errors if playwright not installed
        from qontinui.extraction.web.playwright_collector import (
            SafePlaywrightStateCollector,
        )
        from qontinui.extraction.web.safety import ActionRisk, SafetyConfig

        # Build safety config from request
        safety_config = SafetyConfig(
            max_auto_click_risk=ActionRisk[request.max_risk_level.value.upper()],
            dry_run=request.dry_run,
        )

        # Add custom keywords/selectors
        for keyword in request.additional_blocked_keywords:
            safety_config.add_dangerous_keyword(keyword)
        for keyword in request.additional_safe_keywords:
            safety_config.add_safe_keyword(keyword)
        for selector in request.blocked_selectors:
            safety_config.add_blocked_selector(selector)

        collector = SafePlaywrightStateCollector(
            safety_config=safety_config,
            verify_extractions=request.verify_extractions,
            verification_threshold=request.verification_threshold,
        )

        # Progress callback
        def on_progress(stage: str, percent: int) -> None:
            extraction_jobs[job_id]["progress"] = {"stage": stage, "percent": percent}

        result = await collector.collect(
            url=request.url,
            max_depth=request.max_depth,
            max_elements_per_page=request.max_elements_per_page,
            on_progress=on_progress,
        )

        # Convert result to serializable format
        extraction_jobs[job_id]["status"] = ExtractionStatus.completed
        extraction_jobs[job_id]["result"] = {
            "clickables": [_serialize_clickable(c) for c in result.clickables],
            "skipped_dangerous": result.skipped_dangerous,
            "metrics": result.metrics,
            "pages_visited": result.pages_visited,
            "errors": result.errors,
        }

    except Exception as e:
        logger.exception(f"Extraction failed for job {job_id}")
        extraction_jobs[job_id]["status"] = ExtractionStatus.failed
        extraction_jobs[job_id]["error"] = str(e)


async def _run_single_page_extraction(job_id: str, request: SinglePageExtractionRequest) -> None:
    """Background task to run single-page extraction."""
    try:
        extraction_jobs[job_id]["status"] = ExtractionStatus.running

        from qontinui.extraction.web.playwright_collector import (
            SafePlaywrightStateCollector,
        )
        from qontinui.extraction.web.safety import SafetyConfig

        # Build safety config
        safety_config = SafetyConfig(dry_run=True)  # Never click in single-page mode

        for keyword in request.additional_blocked_keywords:
            safety_config.add_dangerous_keyword(keyword)

        collector = SafePlaywrightStateCollector(
            safety_config=safety_config,
            verify_extractions=request.verify_extractions,
            verification_threshold=request.verification_threshold,
        )

        result = await collector.collect_single_page(
            url=request.url,
            max_elements=request.max_elements,
        )

        # Convert result to serializable format
        extraction_jobs[job_id]["status"] = ExtractionStatus.completed
        extraction_jobs[job_id]["result"] = {
            "clickables": [_serialize_clickable(c) for c in result.clickables],
            "skipped_dangerous": result.skipped_dangerous,
            "metrics": result.metrics,
            "pages_visited": result.pages_visited,
            "errors": result.errors,
        }

    except Exception as e:
        logger.exception(f"Single-page extraction failed for job {job_id}")
        extraction_jobs[job_id]["status"] = ExtractionStatus.failed
        extraction_jobs[job_id]["error"] = str(e)


def _serialize_clickable(clickable) -> dict[str, Any]:
    """Serialize an ExtractedClickable to JSON-compatible dict."""
    import io

    from PIL import Image as PILImage

    result = clickable.to_dict()

    # Convert screenshot to base64 if present
    if clickable.screenshot is not None:
        try:
            pil_img = PILImage.fromarray(clickable.screenshot)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            result["screenshot_base64"] = base64.b64encode(buffer.getvalue()).decode()
        except Exception:
            result["screenshot_base64"] = None

    return result
