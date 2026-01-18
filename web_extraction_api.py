"""
Web Extraction API endpoints for State Machine data collection.

This module provides REST API endpoints for:
- Starting web extraction jobs
- Monitoring job progress
- Retrieving extracted elements
- Building State Machine states from extractions
"""

import asyncio
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
    pages: dict[str, list[dict[str, Any]]] = {}
    for clickable in clickables:
        # Use a default page key if no URL info
        page_key = "default"
        pages.setdefault(page_key, []).append(clickable)

    # Build states
    states = []
    for i, (_page_key, page_clickables) in enumerate(pages.items()):
        state_images: list[dict[str, Any]] = []
        state_locations: list[dict[str, Any]] = []

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
            state_images.append(state_image)

            # Create StateLocation entry (click point)
            bbox = clickable["bounding_box"]
            state_location = {
                "id": f"loc_{clickable['element_id']}",
                "name": f"click_{clickable['element_id']}",
                "x": bbox["x"] + bbox["width"] // 2,
                "y": bbox["y"] + bbox["height"] // 2,
            }
            state_locations.append(state_location)

        state = {
            "id": f"state_{uuid.uuid4().hex[:8]}",
            "name": f"{state_name_prefix}_{i + 1}",
            "description": f"Auto-discovered state with {len(page_clickables)} elements",
            "stateImages": state_images,
            "stateLocations": state_locations,
        }
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


def _verify_element_in_screenshot(
    element_image: Any,
    page_screenshot: Any,
    threshold: float = 0.85,
) -> tuple[bool, float]:
    """Verify an element can be found in the page screenshot using template matching.

    Args:
        element_image: The element's screenshot as numpy array
        page_screenshot: The full page screenshot as numpy array
        threshold: Minimum similarity threshold for verification

    Returns:
        Tuple of (is_verified, match_confidence)
    """
    import cv2
    import numpy as np

    try:
        # Ensure both images are valid numpy arrays
        if element_image is None or page_screenshot is None:
            return False, 0.0

        if not isinstance(element_image, np.ndarray) or not isinstance(page_screenshot, np.ndarray):
            return False, 0.0

        # Convert to BGR if needed (OpenCV format)
        if len(element_image.shape) == 2:
            # Grayscale
            element_bgr = cv2.cvtColor(element_image, cv2.COLOR_GRAY2BGR)
        elif element_image.shape[2] == 4:
            # RGBA -> BGR
            element_bgr = cv2.cvtColor(element_image, cv2.COLOR_RGBA2BGR)
        else:
            # RGB -> BGR
            element_bgr = cv2.cvtColor(element_image, cv2.COLOR_RGB2BGR)

        if len(page_screenshot.shape) == 2:
            page_bgr = cv2.cvtColor(page_screenshot, cv2.COLOR_GRAY2BGR)
        elif page_screenshot.shape[2] == 4:
            page_bgr = cv2.cvtColor(page_screenshot, cv2.COLOR_RGBA2BGR)
        else:
            page_bgr = cv2.cvtColor(page_screenshot, cv2.COLOR_RGB2BGR)

        # Check dimensions - template must be smaller than the image
        if element_bgr.shape[0] > page_bgr.shape[0] or element_bgr.shape[1] > page_bgr.shape[1]:
            return False, 0.0

        # Skip very small templates that might cause issues
        if element_bgr.shape[0] < 5 or element_bgr.shape[1] < 5:
            return False, 0.0

        # Perform template matching
        result = cv2.matchTemplate(page_bgr, element_bgr, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # max_val is the best match score (0.0 to 1.0)
        match_confidence = float(max_val)
        is_verified = match_confidence >= threshold

        return is_verified, match_confidence

    except Exception as e:
        logger.debug(f"Verification failed: {e}")
        return False, 0.0


def _run_extraction_with_sync_playwright(
    job_id: str,
    url: str,
    max_elements: int,
    verify_extractions: bool,
    verification_threshold: float,
    additional_blocked_keywords: list[str],
    safety_config_dict: dict[str, Any] | None = None,
    max_depth: int = 0,
    max_elements_per_page: int = 50,
) -> dict[str, Any]:
    """Run extraction using Playwright's synchronous API.

    This function runs Playwright in sync mode, which handles its own event loop
    internally and works correctly on Windows without the subprocess issues.
    """
    import io

    import numpy as np
    from PIL import Image as PILImage
    from playwright.sync_api import sync_playwright

    logger.info(
        f"Starting sync Playwright extraction for job {job_id}, URL: {url}, verify={verify_extractions}, threshold={verification_threshold}"
    )

    clickables = []
    skipped = []
    errors = []
    page_screenshots: dict[str, bytes] = {}  # Store full-page screenshots

    try:
        logger.info("Launching sync_playwright context manager...")
        with sync_playwright() as p:
            logger.info("sync_playwright started, launching browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            # Navigate to URL
            page.goto(url)
            page.wait_for_load_state("networkidle")

            # Take page screenshot and store it with a unique ID
            screenshot_bytes = page.screenshot(full_page=True)
            screenshot_id = f"page_{uuid.uuid4().hex[:8]}"
            page_screenshots[screenshot_id] = screenshot_bytes

            pil_image = PILImage.open(io.BytesIO(screenshot_bytes))
            page_screenshot = np.array(pil_image)

            # CSS selectors for clickable elements
            clickable_selectors = [
                "button",
                "a[href]",
                '[role="button"]',
                '[role="link"]',
                '[role="menuitem"]',
                '[role="tab"]',
                'input[type="submit"]',
                'input[type="button"]',
                "[onclick]",
                '[tabindex]:not([tabindex="-1"])',
            ]

            # Find clickable elements
            element_counter = 0
            for selector in clickable_selectors:
                try:
                    elements = page.query_selector_all(selector)
                    for element in elements[: max_elements - len(clickables)]:
                        if len(clickables) >= max_elements:
                            break

                        try:
                            # Check if element is visible and interactive
                            if not element.is_visible():
                                continue

                            bbox = element.bounding_box()
                            if not bbox or bbox["width"] < 10 or bbox["height"] < 10:
                                continue

                            # Get element info
                            tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                            text_content = element.text_content()
                            aria_label = element.get_attribute("aria-label")

                            # Generate unique selector
                            try:
                                unique_selector = element.evaluate(
                                    """(el) => {
                                    if (el.id) return '#' + CSS.escape(el.id);
                                    let selector = el.tagName.toLowerCase();
                                    if (el.className && typeof el.className === 'string') {
                                        const classes = el.className.split(' ').filter(c => c).slice(0, 2);
                                        if (classes.length > 0) selector += '.' + classes.map(c => CSS.escape(c)).join('.');
                                    }
                                    return selector;
                                }"""
                                )
                            except Exception:
                                unique_selector = selector

                            # Take element screenshot
                            element_image = None
                            try:
                                element_screenshot_bytes = element.screenshot()
                                element_pil = PILImage.open(io.BytesIO(element_screenshot_bytes))
                                element_image = np.array(element_pil)
                            except Exception:
                                element_image = None

                            # Verify element if requested and we have screenshots
                            is_verified = False
                            match_confidence = 0.0
                            if (
                                verify_extractions
                                and element_image is not None
                                and page_screenshot is not None
                            ):
                                is_verified, match_confidence = _verify_element_in_screenshot(
                                    element_image,
                                    page_screenshot,
                                    verification_threshold,
                                )

                            element_counter += 1
                            clickable = {
                                "element_id": f"elem_{element_counter:06d}",
                                "selector": unique_selector,
                                "tag_name": tag_name,
                                "text": (text_content.strip()[:200] if text_content else None),
                                "aria_label": aria_label,
                                "bounding_box": {
                                    "x": int(bbox["x"]),
                                    "y": int(bbox["y"]),
                                    "width": int(bbox["width"]),
                                    "height": int(bbox["height"]),
                                },
                                "risk_level": "safe",
                                "risk_reason": "Element appears safe",
                                "was_clicked": False,
                                "is_verified": is_verified,
                                "match_confidence": match_confidence,
                                "screenshot": element_image,
                                "page_screenshot_before": page_screenshot,
                                "screenshot_id": screenshot_id,  # Link to full-page screenshot
                            }
                            clickables.append(clickable)

                        except Exception as e:
                            logger.debug(f"Failed to extract element: {e}")
                            continue

                except Exception as e:
                    logger.debug(f"Failed to query selector {selector}: {e}")
                    continue

            browser.close()

    except Exception as e:
        import traceback

        error_msg = f"Collection failed: {type(e).__name__}: {e}"
        full_traceback = traceback.format_exc()
        errors.append(error_msg)
        errors.append(full_traceback)
        logger.error(f"{error_msg}\n{full_traceback}")

    # Convert page screenshots to base64 for JSON serialization
    page_screenshots_base64: dict[str, str] = {}
    for ss_id, ss_bytes in page_screenshots.items():
        page_screenshots_base64[ss_id] = base64.b64encode(ss_bytes).decode()

    # Calculate verification metrics
    verified_count = sum(1 for c in clickables if c.get("is_verified", False))
    failed_count = len(clickables) - verified_count
    total_confidence = sum(c.get("match_confidence", 0.0) for c in clickables)
    avg_confidence = total_confidence / len(clickables) if clickables else 0.0

    return {
        "clickables": clickables,
        "skipped_dangerous": skipped,
        "metrics": {
            "total_found": len(clickables),
            "clicked": 0,
            "skipped_dangerous": len(skipped),
            "pages_visited": 1,
            "errors": len(errors),
            "verified": verified_count,
            "failed": failed_count,
            "avg_confidence": avg_confidence,
            "verification_rate": verified_count / len(clickables) if clickables else 0.0,
        },
        "pages_visited": [url],
        "errors": errors,
        "page_screenshots": page_screenshots_base64,  # Full-page screenshots as base64
    }


async def _run_extraction(job_id: str, request: ExtractionRequest) -> None:
    """Background task to run full extraction using sync Playwright."""
    try:
        extraction_jobs[job_id]["status"] = ExtractionStatus.running

        # Run Playwright extraction in a thread using sync API
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _run_extraction_with_sync_playwright,
            job_id,
            request.url,
            request.max_elements_per_page,
            request.verify_extractions,
            request.verification_threshold,
            request.additional_blocked_keywords,
            None,  # safety_config_dict
            request.max_depth,
            request.max_elements_per_page,
        )

        # Serialize clickables (screenshot images to base64)
        serialized_clickables = []
        for c in result["clickables"]:
            serialized = _serialize_clickable_dict(c)
            serialized_clickables.append(serialized)

        # Convert result to serializable format
        extraction_jobs[job_id]["status"] = ExtractionStatus.completed
        extraction_jobs[job_id]["result"] = {
            "clickables": serialized_clickables,
            "skipped_dangerous": result["skipped_dangerous"],
            "metrics": result["metrics"],
            "pages_visited": result["pages_visited"],
            "errors": result["errors"],
            "page_screenshots": result.get("page_screenshots", {}),
        }

    except Exception as e:
        logger.exception(f"Extraction failed for job {job_id}")
        extraction_jobs[job_id]["status"] = ExtractionStatus.failed
        extraction_jobs[job_id]["error"] = str(e)


async def _run_single_page_extraction(job_id: str, request: SinglePageExtractionRequest) -> None:
    """Background task to run single-page extraction using sync Playwright."""
    try:
        extraction_jobs[job_id]["status"] = ExtractionStatus.running

        # Run Playwright extraction in a thread using sync API
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _run_extraction_with_sync_playwright,
            job_id,
            request.url,
            request.max_elements,
            request.verify_extractions,
            request.verification_threshold,
            request.additional_blocked_keywords,
            None,  # safety_config_dict
            0,  # max_depth (single page)
            request.max_elements,
        )

        # Serialize clickables (screenshot images to base64)
        serialized_clickables = []
        for c in result["clickables"]:
            serialized = _serialize_clickable_dict(c)
            serialized_clickables.append(serialized)

        # Convert result to serializable format
        extraction_jobs[job_id]["status"] = ExtractionStatus.completed
        extraction_jobs[job_id]["result"] = {
            "clickables": serialized_clickables,
            "skipped_dangerous": result["skipped_dangerous"],
            "metrics": result["metrics"],
            "pages_visited": result["pages_visited"],
            "errors": result["errors"],
            "page_screenshots": result.get("page_screenshots", {}),
        }

    except Exception as e:
        logger.exception(f"Single-page extraction failed for job {job_id}")
        extraction_jobs[job_id]["status"] = ExtractionStatus.failed
        extraction_jobs[job_id]["error"] = str(e)


def _serialize_clickable_dict(clickable: dict[str, Any]) -> dict[str, Any]:
    """Serialize a clickable dict (from sync extraction) to JSON-compatible dict."""
    import io

    from PIL import Image as PILImage

    result = dict(clickable)

    # Convert screenshot to base64 if present
    if clickable.get("screenshot") is not None:
        try:
            pil_img = PILImage.fromarray(clickable["screenshot"])
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            result["screenshot_base64"] = base64.b64encode(buffer.getvalue()).decode()
        except Exception:
            result["screenshot_base64"] = None

    # Remove numpy arrays from result
    result.pop("screenshot", None)
    result.pop("page_screenshot_before", None)
    result.pop("page_screenshot_after", None)

    return result


def _serialize_clickable(clickable: Any) -> dict[str, Any]:
    """Serialize an ExtractedClickable to JSON-compatible dict."""
    import io

    from PIL import Image as PILImage

    result: dict[str, Any] = clickable.to_dict()

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
