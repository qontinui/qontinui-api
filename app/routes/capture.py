"""API endpoints for capture sessions and historical data.

These endpoints support:
- Capture session management
- Input event recording and retrieval
- Historical data querying for integration testing
- Frame extraction for visual playback
- Live screenshot capture from connected monitors
"""

import asyncio
import base64
import io
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.capture_service import CaptureService

router = APIRouter(prefix="/capture", tags=["capture"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a new capture session."""

    session_id: str = Field(..., description="Unique session identifier (UUID)")
    video_width: int = Field(..., description="Video width in pixels")
    video_height: int = Field(..., description="Video height in pixels")
    video_fps: float = Field(30.0, description="Video frames per second")
    monitor_id: int | None = Field(None, description="Monitor identifier")
    monitor_name: str | None = Field(None, description="Monitor name")
    workflow_id: int | None = Field(None, description="Associated workflow ID")
    project_id: int | None = Field(None, description="Associated project ID")
    metadata: dict | None = Field(None, description="Additional metadata")


class SessionResponse(BaseModel):
    """Response for capture session."""

    id: int
    session_id: str
    started_at: datetime
    ended_at: datetime | None
    duration_ms: int | None
    video_width: int
    video_height: int
    video_fps: float
    is_complete: bool
    is_processed: bool
    workflow_id: int | None
    project_id: int | None

    class Config:
        from_attributes = True


class InputEventRequest(BaseModel):
    """Request for recording input events."""

    timestamp_ms: int = Field(..., description="Timestamp in milliseconds from session start")
    event_type: str = Field(..., description="Event type (mouse_click, key_press, etc.)")
    mouse_x: int | None = None
    mouse_y: int | None = None
    mouse_button: int | None = None
    scroll_dx: int | None = None
    scroll_dy: int | None = None
    key_code: str | None = None
    key_name: str | None = None
    key_char: str | None = None
    shift_pressed: bool = False
    ctrl_pressed: bool = False
    alt_pressed: bool = False
    meta_pressed: bool = False
    extra_data: dict | None = None


class RecordEventsRequest(BaseModel):
    """Request to record multiple input events."""

    session_id: str
    events: list[InputEventRequest]


class InputEventResponse(BaseModel):
    """Response for input event."""

    id: int
    timestamp_ms: int
    event_type: str
    mouse_x: int | None
    mouse_y: int | None
    mouse_button: int | None
    key_code: str | None
    key_name: str | None

    class Config:
        from_attributes = True


class HistoricalResultResponse(BaseModel):
    """Response for historical result."""

    id: int
    pattern_id: str | None
    pattern_name: str | None
    action_type: str
    active_states: list[str] | None
    success: bool
    match_count: int | None
    best_match_score: float | None
    match_x: int | None
    match_y: int | None
    match_width: int | None
    match_height: int | None
    frame_timestamp_ms: int | None
    recorded_at: datetime
    has_frame: bool = False

    class Config:
        from_attributes = True


class RandomResultRequest(BaseModel):
    """Request for random historical result."""

    pattern_id: str | None = None
    action_type: str | None = None
    active_states: list[str] | None = None
    success_only: bool = True
    workflow_id: int | None = None
    project_id: int | None = None


class FrameResponse(BaseModel):
    """Response with frame data."""

    historical_result_id: int
    action_type: str
    pattern_id: str | None
    pattern_name: str | None
    success: bool
    match_x: int | None
    match_y: int | None
    match_width: int | None
    match_height: int | None
    timestamp_ms: int | None
    frame_base64: str | None  # Base64 encoded JPEG
    has_frame: bool


class IntegrationTestPlaybackRequest(BaseModel):
    """Request for integration test playback frames."""

    historical_result_ids: list[int]


# =============================================================================
# Capture Session Endpoints
# =============================================================================


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest, db: Session = Depends(get_db)):
    """Create a new capture session."""
    service = CaptureService(db)

    try:
        session = await service.create_session(
            session_id=request.session_id,
            video_width=request.video_width,
            video_height=request.video_height,
            video_fps=request.video_fps,
            monitor_id=request.monitor_id,
            monitor_name=request.monitor_name,
            workflow_id=request.workflow_id,
            project_id=request.project_id,
            metadata=request.metadata,
        )
        return session
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/sessions/{session_id}/complete", response_model=SessionResponse)
async def complete_session(
    session_id: str,
    video: UploadFile = File(...),
    snapshot_run_id: int | None = None,
    total_frames: int | None = None,
    db: Session = Depends(get_db),
):
    """Complete a capture session and upload the video."""
    service = CaptureService(db)

    try:
        session = await service.complete_session(
            session_id=session_id,
            video_data=video.file,
            video_filename=video.filename or f"{session_id}.mp4",
            total_frames=total_frames,
            snapshot_run_id=snapshot_run_id,
        )
        return session
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/sessions/{session_id}/build-index")
async def build_frame_index(session_id: str, db: Session = Depends(get_db)):
    """Build frame index for a capture session."""
    service = CaptureService(db)

    try:
        count = await service.build_frame_index(session_id)
        return {"session_id": session_id, "frames_indexed": count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get a capture session by ID."""
    service = CaptureService(db)
    session = await service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.get("/sessions/workflow/{workflow_id}", response_model=list[SessionResponse])
async def get_sessions_for_workflow(
    workflow_id: int, limit: int = Query(50, le=100), db: Session = Depends(get_db)
):
    """Get capture sessions for a workflow."""
    service = CaptureService(db)
    return await service.get_sessions_for_workflow(workflow_id, limit)


@router.get("/sessions/project/{project_id}", response_model=list[SessionResponse])
async def get_sessions_for_project(
    project_id: int, limit: int = Query(100, le=500), db: Session = Depends(get_db)
):
    """Get capture sessions for a project."""
    service = CaptureService(db)
    return await service.get_sessions_for_project(project_id, limit)


# =============================================================================
# Input Event Endpoints
# =============================================================================


@router.post("/events")
async def record_input_events(request: RecordEventsRequest, db: Session = Depends(get_db)):
    """Record input events for a capture session."""
    service = CaptureService(db)

    try:
        events_data = [event.model_dump() for event in request.events]
        count = await service.record_input_events(request.session_id, events_data)
        return {"recorded": count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/events/{session_id}", response_model=list[InputEventResponse])
async def get_input_events(
    session_id: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    event_types: str | None = None,  # Comma-separated
    db: Session = Depends(get_db),
):
    """Get input events for a capture session."""
    service = CaptureService(db)

    type_list = event_types.split(",") if event_types else None

    try:
        events = await service.get_input_events(
            session_id, start_ms=start_ms, end_ms=end_ms, event_types=type_list
        )
        return [
            InputEventResponse(
                id=e.id,
                timestamp_ms=e.timestamp_ms,
                event_type=e.event_type.value,
                mouse_x=e.mouse_x,
                mouse_y=e.mouse_y,
                mouse_button=e.mouse_button,
                key_code=e.key_code,
                key_name=e.key_name,
            )
            for e in events
        ]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# =============================================================================
# Historical Data Endpoints
# =============================================================================


@router.post("/historical/index/{snapshot_run_id}")
async def index_historical_results(
    snapshot_run_id: int,
    capture_session_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Index historical results from a snapshot run."""
    service = CaptureService(db)

    try:
        count = await service.index_historical_results(snapshot_run_id, capture_session_id)
        return {"indexed": count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/historical/random", response_model=HistoricalResultResponse | None)
async def get_random_historical_result(request: RandomResultRequest, db: Session = Depends(get_db)):
    """Get a random historical result matching criteria.

    This is the key endpoint for integration testing - it returns
    a random result from historical data, making each test run different.
    """
    service = CaptureService(db)

    result = await service.get_random_historical_result(
        pattern_id=request.pattern_id,
        action_type=request.action_type,
        active_states=request.active_states,
        success_only=request.success_only,
        workflow_id=request.workflow_id,
        project_id=request.project_id,
    )

    if not result:
        return None

    return HistoricalResultResponse(
        id=result.id,
        pattern_id=result.pattern_id,
        pattern_name=result.pattern_name,
        action_type=result.action_type,
        active_states=result.active_states,
        success=result.success,
        match_count=result.match_count,
        best_match_score=(float(result.best_match_score) if result.best_match_score else None),
        match_x=result.match_x,
        match_y=result.match_y,
        match_width=result.match_width,
        match_height=result.match_height,
        frame_timestamp_ms=result.frame_timestamp_ms,
        recorded_at=result.recorded_at,
        has_frame=result.capture_session_id is not None,
    )


@router.get("/historical/pattern/{pattern_id}", response_model=list[HistoricalResultResponse])
async def get_historical_results_for_pattern(
    pattern_id: str,
    action_type: str,
    active_states: str | None = None,  # Comma-separated
    limit: int = Query(10, le=100),
    db: Session = Depends(get_db),
):
    """Get historical results for a pattern."""
    service = CaptureService(db)

    state_list = active_states.split(",") if active_states else None

    results = await service.get_historical_results_for_action(
        pattern_id=pattern_id,
        action_type=action_type,
        active_states=state_list,
        limit=limit,
    )

    return [
        HistoricalResultResponse(
            id=r.id,
            pattern_id=r.pattern_id,
            pattern_name=r.pattern_name,
            action_type=r.action_type,
            active_states=r.active_states,
            success=r.success,
            match_count=r.match_count,
            best_match_score=float(r.best_match_score) if r.best_match_score else None,
            match_x=r.match_x,
            match_y=r.match_y,
            match_width=r.match_width,
            match_height=r.match_height,
            frame_timestamp_ms=r.frame_timestamp_ms,
            recorded_at=r.recorded_at,
            has_frame=r.capture_session_id is not None,
        )
        for r in results
    ]


# =============================================================================
# Frame Extraction Endpoints
# =============================================================================


@router.get("/frames/{historical_result_id}")
async def get_frame_for_result(
    historical_result_id: int, frame_type: str = "action", db: Session = Depends(get_db)
):
    """Get the frame image for a historical result.

    Returns the frame as a JPEG image.
    """
    service = CaptureService(db)

    frame_data = await service.get_frame_for_action(historical_result_id, frame_type)

    if not frame_data:
        raise HTTPException(status_code=404, detail="Frame not available")

    return Response(content=frame_data, media_type="image/jpeg")


@router.post("/frames/playback", response_model=list[FrameResponse])
async def get_playback_frames(
    request: IntegrationTestPlaybackRequest, db: Session = Depends(get_db)
):
    """Get frames for integration test playback.

    Returns frame data and metadata for a sequence of historical results,
    suitable for visual playback of an integration test.
    """
    service = CaptureService(db)

    frames = await service.get_frames_for_integration_test(request.historical_result_ids)

    return [
        FrameResponse(
            historical_result_id=f["historical_result_id"],
            action_type=f["action_type"],
            pattern_id=f["pattern_id"],
            pattern_name=f["pattern_name"],
            success=f["success"],
            match_x=f["match_x"],
            match_y=f["match_y"],
            match_width=f["match_width"],
            match_height=f["match_height"],
            timestamp_ms=f["timestamp_ms"],
            frame_base64=(base64.b64encode(f["frame_data"]).decode() if f["frame_data"] else None),
            has_frame=f["has_frame"],
        )
        for f in frames
    ]


@router.get("/frames/session/{session_id}/{timestamp_ms}")
async def get_frame_at_timestamp(
    session_id: str,
    timestamp_ms: int,
    quality: int = Query(90, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Extract a frame at a specific timestamp from a capture session.

    Returns the frame as a JPEG image.
    """
    service = CaptureService(db)

    session = await service.get_session(session_id)
    if not session or not session.video_path:
        raise HTTPException(status_code=404, detail="Session or video not found")

    video_filename = session.video_path.split("/")[-1]

    try:
        frame_data = await service.frame_extractor.extract_frame_by_timestamp(
            session_id, video_filename, timestamp_ms, quality
        )

        return Response(content=frame_data, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Live Screenshot Capture Endpoints
# =============================================================================


class MonitorInfo(BaseModel):
    """Information about a display monitor."""

    index: int = Field(..., description="Monitor index (0-based)")
    x: int = Field(..., description="X position of monitor")
    y: int = Field(..., description="Y position of monitor")
    width: int = Field(..., description="Width in physical pixels")
    height: int = Field(..., description="Height in physical pixels")
    scale: float = Field(1.0, description="DPI scale factor")
    is_primary: bool = Field(False, description="Whether this is the primary monitor")
    name: str | None = Field(None, description="Monitor name")


class MonitorListResponse(BaseModel):
    """Response containing list of available monitors."""

    monitors: list[MonitorInfo]
    count: int


class ScreenshotResponse(BaseModel):
    """Response for screenshot capture with base64 data."""

    screenshot_base64: str = Field(..., description="Base64 encoded PNG image")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    monitor: int | None = Field(None, description="Monitor index that was captured")
    timestamp: datetime = Field(..., description="Capture timestamp")
    format: str = Field("png", description="Image format")


@router.get("/screenshot/monitors", response_model=MonitorListResponse)
async def get_available_monitors():
    """Get list of available monitors for screenshot capture.

    Returns information about all connected monitors including their
    position, size (in physical pixels), and DPI scale factor.

    This endpoint uses the qontinui library's HAL layer which captures
    at physical resolution (not logical/scaled resolution).
    """
    try:
        from qontinui.hal.factory import HALFactory

        screen_capture = HALFactory.get_screen_capture()
        monitors = screen_capture.get_monitors()

        monitor_list = [
            MonitorInfo(
                index=m.index,
                x=m.x,
                y=m.y,
                width=m.width,
                height=m.height,
                scale=m.scale,
                is_primary=m.is_primary,
                name=m.name,
            )
            for m in monitors
        ]

        return MonitorListResponse(monitors=monitor_list, count=len(monitor_list))

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Screen capture not available: qontinui library not installed: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitors: {str(e)}") from e


@router.get("/screenshot/current", response_model=ScreenshotResponse)
async def capture_current_screenshot(
    monitor: int | None = Query(None, description="Monitor index (None for all monitors)"),
    quality: int = Query(95, ge=1, le=100, description="PNG compression level (1-100)"),
    delay_seconds: float = Query(
        0.0,
        ge=0.0,
        le=30.0,
        description="Delay in seconds before capturing (0-30). "
        "Useful for waiting for UI animations to settle.",
    ),
):
    """Capture a screenshot from the current display.

    This endpoint captures at **physical pixel resolution**, matching the
    qontinui library's capture behavior. On a 4K monitor with 150% DPI scaling,
    this returns 3840x2160 pixels (not 2560x1440 logical pixels).

    Args:
        monitor: Monitor index (0-based). None captures all monitors combined.
        quality: PNG compression quality (1-100, higher = better quality, larger file)
        delay_seconds: Delay in seconds before capturing (default 0, max 30).
            This is a deterministic sleep that occurs before the screenshot is taken.

    Returns:
        Base64-encoded PNG screenshot with metadata.
    """
    try:
        # Apply delay before capture if specified
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        from qontinui.hal.factory import HALFactory

        screen_capture = HALFactory.get_screen_capture()

        # Capture the screenshot
        pil_image = screen_capture.capture_screen(monitor=monitor)

        # Convert to PNG bytes
        buffer = io.BytesIO()
        # PNG compression: 0 = no compression, 9 = max compression
        # Map quality 1-100 to compress_level 9-0
        compress_level = max(0, min(9, 9 - (quality - 1) // 11))
        pil_image.save(buffer, format="PNG", compress_level=compress_level)
        buffer.seek(0)

        # Encode as base64
        screenshot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return ScreenshotResponse(
            screenshot_base64=screenshot_base64,
            width=pil_image.width,
            height=pil_image.height,
            monitor=monitor,
            timestamp=datetime.now(),
            format="png",
        )

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Screen capture not available: qontinui library not installed: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to capture screenshot: {str(e)}"
        ) from e


@router.get("/screenshot/current/raw")
async def capture_current_screenshot_raw(
    monitor: int | None = Query(None, description="Monitor index (None for all monitors)"),
    quality: int = Query(95, ge=1, le=100, description="PNG compression level (1-100)"),
    delay_seconds: float = Query(
        0.0,
        ge=0.0,
        le=30.0,
        description="Delay in seconds before capturing (0-30). "
        "Useful for waiting for UI animations to settle.",
    ),
):
    """Capture a screenshot and return as raw PNG image.

    Same as /screenshot/current but returns the image directly as binary PNG
    instead of base64-encoded JSON. More efficient for direct image display.

    Args:
        monitor: Monitor index (0-based). None captures all monitors combined.
        quality: PNG compression quality (1-100)
        delay_seconds: Delay in seconds before capturing (default 0, max 30).

    Returns:
        Raw PNG image data with image/png content type.
    """
    try:
        # Apply delay before capture if specified
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        from qontinui.hal.factory import HALFactory

        screen_capture = HALFactory.get_screen_capture()

        # Capture the screenshot
        pil_image = screen_capture.capture_screen(monitor=monitor)

        # Convert to PNG bytes
        buffer = io.BytesIO()
        compress_level = max(0, min(9, 9 - (quality - 1) // 11))
        pil_image.save(buffer, format="PNG", compress_level=compress_level)
        buffer.seek(0)

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={
                "X-Screenshot-Width": str(pil_image.width),
                "X-Screenshot-Height": str(pil_image.height),
                "X-Screenshot-Monitor": str(monitor) if monitor is not None else "all",
            },
        )

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Screen capture not available: qontinui library not installed: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to capture screenshot: {str(e)}"
        ) from e


@router.get("/screenshot/region")
async def capture_screenshot_region(
    x: int = Query(..., description="X coordinate of top-left corner"),
    y: int = Query(..., description="Y coordinate of top-left corner"),
    width: int = Query(..., gt=0, description="Region width in pixels"),
    height: int = Query(..., gt=0, description="Region height in pixels"),
    monitor: int | None = Query(None, description="Monitor index for relative coordinates"),
    quality: int = Query(95, ge=1, le=100, description="PNG compression level"),
    delay_seconds: float = Query(
        0.0,
        ge=0.0,
        le=30.0,
        description="Delay in seconds before capturing (0-30). "
        "Useful for waiting for UI animations to settle.",
    ),
):
    """Capture a specific region of the screen.

    Coordinates are in physical pixels. If monitor is specified, coordinates
    are relative to that monitor's top-left corner.

    Args:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Region width in pixels
        height: Region height in pixels
        monitor: Optional monitor index for relative coordinates
        quality: PNG compression quality (1-100)
        delay_seconds: Delay in seconds before capturing (default 0, max 30).

    Returns:
        Raw PNG image data of the captured region.
    """
    try:
        # Apply delay before capture if specified
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        from qontinui.hal.factory import HALFactory

        screen_capture = HALFactory.get_screen_capture()

        # Capture the region
        pil_image = screen_capture.capture_region(x, y, width, height, monitor=monitor)

        # Convert to PNG bytes
        buffer = io.BytesIO()
        compress_level = max(0, min(9, 9 - (quality - 1) // 11))
        pil_image.save(buffer, format="PNG", compress_level=compress_level)
        buffer.seek(0)

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={
                "X-Screenshot-Width": str(pil_image.width),
                "X-Screenshot-Height": str(pil_image.height),
                "X-Region-X": str(x),
                "X-Region-Y": str(y),
            },
        )

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Screen capture not available: qontinui library not installed: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture region: {str(e)}") from e
