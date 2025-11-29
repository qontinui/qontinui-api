"""API endpoints for capture sessions and historical data.

These endpoints support:
- Capture session management
- Input event recording and retrieval
- Historical data querying for integration testing
- Frame extraction for visual playback
"""

import base64
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
    snapshot_run_id: int, capture_session_id: int | None = None, db: Session = Depends(get_db)
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
        best_match_score=float(result.best_match_score) if result.best_match_score else None,
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
        pattern_id=pattern_id, action_type=action_type, active_states=state_list, limit=limit
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
            frame_base64=base64.b64encode(f["frame_data"]).decode() if f["frame_data"] else None,
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
