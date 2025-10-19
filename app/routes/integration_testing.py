"""API endpoints for integration testing with mock execution and visualization."""

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.state_coverage import (
    StateCoverageAnalyzer,
)

router = APIRouter(prefix="/integration-testing", tags=["integration-testing"])


# Pydantic models


class ActionSpec(BaseModel):
    """Specification for an action in a process."""

    type: str = Field(..., description="Action type: FIND, CLICK, TYPE, etc.")
    pattern_id: str | None = Field(None, description="Pattern identifier for FIND actions")
    text: str | None = Field(None, description="Text for TYPE actions")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MockExecutionRequest(BaseModel):
    """Request to execute a process in mock mode."""

    process_id: str = Field(..., description="Process identifier")
    process_name: str = Field(..., description="Process name")
    snapshot_run_ids: list[str] = Field(
        ...,
        description="Snapshot run IDs to use for mock data (can specify multiple for larger data pool)",
    )
    initial_states: list[str] = Field(..., description="Initial active states")
    actions: list[ActionSpec] = Field(..., description="List of actions to execute")

    # Backward compatibility: accept single snapshot_run_id and convert to list
    snapshot_run_id: str | None = Field(None, description="(Deprecated) Single snapshot run ID")

    def __init__(self, **data):
        # Convert single snapshot_run_id to snapshot_run_ids list
        if "snapshot_run_id" in data and data["snapshot_run_id"]:
            if "snapshot_run_ids" not in data or not data["snapshot_run_ids"]:
                data["snapshot_run_ids"] = [data["snapshot_run_id"]]
        super().__init__(**data)


class ActionVisualizationResponse(BaseModel):
    """Visualization data for a single action."""

    action_type: str
    screenshot_path: str
    action_location: tuple[int, int] | None = None
    action_region: dict[str, int] | None = None
    success: bool
    matches: list[dict[str, Any]] = Field(default_factory=list)
    text: str = ""
    active_states: list[str]
    timestamp: str
    duration_ms: float


class MockExecutionResponse(BaseModel):
    """Result of mock process execution."""

    process_id: str
    process_name: str
    start_time: str
    end_time: str | None
    total_duration_ms: float
    initial_states: list[str]
    final_states: list[str]
    actions: list[ActionVisualizationResponse]
    success: bool
    success_rate: float
    total_actions: int
    successful_actions: int


class StateScreenshotResponse(BaseModel):
    """Screenshot associated with active states."""

    screenshot_path: str
    active_states: list[str]
    timestamp: str
    width: int
    height: int
    state_hash: str


class StateScreenshotListResponse(BaseModel):
    """List of state screenshots."""

    screenshots: list[StateScreenshotResponse]
    total: int
    unique_state_combinations: int


class CoverageAnalysisRequest(BaseModel):
    """Request to analyze coverage for a process."""

    process_id: str = Field(..., description="Process identifier")
    process_name: str = Field(..., description="Process name")
    snapshot_run_ids: list[str] = Field(..., description="Snapshot run IDs to analyze")
    expected_states: list[str] | None = Field(
        None, description="Optional list of states expected in the process"
    )


class StateCoverageMetricsResponse(BaseModel):
    """Coverage metrics for a single state."""

    state_name: str
    screenshot_count: int
    actions_performed: int
    last_tested: str | None
    coverage_percentage: float
    transitions_to: list[str]
    transitions_from: list[str]
    action_types: list[str]
    patterns_tested: list[str]


class StateTransitionResponse(BaseModel):
    """State transition information."""

    from_state: str
    to_state: str
    count: int
    covered: bool
    last_occurrence: str | None
    actions_triggering: list[str]


class CoverageGapResponse(BaseModel):
    """Coverage gap information."""

    gap_type: str
    severity: str
    description: str
    recommendation: str
    affected_states: list[str]
    metric_value: float | None


class CoverageReportResponse(BaseModel):
    """Complete coverage analysis report."""

    process_id: str
    process_name: str
    snapshot_run_ids: list[str]
    analysis_time: str
    overall_coverage_percentage: float
    total_states: int
    covered_states: int
    uncovered_states: int
    total_transitions: int
    covered_transitions: int
    missing_transitions: int
    state_metrics: dict[str, StateCoverageMetricsResponse]
    transitions: list[StateTransitionResponse]
    coverage_gaps: list[CoverageGapResponse]
    recommendations: list[str]


class ThumbnailResponse(BaseModel):
    """Thumbnail information for a screenshot."""

    url: str
    active_states: list[str]
    action_number: int
    timestamp: str


class ThumbnailListResponse(BaseModel):
    """List of thumbnails for a snapshot run."""

    run_id: str
    thumbnails: list[ThumbnailResponse]
    total_screenshots: int


class StartScreenshotResponse(BaseModel):
    """Start screenshot information for a snapshot run."""

    run_id: str
    screenshot_path: str | None
    screenshot_url: str | None
    initial_states: list[str]
    timestamp: str | None
    found: bool


# API endpoints


@router.post("/execute", response_model=MockExecutionResponse)
def execute_mock_process(
    request: MockExecutionRequest,
    db: Session = Depends(get_db),
):
    """
    Execute a process in mock mode using recorded snapshots.

    This endpoint:
    1. Loads action histories from the specified snapshot run(s)
    2. Executes each action using state-aware snapshot matching
    3. Returns visualization data for each action

    Multiple snapshot runs can be specified to provide a larger pool of
    historical data, improving the chances of finding matching screenshots
    for different state combinations.

    The visualization data includes:
    - Screenshot showing the action context
    - Action location (for clicks, type events)
    - Match regions (for find actions)
    - Active states during the action
    - Success/failure status

    Args:
        request: Mock execution request (supports multiple snapshot_run_ids)
        db: Database session

    Returns:
        Complete execution trace with visualization data

    Raises:
        404: If any snapshot run not found
        500: If execution fails
    """
    from app.services.snapshot_sync import SnapshotSyncService

    service = SnapshotSyncService(db)

    # Load data from all specified snapshot runs
    all_action_histories = {}
    all_screenshot_registries = []

    for run_id in request.snapshot_run_ids:
        # Get snapshot run
        snapshot_run = service.get_snapshot(run_id)

        if not snapshot_run:
            raise HTTPException(status_code=404, detail=f"Snapshot run {run_id} not found")

        snapshot_dir = Path(snapshot_run.run_directory)

        # Build action histories from database
        run_histories = _load_action_histories(db, snapshot_run.id)

        # Merge action histories (combine histories for same pattern_id)
        for pattern_id, history in run_histories.items():
            if pattern_id in all_action_histories:
                # Merge histories for same pattern
                all_action_histories[pattern_id] = _merge_action_histories(
                    all_action_histories[pattern_id], history
                )
            else:
                all_action_histories[pattern_id] = history

        # Build screenshot registry from snapshot files
        screenshot_registry = _load_screenshot_registry(snapshot_dir)
        all_screenshot_registries.append(screenshot_registry)

    # Merge all screenshot registries into one
    merged_registry = _merge_screenshot_registries(all_screenshot_registries)

    # Execute process in mock mode
    try:
        from qontinui.mock.mock_executor import MockExecutor

        executor = MockExecutor(
            action_histories=all_action_histories,
            screenshot_registry=merged_registry,
        )

        result = executor.execute_process(
            process_id=request.process_id,
            process_name=request.process_name,
            actions=[action.dict() for action in request.actions],
            initial_states=set(request.initial_states),
        )

        # Convert to response model
        return MockExecutionResponse(
            process_id=result.process_id,
            process_name=result.process_name,
            start_time=result.start_time.isoformat(),
            end_time=result.end_time.isoformat() if result.end_time else None,
            total_duration_ms=result.total_duration_ms,
            initial_states=sorted(result.initial_states),
            final_states=sorted(result.final_states),
            actions=[
                ActionVisualizationResponse(
                    action_type=a.action_type,
                    screenshot_path=a.screenshot_path,
                    action_location=a.action_location,
                    action_region=a.action_region,
                    success=a.success,
                    matches=a.matches,
                    text=a.text,
                    active_states=sorted(a.active_states),
                    timestamp=a.timestamp.isoformat(),
                    duration_ms=a.duration_ms,
                )
                for a in result.actions
            ],
            success=result.success,
            success_rate=result.success_rate,
            total_actions=len(result.actions),
            successful_actions=sum(1 for a in result.actions if a.success),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock execution failed: {str(e)}") from e


@router.get("/snapshots/{run_id}/screenshots", response_model=StateScreenshotListResponse)
def get_state_screenshots(
    run_id: str,
    active_states: str | None = Query(
        None, description="Filter by active states (comma-separated)"
    ),
    db: Session = Depends(get_db),
):
    """
    Get screenshots associated with active states for a snapshot run.

    Returns all screenshots recorded during the snapshot run, with their
    associated active states. Can be filtered to find screenshots matching
    specific state combinations.

    Args:
        run_id: Snapshot run ID
        active_states: Optional comma-separated list of states to filter by
        db: Database session

    Returns:
        List of screenshots with state associations

    Raises:
        404: If snapshot run not found
    """
    from app.services.snapshot_sync import SnapshotSyncService

    service = SnapshotSyncService(db)
    snapshot_run = service.get_snapshot(run_id)

    if not snapshot_run:
        raise HTTPException(status_code=404, detail=f"Snapshot run {run_id} not found")

    snapshot_dir = Path(snapshot_run.run_directory)
    screenshot_registry = _load_screenshot_registry(snapshot_dir)

    # Filter by active states if provided
    if active_states:
        filter_states = {s.strip() for s in active_states.split(",")}
        screenshots = screenshot_registry.find_all_for_states(filter_states)
    else:
        screenshots = screenshot_registry.screenshots

    # Convert to response
    return StateScreenshotListResponse(
        screenshots=[
            StateScreenshotResponse(
                screenshot_path=s.screenshot_path,
                active_states=sorted(s.active_states),
                timestamp=s.timestamp.isoformat(),
                width=s.width,
                height=s.height,
                state_hash=s.state_hash,
            )
            for s in screenshots
        ],
        total=len(screenshots),
        unique_state_combinations=len(screenshot_registry.get_unique_state_combinations()),
    )


@router.get("/snapshots/{run_id}/screenshot/{screenshot_path:path}")
def get_screenshot_file(
    run_id: str,
    screenshot_path: str,
    db: Session = Depends(get_db),
):
    """
    Serve a screenshot file from a snapshot run.

    Args:
        run_id: Snapshot run ID
        screenshot_path: Relative path to screenshot within snapshot directory
        db: Database session

    Returns:
        Screenshot image file

    Raises:
        404: If snapshot or file not found
        403: If path attempts directory traversal
    """
    from app.services.snapshot_sync import SnapshotSyncService

    service = SnapshotSyncService(db)
    snapshot_run = service.get_snapshot(run_id)

    if not snapshot_run:
        raise HTTPException(status_code=404, detail=f"Snapshot run {run_id} not found")

    # Construct full path
    snapshot_dir = Path(snapshot_run.run_directory)
    full_path = snapshot_dir / screenshot_path

    # Security: Prevent directory traversal
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(snapshot_dir.resolve())):
            raise HTTPException(
                status_code=403, detail="Invalid file path (directory traversal attempt)"
            )
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid file path") from e

    # Check if file exists
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return FileResponse(full_path)


@router.get("/snapshots/{run_id}/thumbnails", response_model=ThumbnailListResponse)
def get_snapshot_thumbnails(
    run_id: str,
    limit: int = Query(4, description="Maximum number of thumbnails to return"),
    db: Session = Depends(get_db),
):
    """
    Get thumbnail previews from a snapshot run.

    Returns the first N screenshots from a snapshot run with their metadata.
    Used for quick preview of snapshot contents in the UI.

    Args:
        run_id: Snapshot run ID
        limit: Maximum number of thumbnails to return (default 4)
        db: Database session

    Returns:
        List of thumbnail information including URLs, states, and action numbers

    Raises:
        404: If snapshot run not found
    """
    import json

    from app.services.snapshot_sync import SnapshotSyncService

    service = SnapshotSyncService(db)
    snapshot_run = service.get_snapshot(run_id)

    if not snapshot_run:
        raise HTTPException(status_code=404, detail=f"Snapshot run {run_id} not found")

    snapshot_dir = Path(snapshot_run.run_directory)

    # Load action log to get screenshot metadata
    action_log_file = snapshot_dir / "action_log.json"
    if not action_log_file.exists():
        raise HTTPException(
            status_code=404, detail=f"Action log not found for snapshot run {run_id}"
        )

    try:
        with open(action_log_file) as f:
            action_log = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load action log: {str(e)}") from e

    # Extract thumbnail information from first N actions with screenshots
    thumbnails = []
    action_number = 0

    for action in action_log:
        screenshot_path = action.get("screenshot_path")
        if not screenshot_path:
            continue

        # Verify screenshot file exists
        full_screenshot_path = snapshot_dir / screenshot_path
        if not full_screenshot_path.exists():
            continue

        thumbnails.append(
            ThumbnailResponse(
                url=f"/api/integration-testing/snapshots/{run_id}/screenshot/{screenshot_path}",
                active_states=action.get("active_states", []),
                action_number=action_number,
                timestamp=action.get("timestamp", ""),
            )
        )

        action_number += 1

        if len(thumbnails) >= limit:
            break

    return ThumbnailListResponse(
        run_id=run_id,
        thumbnails=thumbnails,
        total_screenshots=snapshot_run.total_screenshots,
    )


@router.get("/snapshots/{run_id}/start-screenshot", response_model=StartScreenshotResponse)
def get_start_screenshot(
    run_id: str,
    db: Session = Depends(get_db),
):
    """
    Get the start screenshot and initial states for a snapshot run.

    Returns the first screenshot marked as is_start_screenshot=True,
    which represents the initial state of the automation run. This is
    used to automatically determine initial states for integration testing.

    Args:
        run_id: Snapshot run ID
        db: Database session

    Returns:
        Start screenshot information with initial states

    Raises:
        404: If snapshot run not found
    """
    from app.models.snapshot import SnapshotAction
    from app.services.snapshot_sync import SnapshotSyncService

    service = SnapshotSyncService(db)
    snapshot_run = service.get_snapshot(run_id)

    if not snapshot_run:
        raise HTTPException(status_code=404, detail=f"Snapshot run {run_id} not found")

    # Find the first action with is_start_screenshot=True
    start_action = (
        db.query(SnapshotAction)
        .filter_by(snapshot_run_id=snapshot_run.id, is_start_screenshot=True)
        .order_by(SnapshotAction.sequence_number)
        .first()
    )

    if not start_action or not start_action.screenshot_path:
        # Fallback: use the first action with a screenshot
        start_action = (
            db.query(SnapshotAction)
            .filter_by(snapshot_run_id=snapshot_run.id)
            .filter(SnapshotAction.screenshot_path.isnot(None))
            .order_by(SnapshotAction.sequence_number)
            .first()
        )

    if not start_action or not start_action.screenshot_path:
        return StartScreenshotResponse(
            run_id=run_id,
            screenshot_path=None,
            screenshot_url=None,
            initial_states=[],
            timestamp=None,
            found=False,
        )

    return StartScreenshotResponse(
        run_id=run_id,
        screenshot_path=start_action.screenshot_path,
        screenshot_url=f"/api/integration-testing/snapshots/{run_id}/screenshot/{start_action.screenshot_path}",
        initial_states=start_action.active_states or [],
        timestamp=start_action.timestamp.isoformat() if start_action.timestamp else None,
        found=True,
    )


# Helper functions


def _load_action_histories(db: Session, snapshot_run_id: int) -> dict[str, Any]:
    """Load action histories from database for a snapshot run.

    Args:
        db: Database session
        snapshot_run_id: Snapshot run ID

    Returns:
        Dictionary mapping pattern_id to ActionHistory
    """
    from app.models.snapshot import SnapshotAction, SnapshotPattern
    from qontinui.mock.snapshot import ActionHistory, ActionRecord
    from qontinui.model.element.location import Location
    from qontinui.model.element.region import Region
    from qontinui.model.match.match import Match

    # Get all patterns for this run
    patterns = db.query(SnapshotPattern).filter_by(snapshot_run_id=snapshot_run_id).all()

    action_histories = {}

    for pattern in patterns:
        history = ActionHistory()

        # Get all actions for this pattern
        actions = (
            db.query(SnapshotAction)
            .filter_by(snapshot_run_id=snapshot_run_id, pattern_id=pattern.pattern_id)
            .order_by(SnapshotAction.sequence_number)
            .all()
        )

        for action in actions:
            # Reconstruct Match objects from action data
            match_list = []
            action_data = action.action_data_json or {}
            for match_data in action_data.get("matches", []):
                region_data = match_data.get("region")
                if region_data:
                    region = Region(
                        region_data["x"],
                        region_data["y"],
                        region_data["w"],
                        region_data["h"],
                    )
                    location = Location(region=region)
                    match = Match(target=location, score=match_data.get("score", 0.0))
                    match_list.append(match)

            # Create ActionRecord
            record = ActionRecord(
                action_type=action.action_type,
                action_success=action.success,
                match_list=match_list,
                duration=float(action.duration_ms / 1000) if action.duration_ms else 0.0,
                timestamp=action.timestamp,
                active_states=set(action.active_states or []),
                metadata=action_data.get("metadata", {}),
            )

            history.add_record(record)

        action_histories[pattern.pattern_id] = history

    return action_histories


def _load_screenshot_registry(snapshot_dir: Path) -> Any:
    """Load screenshot registry from snapshot directory.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        StateScreenshotRegistry instance
    """
    import json

    from PIL import Image

    from qontinui.mock.state_screenshot import StateScreenshotRegistry

    registry = StateScreenshotRegistry()

    # Load screenshots directory
    screenshots_dir = snapshot_dir / "screenshots"
    if not screenshots_dir.exists():
        return registry

    # Load action log to get active states for each screenshot
    action_log_file = snapshot_dir / "action_log.json"
    if not action_log_file.exists():
        return registry

    with open(action_log_file) as f:
        action_log = json.load(f)

    # Map screenshots to active states
    for action in action_log:
        screenshot_path = action.get("screenshot_path")
        if not screenshot_path:
            continue

        full_screenshot_path = snapshot_dir / screenshot_path
        if not full_screenshot_path.exists():
            continue

        # Get image dimensions
        try:
            with Image.open(full_screenshot_path) as img:
                width, height = img.size
        except Exception:
            continue

        # Register screenshot with active states
        active_states = set(action.get("active_states", []))
        timestamp = datetime.fromisoformat(action["timestamp"])

        registry.register_screenshot(
            screenshot_path=screenshot_path,
            active_states=active_states,
            timestamp=timestamp,
            width=width,
            height=height,
        )

    return registry


def _merge_action_histories(history1: Any, history2: Any) -> Any:
    """Merge two ActionHistory objects for the same pattern.

    Combines all action records from both histories, providing a larger
    pool of historical data for matching.

    Args:
        history1: First ActionHistory
        history2: Second ActionHistory

    Returns:
        Merged ActionHistory containing records from both
    """
    from qontinui.mock.snapshot import ActionHistory

    merged = ActionHistory()

    # Add all records from both histories
    for record in history1.snapshots:
        merged.add_record(record)

    for record in history2.snapshots:
        merged.add_record(record)

    return merged


def _merge_screenshot_registries(registries: list[Any]) -> Any:
    """Merge multiple StateScreenshotRegistry objects.

    Combines screenshots from all registries, providing a larger pool
    of state-screenshot associations for matching.

    Args:
        registries: List of StateScreenshotRegistry instances

    Returns:
        Merged StateScreenshotRegistry containing all screenshots
    """
    from qontinui.mock.state_screenshot import StateScreenshotRegistry

    if not registries:
        return StateScreenshotRegistry()

    if len(registries) == 1:
        return registries[0]

    # Create new registry and add all screenshots
    merged = StateScreenshotRegistry()

    for registry in registries:
        for screenshot in registry.screenshots:
            merged.register_screenshot(
                screenshot_path=screenshot.screenshot_path,
                active_states=screenshot.active_states,
                timestamp=screenshot.timestamp,
                width=screenshot.width,
                height=screenshot.height,
            )

    return merged


@router.post("/coverage/analyze", response_model=CoverageReportResponse)
def analyze_coverage(
    request: CoverageAnalysisRequest,
    db: Session = Depends(get_db),
):
    """
    Analyze state coverage for a process across snapshot runs.

    This endpoint analyzes:
    - Which states are covered/uncovered
    - State transitions (covered/missing)
    - Action coverage per state
    - Coverage percentages
    - Coverage gaps and recommendations

    Args:
        request: Coverage analysis request
        db: Database session

    Returns:
        Complete coverage report with metrics and recommendations

    Raises:
        404: If snapshot runs not found
        500: If analysis fails
    """
    try:
        analyzer = StateCoverageAnalyzer(db)
        report = analyzer.analyze_coverage(
            process_id=request.process_id,
            process_name=request.process_name,
            snapshot_run_ids=request.snapshot_run_ids,
            expected_states=request.expected_states,
        )

        # Convert to response model
        return CoverageReportResponse(
            process_id=report.process_id,
            process_name=report.process_name,
            snapshot_run_ids=report.snapshot_run_ids,
            analysis_time=report.analysis_time.isoformat(),
            overall_coverage_percentage=report.overall_coverage_percentage,
            total_states=report.total_states,
            covered_states=report.covered_states,
            uncovered_states=report.uncovered_states,
            total_transitions=report.total_transitions,
            covered_transitions=report.covered_transitions,
            missing_transitions=report.missing_transitions,
            state_metrics={
                state: StateCoverageMetricsResponse(
                    state_name=metric.state_name,
                    screenshot_count=metric.screenshot_count,
                    actions_performed=metric.actions_performed,
                    last_tested=metric.last_tested.isoformat() if metric.last_tested else None,
                    coverage_percentage=metric.coverage_percentage,
                    transitions_to=sorted(metric.transitions_to),
                    transitions_from=sorted(metric.transitions_from),
                    action_types=sorted(metric.action_types),
                    patterns_tested=sorted(metric.patterns_tested),
                )
                for state, metric in report.state_metrics.items()
            },
            transitions=[
                StateTransitionResponse(
                    from_state=t.from_state,
                    to_state=t.to_state,
                    count=t.count,
                    covered=t.covered,
                    last_occurrence=t.last_occurrence.isoformat() if t.last_occurrence else None,
                    actions_triggering=t.actions_triggering,
                )
                for t in report.transitions
            ],
            coverage_gaps=[
                CoverageGapResponse(
                    gap_type=g.gap_type,
                    severity=g.severity,
                    description=g.description,
                    recommendation=g.recommendation,
                    affected_states=g.affected_states,
                    metric_value=g.metric_value,
                )
                for g in report.coverage_gaps
            ],
            recommendations=report.recommendations,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coverage analysis failed: {str(e)}") from e


@router.get("/coverage/report/{process_id}", response_model=CoverageReportResponse)
def get_coverage_report(
    process_id: str,
    snapshot_run_ids: str = Query(..., description="Comma-separated list of snapshot run IDs"),
    expected_states: str | None = Query(
        None, description="Optional comma-separated expected states"
    ),
    db: Session = Depends(get_db),
):
    """
    Get coverage report for a process.

    This is a convenience endpoint that wraps analyze_coverage
    with query parameters instead of a request body.

    Args:
        process_id: Process identifier
        snapshot_run_ids: Comma-separated snapshot run IDs
        expected_states: Optional comma-separated expected states
        db: Database session

    Returns:
        Complete coverage report

    Raises:
        404: If snapshot runs not found
        500: If analysis fails
    """
    # Parse snapshot run IDs
    run_ids = [s.strip() for s in snapshot_run_ids.split(",") if s.strip()]

    if not run_ids:
        raise HTTPException(status_code=400, detail="At least one snapshot_run_id is required")

    # Parse expected states
    states = None
    if expected_states:
        states = [s.strip() for s in expected_states.split(",") if s.strip()]

    # Create analysis request
    request = CoverageAnalysisRequest(
        process_id=process_id,
        process_name=process_id,  # Use process_id as name for GET requests
        snapshot_run_ids=run_ids,
        expected_states=states,
    )

    # Delegate to analyze_coverage
    return analyze_coverage(request, db)


# Smart Snapshot Selection API endpoints


class SnapshotAnalysisResponse(BaseModel):
    """Response for snapshot analysis."""

    run_id: str
    state_coverage: dict[str, Any]
    action_coverage: dict[str, Any]
    transition_coverage: dict[str, Any]


class SnapshotRecommendationRequest(BaseModel):
    """Request for smart snapshot recommendations."""

    max_snapshots: int = Field(3, description="Maximum number of snapshots to recommend")
    strategy: str | None = Field(
        None,
        description="Specific strategy (max_coverage, min_overlap, recent_diverse, priority_weighted)",
    )
    workflow_id: int | None = Field(None, description="Filter by workflow ID")
    execution_mode: str | None = Field(None, description="Filter by execution mode")
    min_actions: int | None = Field(None, description="Minimum number of actions")
    start_date: str | None = Field(None, description="Filter snapshots from this date")
    recency_weight: float | None = Field(
        0.5, description="Weight for recency (recent_diverse strategy)"
    )
    recency_days: int | None = Field(
        30, description="Days for recency decay (recent_diverse strategy)"
    )
    priority_weight: float | None = Field(
        0.3, description="Weight for priority (priority_weighted strategy)"
    )


class RecommendationResponse(BaseModel):
    """Single recommendation response."""

    strategy: str
    recommended_run_ids: list[str]
    score: float
    reason: str


class SnapshotRecommendationsResponse(BaseModel):
    """Response with snapshot recommendations."""

    timestamp: str
    available_snapshots: int
    max_snapshots: int
    recommendations: list[RecommendationResponse]


class DuplicateGroupResponse(BaseModel):
    """Response for duplicate snapshot group."""

    representative_run_id: str
    duplicate_run_ids: list[str]
    duplicate_count: int
    similarity_score: float
    duplicate_reasons: list[str]
    runs: list[dict[str, Any]]


class DuplicatesResponse(BaseModel):
    """Response with all duplicate groups."""

    timestamp: str
    duplicate_groups: list[DuplicateGroupResponse]
    total_duplicates: int


class SetPriorityRequest(BaseModel):
    """Request to set snapshot priority."""

    priority: int = Field(..., description="Priority value (default 50, higher = more preferred)")


class CoverageReportRequest(BaseModel):
    """Request for coverage report."""

    run_ids: list[str] = Field(..., description="Snapshot run IDs to analyze")


@router.post("/snapshots/analyze", response_model=SnapshotAnalysisResponse)
def analyze_snapshot(
    run_id: str = Query(..., description="Snapshot run ID"),
    db: Session = Depends(get_db),
):
    """
    Analyze a single snapshot run for coverage metrics.

    Calculates:
    - State coverage (unique states, state combinations)
    - Action type coverage
    - State transition coverage

    Args:
        run_id: Snapshot run ID
        db: Database session

    Returns:
        Analysis with coverage metrics

    Raises:
        404: If snapshot not found
        500: If analysis fails
    """
    try:
        from app.services.snapshot_analysis import SnapshotAnalysisService

        analyzer = SnapshotAnalysisService(db)

        state_coverage = analyzer.calculate_state_coverage(run_id)
        action_coverage = analyzer.calculate_action_type_coverage(run_id)
        transition_coverage = analyzer.calculate_state_transitions(run_id)

        return SnapshotAnalysisResponse(
            run_id=run_id,
            state_coverage=state_coverage,
            action_coverage=action_coverage,
            transition_coverage=transition_coverage,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot analysis failed: {str(e)}") from e


@router.post("/snapshots/recommend", response_model=SnapshotRecommendationsResponse)
def recommend_snapshots(
    request: SnapshotRecommendationRequest,
    db: Session = Depends(get_db),
):
    """
    Get smart snapshot recommendations using various strategies.

    Strategies:
    - max_coverage: Maximize state coverage
    - min_overlap: Minimize overlap (most diverse)
    - recent_diverse: Balance recency and diversity
    - priority_weighted: Use priority weights and coverage

    Args:
        request: Recommendation request with filters and parameters
        db: Database session

    Returns:
        Recommendations from one or all strategies

    Raises:
        500: If recommendation generation fails
    """
    try:
        from datetime import datetime

        from app.services.snapshot_recommendations import SnapshotRecommendationService

        service = SnapshotRecommendationService(db)

        # Build filters
        filters = {}
        if request.workflow_id is not None:
            filters["workflow_id"] = request.workflow_id
        if request.execution_mode:
            filters["execution_mode"] = request.execution_mode
        if request.min_actions is not None:
            filters["min_actions"] = request.min_actions
        if request.start_date:
            filters["start_date"] = datetime.fromisoformat(request.start_date)

        # Build strategy parameters
        strategy_params = {}
        if request.recency_weight is not None:
            strategy_params["recency_weight"] = request.recency_weight
        if request.recency_days is not None:
            strategy_params["recency_days"] = request.recency_days
        if request.priority_weight is not None:
            strategy_params["priority_weight"] = request.priority_weight

        # Get recommendations
        result = service.get_recommendations(
            max_snapshots=request.max_snapshots,
            strategy=request.strategy,
            filters=filters if filters else None,
            **strategy_params,
        )

        return SnapshotRecommendationsResponse(
            timestamp=result["timestamp"],
            available_snapshots=result["available_snapshots"],
            max_snapshots=result["max_snapshots"],
            recommendations=[RecommendationResponse(**rec) for rec in result["recommendations"]],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {str(e)}"
        ) from e


@router.get("/snapshots/duplicates", response_model=DuplicatesResponse)
def get_duplicates(
    workflow_id: int | None = Query(None, description="Filter by workflow ID"),
    include_unmarked: bool = Query(True, description="Include unmarked duplicates"),
    db: Session = Depends(get_db),
):
    """
    Get duplicate snapshot groups.

    Detects duplicates based on:
    - State coverage similarity (>95%)
    - Action sequence similarity (>90%)
    - Screenshot similarity (>85%)

    Args:
        workflow_id: Optional workflow filter
        include_unmarked: Include runs not marked as duplicates
        db: Database session

    Returns:
        List of duplicate groups with similarity scores

    Raises:
        500: If duplicate detection fails
    """
    try:
        from app.services.snapshot_deduplication import SnapshotDeduplicationService

        service = SnapshotDeduplicationService(db)

        duplicate_groups = service.get_duplicate_groups(
            workflow_id=workflow_id,
            include_unmarked=include_unmarked,
        )

        return DuplicatesResponse(
            timestamp=datetime.utcnow().isoformat(),
            duplicate_groups=[DuplicateGroupResponse(**group) for group in duplicate_groups],
            total_duplicates=sum(g["duplicate_count"] for g in duplicate_groups),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate detection failed: {str(e)}") from e


@router.put("/snapshots/{run_id}/priority")
def set_snapshot_priority(
    run_id: str,
    request: SetPriorityRequest,
    db: Session = Depends(get_db),
):
    """
    Set priority for a snapshot run.

    Priority influences recommendations:
    - Higher priority = preferred in priority_weighted strategy
    - Default priority is 50
    - Recommended range: 1-100

    Args:
        run_id: Snapshot run ID
        request: Priority value
        db: Database session

    Returns:
        Success message with updated priority

    Raises:
        404: If snapshot not found
        500: If update fails
    """
    try:
        from app.models.snapshot import SnapshotRun

        snapshot = db.query(SnapshotRun).filter_by(run_id=run_id).first()

        if not snapshot:
            raise HTTPException(status_code=404, detail=f"Snapshot run {run_id} not found")

        snapshot.priority = request.priority
        db.commit()

        return {
            "run_id": run_id,
            "priority": request.priority,
            "message": f"Priority updated to {request.priority}",
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update priority: {str(e)}") from e


@router.get("/snapshots/coverage")
def get_coverage_report_multi(
    run_ids: str = Query(..., description="Comma-separated snapshot run IDs"),
    db: Session = Depends(get_db),
):
    """
    Get comprehensive coverage report for multiple snapshots.

    Analyzes:
    - Individual snapshot coverage metrics
    - Combined coverage across all snapshots
    - Complementarity analysis (overlap, diversity)

    Args:
        run_ids: Comma-separated snapshot run IDs
        db: Database session

    Returns:
        Comprehensive coverage report

    Raises:
        400: If no run IDs provided
        500: If analysis fails
    """
    try:
        from app.services.snapshot_analysis import SnapshotAnalysisService

        # Parse run IDs
        run_id_list = [r.strip() for r in run_ids.split(",") if r.strip()]

        if not run_id_list:
            raise HTTPException(status_code=400, detail="At least one run_id is required")

        analyzer = SnapshotAnalysisService(db)

        # Generate coverage report
        report = analyzer.get_coverage_report(run_id_list)

        return report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Coverage report generation failed: {str(e)}"
        ) from e
