"""API endpoints for snapshot search and analytics."""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.snapshot_query import SnapshotQueryService

router = APIRouter(prefix="/snapshots", tags=["snapshot-search"])


# Pydantic models for request/response


class SearchRequest(BaseModel):
    """Request body for advanced snapshot search."""

    query: str | None = Field(None, description="Text search query")
    filters: dict[str, Any] | None = Field(None, description="Filter specifications")
    sort_by: str = Field("start_time", description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order: asc or desc")
    limit: int = Field(50, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class SnapshotSearchResult(BaseModel):
    """Search result for a single snapshot."""

    id: int
    run_id: str
    start_time: str
    end_time: str | None
    execution_mode: str
    total_actions: int
    successful_actions: int
    success_rate: float
    tags: list[str] | None
    notes: str | None

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Response for snapshot search."""

    total: int
    snapshots: list[SnapshotSearchResult]
    limit: int
    offset: int


class SnapshotStatistics(BaseModel):
    """Detailed statistics for a snapshot."""

    run_id: str
    action_breakdown: dict[str, dict[str, int]]
    pattern_usage: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    performance: dict[str, Any]


class PatternAnalytics(BaseModel):
    """Analytics for a specific pattern."""

    pattern_id: str
    pattern_name: str
    total_snapshots: int
    total_finds: int
    successful_finds: int
    success_rate: float
    avg_duration_ms: float
    match_rate: float
    reliability_trend: list[dict[str, Any]]


class ExecutionTrends(BaseModel):
    """Execution trends over time."""

    period_days: int
    group_by: str
    success_rate_trend: list[dict[str, Any]]
    execution_count_trend: list[dict[str, Any]]
    common_failures: list[dict[str, Any]]
    pattern_usage_trend: list[dict[str, Any]]


# API endpoints


@router.post("/search", response_model=SearchResponse)
def search_snapshots(
    request: SearchRequest,
    db: Session = Depends(get_db),
):
    """
    Advanced snapshot search with filtering and sorting.

    This endpoint supports:
    - Text search across run_id, execution_mode, notes, and metadata
    - Complex filtering (date ranges, execution mode, success rate, tags, etc.)
    - Sorting by any field
    - Pagination

    Example filters:
    ```json
    {
      "filters": {
        "start_time": {"gte": "2025-01-01T00:00:00", "lte": "2025-01-31T23:59:59"},
        "execution_mode": {"in": ["real", "hybrid"]},
        "tags": {"contains": ["regression"]},
        "metadata_json": {"jsonb_contains": {"execution_mode": "real"}}
      }
    }
    ```

    Args:
        request: Search request with query, filters, and pagination
        db: Database session

    Returns:
        Search results with total count
    """
    service = SnapshotQueryService(db)

    try:
        snapshots, total = service.search_snapshots(
            query_text=request.query,
            filters=request.filters,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
            limit=request.limit,
            offset=request.offset,
        )

        # Convert to response models
        results = []
        for snapshot in snapshots:
            success_rate = (
                snapshot.successful_actions / snapshot.total_actions
                if snapshot.total_actions > 0
                else 0.0
            )

            results.append(
                SnapshotSearchResult(
                    id=snapshot.id,
                    run_id=snapshot.run_id,
                    start_time=snapshot.start_time.isoformat(),
                    end_time=snapshot.end_time.isoformat() if snapshot.end_time else None,
                    execution_mode=snapshot.execution_mode,
                    total_actions=snapshot.total_actions,
                    successful_actions=snapshot.successful_actions,
                    success_rate=round(success_rate, 3),
                    tags=snapshot.tags,
                    notes=snapshot.notes,
                )
            )

        return SearchResponse(
            total=total,
            snapshots=results,
            limit=request.limit,
            offset=request.offset,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.get("/{run_id}/statistics", response_model=SnapshotStatistics)
def get_snapshot_statistics(
    run_id: str,
    db: Session = Depends(get_db),
):
    """
    Get detailed statistics for a specific snapshot.

    Returns:
    - Action breakdown by type (find, click, type, etc.)
    - Pattern usage with success rates
    - Chronological action timeline
    - Performance metrics (durations, slowest/fastest actions)

    Args:
        run_id: Snapshot run ID
        db: Database session

    Returns:
        Detailed snapshot statistics

    Raises:
        404: If snapshot not found
    """
    service = SnapshotQueryService(db)

    stats = service.get_snapshot_statistics(run_id)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Snapshot {run_id} not found")

    return SnapshotStatistics(**stats)


@router.get("/patterns/{pattern_id}/analytics", response_model=PatternAnalytics)
def get_pattern_analytics(
    pattern_id: str,
    start_date: str | None = Query(None, description="Start date (ISO format: YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (ISO format: YYYY-MM-DD)"),
    workflow_id: int | None = Query(None, description="Filter by workflow ID"),
    db: Session = Depends(get_db),
):
    """
    Get analytics for a specific pattern across all snapshots.

    Returns pattern performance metrics:
    - Success rate across snapshots
    - Average match duration
    - Match rate (avg matches per find)
    - Reliability trend over time

    Args:
        pattern_id: Pattern identifier
        start_date: Optional start date filter
        end_date: Optional end date filter
        workflow_id: Optional workflow filter
        db: Database session

    Returns:
        Pattern analytics data

    Raises:
        404: If pattern not found in any snapshot
    """
    service = SnapshotQueryService(db)

    # Parse dates
    start_datetime = datetime.fromisoformat(start_date) if start_date else None
    end_datetime = datetime.fromisoformat(end_date) if end_date else None

    analytics = service.get_pattern_analytics(
        pattern_id=pattern_id,
        start_date=start_datetime,
        end_date=end_datetime,
        workflow_id=workflow_id,
    )

    if not analytics:
        raise HTTPException(
            status_code=404,
            detail=f"Pattern {pattern_id} not found in any snapshot",
        )

    return PatternAnalytics(**analytics)


@router.get("/analytics/trends", response_model=ExecutionTrends)
def get_execution_trends(
    period_days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    group_by: str = Query("day", description="Group by: day or week"),
    execution_mode: str | None = Query(None, description="Filter by execution mode"),
    workflow_id: int | None = Query(None, description="Filter by workflow ID"),
    db: Session = Depends(get_db),
):
    """
    Get execution trends and statistics over time.

    Returns:
    - Success rate trend (daily/weekly)
    - Execution count trend by mode
    - Common failure patterns
    - Most used patterns

    Args:
        period_days: Number of days to analyze (1-365)
        group_by: Grouping interval (day or week)
        execution_mode: Optional execution mode filter
        workflow_id: Optional workflow filter
        db: Database session

    Returns:
        Execution trends data
    """
    service = SnapshotQueryService(db)

    # Build filters
    filters = {}
    if execution_mode:
        filters["execution_mode"] = execution_mode
    if workflow_id:
        filters["workflow_id"] = workflow_id

    trends = service.get_execution_trends(
        period_days=period_days,
        group_by=group_by,
        filters=filters if filters else None,
    )

    return ExecutionTrends(**trends)


@router.get("/analytics/patterns/top")
def get_top_patterns(
    metric: str = Query(
        "usage_count",
        description="Metric to rank by: usage_count, success_rate, or failure_count",
    ),
    limit: int = Query(10, ge=1, le=100, description="Number of patterns to return"),
    period_days: int = Query(30, ge=1, le=365, description="Period to analyze"),
    ascending: bool = Query(False, description="Sort ascending (lowest first)"),
    db: Session = Depends(get_db),
):
    """
    Get top/bottom patterns by various metrics.

    Metrics:
    - usage_count: Most/least used patterns
    - success_rate: Most/least reliable patterns
    - failure_count: Patterns with most/least failures

    Args:
        metric: Metric to rank by
        limit: Number of results
        period_days: Period to analyze
        ascending: If True, return bottom patterns (lowest metric)
        db: Database session

    Returns:
        List of patterns ranked by metric
    """
    service = SnapshotQueryService(db)
    start_date = datetime.now() - timedelta(days=period_days)

    # Get trends (includes top failures and usage)
    trends = service.get_execution_trends(
        period_days=period_days,
        group_by="day",
        filters=None,
    )

    if metric == "failure_count":
        patterns = trends["common_failures"]
        if ascending:
            patterns = list(reversed(patterns))
        return {"metric": metric, "patterns": patterns[:limit]}

    elif metric == "usage_count":
        patterns = trends["pattern_usage_trend"]
        if ascending:
            patterns = list(reversed(patterns))
        return {"metric": metric, "patterns": patterns[:limit]}

    elif metric == "success_rate":
        # Need to calculate success rates for all patterns
        from sqlalchemy import func

        from app.models.snapshot import SnapshotPattern, SnapshotRun

        results = (
            db.query(
                SnapshotPattern.pattern_id,
                SnapshotPattern.pattern_name,
                func.sum(SnapshotPattern.successful_finds).label("successful"),
                func.sum(SnapshotPattern.total_finds).label("total"),
            )
            .join(SnapshotRun, SnapshotPattern.snapshot_run_id == SnapshotRun.id)
            .filter(SnapshotRun.start_time >= start_date)
            .group_by(SnapshotPattern.pattern_id, SnapshotPattern.pattern_name)
            .having(func.sum(SnapshotPattern.total_finds) >= 5)  # Minimum 5 uses
            .all()
        )

        patterns = []
        for pattern_id, pattern_name, successful, total in results:
            success_rate = successful / total if total > 0 else 0.0
            patterns.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_name": pattern_name,
                    "success_rate": round(success_rate, 3),
                    "total_finds": total,
                }
            )

        # Sort by success rate
        patterns.sort(key=lambda x: x["success_rate"], reverse=not ascending)

        return {"metric": metric, "patterns": patterns[:limit]}

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric: {metric}. Choose from: usage_count, success_rate, failure_count",
        )
