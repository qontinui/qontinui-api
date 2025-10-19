"""API endpoints for snapshot management."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.snapshot_sync import SnapshotSyncService

router = APIRouter(prefix="/snapshots", tags=["snapshots"])


# Pydantic models for request/response


class SyncSnapshotRequest(BaseModel):
    """Request body for syncing a snapshot directory."""

    snapshot_directory: str = Field(..., description="Path to snapshot directory")
    workflow_id: int | None = Field(None, description="Associated workflow ID")
    created_by: int | None = Field(None, description="User ID who created this")
    tags: list[str] | None = Field(None, description="Tags for categorization")
    notes: str | None = Field(None, description="Optional notes")


class SnapshotRunResponse(BaseModel):
    """Response model for snapshot run."""

    id: int
    run_id: str
    run_directory: str
    start_time: str
    end_time: str | None
    duration_seconds: float | None
    execution_mode: str
    total_actions: int
    successful_actions: int
    failed_actions: int
    total_screenshots: int
    patterns_count: int
    workflow_id: int | None
    created_by: int | None
    tags: list[str] | None
    notes: str | None

    class Config:
        from_attributes = True


class SnapshotListResponse(BaseModel):
    """Response model for listing snapshots."""

    total: int
    snapshots: list[SnapshotRunResponse]


class SnapshotDetailResponse(SnapshotRunResponse):
    """Detailed response including metadata."""

    metadata_json: dict

    class Config:
        from_attributes = True


# API endpoints


@router.post("/sync", response_model=SnapshotRunResponse, status_code=201)
def sync_snapshot(
    request: SyncSnapshotRequest,
    db: Session = Depends(get_db),
):
    """
    Sync a snapshot directory to the database.

    This endpoint reads snapshot files (metadata.json, action_log.json, patterns/*)
    and imports them into the database for fast querying and analysis.

    Args:
        request: Sync request with snapshot directory path
        db: Database session

    Returns:
        The created snapshot run

    Raises:
        404: If snapshot directory or metadata.json not found
        409: If snapshot already exists in database
        500: If sync fails due to other errors
    """
    service = SnapshotSyncService(db)

    try:
        snapshot_dir = Path(request.snapshot_directory)
        snapshot_run = service.sync_snapshot_directory(
            snapshot_dir=snapshot_dir,
            workflow_id=request.workflow_id,
            created_by=request.created_by,
            tags=request.tags,
            notes=request.notes,
        )

        return SnapshotRunResponse(
            id=snapshot_run.id,
            run_id=snapshot_run.run_id,
            run_directory=snapshot_run.run_directory,
            start_time=snapshot_run.start_time.isoformat(),
            end_time=(snapshot_run.end_time.isoformat() if snapshot_run.end_time else None),
            duration_seconds=(
                float(snapshot_run.duration_seconds) if snapshot_run.duration_seconds else None
            ),
            execution_mode=snapshot_run.execution_mode,
            total_actions=snapshot_run.total_actions,
            successful_actions=snapshot_run.successful_actions,
            failed_actions=snapshot_run.failed_actions,
            total_screenshots=snapshot_run.total_screenshots,
            patterns_count=snapshot_run.patterns_count,
            workflow_id=snapshot_run.workflow_id,
            created_by=snapshot_run.created_by,
            tags=snapshot_run.tags,
            notes=snapshot_run.notes,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=409, detail=str(e)) from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync snapshot: {str(e)}") from e


@router.post("/import", response_model=SnapshotRunResponse, status_code=201)
def import_snapshot(
    request: SyncSnapshotRequest,
    db: Session = Depends(get_db),
):
    """
    Import a snapshot directory to the database.

    Alias for /sync endpoint with more intuitive naming for frontend.
    This endpoint reads snapshot files (metadata.json, action_log.json, patterns/*)
    and imports them into the database for integration testing.

    Args:
        request: Import request with snapshot directory path
        db: Database session

    Returns:
        The created snapshot run

    Raises:
        404: If snapshot directory or metadata.json not found
        409: If snapshot already exists in database
        500: If import fails due to other errors
    """
    # Delegate to sync_snapshot function
    return sync_snapshot(request=request, db=db)


@router.get("", response_model=SnapshotListResponse)
def list_snapshots(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    workflow_id: int | None = Query(None, description="Filter by workflow ID"),
    created_by: int | None = Query(None, description="Filter by creator user ID"),
    tags: str | None = Query(None, description="Comma-separated tags (must have all)"),
    db: Session = Depends(get_db),
):
    """
    List snapshots with filtering and pagination.

    Args:
        limit: Maximum number of results (1-100)
        offset: Number of results to skip
        workflow_id: Filter by workflow ID
        created_by: Filter by creator user ID
        tags: Filter by tags (comma-separated, must have all)
        db: Database session

    Returns:
        List of snapshots with total count
    """
    service = SnapshotSyncService(db)

    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None

    # Get snapshots
    snapshots = service.list_snapshots(
        limit=limit,
        offset=offset,
        workflow_id=workflow_id,
        created_by=created_by,
        tags=tag_list,
    )

    # Convert to response models
    snapshot_responses = [
        SnapshotRunResponse(
            id=snap.id,
            run_id=snap.run_id,
            run_directory=snap.run_directory,
            start_time=snap.start_time.isoformat(),
            end_time=snap.end_time.isoformat() if snap.end_time else None,
            duration_seconds=float(snap.duration_seconds) if snap.duration_seconds else None,
            execution_mode=snap.execution_mode,
            total_actions=snap.total_actions,
            successful_actions=snap.successful_actions,
            failed_actions=snap.failed_actions,
            total_screenshots=snap.total_screenshots,
            patterns_count=snap.patterns_count,
            workflow_id=snap.workflow_id,
            created_by=snap.created_by,
            tags=snap.tags,
            notes=snap.notes,
        )
        for snap in snapshots
    ]

    return SnapshotListResponse(
        total=len(snapshot_responses),
        snapshots=snapshot_responses,
    )


@router.get("/{run_id}", response_model=SnapshotDetailResponse)
def get_snapshot(
    run_id: str,
    db: Session = Depends(get_db),
):
    """
    Get detailed snapshot information.

    Args:
        run_id: The snapshot run ID
        db: Database session

    Returns:
        Detailed snapshot information including full metadata

    Raises:
        404: If snapshot not found
    """
    service = SnapshotSyncService(db)
    snapshot = service.get_snapshot(run_id)

    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Snapshot {run_id} not found")

    return SnapshotDetailResponse(
        id=snapshot.id,
        run_id=snapshot.run_id,
        run_directory=snapshot.run_directory,
        start_time=snapshot.start_time.isoformat(),
        end_time=snapshot.end_time.isoformat() if snapshot.end_time else None,
        duration_seconds=float(snapshot.duration_seconds) if snapshot.duration_seconds else None,
        execution_mode=snapshot.execution_mode,
        total_actions=snapshot.total_actions,
        successful_actions=snapshot.successful_actions,
        failed_actions=snapshot.failed_actions,
        total_screenshots=snapshot.total_screenshots,
        patterns_count=snapshot.patterns_count,
        workflow_id=snapshot.workflow_id,
        created_by=snapshot.created_by,
        tags=snapshot.tags,
        notes=snapshot.notes,
        metadata_json=snapshot.metadata_json,
    )


@router.get("/{run_id}/files/{file_path:path}")
def get_snapshot_file(
    run_id: str,
    file_path: str,
    db: Session = Depends(get_db),
):
    """
    Serve a file from a snapshot directory.

    This endpoint allows accessing screenshots and other files
    from the snapshot directory.

    Args:
        run_id: The snapshot run ID
        file_path: Relative path to file (e.g., "screenshots/screenshot-001.png")
        db: Database session

    Returns:
        The requested file

    Raises:
        404: If snapshot or file not found
        403: If file path attempts directory traversal
    """
    service = SnapshotSyncService(db)
    snapshot = service.get_snapshot(run_id)

    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Snapshot {run_id} not found")

    # Construct full path
    snapshot_dir = Path(snapshot.run_directory)
    full_path = snapshot_dir / file_path

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
        raise HTTPException(status_code=404, detail="File not found")

    # Serve file
    return FileResponse(full_path)


@router.delete("/{run_id}", status_code=204)
def delete_snapshot(
    run_id: str,
    delete_files: bool = Query(False, description="Also delete snapshot files from disk"),
    db: Session = Depends(get_db),
):
    """
    Delete a snapshot from the database.

    Args:
        run_id: The snapshot run ID
        delete_files: If True, also delete snapshot files from disk
        db: Database session

    Returns:
        No content (204)

    Raises:
        404: If snapshot not found
    """
    service = SnapshotSyncService(db)
    deleted = service.delete_snapshot(run_id, delete_files=delete_files)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Snapshot {run_id} not found")

    return None
