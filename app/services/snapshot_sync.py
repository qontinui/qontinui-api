"""Service for syncing snapshot files to database."""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.snapshot import (
    SnapshotAction,
    SnapshotMatch,
    SnapshotPattern,
    SnapshotRun,
)


class SnapshotSyncService:
    """Service for syncing snapshot data from files to database."""

    def __init__(self, db: Session):
        """Initialize the sync service.

        Args:
            db: Database session
        """
        self.db = db

    def sync_snapshot_directory(
        self,
        snapshot_dir: Path,
        workflow_id: int | None = None,
        created_by: int | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> SnapshotRun:
        """Sync a snapshot directory to the database.

        Args:
            snapshot_dir: Path to snapshot directory
            workflow_id: Optional workflow ID to associate
            created_by: Optional user ID who created this snapshot
            tags: Optional tags for categorization
            notes: Optional notes about this snapshot

        Returns:
            The created SnapshotRun instance

        Raises:
            FileNotFoundError: If snapshot directory or metadata.json not found
            ValueError: If snapshot already exists in database
        """
        snapshot_dir = Path(snapshot_dir)
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

        # Load metadata.json
        metadata_file = snapshot_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {snapshot_dir}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        run_id = metadata.get("run_id")
        if not run_id:
            raise ValueError("run_id not found in metadata.json")

        # Check if already exists
        existing = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if existing:
            raise ValueError(f"Snapshot run {run_id} already exists in database")

        # Parse timestamps
        start_time = datetime.fromisoformat(metadata["start_time"])
        end_time = (
            datetime.fromisoformat(metadata["end_time"]) if metadata.get("end_time") else None
        )

        # Calculate statistics from metadata
        statistics = metadata.get("statistics", {})

        # Create SnapshotRun
        snapshot_run = SnapshotRun(
            run_id=run_id,
            run_directory=str(snapshot_dir.absolute()),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=metadata.get("duration_seconds"),
            execution_mode=metadata.get("execution_mode", "unknown"),
            total_actions=statistics.get("total_actions", 0),
            successful_actions=statistics.get("successful_actions", 0),
            failed_actions=statistics.get("failed_actions", 0),
            total_screenshots=statistics.get("total_screenshots", 0),
            patterns_count=len(metadata.get("patterns", {})),
            metadata_json=metadata,
            workflow_id=workflow_id,
            created_by=created_by,
            tags=tags,
            notes=notes,
        )
        self.db.add(snapshot_run)
        self.db.flush()  # Get the ID without committing

        # Sync action log
        action_log_file = snapshot_dir / "action_log.json"
        if action_log_file.exists():
            self._sync_action_log(snapshot_run, action_log_file)

        # Sync patterns
        patterns_dir = snapshot_dir / "patterns"
        if patterns_dir.exists():
            self._sync_patterns(snapshot_run, patterns_dir)

        self.db.commit()
        self.db.refresh(snapshot_run)

        return snapshot_run

    def _sync_action_log(self, snapshot_run: SnapshotRun, action_log_file: Path):
        """Sync action log to database.

        Args:
            snapshot_run: The snapshot run instance
            action_log_file: Path to action_log.json
        """
        with open(action_log_file) as f:
            actions = json.load(f)

        for idx, action_data in enumerate(actions):
            # Parse timestamp
            timestamp = datetime.fromisoformat(action_data["timestamp"])

            # Create action record
            action = SnapshotAction(
                snapshot_run_id=snapshot_run.id,
                sequence_number=idx,
                timestamp=timestamp,
                action_type=action_data.get("action_type", "unknown"),
                pattern_id=action_data.get("pattern_id"),
                pattern_name=action_data.get("pattern_name"),
                success=action_data.get("success", True),
                match_count=action_data.get("match_count"),
                duration_ms=action_data.get("duration_ms"),
                active_states=action_data.get("active_states", []),
                screenshot_path=action_data.get("screenshot_path"),
                is_start_screenshot=action_data.get("is_start_screenshot", False),
                action_data_json=action_data,
            )
            self.db.add(action)

    def _sync_patterns(self, snapshot_run: SnapshotRun, patterns_dir: Path):
        """Sync pattern data to database.

        Args:
            snapshot_run: The snapshot run instance
            patterns_dir: Path to patterns directory
        """
        # Iterate through pattern directories
        for pattern_dir in patterns_dir.iterdir():
            if not pattern_dir.is_dir():
                continue

            pattern_id = pattern_dir.name

            # Load pattern metadata
            metadata_file = pattern_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                pattern_metadata = json.load(f)

            # Calculate statistics
            total_finds = pattern_metadata.get("total_finds", 0)
            successful_finds = pattern_metadata.get("successful_finds", 0)
            failed_finds = total_finds - successful_finds
            total_matches = pattern_metadata.get("total_matches", 0)

            # Create pattern record
            pattern = SnapshotPattern(
                snapshot_run_id=snapshot_run.id,
                pattern_id=pattern_id,
                pattern_name=pattern_metadata.get("pattern_name", pattern_id),
                total_finds=total_finds,
                successful_finds=successful_finds,
                failed_finds=failed_finds,
                total_matches=total_matches,
                avg_duration_ms=pattern_metadata.get("avg_duration_ms"),
                pattern_data_json=pattern_metadata,
            )
            self.db.add(pattern)
            self.db.flush()  # Get the ID

            # Load and sync match history
            history_file = pattern_dir / "history.json"
            if history_file.exists():
                self._sync_pattern_matches(pattern, history_file)

    def _sync_pattern_matches(self, pattern: SnapshotPattern, history_file: Path):
        """Sync pattern match history to database.

        Args:
            pattern: The snapshot pattern instance
            history_file: Path to history.json
        """
        with open(history_file) as f:
            history = json.load(f)

        for find_record in history:
            matches = find_record.get("matches", [])

            # Get corresponding action ID from action log
            # This requires looking up by timestamp or sequence
            # For now, we'll store without action_id link
            # TODO: Link matches to actions by timestamp correlation

            for match_idx, match_data in enumerate(matches):
                match = SnapshotMatch(
                    pattern_id=pattern.id,
                    action_id=None,  # TODO: Link to action
                    match_index=match_idx,
                    x=match_data.get("x", 0),
                    y=match_data.get("y", 0),
                    width=match_data.get("width", 0),
                    height=match_data.get("height", 0),
                    score=match_data.get("score"),
                    match_data_json=match_data,
                )
                self.db.add(match)

    def delete_snapshot(self, run_id: str, delete_files: bool = False) -> bool:
        """Delete a snapshot from the database.

        Args:
            run_id: The snapshot run ID
            delete_files: If True, also delete the snapshot directory

        Returns:
            True if deleted, False if not found
        """
        snapshot_run = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if not snapshot_run:
            return False

        run_directory = snapshot_run.run_directory

        # Delete from database (CASCADE will handle related records)
        self.db.delete(snapshot_run)
        self.db.commit()

        # Optionally delete files
        if delete_files and run_directory:
            import shutil

            run_path = Path(run_directory)
            if run_path.exists():
                shutil.rmtree(run_path)

        return True

    def get_snapshot(self, run_id: str) -> SnapshotRun | None:
        """Get a snapshot by run ID.

        Args:
            run_id: The snapshot run ID

        Returns:
            The SnapshotRun instance or None if not found
        """
        return self.db.query(SnapshotRun).filter_by(run_id=run_id).first()

    def list_snapshots(
        self,
        limit: int = 50,
        offset: int = 0,
        workflow_id: int | None = None,
        created_by: int | None = None,
        tags: list[str] | None = None,
    ) -> list[SnapshotRun]:
        """List snapshots with filtering and pagination.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            workflow_id: Filter by workflow ID
            created_by: Filter by creator user ID
            tags: Filter by tags (must have all specified tags)

        Returns:
            List of SnapshotRun instances
        """
        query = self.db.query(SnapshotRun)

        if workflow_id is not None:
            query = query.filter_by(workflow_id=workflow_id)

        if created_by is not None:
            query = query.filter_by(created_by=created_by)

        if tags:
            for tag in tags:
                query = query.filter(SnapshotRun.tags.contains([tag]))

        query = query.order_by(SnapshotRun.start_time.desc())
        query = query.limit(limit).offset(offset)

        return query.all()
