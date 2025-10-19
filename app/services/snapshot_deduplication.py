"""Duplicate detection service for identifying redundant snapshot runs."""

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.snapshot import SnapshotRun
from app.services.snapshot_analysis import SnapshotAnalysisService


class DuplicateGroup:
    """Represents a group of duplicate snapshot runs."""

    def __init__(self, representative_run_id: str, similarity_score: float):
        """
        Initialize a duplicate group.

        Args:
            representative_run_id: The "best" run in this group
            similarity_score: Average similarity score within group
        """
        self.representative_run_id = representative_run_id
        self.duplicate_run_ids: list[str] = []
        self.similarity_score = similarity_score
        self.duplicate_reasons: list[str] = []

    def add_duplicate(self, run_id: str, reason: str):
        """Add a duplicate run to this group."""
        self.duplicate_run_ids.append(run_id)
        self.duplicate_reasons.append(reason)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "representative_run_id": self.representative_run_id,
            "duplicate_run_ids": self.duplicate_run_ids,
            "duplicate_count": len(self.duplicate_run_ids),
            "similarity_score": round(self.similarity_score, 4),
            "duplicate_reasons": self.duplicate_reasons,
        }


class SnapshotDeduplicationService:
    """
    Service for detecting duplicate snapshot runs.

    Identifies duplicates based on:
    - State coverage similarity (>95% default)
    - Screenshot hash similarity
    - Action sequence similarity
    """

    def __init__(
        self,
        db: Session,
        state_similarity_threshold: float = 0.95,
        action_similarity_threshold: float = 0.90,
        screenshot_similarity_threshold: float = 0.85,
    ):
        """
        Initialize the deduplication service.

        Args:
            db: Database session
            state_similarity_threshold: Threshold for state coverage similarity
            action_similarity_threshold: Threshold for action sequence similarity
            screenshot_similarity_threshold: Threshold for screenshot hash similarity
        """
        self.db = db
        self.state_threshold = state_similarity_threshold
        self.action_threshold = action_similarity_threshold
        self.screenshot_threshold = screenshot_similarity_threshold
        self.analyzer = SnapshotAnalysisService(db)

    def detect_duplicates(
        self,
        run_ids: list[str] | None = None,
        workflow_id: int | None = None,
    ) -> list[DuplicateGroup]:
        """
        Detect duplicate snapshot runs.

        Args:
            run_ids: Specific run IDs to check (None = all runs)
            workflow_id: Filter by workflow ID

        Returns:
            List of duplicate groups
        """
        # Get snapshot runs to analyze
        query = self.db.query(SnapshotRun)

        if run_ids:
            query = query.filter(SnapshotRun.run_id.in_(run_ids))
        if workflow_id:
            query = query.filter_by(workflow_id=workflow_id)

        snapshots = query.all()

        if len(snapshots) < 2:
            return []

        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(snapshots)

        # Find duplicate groups
        duplicate_groups = self._find_duplicate_groups(snapshots, similarity_matrix)

        return duplicate_groups

    def mark_as_duplicate(
        self,
        run_id: str,
        is_duplicate: bool = True,
        duplicate_of: str | None = None,
    ):
        """
        Mark a snapshot run as duplicate in the database.

        Args:
            run_id: Snapshot run ID to mark
            is_duplicate: Whether this is a duplicate
            duplicate_of: The run this is a duplicate of
        """
        snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if not snapshot:
            raise ValueError(f"Snapshot run {run_id} not found")

        # Add duplicate info to metadata
        if snapshot.metadata_json is None:
            snapshot.metadata_json = {}

        snapshot.metadata_json["is_duplicate"] = is_duplicate
        if duplicate_of:
            snapshot.metadata_json["duplicate_of"] = duplicate_of

        self.db.commit()

    def get_duplicate_groups(
        self,
        workflow_id: int | None = None,
        include_unmarked: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get all duplicate groups.

        Args:
            workflow_id: Filter by workflow
            include_unmarked: Include runs not marked as duplicates

        Returns:
            List of duplicate group dictionaries
        """
        # Detect duplicates
        duplicate_groups = self.detect_duplicates(workflow_id=workflow_id)

        # Convert to dictionaries
        results = []
        for group in duplicate_groups:
            group_dict = group.to_dict()

            # Add metadata for each run
            all_run_ids = [group.representative_run_id] + group.duplicate_run_ids
            runs_info = []

            for run_id in all_run_ids:
                snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
                if snapshot:
                    is_marked = snapshot.metadata_json and snapshot.metadata_json.get(
                        "is_duplicate", False
                    )

                    if include_unmarked or is_marked:
                        runs_info.append(
                            {
                                "run_id": run_id,
                                "start_time": snapshot.start_time.isoformat(),
                                "total_actions": snapshot.total_actions,
                                "duration_seconds": float(snapshot.duration_seconds or 0),
                                "is_marked_duplicate": is_marked,
                                "priority": snapshot.priority,
                            }
                        )

            group_dict["runs"] = runs_info
            results.append(group_dict)

        return results

    def get_recommended_cleanup(
        self,
        workflow_id: int | None = None,
        keep_strategy: str = "recent",
    ) -> dict[str, Any]:
        """
        Get recommended snapshots to keep/delete based on duplicates.

        Args:
            workflow_id: Filter by workflow
            keep_strategy: Strategy for choosing which duplicate to keep:
                - "recent": Keep most recent
                - "priority": Keep highest priority
                - "complete": Keep most complete (most actions)

        Returns:
            Dictionary with keep/delete recommendations
        """
        duplicate_groups = self.detect_duplicates(workflow_id=workflow_id)

        keep_runs = []
        delete_runs = []

        for group in duplicate_groups:
            all_run_ids = [group.representative_run_id] + group.duplicate_run_ids
            snapshots = self.db.query(SnapshotRun).filter(SnapshotRun.run_id.in_(all_run_ids)).all()

            # Choose which to keep based on strategy
            if keep_strategy == "recent":
                keeper = max(snapshots, key=lambda s: s.start_time)
            elif keep_strategy == "priority":
                keeper = max(snapshots, key=lambda s: s.priority)
            elif keep_strategy == "complete":
                keeper = max(snapshots, key=lambda s: s.total_actions)
            else:
                keeper = snapshots[0]

            keep_runs.append(
                {
                    "run_id": keeper.run_id,
                    "start_time": keeper.start_time.isoformat(),
                    "reason": f"Selected by {keep_strategy} strategy",
                }
            )

            # Mark others for deletion
            for snapshot in snapshots:
                if snapshot.run_id != keeper.run_id:
                    delete_runs.append(
                        {
                            "run_id": snapshot.run_id,
                            "start_time": snapshot.start_time.isoformat(),
                            "duplicate_of": keeper.run_id,
                            "reason": f"Duplicate of {keeper.run_id}",
                        }
                    )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": keep_strategy,
            "duplicate_groups": len(duplicate_groups),
            "total_duplicates": len(delete_runs),
            "space_saving_estimate": self._estimate_space_savings(delete_runs),
            "keep": keep_runs,
            "delete": delete_runs,
        }

    def _build_similarity_matrix(
        self, snapshots: list[SnapshotRun]
    ) -> dict[tuple[str, str], dict[str, float]]:
        """
        Build similarity matrix for all snapshot pairs.

        Returns:
            Dictionary mapping (run_id1, run_id2) to similarity metrics
        """
        similarity_matrix = {}

        for i, snap1 in enumerate(snapshots):
            for snap2 in snapshots[i + 1 :]:
                # Calculate state coverage similarity
                state_sim = self.analyzer.calculate_jaccard_similarity(
                    snap1.run_id, snap2.run_id, metric="states"
                )

                # Calculate action sequence similarity
                action_sim = self.analyzer.calculate_action_sequence_similarity(
                    snap1.run_id, snap2.run_id
                )

                # Calculate screenshot similarity (based on action count/types)
                screenshot_sim = self._calculate_screenshot_similarity(snap1, snap2)

                # Store similarities
                key = (snap1.run_id, snap2.run_id)
                similarity_matrix[key] = {
                    "state": state_sim,
                    "action": action_sim,
                    "screenshot": screenshot_sim,
                    "average": (state_sim + action_sim + screenshot_sim) / 3,
                }

        return similarity_matrix

    def _calculate_screenshot_similarity(self, snap1: SnapshotRun, snap2: SnapshotRun) -> float:
        """
        Calculate screenshot similarity between two runs.

        Uses screenshot counts and timing as proxy for full hash comparison.
        """
        # Compare screenshot counts
        if snap1.total_screenshots == 0 and snap2.total_screenshots == 0:
            return 1.0
        if snap1.total_screenshots == 0 or snap2.total_screenshots == 0:
            return 0.0

        count_similarity = min(snap1.total_screenshots, snap2.total_screenshots) / max(
            snap1.total_screenshots, snap2.total_screenshots
        )

        # Compare action counts (proxy for screenshot content)
        if snap1.total_actions == 0 and snap2.total_actions == 0:
            action_similarity = 1.0
        elif snap1.total_actions == 0 or snap2.total_actions == 0:
            action_similarity = 0.0
        else:
            action_similarity = min(snap1.total_actions, snap2.total_actions) / max(
                snap1.total_actions, snap2.total_actions
            )

        return (count_similarity + action_similarity) / 2

    def _find_duplicate_groups(
        self,
        snapshots: list[SnapshotRun],
        similarity_matrix: dict[tuple[str, str], dict[str, float]],
    ) -> list[DuplicateGroup]:
        """
        Find groups of duplicate snapshots from similarity matrix.

        Uses greedy clustering based on similarity thresholds.
        """
        # Track which runs are already in a group
        assigned: set[str] = set()
        duplicate_groups: list[DuplicateGroup] = []

        # Sort runs by start time (newer first)
        sorted_snapshots = sorted(snapshots, key=lambda s: s.start_time, reverse=True)

        for snap in sorted_snapshots:
            if snap.run_id in assigned:
                continue

            # Find all duplicates of this snapshot
            duplicates = []

            for other_snap in snapshots:
                if other_snap.run_id == snap.run_id:
                    continue
                if other_snap.run_id in assigned:
                    continue

                # Get similarity metrics
                key = (
                    (snap.run_id, other_snap.run_id)
                    if (snap.run_id, other_snap.run_id) in similarity_matrix
                    else (other_snap.run_id, snap.run_id)
                )

                if key not in similarity_matrix:
                    continue

                metrics = similarity_matrix[key]

                # Check if duplicate based on thresholds
                reasons = []
                is_duplicate = False

                if metrics["state"] >= self.state_threshold:
                    reasons.append(f"State coverage {metrics['state']:.1%} similar")
                    is_duplicate = True

                if metrics["action"] >= self.action_threshold:
                    reasons.append(f"Action sequence {metrics['action']:.1%} similar")
                    is_duplicate = True

                if metrics["screenshot"] >= self.screenshot_threshold:
                    reasons.append(f"Screenshots {metrics['screenshot']:.1%} similar")

                if is_duplicate:
                    duplicates.append(
                        {
                            "run_id": other_snap.run_id,
                            "similarity": metrics["average"],
                            "reasons": reasons,
                        }
                    )

            # Create group if duplicates found
            if duplicates:
                avg_similarity = sum(d["similarity"] for d in duplicates) / len(duplicates)
                group = DuplicateGroup(snap.run_id, avg_similarity)

                for dup in duplicates:
                    group.add_duplicate(dup["run_id"], "; ".join(dup["reasons"]))
                    assigned.add(dup["run_id"])

                assigned.add(snap.run_id)
                duplicate_groups.append(group)

        return duplicate_groups

    def _estimate_space_savings(self, delete_runs: list[dict[str, Any]]) -> str:
        """
        Estimate disk space savings from deleting duplicate runs.

        Returns:
            Human-readable space estimate
        """
        total_bytes = 0

        for run_info in delete_runs:
            snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_info["run_id"]).first()
            if snapshot:
                # Estimate based on screenshot count (assume ~100KB per screenshot)
                estimated_bytes = snapshot.total_screenshots * 100 * 1024

                # Add action log and pattern data (~1KB per action)
                estimated_bytes += snapshot.total_actions * 1024

                total_bytes += estimated_bytes

        # Convert to human-readable format
        if total_bytes < 1024:
            return f"{total_bytes} bytes"
        elif total_bytes < 1024 * 1024:
            return f"{total_bytes / 1024:.1f} KB"
        elif total_bytes < 1024 * 1024 * 1024:
            return f"{total_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{total_bytes / (1024 * 1024 * 1024):.1f} GB"
