"""Snapshot analysis service for calculating coverage metrics and similarities."""

import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.snapshot import SnapshotAction, SnapshotPattern, SnapshotRun


class SnapshotAnalysisService:
    """
    Service for analyzing snapshot runs and calculating coverage metrics.

    Provides methods to:
    - Calculate state coverage for each snapshot
    - Calculate action type coverage
    - Calculate state transition coverage
    - Compute similarity between snapshots (Jaccard similarity)
    - Analyze complementarity between snapshot runs
    """

    def __init__(self, db: Session):
        """
        Initialize the analysis service.

        Args:
            db: Database session
        """
        self.db = db

    def calculate_state_coverage(self, run_id: str) -> dict[str, Any]:
        """
        Calculate state coverage for a snapshot run.

        Args:
            run_id: Snapshot run ID

        Returns:
            Dictionary with state coverage metrics:
            - unique_states: Set of unique states observed
            - state_count: Number of unique states
            - state_frequencies: Dict mapping state to occurrence count
            - state_combinations: List of unique state combinations
            - combination_count: Number of unique state combinations
        """
        snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if not snapshot:
            raise ValueError(f"Snapshot run {run_id} not found")

        # Get all actions for this snapshot
        actions = self.db.query(SnapshotAction).filter_by(snapshot_run_id=snapshot.id).all()

        unique_states: set[str] = set()
        state_frequencies: dict[str, int] = defaultdict(int)
        state_combinations: set[tuple[str, ...]] = set()

        for action in actions:
            if action.active_states:
                # Track individual states
                for state in action.active_states:
                    unique_states.add(state)
                    state_frequencies[state] += 1

                # Track state combinations (sorted for consistency)
                combination = tuple(sorted(action.active_states))
                state_combinations.add(combination)

        return {
            "run_id": run_id,
            "unique_states": sorted(unique_states),
            "state_count": len(unique_states),
            "state_frequencies": dict(state_frequencies),
            "state_combinations": [list(c) for c in sorted(state_combinations)],
            "combination_count": len(state_combinations),
        }

    def calculate_action_type_coverage(self, run_id: str) -> dict[str, Any]:
        """
        Calculate action type coverage for a snapshot run.

        Args:
            run_id: Snapshot run ID

        Returns:
            Dictionary with action type metrics:
            - action_types: List of unique action types
            - type_counts: Dict mapping action type to count
            - type_success_rates: Dict mapping action type to success rate
        """
        snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if not snapshot:
            raise ValueError(f"Snapshot run {run_id} not found")

        actions = self.db.query(SnapshotAction).filter_by(snapshot_run_id=snapshot.id).all()

        type_counts: dict[str, int] = defaultdict(int)
        type_successes: dict[str, int] = defaultdict(int)

        for action in actions:
            action_type = action.action_type
            type_counts[action_type] += 1
            if action.success:
                type_successes[action_type] += 1

        type_success_rates = {
            action_type: type_successes[action_type] / count if count > 0 else 0.0
            for action_type, count in type_counts.items()
        }

        return {
            "run_id": run_id,
            "action_types": sorted(type_counts.keys()),
            "type_counts": dict(type_counts),
            "type_success_rates": type_success_rates,
        }

    def calculate_state_transitions(self, run_id: str) -> dict[str, Any]:
        """
        Calculate state transition coverage for a snapshot run.

        Args:
            run_id: Snapshot run ID

        Returns:
            Dictionary with state transition metrics:
            - transitions: List of (from_states, to_states) tuples
            - transition_count: Number of unique transitions
            - transition_frequencies: Dict mapping transition to count
        """
        snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if not snapshot:
            raise ValueError(f"Snapshot run {run_id} not found")

        actions = (
            self.db.query(SnapshotAction)
            .filter_by(snapshot_run_id=snapshot.id)
            .order_by(SnapshotAction.sequence_number)
            .all()
        )

        transitions: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        transition_frequencies: dict[str, int] = defaultdict(int)

        for i in range(len(actions) - 1):
            from_states = tuple(sorted(actions[i].active_states or []))
            to_states = tuple(sorted(actions[i + 1].active_states or []))

            # Only track transitions where states actually change
            if from_states != to_states:
                transitions.append((from_states, to_states))
                transition_key = f"{from_states} -> {to_states}"
                transition_frequencies[transition_key] += 1

        return {
            "run_id": run_id,
            "transitions": [{"from": list(f), "to": list(t)} for f, t in transitions],
            "transition_count": len(set(transitions)),
            "transition_frequencies": dict(transition_frequencies),
        }

    def calculate_jaccard_similarity(
        self, run_id1: str, run_id2: str, metric: str = "states"
    ) -> float:
        """
        Calculate Jaccard similarity between two snapshot runs.

        Jaccard similarity = |A ∩ B| / |A ∪ B|

        Args:
            run_id1: First snapshot run ID
            run_id2: Second snapshot run ID
            metric: Type of similarity to calculate:
                - "states": Compare unique states
                - "actions": Compare action types
                - "patterns": Compare pattern IDs
                - "combinations": Compare state combinations

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if metric == "states":
            set1 = set(self.calculate_state_coverage(run_id1)["unique_states"])
            set2 = set(self.calculate_state_coverage(run_id2)["unique_states"])
        elif metric == "actions":
            set1 = set(self.calculate_action_type_coverage(run_id1)["action_types"])
            set2 = set(self.calculate_action_type_coverage(run_id2)["action_types"])
        elif metric == "patterns":
            set1 = set(self._get_pattern_ids(run_id1))
            set2 = set(self._get_pattern_ids(run_id2))
        elif metric == "combinations":
            cov1 = self.calculate_state_coverage(run_id1)
            cov2 = self.calculate_state_coverage(run_id2)
            set1 = {tuple(sorted(c)) for c in cov1["state_combinations"]}
            set2 = {tuple(sorted(c)) for c in cov2["state_combinations"]}
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def calculate_complementarity(self, run_ids: list[str]) -> dict[str, Any]:
        """
        Analyze how well a set of snapshots complement each other.

        Args:
            run_ids: List of snapshot run IDs to analyze

        Returns:
            Dictionary with complementarity metrics:
            - combined_state_coverage: Total unique states across all runs
            - combined_action_coverage: Total unique action types
            - combined_pattern_coverage: Total unique patterns
            - overlap_matrix: Pairwise similarity matrix
            - diversity_score: Overall diversity score (1 - avg similarity)
            - redundancy_score: Measure of redundancy (avg similarity)
        """
        if not run_ids:
            return {
                "combined_state_coverage": 0,
                "combined_action_coverage": 0,
                "combined_pattern_coverage": 0,
                "overlap_matrix": {},
                "diversity_score": 0.0,
                "redundancy_score": 0.0,
            }

        # Calculate combined coverage
        all_states: set[str] = set()
        all_actions: set[str] = set()
        all_patterns: set[str] = set()

        for run_id in run_ids:
            state_cov = self.calculate_state_coverage(run_id)
            action_cov = self.calculate_action_type_coverage(run_id)

            all_states.update(state_cov["unique_states"])
            all_actions.update(action_cov["action_types"])
            all_patterns.update(self._get_pattern_ids(run_id))

        # Calculate pairwise similarities
        overlap_matrix = {}
        similarities = []

        for i, run_id1 in enumerate(run_ids):
            overlap_matrix[run_id1] = {}
            for j, run_id2 in enumerate(run_ids):
                if i != j:
                    similarity = self.calculate_jaccard_similarity(
                        run_id1, run_id2, metric="states"
                    )
                    overlap_matrix[run_id1][run_id2] = similarity
                    if i < j:  # Only count each pair once
                        similarities.append(similarity)
                else:
                    overlap_matrix[run_id1][run_id2] = 1.0

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        return {
            "run_ids": run_ids,
            "combined_state_coverage": len(all_states),
            "combined_action_coverage": len(all_actions),
            "combined_pattern_coverage": len(all_patterns),
            "overlap_matrix": overlap_matrix,
            "diversity_score": 1.0 - avg_similarity,
            "redundancy_score": avg_similarity,
        }

    def calculate_screenshot_hash(self, screenshot_path: Path) -> str | None:
        """
        Calculate MD5 hash of a screenshot for duplicate detection.

        Args:
            screenshot_path: Path to screenshot file

        Returns:
            MD5 hash string or None if file doesn't exist
        """
        if not screenshot_path.exists():
            return None

        md5_hash = hashlib.md5()
        with open(screenshot_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        return md5_hash.hexdigest()

    def calculate_action_sequence_similarity(self, run_id1: str, run_id2: str) -> float:
        """
        Calculate similarity between action sequences of two snapshots.

        Uses longest common subsequence (LCS) algorithm.

        Args:
            run_id1: First snapshot run ID
            run_id2: Second snapshot run ID

        Returns:
            Similarity score (0.0 to 1.0)
        """
        seq1 = self._get_action_sequence(run_id1)
        seq2 = self._get_action_sequence(run_id2)

        if not seq1 or not seq2:
            return 0.0

        # LCS dynamic programming
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        max_length = max(len(seq1), len(seq2))

        return lcs_length / max_length if max_length > 0 else 0.0

    def get_coverage_report(self, run_ids: list[str]) -> dict[str, Any]:
        """
        Generate comprehensive coverage report for multiple snapshots.

        Args:
            run_ids: List of snapshot run IDs

        Returns:
            Comprehensive coverage report with individual and combined metrics
        """
        individual_reports = []

        for run_id in run_ids:
            snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
            if not snapshot:
                continue

            state_cov = self.calculate_state_coverage(run_id)
            action_cov = self.calculate_action_type_coverage(run_id)
            transition_cov = self.calculate_state_transitions(run_id)

            individual_reports.append(
                {
                    "run_id": run_id,
                    "start_time": snapshot.start_time.isoformat(),
                    "duration_seconds": float(snapshot.duration_seconds or 0),
                    "total_actions": snapshot.total_actions,
                    "state_coverage": state_cov,
                    "action_coverage": action_cov,
                    "transition_coverage": transition_cov,
                    "priority": snapshot.priority,
                }
            )

        # Calculate complementarity
        complementarity = self.calculate_complementarity(run_ids)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "snapshot_count": len(run_ids),
            "individual_reports": individual_reports,
            "complementarity": complementarity,
        }

    def _get_pattern_ids(self, run_id: str) -> list[str]:
        """Get list of pattern IDs used in a snapshot run."""
        snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if not snapshot:
            return []

        patterns = (
            self.db.query(SnapshotPattern.pattern_id)
            .filter_by(snapshot_run_id=snapshot.id)
            .distinct()
            .all()
        )

        return [p.pattern_id for p in patterns]

    def _get_action_sequence(self, run_id: str) -> list[str]:
        """Get ordered sequence of action types for a snapshot run."""
        snapshot = self.db.query(SnapshotRun).filter_by(run_id=run_id).first()
        if not snapshot:
            return []

        actions = (
            self.db.query(SnapshotAction.action_type)
            .filter_by(snapshot_run_id=snapshot.id)
            .order_by(SnapshotAction.sequence_number)
            .all()
        )

        return [a.action_type for a in actions]
