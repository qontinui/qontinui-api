"""Smart recommendation engine for selecting optimal snapshot combinations."""

from datetime import datetime
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.snapshot import SnapshotRun
from app.services.snapshot_analysis import SnapshotAnalysisService


class RecommendationStrategy:
    """Base class for recommendation strategies."""

    def __init__(self, db: Session, analyzer: SnapshotAnalysisService):
        """
        Initialize the strategy.

        Args:
            db: Database session
            analyzer: Snapshot analysis service
        """
        self.db = db
        self.analyzer = analyzer

    def recommend(
        self, available_runs: list[SnapshotRun], max_snapshots: int, **kwargs
    ) -> tuple[list[str], float, str]:
        """
        Generate recommendation.

        Args:
            available_runs: List of available snapshot runs
            max_snapshots: Maximum number of snapshots to recommend
            **kwargs: Strategy-specific parameters

        Returns:
            Tuple of (recommended_run_ids, score, reason)
        """
        raise NotImplementedError


class MaximumCoverageStrategy(RecommendationStrategy):
    """Strategy to maximize state coverage."""

    def recommend(
        self, available_runs: list[SnapshotRun], max_snapshots: int, **kwargs
    ) -> tuple[list[str], float, str]:
        """
        Select snapshots that maximize combined state coverage.

        Uses greedy algorithm to iteratively select snapshot that adds
        the most new states to the current coverage.

        Returns:
            (run_ids, coverage_score, reason)
        """
        if not available_runs or max_snapshots < 1:
            return [], 0.0, "No snapshots available"

        selected_runs = []
        covered_states: set[str] = set()
        total_possible_states: set[str] = set()

        # Calculate total possible states across all runs
        for run in available_runs:
            coverage = self.analyzer.calculate_state_coverage(run.run_id)
            total_possible_states.update(coverage["unique_states"])

        # Greedy selection: pick run that adds most new states
        for _ in range(max_snapshots):
            best_run = None
            best_new_states = 0

            for run in available_runs:
                if run.run_id in selected_runs:
                    continue

                coverage = self.analyzer.calculate_state_coverage(run.run_id)
                run_states = set(coverage["unique_states"])
                new_states = len(run_states - covered_states)

                if new_states > best_new_states:
                    best_new_states = new_states
                    best_run = run

            if best_run is None:
                break

            selected_runs.append(best_run.run_id)
            coverage = self.analyzer.calculate_state_coverage(best_run.run_id)
            covered_states.update(coverage["unique_states"])

        # Calculate coverage score
        coverage_score = (
            len(covered_states) / len(total_possible_states) if total_possible_states else 0.0
        )

        reason = (
            f"Selected {len(selected_runs)} snapshots covering "
            f"{len(covered_states)} unique states "
            f"({coverage_score:.1%} of total {len(total_possible_states)} states)"
        )

        return selected_runs, coverage_score, reason


class MinimumOverlapStrategy(RecommendationStrategy):
    """Strategy to minimize overlap (maximize diversity)."""

    def recommend(
        self, available_runs: list[SnapshotRun], max_snapshots: int, **kwargs
    ) -> tuple[list[str], float, str]:
        """
        Select most diverse snapshots with minimal overlap.

        Uses greedy algorithm to select snapshots that are most different
        from already selected ones.

        Returns:
            (run_ids, diversity_score, reason)
        """
        if not available_runs or max_snapshots < 1:
            return [], 0.0, "No snapshots available"

        # Start with run that has most states
        selected_runs = []
        coverage_scores = {}

        for run in available_runs:
            coverage = self.analyzer.calculate_state_coverage(run.run_id)
            coverage_scores[run.run_id] = coverage["state_count"]

        # Select first run (most states)
        first_run = max(available_runs, key=lambda r: coverage_scores[r.run_id])
        selected_runs.append(first_run.run_id)

        # Greedy selection: pick run with lowest similarity to selected runs
        for _ in range(max_snapshots - 1):
            best_run = None
            lowest_avg_similarity = float("inf")

            for run in available_runs:
                if run.run_id in selected_runs:
                    continue

                # Calculate average similarity to selected runs
                similarities = []
                for selected_id in selected_runs:
                    similarity = self.analyzer.calculate_jaccard_similarity(
                        run.run_id, selected_id, metric="states"
                    )
                    similarities.append(similarity)

                avg_similarity = sum(similarities) / len(similarities)

                if avg_similarity < lowest_avg_similarity:
                    lowest_avg_similarity = avg_similarity
                    best_run = run

            if best_run is None:
                break

            selected_runs.append(best_run.run_id)

        # Calculate diversity score (1 - avg pairwise similarity)
        if len(selected_runs) > 1:
            similarities = []
            for i, run_id1 in enumerate(selected_runs):
                for run_id2 in selected_runs[i + 1 :]:
                    similarity = self.analyzer.calculate_jaccard_similarity(
                        run_id1, run_id2, metric="states"
                    )
                    similarities.append(similarity)
            avg_similarity = sum(similarities) / len(similarities)
            diversity_score = 1.0 - avg_similarity
        else:
            diversity_score = 1.0

        reason = (
            f"Selected {len(selected_runs)} diverse snapshots with "
            f"{diversity_score:.1%} diversity score (minimal overlap)"
        )

        return selected_runs, diversity_score, reason


class RecentAndDiverseStrategy(RecommendationStrategy):
    """Strategy combining recency with diversity."""

    def recommend(
        self,
        available_runs: list[SnapshotRun],
        max_snapshots: int,
        recency_weight: float = 0.5,
        recency_days: int = 30,
        **kwargs,
    ) -> tuple[list[str], float, str]:
        """
        Select snapshots balancing recency and diversity.

        Args:
            available_runs: Available snapshot runs
            max_snapshots: Maximum snapshots to select
            recency_weight: Weight for recency (0-1), rest is diversity weight
            recency_days: Days to consider for recency decay

        Returns:
            (run_ids, combined_score, reason)
        """
        if not available_runs or max_snapshots < 1:
            return [], 0.0, "No snapshots available"

        diversity_weight = 1.0 - recency_weight
        now = datetime.utcnow()

        # Calculate recency scores
        recency_scores = {}
        for run in available_runs:
            age_days = (now - run.start_time).total_seconds() / (24 * 3600)
            # Exponential decay: score = exp(-age / recency_days)
            import math

            recency_score = math.exp(-age_days / recency_days)
            recency_scores[run.run_id] = recency_score

        selected_runs = []

        # Greedy selection with combined score
        for _ in range(max_snapshots):
            best_run = None
            best_score = -1.0

            for run in available_runs:
                if run.run_id in selected_runs:
                    continue

                # Calculate diversity component
                if selected_runs:
                    similarities = []
                    for selected_id in selected_runs:
                        similarity = self.analyzer.calculate_jaccard_similarity(
                            run.run_id, selected_id, metric="states"
                        )
                        similarities.append(similarity)
                    avg_similarity = sum(similarities) / len(similarities)
                    diversity_score = 1.0 - avg_similarity
                else:
                    diversity_score = 1.0

                # Calculate combined score
                combined_score = (
                    recency_weight * recency_scores[run.run_id] + diversity_weight * diversity_score
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_run = run

            if best_run is None:
                break

            selected_runs.append(best_run.run_id)

        # Calculate final combined score
        if selected_runs:
            avg_recency = sum(recency_scores[r] for r in selected_runs) / len(selected_runs)

            if len(selected_runs) > 1:
                similarities = []
                for i, run_id1 in enumerate(selected_runs):
                    for run_id2 in selected_runs[i + 1 :]:
                        similarity = self.analyzer.calculate_jaccard_similarity(
                            run_id1, run_id2, metric="states"
                        )
                        similarities.append(similarity)
                avg_diversity = 1.0 - (sum(similarities) / len(similarities))
            else:
                avg_diversity = 1.0

            final_score = recency_weight * avg_recency + diversity_weight * avg_diversity
        else:
            final_score = 0.0

        reason = (
            f"Selected {len(selected_runs)} snapshots balancing "
            f"recency ({recency_weight:.0%}) and diversity ({diversity_weight:.0%})"
        )

        return selected_runs, final_score, reason


class PriorityWeightedStrategy(RecommendationStrategy):
    """Strategy using priority weights and coverage."""

    def recommend(
        self,
        available_runs: list[SnapshotRun],
        max_snapshots: int,
        priority_weight: float = 0.3,
        **kwargs,
    ) -> tuple[list[str], float, str]:
        """
        Select snapshots using priority weights and coverage.

        Args:
            available_runs: Available snapshot runs
            max_snapshots: Maximum snapshots to select
            priority_weight: Weight for priority (0-1), rest is coverage weight

        Returns:
            (run_ids, weighted_score, reason)
        """
        if not available_runs or max_snapshots < 1:
            return [], 0.0, "No snapshots available"

        coverage_weight = 1.0 - priority_weight

        # Normalize priority scores (0-1 range)
        priorities = [run.priority for run in available_runs]
        min_priority = min(priorities)
        max_priority = max(priorities)
        priority_range = max_priority - min_priority if max_priority > min_priority else 1

        normalized_priorities = {
            run.run_id: (run.priority - min_priority) / priority_range for run in available_runs
        }

        selected_runs = []
        covered_states: set[str] = set()

        # Calculate total possible states
        total_possible_states: set[str] = set()
        for run in available_runs:
            coverage = self.analyzer.calculate_state_coverage(run.run_id)
            total_possible_states.update(coverage["unique_states"])

        # Greedy selection with weighted score
        for _ in range(max_snapshots):
            best_run = None
            best_score = -1.0

            for run in available_runs:
                if run.run_id in selected_runs:
                    continue

                # Calculate coverage contribution
                coverage = self.analyzer.calculate_state_coverage(run.run_id)
                run_states = set(coverage["unique_states"])
                new_states = len(run_states - covered_states)
                coverage_score = (
                    new_states / len(total_possible_states) if total_possible_states else 0.0
                )

                # Calculate weighted score
                weighted_score = (
                    priority_weight * normalized_priorities[run.run_id]
                    + coverage_weight * coverage_score
                )

                if weighted_score > best_score:
                    best_score = weighted_score
                    best_run = run

            if best_run is None:
                break

            selected_runs.append(best_run.run_id)
            coverage = self.analyzer.calculate_state_coverage(best_run.run_id)
            covered_states.update(coverage["unique_states"])

        # Calculate final score
        final_score = (
            len(covered_states) / len(total_possible_states) if total_possible_states else 0.0
        )

        reason = (
            f"Selected {len(selected_runs)} snapshots using priority ({priority_weight:.0%}) "
            f"and coverage ({coverage_weight:.0%}) weights, "
            f"covering {len(covered_states)} states"
        )

        return selected_runs, final_score, reason


class SnapshotRecommendationService:
    """
    Service for generating smart snapshot recommendations.

    Provides multiple strategies for selecting optimal snapshot combinations:
    - Maximum state coverage
    - Minimum overlap (most diverse)
    - Recent + diverse
    - Priority-weighted selection
    """

    def __init__(self, db: Session):
        """
        Initialize the recommendation service.

        Args:
            db: Database session
        """
        self.db = db
        self.analyzer = SnapshotAnalysisService(db)

        # Initialize strategies
        self.strategies = {
            "max_coverage": MaximumCoverageStrategy(db, self.analyzer),
            "min_overlap": MinimumOverlapStrategy(db, self.analyzer),
            "recent_diverse": RecentAndDiverseStrategy(db, self.analyzer),
            "priority_weighted": PriorityWeightedStrategy(db, self.analyzer),
        }

    def get_recommendations(
        self,
        max_snapshots: int = 3,
        strategy: str | None = None,
        filters: dict[str, Any] | None = None,
        **strategy_params,
    ) -> dict[str, Any]:
        """
        Get smart snapshot recommendations.

        Args:
            max_snapshots: Maximum number of snapshots to recommend
            strategy: Specific strategy to use (None = all strategies)
            filters: Optional filters to apply to available snapshots
            **strategy_params: Strategy-specific parameters

        Returns:
            Dictionary with recommendations:
            - If strategy specified: single recommendation
            - If strategy=None: multiple recommendations from all strategies
        """
        # Get available snapshot runs
        query = self.db.query(SnapshotRun).order_by(desc(SnapshotRun.start_time))

        # Apply filters if provided
        if filters:
            if "workflow_id" in filters:
                query = query.filter_by(workflow_id=filters["workflow_id"])
            if "execution_mode" in filters:
                query = query.filter_by(execution_mode=filters["execution_mode"])
            if "min_actions" in filters:
                query = query.filter(SnapshotRun.total_actions >= filters["min_actions"])
            if "start_date" in filters:
                query = query.filter(SnapshotRun.start_time >= filters["start_date"])

        available_runs = query.all()

        if not available_runs:
            return {
                "available_snapshots": 0,
                "recommendations": [],
                "message": "No snapshots available",
            }

        # Generate recommendations
        recommendations = []

        if strategy and strategy in self.strategies:
            # Single strategy
            run_ids, score, reason = self.strategies[strategy].recommend(
                available_runs, max_snapshots, **strategy_params
            )
            recommendations.append(
                {
                    "strategy": strategy,
                    "recommended_run_ids": run_ids,
                    "score": round(score, 4),
                    "reason": reason,
                }
            )
        else:
            # All strategies
            for strategy_name, strategy_obj in self.strategies.items():
                run_ids, score, reason = strategy_obj.recommend(
                    available_runs, max_snapshots, **strategy_params
                )
                recommendations.append(
                    {
                        "strategy": strategy_name,
                        "recommended_run_ids": run_ids,
                        "score": round(score, 4),
                        "reason": reason,
                    }
                )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "available_snapshots": len(available_runs),
            "max_snapshots": max_snapshots,
            "recommendations": recommendations,
        }

    def get_best_recommendation(
        self, max_snapshots: int = 3, filters: dict[str, Any] | None = None, **strategy_params
    ) -> dict[str, Any]:
        """
        Get the single best recommendation across all strategies.

        Selects the recommendation with the highest score.

        Args:
            max_snapshots: Maximum number of snapshots
            filters: Optional filters
            **strategy_params: Strategy parameters

        Returns:
            Best recommendation with metadata
        """
        all_recommendations = self.get_recommendations(
            max_snapshots=max_snapshots, strategy=None, filters=filters, **strategy_params
        )

        if not all_recommendations["recommendations"]:
            return {
                "recommended_run_ids": [],
                "strategy": None,
                "score": 0.0,
                "reason": "No snapshots available",
            }

        # Find recommendation with highest score
        best = max(all_recommendations["recommendations"], key=lambda r: r["score"])

        return best
