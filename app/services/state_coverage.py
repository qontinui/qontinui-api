"""State coverage analysis service for integration testing.

Analyzes coverage metrics across snapshot runs to identify:
- Which states are covered/uncovered
- State transitions (covered/missing)
- Action coverage per state
- Coverage gaps and recommendations
"""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.snapshot import SnapshotAction, SnapshotRun


@dataclass
class StateCoverageMetrics:
    """Metrics for a single state."""

    state_name: str
    screenshot_count: int = 0
    actions_performed: int = 0
    last_tested: datetime | None = None
    coverage_percentage: float = 0.0
    transitions_to: set[str] = field(default_factory=set)
    transitions_from: set[str] = field(default_factory=set)
    action_types: set[str] = field(default_factory=set)
    patterns_tested: set[str] = field(default_factory=set)


@dataclass
class StateTransition:
    """Represents a transition between two states."""

    from_state: str
    to_state: str
    count: int = 0
    covered: bool = False
    last_occurrence: datetime | None = None
    actions_triggering: list[str] = field(default_factory=list)


@dataclass
class CoverageGap:
    """Identifies a gap in test coverage."""

    gap_type: str  # "uncovered_state", "missing_transition", "low_action_coverage"
    severity: str  # "high", "medium", "low"
    description: str
    recommendation: str
    affected_states: list[str]
    metric_value: float | None = None


@dataclass
class CoverageReport:
    """Complete coverage analysis report."""

    process_id: str
    process_name: str
    snapshot_run_ids: list[str]
    analysis_time: datetime
    overall_coverage_percentage: float
    total_states: int
    covered_states: int
    uncovered_states: int
    total_transitions: int
    covered_transitions: int
    missing_transitions: int
    state_metrics: dict[str, StateCoverageMetrics]
    transitions: list[StateTransition]
    coverage_gaps: list[CoverageGap]
    recommendations: list[str]


class StateCoverageAnalyzer:
    """Analyzes state coverage across snapshot runs."""

    def __init__(self, db: Session):
        """Initialize analyzer with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def analyze_coverage(
        self,
        process_id: str,
        process_name: str,
        snapshot_run_ids: list[str],
        expected_states: list[str] | None = None,
    ) -> CoverageReport:
        """Analyze coverage for a process across snapshot runs.

        Args:
            process_id: Process identifier
            process_name: Process name
            snapshot_run_ids: List of snapshot run IDs to analyze
            expected_states: Optional list of states expected in the process

        Returns:
            Complete coverage report

        Raises:
            ValueError: If no snapshot runs found
        """
        # Verify snapshot runs exist
        snapshot_runs = (
            self.db.query(SnapshotRun).filter(SnapshotRun.run_id.in_(snapshot_run_ids)).all()
        )

        if not snapshot_runs:
            raise ValueError(f"No snapshot runs found for IDs: {snapshot_run_ids}")

        snapshot_ids = [run.id for run in snapshot_runs]

        # Gather all states from actions
        discovered_states = self._discover_states(snapshot_ids)

        # Combine with expected states
        all_states = discovered_states.copy()
        if expected_states:
            all_states.update(expected_states)

        # Analyze each state
        state_metrics = self._analyze_states(snapshot_ids, all_states)

        # Analyze transitions
        transitions = self._analyze_transitions(snapshot_ids)

        # Calculate overall metrics
        covered_states = sum(1 for m in state_metrics.values() if m.screenshot_count > 0)
        uncovered_states = len(all_states) - covered_states
        covered_transitions = sum(1 for t in transitions if t.covered)

        # Calculate overall coverage percentage
        if len(all_states) > 0:
            state_coverage = (covered_states / len(all_states)) * 100
        else:
            state_coverage = 0.0

        if len(transitions) > 0:
            transition_coverage = (covered_transitions / len(transitions)) * 100
        else:
            transition_coverage = 100.0  # No transitions expected = 100%

        # Weighted average: 60% state coverage, 40% transition coverage
        overall_coverage = (state_coverage * 0.6) + (transition_coverage * 0.4)

        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps(
            state_metrics,
            transitions,
            expected_states or list(discovered_states),
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(coverage_gaps, state_metrics)

        return CoverageReport(
            process_id=process_id,
            process_name=process_name,
            snapshot_run_ids=snapshot_run_ids,
            analysis_time=datetime.utcnow(),
            overall_coverage_percentage=round(overall_coverage, 2),
            total_states=len(all_states),
            covered_states=covered_states,
            uncovered_states=uncovered_states,
            total_transitions=len(transitions),
            covered_transitions=covered_transitions,
            missing_transitions=len(transitions) - covered_transitions,
            state_metrics=state_metrics,
            transitions=transitions,
            coverage_gaps=coverage_gaps,
            recommendations=recommendations,
        )

    def _discover_states(self, snapshot_ids: list[int]) -> set[str]:
        """Discover all states from snapshot actions.

        Args:
            snapshot_ids: List of snapshot run IDs

        Returns:
            Set of discovered state names
        """
        actions = (
            self.db.query(SnapshotAction)
            .filter(SnapshotAction.snapshot_run_id.in_(snapshot_ids))
            .all()
        )

        states = set()
        for action in actions:
            if action.active_states:
                states.update(action.active_states)

        return states

    def _analyze_states(
        self,
        snapshot_ids: list[int],
        all_states: set[str],
    ) -> dict[str, StateCoverageMetrics]:
        """Analyze coverage metrics for each state.

        Args:
            snapshot_ids: List of snapshot run IDs
            all_states: Set of all states to analyze

        Returns:
            Dictionary mapping state name to metrics
        """
        # Get all actions
        actions = (
            self.db.query(SnapshotAction)
            .filter(SnapshotAction.snapshot_run_id.in_(snapshot_ids))
            .order_by(SnapshotAction.timestamp)
            .all()
        )

        # Initialize metrics for all states
        metrics = {state: StateCoverageMetrics(state_name=state) for state in all_states}

        # Track state occurrences
        for action in actions:
            if not action.active_states:
                continue

            for state in action.active_states:
                if state not in metrics:
                    metrics[state] = StateCoverageMetrics(state_name=state)

                metric = metrics[state]
                metric.screenshot_count += 1
                metric.actions_performed += 1
                metric.action_types.add(action.action_type)

                if action.pattern_id:
                    metric.patterns_tested.add(action.pattern_id)

                if metric.last_tested is None or action.timestamp > metric.last_tested:
                    metric.last_tested = action.timestamp

        # Calculate coverage percentage for each state
        # Coverage = (screenshot_count + actions_performed + patterns_count) / baseline
        for _state, metric in metrics.items():
            # Simple heuristic: state is fully covered if it has:
            # - At least 3 screenshots
            # - At least 2 different action types
            # - At least 1 pattern tested
            coverage_score = 0.0

            if metric.screenshot_count >= 3:
                coverage_score += 40
            elif metric.screenshot_count > 0:
                coverage_score += (metric.screenshot_count / 3) * 40

            if len(metric.action_types) >= 2:
                coverage_score += 30
            elif len(metric.action_types) > 0:
                coverage_score += (len(metric.action_types) / 2) * 30

            if len(metric.patterns_tested) >= 1:
                coverage_score += 30

            metric.coverage_percentage = min(100.0, round(coverage_score, 2))

        return metrics

    def _analyze_transitions(self, snapshot_ids: list[int]) -> list[StateTransition]:
        """Analyze state transitions.

        Args:
            snapshot_ids: List of snapshot run IDs

        Returns:
            List of state transitions with coverage info
        """
        # Get all actions ordered by timestamp
        actions = (
            self.db.query(SnapshotAction)
            .filter(SnapshotAction.snapshot_run_id.in_(snapshot_ids))
            .order_by(SnapshotAction.timestamp)
            .all()
        )

        # Track transitions
        transitions_map: dict[tuple[str, str], StateTransition] = {}
        previous_states: set[str] | None = None

        for action in actions:
            if not action.active_states:
                continue

            current_states = set(action.active_states)

            if previous_states is not None:
                # Detect transitions
                added_states = current_states - previous_states
                removed_states = previous_states - current_states

                # Track state additions (transitions TO new states)
                for to_state in added_states:
                    for from_state in previous_states:
                        key = (from_state, to_state)
                        if key not in transitions_map:
                            transitions_map[key] = StateTransition(
                                from_state=from_state,
                                to_state=to_state,
                                covered=True,
                            )

                        transition = transitions_map[key]
                        transition.count += 1
                        transition.last_occurrence = action.timestamp
                        transition.actions_triggering.append(action.action_type)

                # Track state removals (transitions FROM old states)
                for from_state in removed_states:
                    for to_state in current_states:
                        key = (from_state, to_state)
                        if key not in transitions_map:
                            transitions_map[key] = StateTransition(
                                from_state=from_state,
                                to_state=to_state,
                                covered=True,
                            )

                        transition = transitions_map[key]
                        transition.count += 1
                        transition.last_occurrence = action.timestamp
                        transition.actions_triggering.append(action.action_type)

            previous_states = current_states

        return list(transitions_map.values())

    def _identify_coverage_gaps(
        self,
        state_metrics: dict[str, StateCoverageMetrics],
        transitions: list[StateTransition],
        expected_states: list[str],
    ) -> list[CoverageGap]:
        """Identify coverage gaps and issues.

        Args:
            state_metrics: State coverage metrics
            transitions: State transitions
            expected_states: List of expected states

        Returns:
            List of identified coverage gaps
        """
        gaps = []

        # Identify uncovered states
        for state in expected_states:
            metric = state_metrics.get(state)
            if not metric or metric.screenshot_count == 0:
                gaps.append(
                    CoverageGap(
                        gap_type="uncovered_state",
                        severity="high",
                        description=f"State '{state}' has never been tested",
                        recommendation=f"Add test scenarios that activate state '{state}'",
                        affected_states=[state],
                        metric_value=0.0,
                    )
                )

        # Identify low-coverage states
        for state, metric in state_metrics.items():
            if 0 < metric.coverage_percentage < 50:
                gaps.append(
                    CoverageGap(
                        gap_type="low_coverage_state",
                        severity="medium",
                        description=f"State '{state}' has low coverage ({metric.coverage_percentage}%)",
                        recommendation=f"Increase test scenarios for state '{state}' - add more actions and patterns",
                        affected_states=[state],
                        metric_value=metric.coverage_percentage,
                    )
                )

        # Identify states with limited action coverage
        for state, metric in state_metrics.items():
            if metric.screenshot_count > 0 and len(metric.action_types) < 2:
                gaps.append(
                    CoverageGap(
                        gap_type="low_action_coverage",
                        severity="low",
                        description=f"State '{state}' tested with only {len(metric.action_types)} action type(s)",
                        recommendation=f"Test state '{state}' with different action types (FIND, CLICK, TYPE, etc.)",
                        affected_states=[state],
                        metric_value=float(len(metric.action_types)),
                    )
                )

        # Check for isolated states (no transitions)
        for state, metric in state_metrics.items():
            if metric.screenshot_count > 0:
                has_transition = any(
                    state == t.from_state or state == t.to_state for t in transitions
                )
                if not has_transition:
                    gaps.append(
                        CoverageGap(
                            gap_type="isolated_state",
                            severity="medium",
                            description=f"State '{state}' has no transitions to/from other states",
                            recommendation=f"Add test scenarios showing how to enter and exit state '{state}'",
                            affected_states=[state],
                        )
                    )

        return sorted(gaps, key=lambda g: {"high": 0, "medium": 1, "low": 2}[g.severity])

    def _generate_recommendations(
        self,
        coverage_gaps: list[CoverageGap],
        state_metrics: dict[str, StateCoverageMetrics],
    ) -> list[str]:
        """Generate actionable recommendations based on coverage analysis.

        Args:
            coverage_gaps: Identified coverage gaps
            state_metrics: State coverage metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        # High priority: uncovered states
        uncovered = [g for g in coverage_gaps if g.gap_type == "uncovered_state"]
        if uncovered:
            states = ", ".join([g.affected_states[0] for g in uncovered[:3]])
            recommendations.append(
                f"HIGH PRIORITY: Add test coverage for uncovered states: {states}"
            )

        # Medium priority: low coverage states
        low_coverage = [g for g in coverage_gaps if g.gap_type == "low_coverage_state"]
        if low_coverage:
            states = ", ".join([g.affected_states[0] for g in low_coverage[:3]])
            recommendations.append(f"MEDIUM PRIORITY: Improve coverage for states: {states}")

        # Action diversity
        action_gaps = [g for g in coverage_gaps if g.gap_type == "low_action_coverage"]
        if len(action_gaps) > 3:
            recommendations.append(
                f"Increase action diversity - {len(action_gaps)} states tested with limited action types"
            )

        # Transition coverage
        isolated = [g for g in coverage_gaps if g.gap_type == "isolated_state"]
        if isolated:
            recommendations.append(
                f"Add state transition tests - {len(isolated)} states are isolated"
            )

        # Best practices
        if not recommendations:
            recommendations.append("Coverage looks good! Consider adding edge case scenarios.")

        return recommendations

    def get_cached_report(
        self,
        process_id: str,
        snapshot_run_ids: list[str],
    ) -> CoverageReport | None:
        """Get cached coverage report if available.

        Note: This is a placeholder for future caching implementation.

        Args:
            process_id: Process identifier
            snapshot_run_ids: List of snapshot run IDs

        Returns:
            Cached report if available, None otherwise
        """
        # TODO: Implement caching using Redis or database
        return None

    def cache_report(self, report: CoverageReport) -> None:
        """Cache coverage report for future use.

        Note: This is a placeholder for future caching implementation.

        Args:
            report: Coverage report to cache
        """
        # TODO: Implement caching using Redis or database
        pass
