"""Deficiency detection logic for PathTracker."""

import logging
from datetime import datetime
from typing import Any

from app.testing.enums import DeficiencyCategory, DeficiencySeverity, ExecutionStatus
from app.testing.models import Deficiency, TransitionExecution, TransitionStatistics

logger = logging.getLogger(__name__)


class DeficiencyDetector:
    """Detects and categorizes deficiencies from transition executions."""

    def __init__(
        self,
        performance_threshold_ms: float = 5000.0,
        stability_threshold: float = 0.95,
        min_attempts_for_stability: int = 5,
    ):
        """Initialize DeficiencyDetector.

        Args:
            performance_threshold_ms: Threshold for slow transition detection (milliseconds)
            stability_threshold: Success rate threshold for stability (0.95 = 95%)
            min_attempts_for_stability: Minimum attempts before checking stability
        """
        self.performance_threshold_ms = performance_threshold_ms
        self.stability_threshold = stability_threshold
        self.min_attempts_for_stability = min_attempts_for_stability

    def detect_execution_deficiencies(
        self, execution: TransitionExecution
    ) -> list[tuple[DeficiencyCategory, DeficiencySeverity, str, str]]:
        """Detect deficiencies from a single execution.

        Args:
            execution: Transition execution to analyze

        Returns:
            List of tuples: (category, severity, title, description)
        """
        deficiencies = []

        # Check for slow transitions
        if execution.duration_ms > self.performance_threshold_ms:
            deficiencies.append(
                (
                    DeficiencyCategory.SLOW_TRANSITION,
                    DeficiencySeverity.MEDIUM,
                    f"Slow transition: {execution.from_state} -> {execution.to_state}",
                    f"Transition took {execution.duration_ms:.0f}ms "
                    f"(threshold: {self.performance_threshold_ms:.0f}ms)",
                )
            )

        # Check for errors
        if execution.status == ExecutionStatus.ERROR:
            # Determine severity based on error type
            severity = DeficiencySeverity.HIGH
            category = DeficiencyCategory.UNSTABLE_TRANSITION

            if execution.error_message and "timeout" in execution.error_message.lower():
                category = DeficiencyCategory.TIMEOUT
                severity = DeficiencySeverity.HIGH
            elif execution.error_message and "element" in execution.error_message.lower():
                category = DeficiencyCategory.UI_ELEMENT_MISSING
                severity = DeficiencySeverity.HIGH

            deficiencies.append(
                (
                    category,
                    severity,
                    f"Transition error: {execution.from_state} -> {execution.to_state}",
                    execution.error_message or "Unknown error",
                )
            )

        # Check for unexpected state
        if (
            execution.actual_end_state
            and execution.actual_end_state != execution.to_state
            and not execution.is_successful
        ):
            deficiencies.append(
                (
                    DeficiencyCategory.UNEXPECTED_STATE,
                    DeficiencySeverity.MEDIUM,
                    f"Unexpected state: {execution.from_state} -> {execution.actual_end_state}",
                    f"Expected to reach {execution.to_state}, "
                    f"but ended in {execution.actual_end_state}",
                )
            )

        return deficiencies

    def detect_transition_deficiencies(
        self, stats: TransitionStatistics
    ) -> list[tuple[DeficiencyCategory, DeficiencySeverity, str, str]]:
        """Detect deficiencies from transition statistics.

        Args:
            stats: Transition statistics to analyze

        Returns:
            List of tuples: (category, severity, title, description)
        """
        deficiencies = []

        # Check stability only if enough attempts
        if stats.total_attempts >= self.min_attempts_for_stability:
            success_rate = stats.success_rate / 100.0  # Convert to 0-1 range

            if success_rate < self.stability_threshold:
                # Determine severity based on success rate
                if success_rate < 0.50:
                    severity = DeficiencySeverity.CRITICAL
                elif success_rate < 0.70:
                    severity = DeficiencySeverity.HIGH
                else:
                    severity = DeficiencySeverity.MEDIUM

                deficiencies.append(
                    (
                        DeficiencyCategory.UNSTABLE_TRANSITION,
                        severity,
                        f"Unstable transition: {stats.from_state} -> {stats.to_state}",
                        f"Success rate: {stats.success_rate:.1f}% "
                        f"({stats.successes}/{stats.total_attempts} attempts)",
                    )
                )

        # Check for consistently slow transitions
        if (
            stats.total_attempts >= 3
            and stats.avg_duration_ms > self.performance_threshold_ms
        ):
            deficiencies.append(
                (
                    DeficiencyCategory.SLOW_TRANSITION,
                    DeficiencySeverity.MEDIUM,
                    f"Consistently slow transition: {stats.from_state} -> {stats.to_state}",
                    f"Average duration: {stats.avg_duration_ms:.0f}ms "
                    f"(threshold: {self.performance_threshold_ms:.0f}ms)",
                )
            )

        return deficiencies

    def detect_graph_deficiencies(
        self,
        state_graph: Any,
        visited_states: set[str],
        executed_transitions: set[tuple[str, str]],
    ) -> list[tuple[DeficiencyCategory, DeficiencySeverity, str, str, list[str]]]:
        """Detect structural deficiencies in state graph coverage.

        Args:
            state_graph: State graph object
            visited_states: Set of visited state names
            executed_transitions: Set of executed (from_state, to_state) transitions

        Returns:
            List of tuples: (category, severity, title, description, affected_states)
        """
        deficiencies = []

        # Check for unreachable states (if graph has reachability info)
        if hasattr(state_graph, "get_reachable_states"):
            try:
                reachable = state_graph.get_reachable_states()
                all_states = set(state_graph.states.keys()) if hasattr(state_graph, "states") else set()

                for state_name in all_states:
                    if state_name not in reachable and state_name not in visited_states:
                        deficiencies.append(
                            (
                                DeficiencyCategory.UNREACHABLE_STATE,
                                DeficiencySeverity.HIGH,
                                f"Unreachable state: {state_name}",
                                f"State {state_name} cannot be reached from initial state",
                                [state_name],
                            )
                        )
            except Exception as e:
                logger.debug(f"Could not check reachability: {e}")

        # Check for dead-end states (states with no outgoing transitions)
        if hasattr(state_graph, "states"):
            try:
                for state_name, state_obj in state_graph.states.items():
                    if state_name in visited_states:
                        # Get outgoing transitions
                        outgoing = []
                        if hasattr(state_obj, "transitions"):
                            outgoing = state_obj.transitions
                        elif hasattr(state_obj, "get_transitions"):
                            outgoing = state_obj.get_transitions()

                        if not outgoing and state_name not in ["final", "exit", "end"]:
                            deficiencies.append(
                                (
                                    DeficiencyCategory.DEAD_END_STATE,
                                    DeficiencySeverity.MEDIUM,
                                    f"Dead-end state: {state_name}",
                                    f"State {state_name} has no outgoing transitions",
                                    [state_name],
                                )
                            )
            except Exception as e:
                logger.debug(f"Could not check dead-end states: {e}")

        return deficiencies

    def create_deficiency(
        self,
        category: DeficiencyCategory,
        severity: DeficiencySeverity,
        title: str,
        description: str,
        affected_transitions: list[tuple[str, str]],
        execution_id: str,
        timestamp: datetime,
        screenshot_path: str | None = None,
        affected_states: list[str] | None = None,
    ) -> Deficiency:
        """Create a Deficiency object.

        Args:
            category: Deficiency category
            severity: Severity level
            title: Short title
            description: Detailed description
            affected_transitions: List of affected transitions
            execution_id: Execution ID
            timestamp: When observed
            screenshot_path: Optional screenshot
            affected_states: Optional list of affected states

        Returns:
            Deficiency object
        """
        import uuid

        return Deficiency(
            deficiency_id=str(uuid.uuid4()),
            category=category,
            severity=severity,
            title=title,
            description=description,
            affected_states=affected_states or [],
            affected_transitions=affected_transitions,
            first_observed=timestamp,
            last_observed=timestamp,
            execution_ids=[execution_id],
            evidence=[screenshot_path] if screenshot_path else [],
        )

    def get_deficiency_key(self, category: DeficiencyCategory, title: str) -> str:
        """Generate a unique key for deficiency deduplication.

        Args:
            category: Deficiency category
            title: Deficiency title

        Returns:
            Unique key string
        """
        return f"{category.value}:{title}"
