"""Data models for PathTracker."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.testing.enums import DeficiencyCategory, DeficiencySeverity, ExecutionStatus


@dataclass
class TransitionExecution:
    """Record of a single transition execution attempt.

    This class represents one attempt to execute a transition between states.
    Multiple executions may exist for the same transition (from_state, to_state).
    """

    execution_id: str
    """Unique identifier for this execution (UUID)."""

    from_state: str
    """Source state name."""

    to_state: str
    """Target state name."""

    status: ExecutionStatus
    """Execution result status."""

    timestamp: datetime
    """When the execution occurred."""

    duration_ms: float
    """Execution duration in milliseconds."""

    attempt_number: int = 1
    """Attempt number for this transition (1-based)."""

    error_message: str | None = None
    """Error message if status is ERROR."""

    actual_end_state: str | None = None
    """Actual state reached (may differ from to_state if failed)."""

    screenshot_path: str | None = None
    """Path to screenshot captured during execution."""

    variables_snapshot: dict[str, Any] | None = None
    """Snapshot of execution variables at time of transition."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (action details, element coordinates, etc.)."""

    @property
    def is_successful(self) -> bool:
        """Check if execution was successful.

        Returns:
            True if status is SUCCESS
        """
        return self.status == ExecutionStatus.SUCCESS

    @property
    def transition_key(self) -> str:
        """Get unique key for this transition.

        Returns:
            String in format "from_state -> to_state"
        """
        return f"{self.from_state} -> {self.to_state}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "execution_id": self.execution_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "attempt_number": self.attempt_number,
            "error_message": self.error_message,
            "actual_end_state": self.actual_end_state,
            "screenshot_path": self.screenshot_path,
            "variables_snapshot": self.variables_snapshot,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionExecution":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            TransitionExecution instance
        """
        return cls(
            execution_id=data["execution_id"],
            from_state=data["from_state"],
            to_state=data["to_state"],
            status=ExecutionStatus(data["status"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_ms=data["duration_ms"],
            attempt_number=data.get("attempt_number", 1),
            error_message=data.get("error_message"),
            actual_end_state=data.get("actual_end_state"),
            screenshot_path=data.get("screenshot_path"),
            variables_snapshot=data.get("variables_snapshot"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CoverageMetrics:
    """Coverage metrics for state graph exploration.

    Provides comprehensive metrics about which states and transitions
    have been visited during test execution.
    """

    # State Coverage
    total_states: int
    """Total number of states in graph."""

    visited_states: int
    """Number of states visited at least once."""

    unvisited_states: list[str]
    """List of state names never visited."""

    # Transition Coverage
    total_transitions: int
    """Total number of transitions in graph."""

    executed_transitions: int
    """Number of transitions executed at least once."""

    unexecuted_transitions: list[tuple[str, str]]
    """List of (from_state, to_state) pairs never executed."""

    # Success Metrics
    total_executions: int
    """Total number of transition execution attempts."""

    successful_executions: int
    """Number of successful executions."""

    failed_executions: int
    """Number of failed executions."""

    error_executions: int
    """Number of executions with errors."""

    # Path Metrics
    unique_paths: int
    """Number of unique paths discovered."""

    longest_path_length: int
    """Length of longest path traversed."""

    average_path_length: float
    """Average path length."""

    # Time Metrics
    total_execution_time_ms: float
    """Total time spent executing transitions."""

    average_transition_time_ms: float
    """Average transition execution time."""

    # Timestamp
    calculated_at: datetime
    """When these metrics were calculated."""

    @property
    def state_coverage_percent(self) -> float:
        """Calculate state coverage percentage.

        Returns:
            Percentage of states visited (0-100)
        """
        if self.total_states == 0:
            return 0.0
        return (self.visited_states / self.total_states) * 100

    @property
    def transition_coverage_percent(self) -> float:
        """Calculate transition coverage percentage.

        Returns:
            Percentage of transitions executed (0-100)
        """
        if self.total_transitions == 0:
            return 0.0
        return (self.executed_transitions / self.total_transitions) * 100

    @property
    def success_rate_percent(self) -> float:
        """Calculate overall success rate.

        Returns:
            Percentage of successful executions (0-100)
        """
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100

    @property
    def is_complete_coverage(self) -> bool:
        """Check if complete coverage achieved.

        Returns:
            True if 100% state and transition coverage
        """
        return self.state_coverage_percent == 100.0 and self.transition_coverage_percent == 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "total_states": self.total_states,
            "visited_states": self.visited_states,
            "unvisited_states": self.unvisited_states,
            "total_transitions": self.total_transitions,
            "executed_transitions": self.executed_transitions,
            "unexecuted_transitions": self.unexecuted_transitions,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "error_executions": self.error_executions,
            "unique_paths": self.unique_paths,
            "longest_path_length": self.longest_path_length,
            "average_path_length": self.average_path_length,
            "total_execution_time_ms": self.total_execution_time_ms,
            "average_transition_time_ms": self.average_transition_time_ms,
            "calculated_at": self.calculated_at.isoformat(),
            "state_coverage_percent": self.state_coverage_percent,
            "transition_coverage_percent": self.transition_coverage_percent,
            "success_rate_percent": self.success_rate_percent,
        }


@dataclass
class Deficiency:
    """Represents a deficiency or issue found during testing.

    A deficiency is any deviation from expected behavior, including
    bugs, performance issues, missing features, or unstable behavior.
    """

    deficiency_id: str
    """Unique identifier for this deficiency (UUID)."""

    category: DeficiencyCategory
    """Category of deficiency."""

    severity: DeficiencySeverity
    """Severity level."""

    title: str
    """Short, descriptive title."""

    description: str
    """Detailed description of the issue."""

    affected_states: list[str]
    """States affected by this deficiency."""

    affected_transitions: list[tuple[str, str]]
    """Transitions affected by this deficiency."""

    first_observed: datetime
    """When deficiency was first detected."""

    last_observed: datetime
    """When deficiency was last detected."""

    occurrence_count: int = 1
    """Number of times this deficiency occurred."""

    evidence: list[str] = field(default_factory=list)
    """List of screenshot paths or evidence references."""

    execution_ids: list[str] = field(default_factory=list)
    """List of execution IDs that exhibit this deficiency."""

    suggested_fix: str | None = None
    """Suggested fix or workaround."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (error traces, timing data, etc.)."""

    @property
    def is_critical(self) -> bool:
        """Check if deficiency is critical severity.

        Returns:
            True if severity is CRITICAL
        """
        return self.severity == DeficiencySeverity.CRITICAL

    def add_occurrence(
        self, execution_id: str, timestamp: datetime, screenshot_path: str | None = None
    ) -> None:
        """Record another occurrence of this deficiency.

        Args:
            execution_id: ID of execution exhibiting this deficiency
            timestamp: When the occurrence was observed
            screenshot_path: Optional screenshot evidence
        """
        self.occurrence_count += 1
        self.last_observed = timestamp
        self.execution_ids.append(execution_id)
        if screenshot_path:
            self.evidence.append(screenshot_path)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "deficiency_id": self.deficiency_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_states": self.affected_states,
            "affected_transitions": self.affected_transitions,
            "first_observed": self.first_observed.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "occurrence_count": self.occurrence_count,
            "evidence": self.evidence,
            "execution_ids": self.execution_ids,
            "suggested_fix": self.suggested_fix,
            "metadata": self.metadata,
        }


@dataclass
class PathHistory:
    """Complete history of state transitions during execution.

    Represents a sequential path through the state graph, tracking
    the exact order of states visited and transitions executed.
    """

    path_id: str
    """Unique identifier for this path (UUID)."""

    start_state: str
    """Initial state of the path."""

    end_state: str
    """Final state of the path."""

    states: list[str]
    """Ordered list of states visited."""

    transitions: list[str]
    """Ordered list of transition execution IDs."""

    start_time: datetime
    """When path traversal started."""

    end_time: datetime | None = None
    """When path traversal ended (None if in progress)."""

    total_duration_ms: float = 0.0
    """Total duration of path traversal."""

    success: bool = True
    """Whether path completed successfully."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the path."""

    @property
    def length(self) -> int:
        """Get length of path (number of transitions).

        Returns:
            Number of transitions in path
        """
        return len(self.transitions)

    @property
    def is_complete(self) -> bool:
        """Check if path traversal is complete.

        Returns:
            True if end_time is set
        """
        return self.end_time is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "path_id": self.path_id,
            "start_state": self.start_state,
            "end_state": self.end_state,
            "states": self.states,
            "transitions": self.transitions,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class TransitionStatistics:
    """Statistical data for a specific transition.

    Aggregates data across all executions of a transition
    to identify patterns, performance issues, and reliability.
    """

    from_state: str
    """Source state."""

    to_state: str
    """Target state."""

    total_attempts: int = 0
    """Total number of execution attempts."""

    successes: int = 0
    """Number of successful executions."""

    failures: int = 0
    """Number of failed executions."""

    errors: int = 0
    """Number of executions with errors."""

    min_duration_ms: float = float("inf")
    """Minimum execution duration."""

    max_duration_ms: float = 0.0
    """Maximum execution duration."""

    avg_duration_ms: float = 0.0
    """Average execution duration."""

    std_dev_duration_ms: float = 0.0
    """Standard deviation of duration."""

    first_executed: datetime | None = None
    """When first executed."""

    last_executed: datetime | None = None
    """When last executed."""

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage.

        Returns:
            Success rate (0-100)
        """
        if self.total_attempts == 0:
            return 0.0
        return (self.successes / self.total_attempts) * 100

    @property
    def is_stable(self) -> bool:
        """Check if transition is stable (>95% success rate).

        Returns:
            True if success rate >= 95%
        """
        return self.success_rate >= 95.0

    @property
    def is_unreliable(self) -> bool:
        """Check if transition is unreliable (<80% success rate).

        Returns:
            True if success rate < 80%
        """
        return self.success_rate < 80.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "total_attempts": self.total_attempts,
            "successes": self.successes,
            "failures": self.failures,
            "errors": self.errors,
            "min_duration_ms": (
                self.min_duration_ms if self.min_duration_ms != float("inf") else None
            ),
            "max_duration_ms": self.max_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "std_dev_duration_ms": self.std_dev_duration_ms,
            "first_executed": self.first_executed.isoformat() if self.first_executed else None,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "success_rate": self.success_rate,
        }
