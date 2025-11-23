"""Enumerations for PathTracker."""

from enum import Enum


class ExecutionStatus(Enum):
    """Status of a transition execution attempt."""

    SUCCESS = "success"
    """Transition completed successfully."""

    FAILURE = "failure"
    """Transition failed (expected target state not reached)."""

    ERROR = "error"
    """Transition encountered an error (exception, timeout, etc.)."""

    SKIPPED = "skipped"
    """Transition was skipped (conditional branch not taken)."""

    PARTIAL = "partial"
    """Transition partially succeeded (intermediate state reached)."""


class DeficiencySeverity(Enum):
    """Severity level for deficiencies."""

    CRITICAL = "critical"
    """Critical issue blocking core functionality."""

    HIGH = "high"
    """High severity issue affecting important features."""

    MEDIUM = "medium"
    """Medium severity issue with workarounds available."""

    LOW = "low"
    """Low severity issue or minor inconvenience."""

    INFO = "info"
    """Informational finding, not necessarily a defect."""


class DeficiencyCategory(Enum):
    """Category of deficiency."""

    UNREACHABLE_STATE = "unreachable_state"
    """State that cannot be reached from initial state."""

    DEAD_END_STATE = "dead_end_state"
    """State with no outgoing transitions."""

    UNSTABLE_TRANSITION = "unstable_transition"
    """Transition with inconsistent success rate."""

    SLOW_TRANSITION = "slow_transition"
    """Transition that exceeds performance threshold."""

    MISSING_TRANSITION = "missing_transition"
    """Expected transition not found in state graph."""

    STATE_DETECTION_FAILURE = "state_detection_failure"
    """State detection failed consistently."""

    TIMEOUT = "timeout"
    """Operation exceeded timeout threshold."""

    UNEXPECTED_STATE = "unexpected_state"
    """Reached a state not defined in state graph."""

    UI_ELEMENT_MISSING = "ui_element_missing"
    """Expected UI element not found in state."""
