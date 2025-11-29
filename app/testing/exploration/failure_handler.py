"""Failure-aware exploration with intelligent retry logic.

This module provides failure handling, retry mechanisms, and adaptive
strategies for dealing with unstable or flaky transitions.
"""

import logging
import time
from collections import defaultdict
from typing import Any

from app.testing.config import ExplorationConfig
from app.testing.path_tracker import PathTracker

logger = logging.getLogger(__name__)


class FailureAwareExplorer:
    """Handles failures intelligently during exploration.

    Implements retry logic with exponential backoff, failure tracking,
    and adaptive strategies for unstable transitions.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize failure-aware explorer.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance
        """
        self.config = config
        self.tracker = tracker
        self.max_retries = config.failure_max_retries
        self.backoff_base_ms = config.failure_backoff_base_ms
        self.backoff_multiplier = config.failure_backoff_multiplier
        self.skip_threshold = config.failure_skip_threshold
        self.cooldown_iterations = config.failure_cooldown_iterations

        # Failure tracking
        self.consecutive_failures: dict[tuple[str, str], int] = defaultdict(int)
        self.total_failures: dict[tuple[str, str], int] = defaultdict(int)
        self.skipped_transitions: dict[tuple[str, str], int] = defaultdict(int)
        self.last_retry_time: dict[tuple[str, str], float] = {}

    def should_retry_transition(self, from_state: str, to_state: str, attempt: int) -> bool:
        """Determine if a failed transition should be retried.

        Args:
            from_state: Source state
            to_state: Target state
            attempt: Current attempt number (1-based)

        Returns:
            True if transition should be retried
        """
        transition_key = (from_state, to_state)

        # Check if transition is currently skipped
        if transition_key in self.skipped_transitions:
            cooldown_remaining = self.skipped_transitions[transition_key]
            if cooldown_remaining > 0:
                logger.debug(
                    f"Transition {from_state} -> {to_state} is in cooldown "
                    f"({cooldown_remaining} iterations remaining)"
                )
                return False
            else:
                # Cooldown expired, allow retry
                del self.skipped_transitions[transition_key]
                self.consecutive_failures[transition_key] = 0

        # Check max retries
        if attempt > self.max_retries:
            logger.warning(
                f"Max retries ({self.max_retries}) exceeded for " f"{from_state} -> {to_state}"
            )
            return False

        # Check consecutive failure threshold
        consecutive = self.consecutive_failures[transition_key]
        if consecutive >= self.skip_threshold:
            # Skip this transition for cooldown period
            self.skipped_transitions[transition_key] = self.cooldown_iterations
            logger.warning(
                f"Skipping transition {from_state} -> {to_state} after "
                f"{consecutive} consecutive failures (cooldown: {self.cooldown_iterations})"
            )
            return False

        return True

    def calculate_backoff_time(self, from_state: str, to_state: str, attempt: int) -> float:
        """Calculate exponential backoff time for retry.

        Args:
            from_state: Source state
            to_state: Target state
            attempt: Current attempt number (1-based)

        Returns:
            Backoff time in milliseconds
        """
        # Exponential backoff: base * multiplier^(attempt-1)
        backoff_ms = self.backoff_base_ms * (self.backoff_multiplier ** (attempt - 1))

        # Add jitter (±10%) to prevent thundering herd
        import random

        jitter = random.uniform(-0.1, 0.1)
        backoff_ms *= 1 + jitter

        logger.debug(
            f"Backoff for {from_state} -> {to_state} (attempt {attempt}): " f"{backoff_ms:.0f}ms"
        )

        return backoff_ms

    def wait_for_backoff(self, from_state: str, to_state: str, attempt: int) -> None:
        """Wait for exponential backoff period.

        Args:
            from_state: Source state
            to_state: Target state
            attempt: Current attempt number
        """
        backoff_ms = self.calculate_backoff_time(from_state, to_state, attempt)
        transition_key = (from_state, to_state)

        # Record retry time
        self.last_retry_time[transition_key] = time.time()

        # Wait
        time.sleep(backoff_ms / 1000.0)

    def record_failure(self, from_state: str, to_state: str) -> None:
        """Record a transition failure.

        Args:
            from_state: Source state
            to_state: Target state
        """
        transition_key = (from_state, to_state)

        self.consecutive_failures[transition_key] += 1
        self.total_failures[transition_key] += 1

        logger.debug(
            f"Recorded failure: {from_state} -> {to_state} "
            f"(consecutive: {self.consecutive_failures[transition_key]}, "
            f"total: {self.total_failures[transition_key]})"
        )

    def record_success(self, from_state: str, to_state: str) -> None:
        """Record a transition success (resets consecutive failures).

        Args:
            from_state: Source state
            to_state: Target state
        """
        transition_key = (from_state, to_state)

        # Reset consecutive failures on success
        if transition_key in self.consecutive_failures:
            self.consecutive_failures[transition_key] = 0

        logger.debug(f"Recorded success: {from_state} -> {to_state}")

    def update_cooldowns(self) -> None:
        """Update cooldown counters for skipped transitions.

        Should be called once per iteration.
        """
        transitions_to_remove = []

        for transition_key, cooldown in self.skipped_transitions.items():
            self.skipped_transitions[transition_key] = cooldown - 1

            if self.skipped_transitions[transition_key] <= 0:
                transitions_to_remove.append(transition_key)

        # Remove expired cooldowns
        for transition_key in transitions_to_remove:
            del self.skipped_transitions[transition_key]
            from_state, to_state = transition_key
            logger.info(
                f"Cooldown expired for {from_state} -> {to_state}, " "transition available again"
            )

    def get_failure_statistics(self) -> dict[str, Any]:
        """Get failure statistics.

        Returns:
            Dictionary with failure metrics
        """
        total_failed_transitions = len(self.total_failures)
        total_failure_count = sum(self.total_failures.values())

        most_problematic = sorted(self.total_failures.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return {
            "total_failed_transitions": total_failed_transitions,
            "total_failure_count": total_failure_count,
            "skipped_transitions": len(self.skipped_transitions),
            "most_problematic_transitions": [
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "failures": count,
                    "consecutive": self.consecutive_failures.get((from_state, to_state), 0),
                }
                for (from_state, to_state), count in most_problematic
            ],
        }

    def is_transition_skipped(self, from_state: str, to_state: str) -> bool:
        """Check if a transition is currently skipped.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            True if transition is in cooldown
        """
        transition_key = (from_state, to_state)
        return transition_key in self.skipped_transitions

    def get_reliable_alternative(
        self, from_state: str, avoid_transitions: set[tuple[str, str]] | None = None
    ) -> str | None:
        """Get a reliable alternative transition from a state.

        Avoids skipped and unreliable transitions.

        Args:
            from_state: Source state
            avoid_transitions: Additional transitions to avoid

        Returns:
            Reliable target state, or None
        """
        if avoid_transitions is None:
            avoid_transitions = set()

        # Get all available transitions
        if not hasattr(self.tracker.state_graph, "states"):
            return None

        state_obj = self.tracker.state_graph.states.get(from_state)
        if not state_obj or not hasattr(state_obj, "transitions"):
            return None

        # Score transitions by reliability
        candidates = []

        for transition in state_obj.transitions:
            to_state = getattr(transition, "to_state", None)
            if not to_state:
                continue

            transition_key = (from_state, to_state)

            # Skip if in avoid set or currently skipped
            if transition_key in avoid_transitions:
                continue
            if self.is_transition_skipped(from_state, to_state):
                continue

            # Calculate reliability score
            score = self._calculate_reliability_score(from_state, to_state)
            candidates.append((to_state, score))

        if not candidates:
            return None

        # Return most reliable option
        candidates.sort(key=lambda x: x[1], reverse=True)
        return str(candidates[0][0])

    def _calculate_reliability_score(self, from_state: str, to_state: str) -> float:
        """Calculate reliability score for a transition.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            Reliability score (higher is better)
        """
        transition_key = (from_state, to_state)
        score = 1.0

        # Penalize based on total failures
        if transition_key in self.total_failures:
            failures = self.total_failures[transition_key]
            score -= failures * 0.1

        # Penalize based on consecutive failures
        if transition_key in self.consecutive_failures:
            consecutive = self.consecutive_failures[transition_key]
            score -= consecutive * 0.2

        # Bonus for successful transitions from PathTracker
        if transition_key in self.tracker._transition_stats:
            stats = self.tracker._transition_stats[transition_key]
            success_rate = stats.success_rate / 100.0
            score += success_rate

        return max(0.0, score)

    def reset(self) -> None:
        """Reset failure tracking."""
        self.consecutive_failures.clear()
        self.total_failures.clear()
        self.skipped_transitions.clear()
        self.last_retry_time.clear()
        logger.info("Failure tracking reset")

    def export_failure_report(self) -> dict[str, Any]:
        """Export comprehensive failure report.

        Returns:
            Dictionary with detailed failure information
        """
        stats = self.get_failure_statistics()

        # Add detailed transition data
        transition_details = []

        for (from_state, to_state), failures in self.total_failures.items():
            consecutive = self.consecutive_failures.get((from_state, to_state), 0)
            is_skipped = self.is_transition_skipped(from_state, to_state)
            reliability = self._calculate_reliability_score(from_state, to_state)

            transition_details.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "total_failures": failures,
                    "consecutive_failures": consecutive,
                    "is_skipped": is_skipped,
                    "reliability_score": reliability,
                }
            )

        # Sort by total failures
        transition_details.sort(key=lambda x: int(x["total_failures"]) if isinstance(x["total_failures"], int | float) else 0, reverse=True)  # type: ignore[arg-type, return-value]

        return {
            "summary": stats,
            "transition_details": transition_details,
            "configuration": {
                "max_retries": self.max_retries,
                "skip_threshold": self.skip_threshold,
                "cooldown_iterations": self.cooldown_iterations,
                "backoff_base_ms": self.backoff_base_ms,
                "backoff_multiplier": self.backoff_multiplier,
            },
        }
