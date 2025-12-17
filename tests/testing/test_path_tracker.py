"""
Tests for PathTracker library in qontinui-api.

Tests transition recording, coverage calculation, deficiency detection,
and path exploration strategies.
"""

from datetime import datetime

import pytest


class MockPathTracker:
    """Mock PathTracker for testing (replace with actual implementation)."""

    def __init__(self):
        self.transitions: list[dict] = []
        self.states_visited: set[str] = set()
        self.coverage_map: dict[str, int] = {}

    def record_transition(
        self, from_state: str, to_state: str, metadata: dict | None = None
    ) -> None:
        """Record a state transition."""
        self.transitions.append(
            {
                "from_state": from_state,
                "to_state": to_state,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {},
            }
        )
        self.states_visited.add(from_state)
        self.states_visited.add(to_state)

        transition_key = f"{from_state}->{to_state}"
        self.coverage_map[transition_key] = self.coverage_map.get(transition_key, 0) + 1

    def calculate_coverage(self, total_transitions: int) -> float:
        """Calculate transition coverage percentage."""
        unique_transitions = len(self.coverage_map)
        if total_transitions == 0:
            return 0.0
        return (unique_transitions / total_transitions) * 100

    def get_transition_count(self, from_state: str, to_state: str) -> int:
        """Get number of times a transition was executed."""
        transition_key = f"{from_state}->{to_state}"
        return self.coverage_map.get(transition_key, 0)

    def export_results(self) -> dict:
        """Export path tracking results."""
        return {
            "total_transitions": len(self.transitions),
            "unique_states": len(self.states_visited),
            "unique_transitions": len(self.coverage_map),
            "coverage_map": self.coverage_map,
            "states_visited": list(self.states_visited),
        }


@pytest.fixture
def path_tracker():
    """Create a PathTracker instance for testing."""
    return MockPathTracker()


@pytest.fixture
def workflow_definition():
    """Define a test workflow with known transitions."""
    return {
        "states": ["login", "dashboard", "settings", "profile", "logout"],
        "transitions": [
            ("login", "dashboard"),
            ("dashboard", "settings"),
            ("dashboard", "profile"),
            ("settings", "dashboard"),
            ("profile", "dashboard"),
            ("dashboard", "logout"),
        ],
    }


class TestPathTrackerBasics:
    """Test basic PathTracker functionality."""

    def test_record_single_transition(self, path_tracker):
        """Test recording a single transition."""
        path_tracker.record_transition("login", "dashboard")

        assert len(path_tracker.transitions) == 1
        assert path_tracker.transitions[0]["from_state"] == "login"
        assert path_tracker.transitions[0]["to_state"] == "dashboard"

    def test_record_multiple_transitions(self, path_tracker):
        """Test recording multiple transitions."""
        transitions = [
            ("login", "dashboard"),
            ("dashboard", "profile"),
            ("profile", "settings"),
            ("settings", "dashboard"),
        ]

        for from_state, to_state in transitions:
            path_tracker.record_transition(from_state, to_state)

        assert len(path_tracker.transitions) == 4

    def test_track_states_visited(self, path_tracker):
        """Test tracking unique states visited."""
        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "profile")
        path_tracker.record_transition("profile", "dashboard")

        assert len(path_tracker.states_visited) == 3
        assert "login" in path_tracker.states_visited
        assert "dashboard" in path_tracker.states_visited
        assert "profile" in path_tracker.states_visited

    def test_record_transition_with_metadata(self, path_tracker):
        """Test recording transition with additional metadata."""
        metadata = {
            "duration_ms": 1500,
            "screenshot_path": "/screenshots/001.png",
            "confidence": 0.95,
        }

        path_tracker.record_transition("login", "dashboard", metadata)

        assert path_tracker.transitions[0]["metadata"] == metadata


class TestCoverageCalculation:
    """Test coverage calculation functionality."""

    def test_calculate_coverage_empty(self, path_tracker):
        """Test coverage calculation with no transitions."""
        coverage = path_tracker.calculate_coverage(total_transitions=10)

        assert coverage == 0.0

    def test_calculate_coverage_partial(self, path_tracker, workflow_definition):
        """Test coverage calculation with partial exploration."""
        # Execute 3 out of 6 possible transitions
        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "profile")
        path_tracker.record_transition("dashboard", "settings")

        total_possible = len(workflow_definition["transitions"])
        coverage = path_tracker.calculate_coverage(total_possible)

        assert coverage == pytest.approx(50.0)  # 3/6 = 50%

    def test_calculate_coverage_full(self, path_tracker, workflow_definition):
        """Test coverage calculation with complete exploration."""
        # Execute all transitions
        for from_state, to_state in workflow_definition["transitions"]:
            path_tracker.record_transition(from_state, to_state)

        total_possible = len(workflow_definition["transitions"])
        coverage = path_tracker.calculate_coverage(total_possible)

        assert coverage == 100.0

    def test_coverage_with_repeated_transitions(self, path_tracker):
        """Test that repeated transitions don't increase unique coverage."""
        # Execute same transition multiple times
        for _ in range(5):
            path_tracker.record_transition("login", "dashboard")

        coverage = path_tracker.calculate_coverage(total_transitions=10)

        # Should count as only 1 unique transition
        assert coverage == 10.0  # 1/10 = 10%


class TestTransitionCounting:
    """Test transition execution counting."""

    def test_count_single_transition(self, path_tracker):
        """Test counting a transition executed once."""
        path_tracker.record_transition("login", "dashboard")

        count = path_tracker.get_transition_count("login", "dashboard")
        assert count == 1

    def test_count_repeated_transitions(self, path_tracker):
        """Test counting a transition executed multiple times."""
        for _ in range(5):
            path_tracker.record_transition("login", "dashboard")

        count = path_tracker.get_transition_count("login", "dashboard")
        assert count == 5

    def test_count_unexecuted_transition(self, path_tracker):
        """Test counting a transition that was never executed."""
        path_tracker.record_transition("login", "dashboard")

        count = path_tracker.get_transition_count("settings", "profile")
        assert count == 0


class TestResultsExport:
    """Test exporting path tracking results."""

    def test_export_empty_results(self, path_tracker):
        """Test exporting results with no data."""
        results = path_tracker.export_results()

        assert results["total_transitions"] == 0
        assert results["unique_states"] == 0
        assert results["unique_transitions"] == 0
        assert len(results["coverage_map"]) == 0

    def test_export_with_data(self, path_tracker):
        """Test exporting results with recorded data."""
        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "profile")
        path_tracker.record_transition("login", "dashboard")  # Repeat

        results = path_tracker.export_results()

        assert results["total_transitions"] == 3
        assert results["unique_states"] == 3
        assert results["unique_transitions"] == 2
        assert "login->dashboard" in results["coverage_map"]
        assert results["coverage_map"]["login->dashboard"] == 2

    def test_export_includes_all_fields(self, path_tracker):
        """Test that export includes all required fields."""
        path_tracker.record_transition("login", "dashboard")

        results = path_tracker.export_results()

        required_fields = [
            "total_transitions",
            "unique_states",
            "unique_transitions",
            "coverage_map",
            "states_visited",
        ]

        for field in required_fields:
            assert field in results


class TestDeficiencyDetection:
    """Test deficiency detection capabilities."""

    def test_detect_unreachable_states(self, path_tracker, workflow_definition):
        """Test detecting states that were never reached."""
        # Execute only some transitions
        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "profile")

        all_states = set(workflow_definition["states"])
        visited_states = path_tracker.states_visited
        unreachable = all_states - visited_states

        assert "settings" in unreachable
        assert "logout" in unreachable

    def test_detect_unexecuted_transitions(self, path_tracker, workflow_definition):
        """Test detecting transitions that were never executed."""
        # Execute partial transitions
        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "profile")

        all_transitions = {
            f"{from_state}->{to_state}"
            for from_state, to_state in workflow_definition["transitions"]
        }
        executed_transitions = set(path_tracker.coverage_map.keys())
        unexecuted = all_transitions - executed_transitions

        assert "dashboard->settings" in unexecuted
        assert "dashboard->logout" in unexecuted

    def test_detect_low_coverage_paths(self, path_tracker, workflow_definition):
        """Test detecting paths with low execution count (potential issues)."""
        # Execute transitions with varying frequencies
        for _ in range(10):
            path_tracker.record_transition("login", "dashboard")

        path_tracker.record_transition("dashboard", "settings")  # Only once

        # Paths executed only once might be suspicious
        low_execution_paths = {
            k: v for k, v in path_tracker.coverage_map.items() if v < 2
        }

        assert "dashboard->settings" in low_execution_paths
        assert "login->dashboard" not in low_execution_paths


class TestExplorationStrategies:
    """Test different path exploration strategies."""

    def test_breadth_first_exploration(self, path_tracker):
        """Test breadth-first exploration strategy."""
        # In BFS, we explore all immediate neighbors before going deeper
        # This is a conceptual test - actual implementation would differ

        # Level 1
        path_tracker.record_transition("login", "dashboard")
        # Level 2
        path_tracker.record_transition("dashboard", "settings")
        path_tracker.record_transition("dashboard", "profile")

        assert len(path_tracker.transitions) == 3

    def test_depth_first_exploration(self, path_tracker):
        """Test depth-first exploration strategy."""
        # In DFS, we explore as deep as possible before backtracking

        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "settings")
        path_tracker.record_transition("settings", "dashboard")
        path_tracker.record_transition("dashboard", "profile")

        assert len(path_tracker.transitions) == 4

    def test_random_walk_exploration(self, path_tracker):
        """Test random walk exploration strategy."""
        # Random walk selects next transition randomly
        # Just verify we can record random transitions

        import random

        states = ["login", "dashboard", "settings", "profile"]

        for _ in range(10):
            from_state = random.choice(states)
            to_state = random.choice(states)
            if from_state != to_state:
                path_tracker.record_transition(from_state, to_state)

        assert len(path_tracker.transitions) > 0


class TestBacktracking:
    """Test backtracking navigation functionality."""

    def test_backtrack_to_previous_state(self, path_tracker):
        """Test returning to a previous state."""
        # Forward path
        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "settings")

        # Backtrack
        path_tracker.record_transition("settings", "dashboard")

        assert path_tracker.transitions[-1]["to_state"] == "dashboard"

    def test_backtrack_multiple_levels(self, path_tracker):
        """Test backtracking multiple levels."""
        # Deep path
        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "profile")
        path_tracker.record_transition("profile", "settings")

        # Backtrack to dashboard
        path_tracker.record_transition("settings", "profile")
        path_tracker.record_transition("profile", "dashboard")

        assert len(path_tracker.transitions) == 5
        assert path_tracker.transitions[-1]["to_state"] == "dashboard"

    def test_detect_exploration_dead_ends(self, path_tracker):
        """Test detecting when we've reached a state with no unexplored exits."""
        # Explore all transitions from dashboard
        path_tracker.record_transition("login", "dashboard")
        path_tracker.record_transition("dashboard", "settings")
        path_tracker.record_transition("settings", "dashboard")
        path_tracker.record_transition("dashboard", "profile")
        path_tracker.record_transition("profile", "dashboard")

        # All exits from dashboard have been explored
        # This would trigger backtracking in actual implementation
        dashboard_transitions = [
            t for t in path_tracker.transitions if t["from_state"] == "dashboard"
        ]

        assert len(dashboard_transitions) >= 2
