"""Example usage of PathExplorer and exploration strategies.

This module demonstrates how to use the PathExplorer system for
intelligent state graph exploration.
"""

import logging
import time
from typing import Any

from app.testing.config import ExplorationConfig
from app.testing.exploration import PathExplorer
from app.testing.path_tracker import PathTracker

logger = logging.getLogger(__name__)


def example_basic_exploration():
    """Example: Basic exploration with default configuration."""

    # Assume we have a state graph (mock for example)
    class MockStateGraph:
        def __init__(self):
            self.states = {
                "login": MockState("login", ["dashboard", "error"]),
                "dashboard": MockState("dashboard", ["profile", "settings", "logout"]),
                "profile": MockState("profile", ["edit_profile", "dashboard"]),
                "settings": MockState("settings", ["change_password", "dashboard"]),
                "logout": MockState("logout", ["login"]),
            }
            self.initial_state = "login"

    class MockState:
        def __init__(self, name: str, transitions: list[str]):
            self.name = name
            self.transitions = [MockTransition(t) for t in transitions]

    class MockTransition:
        def __init__(self, to_state: str):
            self.to_state = to_state

    # Create state graph
    state_graph = MockStateGraph()

    # Create configuration
    config = ExplorationConfig(
        strategy="hybrid",
        max_iterations=100,
        coverage_target=0.90,
        enable_backtracking=True,
        enable_diversity=True,
        enable_failure_handling=True,
    )

    # Create tracker
    tracker = PathTracker(state_graph)

    # Create explorer
    explorer = PathExplorer(config, tracker, initial_state="login")

    # Define executor callback (simulates actual transition execution)
    def execute_transition(from_state: str, to_state: str) -> tuple[bool, float, dict[str, Any]]:
        """Simulate transition execution."""
        logger.info(f"Executing: {from_state} -> {to_state}")

        # Simulate execution time
        duration_ms = 100.0 + (hash(from_state + to_state) % 1000)

        # Simulate 95% success rate
        success = (hash(from_state + to_state) % 100) >= 5

        metadata = {
            "action": "click_button",
            "element": to_state,
        }

        # Simulate actual delay
        time.sleep(duration_ms / 10000.0)

        return success, duration_ms, metadata

    # Run exploration
    report = explorer.explore(execute_transition)

    # Print results
    print("\n" + "=" * 80)
    print("EXPLORATION RESULTS")
    print("=" * 80)
    print(f"Strategy: {report['summary']['strategy']}")
    print(f"Iterations: {report['summary']['iterations']}")
    print(f"State Coverage: {report['coverage']['state_coverage_percent']:.1f}%")
    print(f"Transition Coverage: {report['coverage']['transition_coverage_percent']:.1f}%")
    print(f"Success Rate: {report['coverage']['success_rate_percent']:.1f}%")
    print(f"Unique Paths: {report['coverage']['unique_paths']}")
    print("=" * 80 + "\n")


def example_greedy_strategy():
    """Example: Using greedy coverage strategy."""
    # Mock state graph (same as above)
    # ... (omitted for brevity)

    ExplorationConfig(
        strategy="greedy",
        max_iterations=50,
        greedy_unexplored_bonus=3.0,
        greedy_unvisited_state_bonus=2.0,
    )

    # ... rest of setup


def example_adaptive_strategy():
    """Example: Using Q-learning adaptive strategy."""
    ExplorationConfig(
        strategy="adaptive",
        max_iterations=200,
        adaptive_learning_rate=0.1,
        adaptive_discount_factor=0.9,
        adaptive_epsilon_start=1.0,
        adaptive_epsilon_decay=0.99,
        adaptive_reward_new_state=50.0,
    )

    # ... rest of setup


def example_custom_configuration():
    """Example: Using custom YAML configuration."""
    # Save example config to YAML
    config = ExplorationConfig(
        strategy="hybrid",
        max_iterations=500,
        coverage_target=0.95,
        hybrid_phase_iterations=[50, 150, 300],
        hybrid_phase_strategies=["random_walk", "greedy", "adaptive"],
    )

    config.save_yaml("exploration_config.yaml")

    # Later, load from YAML
    loaded_config = ExplorationConfig.from_yaml("exploration_config.yaml")

    print(f"Loaded strategy: {loaded_config.strategy}")
    print(f"Max iterations: {loaded_config.max_iterations}")


def example_path_exploration():
    """Example: Exploring specific path to target state."""
    # Setup (omitted for brevity)
    # explorer = PathExplorer(config, tracker)

    # Define executor
    def execute_transition(from_state: str, to_state: str) -> tuple[bool, float, dict[str, Any]]:
        # Simulate execution
        return True, 150.0, {}

    # Explore path to specific state
    # success = explorer.explore_path("settings", execute_transition)

    # if success:
    #     print("Successfully reached settings page!")


def example_failure_handling():
    """Example: Exploration with failure handling."""
    ExplorationConfig(
        strategy="greedy",
        max_iterations=100,
        enable_failure_handling=True,
        failure_max_retries=3,
        failure_backoff_base_ms=1000.0,
        failure_backoff_multiplier=2.0,
        failure_skip_threshold=5,
    )

    # ... rest of setup

    # Executor with occasional failures
    def execute_transition(from_state: str, to_state: str) -> tuple[bool, float, dict[str, Any]]:
        # Simulate flaky transition
        if to_state == "error_prone_state":
            success = (hash(str(time.time())) % 10) >= 3  # 70% success
        else:
            success = True

        return success, 100.0, {}

    # Explorer will automatically retry failed transitions with backoff


def example_coverage_callbacks():
    """Example: Using coverage milestone callbacks."""
    # Setup tracker and explorer
    # tracker = PathTracker(state_graph)
    # explorer = PathExplorer(config, tracker)

    # Register callback for 50% coverage
    # def on_50_percent_coverage(metrics, milestone):
    #     print(f"Milestone reached: {milestone}% coverage!")
    #     print(f"Visited states: {metrics.visited_states}/{metrics.total_states}")

    # tracker.on_coverage_milestone(on_50_percent_coverage, 50.0)

    # Register callback for deficiency detection
    # def on_deficiency(deficiency):
    #     print(f"Deficiency found: [{deficiency.severity.value}] {deficiency.title}")

    # tracker.on_deficiency_detected(on_deficiency)


def example_exporting_results():
    """Example: Exporting exploration results."""
    ExplorationConfig(
        export_on_completion=True,
        export_format="json",  # or "html", "csv", "markdown"
        export_path="./exploration_results",
    )

    # After exploration completes, results are automatically exported

    # Or export manually
    # tracker.export_results(
    #     output_path="./custom_results.json",
    #     format="json",
    #     include_screenshots=True,
    # )


def example_backtracking():
    """Example: Using backtracking to reach unexplored states."""
    ExplorationConfig(
        strategy="greedy",
        enable_backtracking=True,
        backtracking_max_attempts=3,
        backtracking_prefer_shortest=True,
    )

    # When explorer reaches a dead end or gets stuck,
    # it will automatically backtrack to states with unexplored transitions


def example_diverse_paths():
    """Example: Generating diverse paths."""
    ExplorationConfig(
        enable_diversity=True,
        diversity_k_paths=5,
        diversity_variation_rate=0.3,
        diversity_min_difference=0.2,
    )

    # Diversity engine will generate multiple different paths
    # between states to discover varied behaviors


if __name__ == "__main__":
    # Run basic example
    logging.basicConfig(level=logging.INFO)
    example_basic_exploration()
