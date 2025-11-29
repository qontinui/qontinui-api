"""Main PathExplorer class for intelligent path traversal.

This module provides the high-level PathExplorer interface that orchestrates
exploration strategies, backtracking, diversity, and failure handling.
"""

import logging
from collections.abc import Callable
from typing import Any

from app.testing.config import ExplorationConfig
from app.testing.exploration.backtracking import BacktrackingNavigator
from app.testing.exploration.diversity import PathDiversityEngine
from app.testing.exploration.failure_handler import FailureAwareExplorer
from app.testing.exploration.strategies import (
    AdaptiveExplorer,
    BreadthFirstExplorer,
    DepthFirstExplorer,
    ExplorationStrategy,
    GreedyCoverageExplorer,
    HybridExplorer,
    RandomWalkExplorer,
)
from app.testing.path_tracker import PathTracker

logger = logging.getLogger(__name__)


class PathExplorer:
    """Intelligent path exploration orchestrator.

    PathExplorer is the main entry point for exploring state graphs.
    It coordinates multiple strategies, handles failures, manages backtracking,
    and ensures comprehensive coverage.

    Example:
        >>> config = ExplorationConfig(strategy="hybrid", max_iterations=1000)
        >>> tracker = PathTracker(state_graph=graph)
        >>> explorer = PathExplorer(config, tracker)
        >>> explorer.explore(initial_state="login", executor_callback=execute_transition)
        >>> metrics = tracker.get_coverage_metrics()
        >>> print(f"Coverage: {metrics.transition_coverage_percent:.1f}%")
    """

    def __init__(
        self,
        config: ExplorationConfig,
        tracker: PathTracker,
        initial_state: str | None = None,
    ):
        """Initialize PathExplorer.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance for metrics
            initial_state: Initial state to start from (optional)
        """
        self.config = config
        self.tracker = tracker
        self.state_graph = tracker.state_graph

        # Get initial state
        if initial_state:
            self.initial_state = initial_state
        elif hasattr(self.state_graph, "initial_state"):
            self.initial_state = self.state_graph.initial_state
        else:
            # Try to infer from graph
            if hasattr(self.state_graph, "states") and self.state_graph.states:
                self.initial_state = list(self.state_graph.states.keys())[0]
            else:
                raise ValueError("Could not determine initial state")

        # Initialize strategy
        self.strategy = self._create_strategy(config.strategy)

        # Initialize supporting systems
        self.backtracker: BacktrackingNavigator | None = None
        self.diversity_engine: PathDiversityEngine | None = None
        self.failure_handler: FailureAwareExplorer | None = None

        if config.enable_backtracking:
            self.backtracker = BacktrackingNavigator(config, tracker)

        if config.enable_diversity:
            self.diversity_engine = PathDiversityEngine(config, tracker)

        if config.enable_failure_handling:
            self.failure_handler = FailureAwareExplorer(config, tracker)

        # Exploration state
        self.current_state = self.initial_state
        self.iteration = 0
        self.stuck_count = 0
        self.is_exploring = False

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info(
            f"PathExplorer initialized: strategy={config.strategy}, "
            f"initial_state={self.initial_state}"
        )

    def explore(
        self,
        executor_callback: Callable[[str, str], tuple[bool, float, dict[str, Any]]],
        initial_state: str | None = None,
    ) -> dict[str, Any]:
        """Execute exploration with automatic strategy selection.

        The executor callback is responsible for actually executing transitions
        in the target system and returning results.

        Args:
            executor_callback: Function to execute transitions
                               Args: (from_state, to_state)
                               Returns: (success: bool, duration_ms: float, metadata: dict)
            initial_state: Override initial state (optional)

        Returns:
            Dictionary with exploration results and metrics
        """
        if initial_state:
            self.current_state = initial_state
        else:
            self.current_state = self.initial_state

        self.is_exploring = True
        self.iteration = 0
        self.stuck_count = 0

        logger.info(f"Starting exploration from {self.current_state}")
        self.tracker.start_new_path(self.current_state)

        try:
            while self._should_continue_exploring():
                self.iteration += 1

                # Log progress
                if self.iteration % self.config.log_progress_interval == 0:
                    self._log_progress()

                # Update cooldowns if using failure handler
                if self.failure_handler:
                    self.failure_handler.update_cooldowns()

                # Select next state
                next_state = self._select_next_state()

                if next_state is None:
                    logger.debug("No next state available, attempting recovery")
                    if not self._handle_stuck_state():
                        logger.warning("Recovery failed, ending exploration")
                        break
                    continue

                # Execute transition
                success, duration_ms, metadata = self._execute_transition(
                    self.current_state, next_state, executor_callback
                )

                # Update current state
                if success:
                    self.current_state = next_state
                    self.stuck_count = 0
                else:
                    self.stuck_count += 1

                # Check if stuck
                if self.stuck_count >= self.config.stuck_threshold:
                    if not self._handle_stuck_state():
                        break

        except KeyboardInterrupt:
            logger.info("Exploration interrupted by user")
        except Exception as e:
            logger.error(f"Exploration error: {e}", exc_info=True)
        finally:
            self.is_exploring = False
            self.tracker.end_current_path(success=True)

        # Get final metrics
        metrics = self.tracker.get_coverage_metrics()

        logger.info(
            f"Exploration completed: {self.iteration} iterations, "
            f"{metrics.transition_coverage_percent:.1f}% coverage"
        )

        # Export results if configured
        if self.config.export_on_completion:
            self._export_results()

        return self._generate_exploration_report(metrics)

    def explore_path(
        self,
        target_state: str,
        executor_callback: Callable[[str, str], tuple[bool, float, dict[str, Any]]],
    ) -> bool:
        """Explore a specific path to a target state.

        Args:
            target_state: State to reach
            executor_callback: Function to execute transitions

        Returns:
            True if target state reached successfully
        """
        logger.info(f"Exploring path: {self.current_state} -> {target_state}")

        # Find path using diversity engine or backtracker
        path = None

        if self.diversity_engine:
            paths = self.diversity_engine.generate_diverse_paths(self.current_state, target_state)
            if paths:
                path = paths[0]  # Use first (shortest) path

        if not path and self.backtracker:
            path = self.backtracker.find_backtrack_path(self.current_state, target_state)

        if not path:
            logger.warning(f"No path found to {target_state}")
            return False

        # Execute path
        for i in range(len(path) - 1):
            from_state = path[i]
            to_state = path[i + 1]

            success, _, _ = self._execute_transition(from_state, to_state, executor_callback)

            if not success:
                logger.warning(f"Path execution failed at {from_state} -> {to_state}")
                return False

            self.current_state = to_state

        logger.info(f"Successfully reached {target_state}")
        return True

    def _create_strategy(self, strategy_name: str) -> ExplorationStrategy:
        """Create exploration strategy instance.

        Args:
            strategy_name: Name of strategy

        Returns:
            ExplorationStrategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategies = {
            "random_walk": RandomWalkExplorer,
            "greedy": GreedyCoverageExplorer,
            "dfs": DepthFirstExplorer,
            "bfs": BreadthFirstExplorer,
            "adaptive": AdaptiveExplorer,
            "hybrid": HybridExplorer,
        }

        strategy_class = strategies.get(strategy_name)

        if not strategy_class:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. " f"Available: {list(strategies.keys())}"
            )

        return strategy_class(self.config, self.tracker)

    def _select_next_state(self) -> str | None:
        """Select next state using current strategy.

        Returns:
            Next state name or None
        """
        # Use strategy to select next state
        next_state = self.strategy.select_next_state(self.current_state)

        # If strategy returns None, try backtracking
        if next_state is None and self.backtracker:
            logger.debug("Strategy returned None, attempting backtracking")
            path = self.backtracker.find_backtrack_path(self.current_state)
            if path and len(path) > 1:
                next_state = path[1]  # Next state in backtrack path

        # If still None and failure handler available, try finding reliable alternative
        if next_state is None and self.failure_handler:
            logger.debug("Backtracking failed, trying reliable alternative")
            next_state = self.failure_handler.get_reliable_alternative(self.current_state)

        return next_state

    def _execute_transition(
        self,
        from_state: str,
        to_state: str,
        executor_callback: Callable[[str, str], tuple[bool, float, dict[str, Any]]],
    ) -> tuple[bool, float, dict[str, Any]]:
        """Execute a transition with retry logic.

        Args:
            from_state: Source state
            to_state: Target state
            executor_callback: Callback to execute transition

        Returns:
            Tuple of (success, duration_ms, metadata)
        """
        attempt = 1
        max_attempts = self.config.failure_max_retries + 1

        while attempt <= max_attempts:
            # Check if should retry
            if attempt > 1 and self.failure_handler:
                if not self.failure_handler.should_retry_transition(from_state, to_state, attempt):
                    logger.info(f"Skipping retry for {from_state} -> {to_state}")
                    break

                # Wait for backoff
                self.failure_handler.wait_for_backoff(from_state, to_state, attempt)

            # Execute transition
            try:
                success, duration_ms, metadata = executor_callback(from_state, to_state)

                # Record in PathTracker
                self.tracker.record_transition(
                    from_state=from_state,
                    to_state=to_state,
                    success=success,
                    duration_ms=duration_ms,
                    metadata=metadata,
                    error_message=metadata.get("error_message") if not success else None,
                )

                # Update failure handler
                if self.failure_handler:
                    if success:
                        self.failure_handler.record_success(from_state, to_state)
                    else:
                        self.failure_handler.record_failure(from_state, to_state)

                # Update Q-learning if using adaptive strategy
                if isinstance(self.strategy, AdaptiveExplorer):
                    new_state = to_state not in self.tracker._visited_states
                    new_transition = (
                        from_state,
                        to_state,
                    ) not in self.tracker._executed_transitions
                    self.strategy.update_q_value(success, new_state, new_transition)

                # Return if successful or max attempts reached
                if success or attempt >= max_attempts:
                    return success, duration_ms, metadata

                attempt += 1

            except Exception as e:
                logger.error(f"Transition execution error: {e}", exc_info=True)

                # Record error
                self.tracker.record_transition(
                    from_state=from_state,
                    to_state=to_state,
                    success=False,
                    duration_ms=0.0,
                    error_message=str(e),
                )

                if self.failure_handler:
                    self.failure_handler.record_failure(from_state, to_state)

                if attempt >= max_attempts:
                    return False, 0.0, {"error_message": str(e)}

                attempt += 1

        return False, 0.0, {}

    def _should_continue_exploring(self) -> bool:
        """Check if exploration should continue.

        Returns:
            True if exploration should continue
        """
        if not self.is_exploring:
            return False

        # Check max iterations
        if self.iteration >= self.config.max_iterations:
            logger.info(f"Max iterations ({self.config.max_iterations}) reached")
            return False

        # Check coverage target
        if self.config.early_stopping:
            metrics = self.tracker.get_coverage_metrics()
            coverage = metrics.transition_coverage_percent / 100.0

            if coverage >= self.config.coverage_target:
                logger.info(f"Coverage target ({self.config.coverage_target * 100:.1f}%) reached")
                return False

        return True

    def _handle_stuck_state(self) -> bool:
        """Handle stuck state by restarting or backtracking.

        Returns:
            True if recovery successful
        """
        logger.warning(f"Stuck at {self.current_state} for {self.stuck_count} iterations")

        # Try backtracking first
        if self.backtracker:
            path = self.backtracker.find_backtrack_path(self.current_state)
            if path and len(path) > 1:
                logger.info(f"Backtracking from {self.current_state}")
                self.current_state = path[1]
                self.stuck_count = 0
                return True

        # Try restarting from initial state
        if self.config.restart_on_stuck:
            logger.info(f"Restarting from {self.initial_state}")
            self.current_state = self.initial_state
            self.stuck_count = 0
            self.tracker.start_new_path(self.initial_state)
            return True

        return False

    def _log_progress(self) -> None:
        """Log exploration progress."""
        metrics = self.tracker.get_coverage_metrics()

        logger.info(
            f"Progress - Iteration: {self.iteration}, "
            f"State: {self.current_state}, "
            f"Coverage: {metrics.transition_coverage_percent:.1f}%, "
            f"Paths: {metrics.unique_paths}, "
            f"Success Rate: {metrics.success_rate_percent:.1f}%"
        )

    def _export_results(self) -> None:
        """Export exploration results."""
        import os

        os.makedirs(self.config.export_path, exist_ok=True)

        # Export PathTracker results
        output_file = os.path.join(
            self.config.export_path,
            f"exploration_results.{self.config.export_format}",
        )

        self.tracker.export_results(
            output_path=output_file,
            format=self.config.export_format,
        )

        logger.info(f"Results exported to {output_file}")

        # Export failure report if available
        if self.failure_handler:
            import json

            failure_report = self.failure_handler.export_failure_report()
            failure_file = os.path.join(self.config.export_path, "failure_report.json")

            with open(failure_file, "w") as f:
                json.dump(failure_report, f, indent=2)

            logger.info(f"Failure report exported to {failure_file}")

    def _generate_exploration_report(self, metrics: Any) -> dict[str, Any]:
        """Generate comprehensive exploration report.

        Args:
            metrics: CoverageMetrics instance

        Returns:
            Dictionary with exploration report
        """
        report = {
            "summary": {
                "strategy": self.config.strategy,
                "iterations": self.iteration,
                "initial_state": self.initial_state,
                "final_state": self.current_state,
            },
            "coverage": metrics.to_dict(),
            "configuration": self.config.to_dict(),
        }

        # Add failure statistics if available
        if self.failure_handler:
            report["failures"] = self.failure_handler.get_failure_statistics()

        # Add deficiencies
        deficiencies = self.tracker.get_deficiencies()
        report["deficiencies"] = {
            "total": len(deficiencies),
            "critical": len([d for d in deficiencies if d.is_critical]),
            "by_category": {},
        }

        # Group by category
        from collections import defaultdict

        by_category = defaultdict(int)
        for d in deficiencies:
            by_category[d.category.value] += 1

        report["deficiencies"]["by_category"] = dict(by_category)

        return report

    def reset(self) -> None:
        """Reset explorer state for new exploration run."""
        self.current_state = self.initial_state
        self.iteration = 0
        self.stuck_count = 0
        self.is_exploring = False

        # Reset strategy
        self.strategy.reset()

        # Reset supporting systems
        if self.failure_handler:
            self.failure_handler.reset()

        logger.info("PathExplorer reset")

    def get_exploration_status(self) -> dict[str, Any]:
        """Get current exploration status.

        Returns:
            Dictionary with current status
        """
        metrics = self.tracker.get_coverage_metrics()

        return {
            "is_exploring": self.is_exploring,
            "iteration": self.iteration,
            "current_state": self.current_state,
            "stuck_count": self.stuck_count,
            "strategy": self.config.strategy,
            "coverage_percent": metrics.transition_coverage_percent,
            "states_visited": metrics.visited_states,
            "transitions_executed": metrics.executed_transitions,
            "success_rate": metrics.success_rate_percent,
        }
