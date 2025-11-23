"""Backtracking navigator for returning to unexplored states.

This module provides intelligent backtracking using shortest path algorithms
to navigate back to states with unexplored transitions.
"""

import logging
from typing import Any

from app.testing.config import ExplorationConfig
from app.testing.path_tracker import PathTracker

logger = logging.getLogger(__name__)


class BacktrackingNavigator:
    """Intelligent backtracking to reach unexplored states.

    Uses Dijkstra's algorithm to find shortest paths back to states
    with unexplored transitions, enabling comprehensive coverage.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize backtracking navigator.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance
        """
        self.config = config
        self.tracker = tracker
        self.state_graph = tracker.state_graph
        self.max_attempts = config.backtracking_max_attempts
        self.prefer_shortest = config.backtracking_prefer_shortest

    def find_backtrack_path(
        self, current_state: str, target_state: str | None = None
    ) -> list[str] | None:
        """Find shortest path to target or nearest unexplored state.

        Args:
            current_state: Current state name
            target_state: Target state (if None, finds nearest unexplored)

        Returns:
            List of state names forming path, or None if no path found
        """
        if target_state is None:
            # Find nearest state with unexplored transitions
            target_state = self._find_nearest_unexplored_state(current_state)

            if target_state is None:
                logger.debug("No unexplored states found for backtracking")
                return None

        # Find shortest path using Dijkstra
        path = self._dijkstra_shortest_path(current_state, target_state)

        if path:
            logger.info(
                f"Backtrack path found: {current_state} -> {target_state} "
                f"({len(path)} steps)"
            )
        else:
            logger.warning(f"No backtrack path found from {current_state} to {target_state}")

        return path

    def _find_nearest_unexplored_state(self, start_state: str) -> str | None:
        """Find nearest state with unexplored transitions.

        Args:
            start_state: Starting state

        Returns:
            Nearest state with unexplored transitions, or None
        """
        # Get all states with unexplored transitions
        unexplored_states = self._get_states_with_unexplored_transitions()

        if not unexplored_states:
            return None

        # Find nearest state using BFS
        visited = {start_state}
        queue = [(start_state, 0)]  # (state, distance)
        nearest_state = None
        min_distance = float("inf")

        while queue:
            state, distance = queue.pop(0)

            # Check if this state has unexplored transitions
            if state in unexplored_states:
                if distance < min_distance:
                    min_distance = distance
                    nearest_state = state
                continue

            # Explore neighbors
            if not hasattr(self.state_graph, "states"):
                continue

            state_obj = self.state_graph.states.get(state)
            if not state_obj or not hasattr(state_obj, "transitions"):
                continue

            for transition in state_obj.transitions:
                next_state = getattr(transition, "to_state", None)
                if next_state and next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, distance + 1))

        return nearest_state

    def _get_states_with_unexplored_transitions(self) -> set[str]:
        """Get all states that have unexplored transitions.

        Returns:
            Set of state names with unexplored transitions
        """
        states_with_unexplored = set()

        if not hasattr(self.state_graph, "states"):
            return states_with_unexplored

        executed_transitions = self.tracker._executed_transitions

        for state_name, state_obj in self.state_graph.states.items():
            if not hasattr(state_obj, "transitions"):
                continue

            # Check if state has any unexplored transitions
            for transition in state_obj.transitions:
                to_state = getattr(transition, "to_state", None)
                if to_state:
                    transition_key = (state_name, to_state)
                    if transition_key not in executed_transitions:
                        states_with_unexplored.add(state_name)
                        break

        return states_with_unexplored

    def _dijkstra_shortest_path(self, start: str, goal: str) -> list[str] | None:
        """Find shortest path using Dijkstra's algorithm.

        Args:
            start: Starting state
            goal: Goal state

        Returns:
            List of states forming shortest path, or None if no path exists
        """
        import heapq

        # Build graph from state_graph
        if not hasattr(self.state_graph, "states"):
            return None

        # Priority queue: (cost, state, path)
        pq = [(0, start, [start])]
        visited = set()

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            # Check if goal reached
            if current == goal:
                return path

            # Explore neighbors
            state_obj = self.state_graph.states.get(current)
            if not state_obj or not hasattr(state_obj, "transitions"):
                continue

            for transition in state_obj.transitions:
                next_state = getattr(transition, "to_state", None)

                if next_state and next_state not in visited:
                    # Calculate edge cost (prioritize successful transitions)
                    edge_cost = self._get_edge_cost(current, next_state)
                    new_cost = cost + edge_cost
                    new_path = path + [next_state]

                    heapq.heappush(pq, (new_cost, next_state, new_path))

        # No path found
        return None

    def _get_edge_cost(self, from_state: str, to_state: str) -> float:
        """Get cost of traversing an edge.

        Lower cost for reliable, fast transitions.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            Edge cost (lower is better)
        """
        transition_key = (from_state, to_state)
        base_cost = 1.0

        # Check if we have statistics for this transition
        if transition_key in self.tracker._transition_stats:
            stats = self.tracker._transition_stats[transition_key]

            # Increase cost for unreliable transitions
            if stats.total_attempts > 0:
                failure_rate = 1.0 - (stats.successes / stats.total_attempts)
                base_cost += failure_rate * 5.0

            # Increase cost for slow transitions
            if stats.avg_duration_ms > self.config.performance_threshold_ms:
                base_cost += 2.0

        else:
            # Unexplored transitions have medium cost
            base_cost += 0.5

        return base_cost

    def find_alternative_path(
        self, current_state: str, target_state: str, avoid_states: set[str] | None = None
    ) -> list[str] | None:
        """Find alternative path avoiding certain states.

        Useful when primary path is blocked or unreliable.

        Args:
            current_state: Current state
            target_state: Target state
            avoid_states: States to avoid in path

        Returns:
            Alternative path, or None if no path found
        """
        if avoid_states is None:
            avoid_states = set()

        # Modified Dijkstra avoiding certain states
        import heapq

        if not hasattr(self.state_graph, "states"):
            return None

        pq = [(0, current_state, [current_state])]
        visited = set(avoid_states)  # Pre-populate with states to avoid

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == target_state:
                return path

            state_obj = self.state_graph.states.get(current)
            if not state_obj or not hasattr(state_obj, "transitions"):
                continue

            for transition in state_obj.transitions:
                next_state = getattr(transition, "to_state", None)

                if next_state and next_state not in visited:
                    edge_cost = self._get_edge_cost(current, next_state)
                    new_cost = cost + edge_cost
                    new_path = path + [next_state]

                    heapq.heappush(pq, (new_cost, next_state, new_path))

        return None

    def get_reachable_unexplored_states(self, current_state: str) -> list[str]:
        """Get all reachable states with unexplored transitions.

        Args:
            current_state: Current state

        Returns:
            List of reachable states with unexplored transitions
        """
        unexplored_states = self._get_states_with_unexplored_transitions()
        reachable = []

        for state in unexplored_states:
            path = self._dijkstra_shortest_path(current_state, state)
            if path:
                reachable.append(state)

        return reachable

    def estimate_backtrack_cost(self, current_state: str, target_state: str) -> float:
        """Estimate cost of backtracking to target state.

        Args:
            current_state: Current state
            target_state: Target state

        Returns:
            Estimated cost (time in milliseconds)
        """
        path = self._dijkstra_shortest_path(current_state, target_state)

        if not path:
            return float("inf")

        # Estimate time based on transition statistics
        total_time = 0.0

        for i in range(len(path) - 1):
            from_s = path[i]
            to_s = path[i + 1]
            transition_key = (from_s, to_s)

            if transition_key in self.tracker._transition_stats:
                stats = self.tracker._transition_stats[transition_key]
                total_time += stats.avg_duration_ms
            else:
                # Assume average time for unexplored transitions
                total_time += self.config.performance_threshold_ms / 2

        return total_time
