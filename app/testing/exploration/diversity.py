"""Path diversity engine for generating varied exploration paths.

This module provides algorithms to generate diverse paths through the state
graph, helping discover edge cases and maximize test coverage.
"""

import logging
import random
from typing import Any

from app.testing.config import ExplorationConfig
from app.testing.path_tracker import PathTracker

logger = logging.getLogger(__name__)


class PathDiversityEngine:
    """Generates diverse paths for comprehensive testing.

    Uses k-shortest paths and path variation algorithms to create
    multiple different paths between states, discovering varied behaviors.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize path diversity engine.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance
        """
        self.config = config
        self.tracker = tracker
        self.state_graph = tracker.state_graph
        self.k_paths = config.diversity_k_paths
        self.variation_rate = config.diversity_variation_rate
        self.min_difference = config.diversity_min_difference

    def generate_diverse_paths(
        self, start_state: str, end_state: str
    ) -> list[list[str]]:
        """Generate k diverse paths between two states.

        Args:
            start_state: Starting state
            end_state: Ending state

        Returns:
            List of diverse paths (each path is a list of states)
        """
        # Find k-shortest paths
        k_shortest = self._k_shortest_paths(start_state, end_state, self.k_paths)

        if not k_shortest:
            logger.debug(f"No paths found from {start_state} to {end_state}")
            return []

        # Filter for diversity
        diverse_paths = self._filter_diverse_paths(k_shortest)

        logger.info(
            f"Generated {len(diverse_paths)} diverse paths: "
            f"{start_state} -> {end_state}"
        )

        return diverse_paths

    def generate_path_variations(self, base_path: list[str]) -> list[list[str]]:
        """Generate variations of a base path.

        Creates alternative paths by substituting segments while maintaining
        start and end states.

        Args:
            base_path: Base path to create variations from

        Returns:
            List of path variations
        """
        if len(base_path) < 3:
            return [base_path]  # Can't vary too-short paths

        variations = [base_path]

        # Generate variations by replacing path segments
        num_variations = self.k_paths - 1

        for _ in range(num_variations):
            variation = self._create_path_variation(base_path)
            if variation and variation != base_path:
                variations.append(variation)

        # Filter for diversity
        diverse_variations = self._filter_diverse_paths(variations)

        logger.debug(f"Generated {len(diverse_variations)} path variations")

        return diverse_variations

    def _k_shortest_paths(
        self, start: str, goal: str, k: int
    ) -> list[tuple[float, list[str]]]:
        """Find k shortest paths using Yen's algorithm.

        Args:
            start: Starting state
            goal: Goal state
            k: Number of paths to find

        Returns:
            List of (cost, path) tuples
        """
        if not hasattr(self.state_graph, "states"):
            return []

        # Storage for k-shortest paths
        A = []  # Found paths
        B = []  # Candidate paths

        # Find first shortest path
        first_path = self._dijkstra_path(start, goal)
        if not first_path:
            return []

        first_cost = self._calculate_path_cost(first_path)
        A.append((first_cost, first_path))

        # Find k-1 more paths
        for k_i in range(1, k):
            if not A:
                break

            last_path = A[-1][1]

            # For each node in the last path (except last)
            for i in range(len(last_path) - 1):
                spur_node = last_path[i]
                root_path = last_path[: i + 1]

                # Find spur path avoiding certain edges
                removed_edges = set()

                # Remove edges used in previous paths with same root
                for _, path in A:
                    if len(path) > i and path[: i + 1] == root_path:
                        if len(path) > i + 1:
                            removed_edges.add((path[i], path[i + 1]))

                # Find spur path
                spur_path = self._dijkstra_path(
                    spur_node, goal, avoid_edges=removed_edges
                )

                if spur_path and len(spur_path) > 1:
                    # Combine root and spur paths
                    total_path = root_path[:-1] + spur_path
                    total_cost = self._calculate_path_cost(total_path)

                    # Add to candidates if not duplicate
                    if not any(p == total_path for _, p in B) and not any(
                        p == total_path for _, p in A
                    ):
                        B.append((total_cost, total_path))

            # No more candidates
            if not B:
                break

            # Sort candidates and add best to A
            B.sort(key=lambda x: x[0])
            A.append(B.pop(0))

        return A

    def _dijkstra_path(
        self,
        start: str,
        goal: str,
        avoid_edges: set[tuple[str, str]] | None = None,
    ) -> list[str] | None:
        """Find shortest path with optional edge avoidance.

        Args:
            start: Starting state
            goal: Goal state
            avoid_edges: Set of (from, to) edges to avoid

        Returns:
            Shortest path or None
        """
        import heapq

        if avoid_edges is None:
            avoid_edges = set()

        if not hasattr(self.state_graph, "states"):
            return None

        pq = [(0, start, [start])]
        visited = set()

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                return path

            state_obj = self.state_graph.states.get(current)
            if not state_obj or not hasattr(state_obj, "transitions"):
                continue

            for transition in state_obj.transitions:
                next_state = getattr(transition, "to_state", None)

                if next_state and next_state not in visited:
                    # Skip if edge should be avoided
                    if (current, next_state) in avoid_edges:
                        continue

                    edge_cost = self._get_edge_cost(current, next_state)
                    new_cost = cost + edge_cost
                    new_path = path + [next_state]

                    heapq.heappush(pq, (new_cost, next_state, new_path))

        return None

    def _calculate_path_cost(self, path: list[str]) -> float:
        """Calculate total cost of a path.

        Args:
            path: List of states

        Returns:
            Total path cost
        """
        cost = 0.0

        for i in range(len(path) - 1):
            cost += self._get_edge_cost(path[i], path[i + 1])

        return cost

    def _get_edge_cost(self, from_state: str, to_state: str) -> float:
        """Get cost of an edge.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            Edge cost
        """
        transition_key = (from_state, to_state)
        base_cost = 1.0

        if transition_key in self.tracker._transition_stats:
            stats = self.tracker._transition_stats[transition_key]

            # Factor in reliability
            if stats.total_attempts > 0:
                failure_rate = 1.0 - (stats.successes / stats.total_attempts)
                base_cost += failure_rate

            # Factor in performance
            if stats.avg_duration_ms > self.config.performance_threshold_ms:
                base_cost += 0.5

        return base_cost

    def _create_path_variation(self, base_path: list[str]) -> list[str] | None:
        """Create a variation of a path by modifying segments.

        Args:
            base_path: Original path

        Returns:
            Varied path or None
        """
        if len(base_path) < 3:
            return None

        # Select random segment to vary
        start_idx = random.randint(0, len(base_path) - 3)
        end_idx = random.randint(start_idx + 2, len(base_path) - 1)

        prefix = base_path[:start_idx]
        suffix = base_path[end_idx:]
        segment_start = base_path[start_idx]
        segment_end = base_path[end_idx]

        # Find alternative path for segment
        alternative_segment = self._find_alternative_segment(
            segment_start, segment_end, base_path[start_idx:end_idx + 1]
        )

        if alternative_segment:
            return prefix + alternative_segment + suffix

        return None

    def _find_alternative_segment(
        self, start: str, end: str, avoid_path: list[str]
    ) -> list[str] | None:
        """Find alternative path segment avoiding certain nodes.

        Args:
            start: Segment start state
            end: Segment end state
            avoid_path: Path to avoid

        Returns:
            Alternative segment or None
        """
        import heapq

        if not hasattr(self.state_graph, "states"):
            return None

        # Avoid internal nodes of original path
        avoid_nodes = set(avoid_path[1:-1])

        pq = [(0, start, [start])]
        visited = set(avoid_nodes)

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == end:
                # Check if path is sufficiently different
                if len(path) != len(avoid_path):
                    return path
                overlap = len(set(path) & set(avoid_path))
                if overlap / len(path) < (1 - self.min_difference):
                    return path
                continue

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

    def _filter_diverse_paths(
        self, paths: list[tuple[float, list[str]] | list[str]]
    ) -> list[list[str]]:
        """Filter paths to ensure diversity.

        Args:
            paths: List of paths (with or without costs)

        Returns:
            Filtered list of diverse paths
        """
        if not paths:
            return []

        # Normalize input (handle both (cost, path) and path formats)
        normalized_paths = []
        for item in paths:
            if isinstance(item, tuple):
                _, path = item
                normalized_paths.append(path)
            else:
                normalized_paths.append(item)

        # Start with first path
        diverse = [normalized_paths[0]]

        # Add paths that are sufficiently different
        for path in normalized_paths[1:]:
            if self._is_diverse(path, diverse):
                diverse.append(path)

            if len(diverse) >= self.k_paths:
                break

        return diverse

    def _is_diverse(self, path: list[str], existing_paths: list[list[str]]) -> bool:
        """Check if a path is diverse enough from existing paths.

        Args:
            path: Path to check
            existing_paths: Existing paths

        Returns:
            True if path is sufficiently diverse
        """
        for existing in existing_paths:
            similarity = self._calculate_path_similarity(path, existing)

            if similarity > (1 - self.min_difference):
                return False

        return True

    def _calculate_path_similarity(self, path1: list[str], path2: list[str]) -> float:
        """Calculate similarity between two paths.

        Args:
            path1: First path
            path2: Second path

        Returns:
            Similarity score (0.0 = completely different, 1.0 = identical)
        """
        # Use Jaccard similarity on state sets
        set1 = set(path1)
        set2 = set(path2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def get_least_explored_path(
        self, start_state: str, end_state: str
    ) -> list[str] | None:
        """Get path with most unexplored transitions.

        Args:
            start_state: Starting state
            end_state: Ending state

        Returns:
            Path with most unexplored transitions, or None
        """
        paths = self.generate_diverse_paths(start_state, end_state)

        if not paths:
            return None

        # Score paths by number of unexplored transitions
        executed = self.tracker._executed_transitions

        scored_paths = []
        for path in paths:
            unexplored_count = 0

            for i in range(len(path) - 1):
                transition_key = (path[i], path[i + 1])
                if transition_key not in executed:
                    unexplored_count += 1

            scored_paths.append((unexplored_count, path))

        # Return path with most unexplored transitions
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return scored_paths[0][1]
