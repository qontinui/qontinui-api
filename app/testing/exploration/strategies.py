"""Exploration strategy implementations for path discovery.

This module provides various strategies for exploring state graphs,
from simple random walks to sophisticated reinforcement learning approaches.
"""

import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any

import numpy as np

from app.testing.config import ExplorationConfig
from app.testing.path_tracker import PathTracker

logger = logging.getLogger(__name__)


class ExplorationStrategy(ABC):
    """Base class for exploration strategies.

    All exploration strategies must implement the select_next_state method
    to determine which state to transition to next.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize exploration strategy.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance for metrics and history
        """
        self.config = config
        self.tracker = tracker
        self.state_graph = tracker.state_graph

    @abstractmethod
    def select_next_state(self, current_state: str) -> str | None:
        """Select the next state to transition to.

        Args:
            current_state: Current state name

        Returns:
            Next state name, or None if no valid transition available
        """
        pass

    def get_available_transitions(self, state_name: str) -> list[tuple[str, str]]:
        """Get available transitions from a state.

        Args:
            state_name: State to get transitions from

        Returns:
            List of (from_state, to_state) tuples
        """
        transitions = []

        if not hasattr(self.state_graph, "states"):
            return transitions

        state = self.state_graph.states.get(state_name)
        if not state or not hasattr(state, "transitions"):
            return transitions

        for transition in state.transitions:
            to_state = getattr(transition, "to_state", None)
            if to_state:
                transitions.append((state_name, to_state))

        return transitions

    def reset(self) -> None:
        """Reset strategy state (for stateful strategies)."""
        pass


class RandomWalkExplorer(ExplorationStrategy):
    """Random walk exploration strategy.

    Selects next state randomly from available transitions.
    Useful for baseline comparison and discovering unexpected paths.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize random walk explorer.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance
        """
        super().__init__(config, tracker)

        # Set random seed if provided
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        self.temperature = config.random_walk_temperature

    def select_next_state(self, current_state: str) -> str | None:
        """Select next state randomly.

        Args:
            current_state: Current state name

        Returns:
            Random next state, or None if no transitions available
        """
        transitions = self.get_available_transitions(current_state)

        if not transitions:
            logger.debug(f"No transitions available from {current_state}")
            return None

        # Apply temperature for exploration control
        if self.temperature == 1.0:
            # Uniform random selection
            _, next_state = random.choice(transitions)
        else:
            # Temperature-weighted selection
            # Higher temperature = more uniform
            # Lower temperature = more concentrated
            weights = np.ones(len(transitions))
            weights = np.exp(weights / self.temperature)
            weights /= weights.sum()

            idx = np.random.choice(len(transitions), p=weights)
            _, next_state = transitions[idx]

        logger.debug(f"Random walk: {current_state} -> {next_state}")
        return next_state


class GreedyCoverageExplorer(ExplorationStrategy):
    """Greedy coverage-maximizing exploration strategy.

    Prioritizes unexplored transitions and states to maximize coverage quickly.
    Uses heuristics to score transitions based on exploration value.
    """

    def select_next_state(self, current_state: str) -> str | None:
        """Select next state greedily to maximize coverage.

        Args:
            current_state: Current state name

        Returns:
            Best next state for coverage, or None if no transitions available
        """
        transitions = self.get_available_transitions(current_state)

        if not transitions:
            return None

        # Score each transition
        scored_transitions = []
        executed_transitions = self.tracker._executed_transitions
        visited_states = self.tracker._visited_states
        transition_stats = self.tracker._transition_stats

        for from_state, to_state in transitions:
            score = 1.0
            transition_key = (from_state, to_state)

            # Bonus for unexplored transitions
            if transition_key not in executed_transitions:
                score *= self.config.greedy_unexplored_bonus

            # Bonus for unvisited states
            if to_state not in visited_states:
                score *= self.config.greedy_unvisited_state_bonus

            # Penalty for unstable transitions
            if transition_key in transition_stats:
                stats = transition_stats[transition_key]
                if stats.is_unreliable:
                    score *= self.config.greedy_unstable_penalty

            scored_transitions.append((to_state, score))

        # Select highest scoring transition
        scored_transitions.sort(key=lambda x: x[1], reverse=True)
        next_state, score = scored_transitions[0]

        logger.debug(f"Greedy: {current_state} -> {next_state} (score: {score:.2f})")
        return next_state


class DepthFirstExplorer(ExplorationStrategy):
    """Depth-first search exploration strategy.

    Explores paths deeply before backtracking, useful for finding
    long execution sequences and deep states.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize DFS explorer.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance
        """
        super().__init__(config, tracker)
        self.visited_stack: deque[str] = deque()
        self.depth = 0
        self.max_depth = config.dfs_max_depth

    def select_next_state(self, current_state: str) -> str | None:
        """Select next state using DFS.

        Args:
            current_state: Current state name

        Returns:
            Next state via DFS, or None if max depth reached
        """
        # Check depth limit
        if self.depth >= self.max_depth:
            logger.debug(f"DFS max depth {self.max_depth} reached, backtracking")
            return self._backtrack()

        transitions = self.get_available_transitions(current_state)

        if not transitions:
            return self._backtrack()

        # Prioritize unexplored transitions
        executed_transitions = self.tracker._executed_transitions
        unexplored = [
            (f, t) for f, t in transitions if (f, t) not in executed_transitions
        ]

        if unexplored:
            _, next_state = unexplored[0]
        else:
            # All explored, take first available
            _, next_state = transitions[0]

        # Update stack and depth
        self.visited_stack.append(current_state)
        self.depth += 1

        logger.debug(f"DFS: {current_state} -> {next_state} (depth: {self.depth})")
        return next_state

    def _backtrack(self) -> str | None:
        """Backtrack to previous state.

        Returns:
            Previous state, or None if stack is empty
        """
        if self.visited_stack:
            prev_state = self.visited_stack.pop()
            self.depth = len(self.visited_stack)
            logger.debug(f"DFS backtrack to {prev_state}")
            return prev_state

        return None

    def reset(self) -> None:
        """Reset DFS state."""
        self.visited_stack.clear()
        self.depth = 0


class BreadthFirstExplorer(ExplorationStrategy):
    """Breadth-first search exploration strategy.

    Explores states level by level, useful for finding shortest paths
    and ensuring broad coverage before going deep.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize BFS explorer.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance
        """
        super().__init__(config, tracker)
        self.queue: deque[str] = deque()
        self.visited_this_run: set[str] = set()
        self.max_breadth = config.bfs_max_breadth

    def select_next_state(self, current_state: str) -> str | None:
        """Select next state using BFS.

        Args:
            current_state: Current state name

        Returns:
            Next state via BFS, or None if no more states
        """
        # Add current state to visited
        self.visited_this_run.add(current_state)

        # Get all transitions from current state
        transitions = self.get_available_transitions(current_state)

        # Add unvisited states to queue
        for _, to_state in transitions:
            if to_state not in self.visited_this_run and to_state not in self.queue:
                if len(self.queue) < self.max_breadth:
                    self.queue.append(to_state)

        # Get next state from queue
        if self.queue:
            next_state = self.queue.popleft()
            logger.debug(f"BFS: {current_state} -> {next_state} (queue: {len(self.queue)})")
            return next_state

        logger.debug("BFS queue exhausted")
        return None

    def reset(self) -> None:
        """Reset BFS state."""
        self.queue.clear()
        self.visited_this_run.clear()


class AdaptiveExplorer(ExplorationStrategy):
    """Adaptive exploration using Q-learning.

    Learns optimal transition policies through reinforcement learning,
    balancing exploration and exploitation using epsilon-greedy strategy.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize adaptive explorer.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance
        """
        super().__init__(config, tracker)

        # Q-learning parameters
        self.learning_rate = config.adaptive_learning_rate
        self.discount_factor = config.adaptive_discount_factor
        self.epsilon = config.adaptive_epsilon_start
        self.epsilon_min = config.adaptive_epsilon_min
        self.epsilon_decay = config.adaptive_epsilon_decay

        # Rewards
        self.reward_success = config.adaptive_reward_success
        self.reward_failure = config.adaptive_reward_failure
        self.reward_new_state = config.adaptive_reward_new_state
        self.reward_new_transition = config.adaptive_reward_new_transition

        # Q-table: (state, action) -> Q-value
        self.q_table: dict[tuple[str, str], float] = defaultdict(float)

        # History for learning
        self.last_state: str | None = None
        self.last_action: str | None = None

    def select_next_state(self, current_state: str) -> str | None:
        """Select next state using Q-learning with epsilon-greedy.

        Args:
            current_state: Current state name

        Returns:
            Next state based on Q-values, or None if no transitions
        """
        transitions = self.get_available_transitions(current_state)

        if not transitions:
            return None

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Explore: random action
            _, next_state = random.choice(transitions)
            logger.debug(f"Q-learning explore: {current_state} -> {next_state} (ε={self.epsilon:.3f})")
        else:
            # Exploit: best Q-value
            q_values = [
                (to_state, self.q_table[(current_state, to_state)])
                for _, to_state in transitions
            ]
            q_values.sort(key=lambda x: x[1], reverse=True)
            next_state, q_val = q_values[0]
            logger.debug(f"Q-learning exploit: {current_state} -> {next_state} (Q={q_val:.2f})")

        # Store for learning update
        self.last_state = current_state
        self.last_action = next_state

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return next_state

    def update_q_value(self, success: bool, new_state_discovered: bool = False,
                       new_transition_discovered: bool = False) -> None:
        """Update Q-value based on outcome.

        Args:
            success: Whether transition succeeded
            new_state_discovered: Whether a new state was discovered
            new_transition_discovered: Whether a new transition was executed
        """
        if self.last_state is None or self.last_action is None:
            return

        # Calculate reward
        reward = self.reward_success if success else self.reward_failure

        if new_state_discovered:
            reward += self.reward_new_state

        if new_transition_discovered:
            reward += self.reward_new_transition

        # Get current Q-value
        state_action = (self.last_state, self.last_action)
        current_q = self.q_table[state_action]

        # Get max Q-value for next state
        next_transitions = self.get_available_transitions(self.last_action)
        if next_transitions:
            max_next_q = max(
                self.q_table[(self.last_action, to_state)]
                for _, to_state in next_transitions
            )
        else:
            max_next_q = 0.0

        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_action] = new_q

        logger.debug(
            f"Q-update: ({self.last_state}, {self.last_action}) "
            f"Q: {current_q:.2f} -> {new_q:.2f} (r={reward:.1f})"
        )

    def reset(self) -> None:
        """Reset learning state (keep Q-table for transfer learning)."""
        self.last_state = None
        self.last_action = None


class HybridExplorer(ExplorationStrategy):
    """Hybrid exploration strategy combining multiple approaches.

    Uses different strategies in phases or dynamically switches based
    on coverage progress. Combines strengths of multiple strategies.
    """

    def __init__(self, config: ExplorationConfig, tracker: PathTracker):
        """Initialize hybrid explorer.

        Args:
            config: Exploration configuration
            tracker: PathTracker instance
        """
        super().__init__(config, tracker)

        # Phase configuration
        self.phase_iterations = config.hybrid_phase_iterations
        self.phase_strategies = config.hybrid_phase_strategies
        self.dynamic_switching = config.hybrid_dynamic_switching
        self.switch_threshold = config.hybrid_switch_threshold

        # Current state
        self.iteration = 0
        self.current_phase = 0
        self.last_coverage = 0.0
        self.stagnation_count = 0

        # Initialize sub-strategies
        self.strategies: dict[str, ExplorationStrategy] = {
            "random_walk": RandomWalkExplorer(config, tracker),
            "greedy": GreedyCoverageExplorer(config, tracker),
            "dfs": DepthFirstExplorer(config, tracker),
            "bfs": BreadthFirstExplorer(config, tracker),
            "adaptive": AdaptiveExplorer(config, tracker),
        }

        # Set initial strategy
        self.current_strategy_name = self.phase_strategies[0]
        self.current_strategy = self.strategies[self.current_strategy_name]

        logger.info(f"Hybrid explorer initialized with phases: {self.phase_strategies}")

    def select_next_state(self, current_state: str) -> str | None:
        """Select next state using current strategy.

        Args:
            current_state: Current state name

        Returns:
            Next state from current strategy
        """
        # Update phase if needed
        self._update_phase()

        # Select using current strategy
        next_state = self.current_strategy.select_next_state(current_state)

        self.iteration += 1

        return next_state

    def _update_phase(self) -> None:
        """Update current phase/strategy based on iteration or coverage."""
        # Check if phase transition needed
        if self.current_phase < len(self.phase_iterations):
            if self.iteration >= self.phase_iterations[self.current_phase]:
                self._transition_to_next_phase()
                return

        # Dynamic switching based on coverage
        if self.dynamic_switching:
            metrics = self.tracker.get_coverage_metrics()
            current_coverage = metrics.transition_coverage_percent

            # Check if coverage is stagnating
            improvement = current_coverage - self.last_coverage

            if improvement < self.switch_threshold:
                self.stagnation_count += 1

                # Switch strategy if stagnating
                if self.stagnation_count >= 10:
                    self._switch_strategy_dynamically()
                    self.stagnation_count = 0
            else:
                self.stagnation_count = 0

            self.last_coverage = current_coverage

    def _transition_to_next_phase(self) -> None:
        """Transition to next phase."""
        self.current_phase += 1

        if self.current_phase < len(self.phase_strategies):
            new_strategy_name = self.phase_strategies[self.current_phase]
            self._switch_strategy(new_strategy_name)

            logger.info(
                f"Phase transition: {self.current_strategy_name} -> {new_strategy_name} "
                f"(iteration: {self.iteration})"
            )

    def _switch_strategy(self, strategy_name: str) -> None:
        """Switch to a different strategy.

        Args:
            strategy_name: Name of strategy to switch to
        """
        if strategy_name in self.strategies:
            self.current_strategy_name = strategy_name
            self.current_strategy = self.strategies[strategy_name]

    def _switch_strategy_dynamically(self) -> None:
        """Dynamically switch to a different strategy."""
        # Try strategies in order, skip current one
        available = [s for s in self.strategies.keys() if s != self.current_strategy_name]

        if available:
            new_strategy = available[0]
            self._switch_strategy(new_strategy)

            logger.info(
                f"Dynamic switch: {self.current_strategy_name} -> {new_strategy} "
                f"(coverage stagnation detected)"
            )

    def reset(self) -> None:
        """Reset hybrid explorer state."""
        self.iteration = 0
        self.current_phase = 0
        self.last_coverage = 0.0
        self.stagnation_count = 0

        # Reset all sub-strategies
        for strategy in self.strategies.values():
            strategy.reset()

        # Reset to initial strategy
        self.current_strategy_name = self.phase_strategies[0]
        self.current_strategy = self.strategies[self.current_strategy_name]
