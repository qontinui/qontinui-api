"""Configuration dataclasses for path exploration and testing."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExplorationConfig:
    """Configuration for path exploration strategies.

    This class centralizes all configuration parameters for path exploration,
    making it easy to adjust exploration behavior via YAML/JSON config files.

    Example:
        >>> config = ExplorationConfig(
        ...     strategy="adaptive",
        ...     max_iterations=1000,
        ...     coverage_target=0.95
        ... )
        >>> config.to_dict()
    """

    # Strategy Selection
    strategy: str = "hybrid"
    """Primary exploration strategy: random_walk, greedy, dfs, bfs, adaptive, hybrid."""

    # General Exploration Settings
    max_iterations: int = 1000
    """Maximum number of exploration iterations."""

    max_path_length: int = 100
    """Maximum length of a single path before restart."""

    coverage_target: float = 0.95
    """Target coverage percentage (0.0-1.0) before stopping."""

    enable_backtracking: bool = True
    """Enable intelligent backtracking to reach unexplored states."""

    enable_diversity: bool = True
    """Enable path diversity engine for varied exploration."""

    enable_failure_handling: bool = True
    """Enable failure-aware exploration with retry logic."""

    # Random Walk Settings
    random_walk_temperature: float = 1.0
    """Temperature for random walk (higher = more random)."""

    random_seed: int | None = None
    """Random seed for reproducible exploration (None = random)."""

    # Greedy Coverage Settings
    greedy_unexplored_bonus: float = 2.0
    """Priority bonus for unexplored transitions."""

    greedy_unvisited_state_bonus: float = 1.5
    """Priority bonus for transitions to unvisited states."""

    greedy_unstable_penalty: float = 0.5
    """Priority penalty for unstable transitions."""

    # DFS/BFS Settings
    dfs_max_depth: int = 50
    """Maximum depth for depth-first search."""

    bfs_max_breadth: int = 100
    """Maximum breadth for breadth-first search."""

    # Adaptive (Q-Learning) Settings
    adaptive_learning_rate: float = 0.1
    """Learning rate for Q-learning (alpha)."""

    adaptive_discount_factor: float = 0.9
    """Discount factor for future rewards (gamma)."""

    adaptive_epsilon_start: float = 1.0
    """Initial epsilon for epsilon-greedy exploration."""

    adaptive_epsilon_min: float = 0.01
    """Minimum epsilon value."""

    adaptive_epsilon_decay: float = 0.995
    """Epsilon decay rate per iteration."""

    adaptive_reward_success: float = 10.0
    """Reward for successful transition."""

    adaptive_reward_failure: float = -5.0
    """Penalty for failed transition."""

    adaptive_reward_new_state: float = 20.0
    """Bonus reward for discovering new state."""

    adaptive_reward_new_transition: float = 15.0
    """Bonus reward for executing new transition."""

    # Hybrid Strategy Settings
    hybrid_phase_iterations: list[int] = field(default_factory=lambda: [100, 200, 300])
    """Iteration thresholds for phase transitions."""

    hybrid_phase_strategies: list[str] = field(
        default_factory=lambda: ["random_walk", "greedy", "adaptive"]
    )
    """Strategies for each phase."""

    hybrid_dynamic_switching: bool = True
    """Enable dynamic strategy switching based on coverage progress."""

    hybrid_switch_threshold: float = 0.05
    """Coverage improvement threshold for dynamic switching."""

    # Backtracking Settings
    backtracking_max_attempts: int = 3
    """Maximum backtracking attempts before giving up."""

    backtracking_prefer_shortest: bool = True
    """Prefer shortest path when backtracking."""

    backtracking_timeout_ms: float = 30000.0
    """Maximum time for backtracking path execution (milliseconds)."""

    # Path Diversity Settings
    diversity_k_paths: int = 3
    """Number of diverse paths to generate."""

    diversity_variation_rate: float = 0.3
    """Rate of path variation (0.0-1.0)."""

    diversity_min_difference: float = 0.2
    """Minimum difference ratio between paths."""

    # Failure Handling Settings
    failure_max_retries: int = 3
    """Maximum retry attempts for failed transitions."""

    failure_backoff_base_ms: float = 1000.0
    """Base backoff time in milliseconds."""

    failure_backoff_multiplier: float = 2.0
    """Exponential backoff multiplier."""

    failure_skip_threshold: int = 5
    """Skip transition after this many consecutive failures."""

    failure_cooldown_iterations: int = 10
    """Iterations to wait before retrying skipped transition."""

    # Performance Settings
    performance_threshold_ms: float = 5000.0
    """Threshold for slow transition detection (milliseconds)."""

    stability_threshold: float = 0.95
    """Success rate threshold for transition stability (0.95 = 95%)."""

    min_attempts_for_stability: int = 5
    """Minimum attempts before checking stability."""

    # Screenshot Settings
    enable_screenshots: bool = True
    """Capture screenshots during exploration."""

    screenshot_dir: str = "./screenshots"
    """Directory for screenshot storage."""

    screenshot_on_failure_only: bool = False
    """Only capture screenshots on failures."""

    # Logging Settings
    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR."""

    log_progress_interval: int = 50
    """Log progress every N iterations."""

    # Export Settings
    export_format: str = "json"
    """Default export format: json, html, csv, markdown."""

    export_on_completion: bool = True
    """Automatically export results when exploration completes."""

    export_path: str = "./exploration_results"
    """Path for exported results."""

    # Advanced Settings
    state_timeout_ms: float = 10000.0
    """Timeout for state detection (milliseconds)."""

    transition_timeout_ms: float = 30000.0
    """Timeout for transition execution (milliseconds)."""

    early_stopping: bool = True
    """Stop early if coverage target reached."""

    restart_on_stuck: bool = True
    """Restart from initial state if stuck."""

    stuck_threshold: int = 20
    """Number of iterations without progress before considering stuck."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional custom metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "strategy": self.strategy,
            "max_iterations": self.max_iterations,
            "max_path_length": self.max_path_length,
            "coverage_target": self.coverage_target,
            "enable_backtracking": self.enable_backtracking,
            "enable_diversity": self.enable_diversity,
            "enable_failure_handling": self.enable_failure_handling,
            "random_walk_temperature": self.random_walk_temperature,
            "random_seed": self.random_seed,
            "greedy_unexplored_bonus": self.greedy_unexplored_bonus,
            "greedy_unvisited_state_bonus": self.greedy_unvisited_state_bonus,
            "greedy_unstable_penalty": self.greedy_unstable_penalty,
            "dfs_max_depth": self.dfs_max_depth,
            "bfs_max_breadth": self.bfs_max_breadth,
            "adaptive_learning_rate": self.adaptive_learning_rate,
            "adaptive_discount_factor": self.adaptive_discount_factor,
            "adaptive_epsilon_start": self.adaptive_epsilon_start,
            "adaptive_epsilon_min": self.adaptive_epsilon_min,
            "adaptive_epsilon_decay": self.adaptive_epsilon_decay,
            "adaptive_reward_success": self.adaptive_reward_success,
            "adaptive_reward_failure": self.adaptive_reward_failure,
            "adaptive_reward_new_state": self.adaptive_reward_new_state,
            "adaptive_reward_new_transition": self.adaptive_reward_new_transition,
            "hybrid_phase_iterations": self.hybrid_phase_iterations,
            "hybrid_phase_strategies": self.hybrid_phase_strategies,
            "hybrid_dynamic_switching": self.hybrid_dynamic_switching,
            "hybrid_switch_threshold": self.hybrid_switch_threshold,
            "backtracking_max_attempts": self.backtracking_max_attempts,
            "backtracking_prefer_shortest": self.backtracking_prefer_shortest,
            "backtracking_timeout_ms": self.backtracking_timeout_ms,
            "diversity_k_paths": self.diversity_k_paths,
            "diversity_variation_rate": self.diversity_variation_rate,
            "diversity_min_difference": self.diversity_min_difference,
            "failure_max_retries": self.failure_max_retries,
            "failure_backoff_base_ms": self.failure_backoff_base_ms,
            "failure_backoff_multiplier": self.failure_backoff_multiplier,
            "failure_skip_threshold": self.failure_skip_threshold,
            "failure_cooldown_iterations": self.failure_cooldown_iterations,
            "performance_threshold_ms": self.performance_threshold_ms,
            "stability_threshold": self.stability_threshold,
            "min_attempts_for_stability": self.min_attempts_for_stability,
            "enable_screenshots": self.enable_screenshots,
            "screenshot_dir": self.screenshot_dir,
            "screenshot_on_failure_only": self.screenshot_on_failure_only,
            "log_level": self.log_level,
            "log_progress_interval": self.log_progress_interval,
            "export_format": self.export_format,
            "export_on_completion": self.export_on_completion,
            "export_path": self.export_path,
            "state_timeout_ms": self.state_timeout_ms,
            "transition_timeout_ms": self.transition_timeout_ms,
            "early_stopping": self.early_stopping,
            "restart_on_stuck": self.restart_on_stuck,
            "stuck_threshold": self.stuck_threshold,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExplorationConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            ExplorationConfig instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExplorationConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExplorationConfig instance
        """
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, json_path: str) -> "ExplorationConfig":
        """Load configuration from JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            ExplorationConfig instance
        """
        import json

        with open(json_path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        import yaml

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, json_path: str) -> None:
        """Save configuration to JSON file.

        Args:
            json_path: Path to save JSON file
        """
        import json

        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
