# Path Exploration System

Intelligent path traversal and exploration strategies for comprehensive GUI testing.

## Overview

The Path Exploration System provides sophisticated algorithms for exploring state graphs in GUI applications. It combines multiple exploration strategies, intelligent backtracking, path diversity, and failure handling to achieve maximum test coverage efficiently.

## Architecture

```
app/testing/
├── exploration/
│   ├── __init__.py              # Package exports
│   ├── path_explorer.py         # Main PathExplorer orchestrator
│   ├── strategies.py            # All exploration strategies
│   ├── backtracking.py          # Dijkstra-based backtracking
│   ├── diversity.py             # k-shortest paths & variations
│   ├── failure_handler.py       # Retry logic & failure tracking
│   ├── examples.py              # Usage examples
│   └── README.md                # This file
├── config.py                    # ExplorationConfig dataclass
├── path_tracker.py              # PathTracker (execution tracking)
├── models.py                    # Data models
└── enums.py                     # Enumerations
```

## Key Components

### 1. PathExplorer

The main orchestrator that coordinates all exploration activities.

```python
from app.testing.config import ExplorationConfig
from app.testing.exploration import PathExplorer
from app.testing.path_tracker import PathTracker

# Create configuration
config = ExplorationConfig(
    strategy="hybrid",
    max_iterations=1000,
    coverage_target=0.95
)

# Create tracker and explorer
tracker = PathTracker(state_graph)
explorer = PathExplorer(config, tracker)

# Define execution callback
def execute_transition(from_state, to_state):
    # Execute transition in actual system
    success, duration_ms, metadata = perform_transition(from_state, to_state)
    return success, duration_ms, metadata

# Run exploration
report = explorer.explore(execute_transition)
```

### 2. Exploration Strategies

Six strategies for different exploration scenarios:

#### Random Walk Explorer
- **Use case**: Baseline, discovering unexpected paths
- **Behavior**: Random selection with temperature control
- **Best for**: Initial exploration, stress testing

```python
config = ExplorationConfig(
    strategy="random_walk",
    random_walk_temperature=1.0,
    random_seed=42  # For reproducibility
)
```

#### Greedy Coverage Explorer
- **Use case**: Fast coverage maximization
- **Behavior**: Prioritizes unexplored transitions and states
- **Best for**: Rapid coverage, CI/CD pipelines

```python
config = ExplorationConfig(
    strategy="greedy",
    greedy_unexplored_bonus=2.0,
    greedy_unvisited_state_bonus=1.5
)
```

#### Depth-First Explorer (DFS)
- **Use case**: Finding long execution sequences
- **Behavior**: Explores deeply before backtracking
- **Best for**: Deep state discovery, workflow testing

```python
config = ExplorationConfig(
    strategy="dfs",
    dfs_max_depth=50
)
```

#### Breadth-First Explorer (BFS)
- **Use case**: Finding shortest paths, broad coverage
- **Behavior**: Explores level by level
- **Best for**: Critical path discovery, minimal test sets

```python
config = ExplorationConfig(
    strategy="bfs",
    bfs_max_breadth=100
)
```

#### Adaptive Explorer (Q-Learning)
- **Use case**: Learning optimal exploration policies
- **Behavior**: Reinforcement learning with epsilon-greedy
- **Best for**: Long-running tests, optimization

```python
config = ExplorationConfig(
    strategy="adaptive",
    adaptive_learning_rate=0.1,
    adaptive_discount_factor=0.9,
    adaptive_epsilon_start=1.0,
    adaptive_epsilon_decay=0.995,
    adaptive_reward_new_state=20.0
)
```

#### Hybrid Explorer
- **Use case**: Combining strengths of multiple strategies
- **Behavior**: Phase-based strategy switching
- **Best for**: Comprehensive testing, production use

```python
config = ExplorationConfig(
    strategy="hybrid",
    hybrid_phase_iterations=[100, 200, 300],
    hybrid_phase_strategies=["random_walk", "greedy", "adaptive"],
    hybrid_dynamic_switching=True
)
```

### 3. Backtracking Navigator

Dijkstra-based shortest path finding for reaching unexplored states.

**Features:**
- Finds shortest path to unexplored states
- Cost-weighted edges (reliability, performance)
- Alternative path finding
- Reachability analysis

```python
# Automatically used when enable_backtracking=True
config = ExplorationConfig(
    enable_backtracking=True,
    backtracking_max_attempts=3,
    backtracking_prefer_shortest=True
)
```

### 4. Path Diversity Engine

Generates diverse paths using k-shortest paths and variations.

**Features:**
- Yen's algorithm for k-shortest paths
- Path variation generation
- Diversity filtering
- Least explored path selection

```python
config = ExplorationConfig(
    enable_diversity=True,
    diversity_k_paths=5,
    diversity_variation_rate=0.3,
    diversity_min_difference=0.2
)
```

### 5. Failure-Aware Explorer

Intelligent failure handling with retry logic and cooldowns.

**Features:**
- Exponential backoff retry
- Consecutive failure tracking
- Transition skipping with cooldown
- Reliability scoring

```python
config = ExplorationConfig(
    enable_failure_handling=True,
    failure_max_retries=3,
    failure_backoff_base_ms=1000.0,
    failure_backoff_multiplier=2.0,
    failure_skip_threshold=5,
    failure_cooldown_iterations=10
)
```

## Configuration

### From Code

```python
config = ExplorationConfig(
    strategy="hybrid",
    max_iterations=1000,
    coverage_target=0.95,
    # ... other parameters
)
```

### From YAML

```yaml
# exploration_config.yaml
strategy: hybrid
max_iterations: 1000
coverage_target: 0.95
enable_backtracking: true
enable_diversity: true
enable_failure_handling: true

hybrid_phase_iterations: [100, 200, 300]
hybrid_phase_strategies: [random_walk, greedy, adaptive]

adaptive_learning_rate: 0.1
adaptive_discount_factor: 0.9
```

```python
config = ExplorationConfig.from_yaml("exploration_config.yaml")
```

### From JSON

```json
{
  "strategy": "hybrid",
  "max_iterations": 1000,
  "coverage_target": 0.95,
  "enable_backtracking": true
}
```

```python
config = ExplorationConfig.from_json("exploration_config.json")
```

## Usage Examples

### Basic Exploration

```python
from app.testing.config import ExplorationConfig
from app.testing.exploration import PathExplorer
from app.testing.path_tracker import PathTracker

# Setup
config = ExplorationConfig(strategy="hybrid")
tracker = PathTracker(state_graph)
explorer = PathExplorer(config, tracker)

# Define executor
def execute_transition(from_state, to_state):
    # Your transition execution logic
    success = perform_action(from_state, to_state)
    duration_ms = measure_time()
    metadata = {"action": "click", "element": to_state}
    return success, duration_ms, metadata

# Explore
report = explorer.explore(execute_transition)

# Results
print(f"Coverage: {report['coverage']['transition_coverage_percent']:.1f}%")
print(f"Success Rate: {report['coverage']['success_rate_percent']:.1f}%")
```

### Targeted Path Exploration

```python
# Navigate to specific state
success = explorer.explore_path("settings_page", execute_transition)

if success:
    print("Reached settings page!")
    # Continue exploration from there
```

### With Coverage Callbacks

```python
def on_milestone(metrics, milestone):
    print(f"{milestone}% coverage reached!")
    print(f"States: {metrics.visited_states}/{metrics.total_states}")

tracker.on_coverage_milestone(on_milestone, 50.0)
tracker.on_coverage_milestone(on_milestone, 90.0)

explorer.explore(execute_transition)
```

### With Deficiency Detection

```python
def on_deficiency(deficiency):
    print(f"[{deficiency.severity.value}] {deficiency.title}")
    print(f"  {deficiency.description}")

tracker.on_deficiency_detected(on_deficiency)
```

### Export Results

```python
# Automatic export
config = ExplorationConfig(
    export_on_completion=True,
    export_format="json",
    export_path="./results"
)

# Or manual export
tracker.export_results(
    output_path="./report.html",
    format="html",
    include_screenshots=True
)
```

## Advanced Features

### Dynamic Strategy Switching

The Hybrid strategy can automatically switch strategies based on coverage progress:

```python
config = ExplorationConfig(
    strategy="hybrid",
    hybrid_dynamic_switching=True,
    hybrid_switch_threshold=0.05  # Switch if improvement < 5%
)
```

### Failure Report Generation

```python
if explorer.failure_handler:
    report = explorer.failure_handler.export_failure_report()

    print(f"Total failures: {report['summary']['total_failure_count']}")
    print(f"Skipped transitions: {report['summary']['skipped_transitions']}")

    for transition in report['summary']['most_problematic_transitions']:
        print(f"  {transition['from_state']} -> {transition['to_state']}: "
              f"{transition['failures']} failures")
```

### Real-time Status Monitoring

```python
# During exploration (in another thread)
status = explorer.get_exploration_status()

print(f"Iteration: {status['iteration']}")
print(f"Current state: {status['current_state']}")
print(f"Coverage: {status['coverage_percent']:.1f}%")
```

## Integration with PathTracker

PathExplorer integrates seamlessly with PathTracker:

```python
# After exploration
metrics = tracker.get_coverage_metrics()
deficiencies = tracker.get_deficiencies()
unstable = tracker.get_unstable_transitions()
unexplored = tracker.get_unexplored_transitions()

# Analysis
print(f"State Coverage: {metrics.state_coverage_percent:.1f}%")
print(f"Transition Coverage: {metrics.transition_coverage_percent:.1f}%")
print(f"Deficiencies: {len(deficiencies)}")
print(f"Unstable Transitions: {len(unstable)}")
```

## Performance Considerations

### Optimal Settings for Different Scenarios

**Quick Smoke Test (5-10 minutes):**
```python
config = ExplorationConfig(
    strategy="greedy",
    max_iterations=200,
    coverage_target=0.70,
    enable_backtracking=True,
    enable_failure_handling=True
)
```

**Comprehensive Test (30-60 minutes):**
```python
config = ExplorationConfig(
    strategy="hybrid",
    max_iterations=1000,
    coverage_target=0.95,
    enable_backtracking=True,
    enable_diversity=True,
    enable_failure_handling=True
)
```

**Long-running Optimization (hours):**
```python
config = ExplorationConfig(
    strategy="adaptive",
    max_iterations=5000,
    coverage_target=0.99,
    adaptive_learning_rate=0.05,  # Lower for stability
    enable_backtracking=True,
    enable_diversity=True,
    enable_failure_handling=True
)
```

## Troubleshooting

### Explorer Gets Stuck

**Solution 1**: Enable restart on stuck
```python
config.restart_on_stuck = True
config.stuck_threshold = 20
```

**Solution 2**: Enable backtracking
```python
config.enable_backtracking = True
```

### Low Coverage

**Solution**: Use hybrid strategy with aggressive exploration
```python
config.strategy = "hybrid"
config.greedy_unexplored_bonus = 3.0
config.enable_diversity = True
```

### Flaky Transitions

**Solution**: Increase retry attempts and backoff
```python
config.failure_max_retries = 5
config.failure_backoff_base_ms = 2000.0
config.failure_skip_threshold = 10
```

### Memory Issues (Large State Graphs)

**Solution**: Limit history size
```python
tracker = PathTracker(
    state_graph,
    max_history_size=5000  # Default: 10000
)
```

## Best Practices

1. **Start with Greedy**: Use greedy strategy for initial runs to quickly understand coverage
2. **Use Hybrid for Production**: Hybrid strategy combines benefits of multiple approaches
3. **Enable All Features**: Enable backtracking, diversity, and failure handling for comprehensive testing
4. **Monitor Progress**: Use callbacks to track progress and detect issues early
5. **Export Results**: Always export results for analysis and reporting
6. **Tune Parameters**: Adjust strategy parameters based on your application's behavior
7. **Handle Failures**: Configure appropriate retry and backoff settings for flaky UIs
8. **Set Coverage Targets**: Use realistic coverage targets (90-95% is often achievable)

## API Reference

See individual module docstrings for detailed API documentation:

- `path_explorer.py`: PathExplorer class
- `strategies.py`: All exploration strategies
- `backtracking.py`: BacktrackingNavigator
- `diversity.py`: PathDiversityEngine
- `failure_handler.py`: FailureAwareExplorer
- `../config.py`: ExplorationConfig

## Contributing

When adding new exploration strategies:

1. Extend `ExplorationStrategy` base class
2. Implement `select_next_state(current_state) -> str | None`
3. Add strategy to `PathExplorer._create_strategy()` dictionary
4. Add configuration parameters to `ExplorationConfig`
5. Update documentation and examples

## License

Part of the Qontinui API project.
