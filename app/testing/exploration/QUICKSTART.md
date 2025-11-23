# Path Explorer Quick Start Guide

Get started with intelligent path exploration in 5 minutes.

## Installation

The path exploration system is already installed as part of qontinui-api. No additional dependencies required beyond the standard qontinui-api requirements.

## Basic Usage

### Step 1: Import Components

```python
from app.testing.config import ExplorationConfig
from app.testing.exploration import PathExplorer
from app.testing.path_tracker import PathTracker
```

### Step 2: Create Configuration

```python
# Use default hybrid strategy (recommended)
config = ExplorationConfig(
    strategy="hybrid",
    max_iterations=1000,
    coverage_target=0.95
)

# Or customize as needed
config = ExplorationConfig(
    strategy="greedy",           # Strategy: random_walk, greedy, dfs, bfs, adaptive, hybrid
    max_iterations=500,           # Stop after 500 iterations
    coverage_target=0.90,         # Or when 90% coverage reached
    enable_backtracking=True,     # Enable intelligent backtracking
    enable_diversity=True,        # Enable diverse path generation
    enable_failure_handling=True  # Enable retry logic
)
```

### Step 3: Create PathTracker

```python
# Assuming you have a state_graph object
tracker = PathTracker(
    state_graph=your_state_graph,
    enable_screenshots=True,
    screenshot_dir="./screenshots"
)
```

### Step 4: Create PathExplorer

```python
explorer = PathExplorer(
    config=config,
    tracker=tracker,
    initial_state="login"  # Your starting state
)
```

### Step 5: Define Executor Callback

```python
def execute_transition(from_state: str, to_state: str) -> tuple[bool, float, dict]:
    """
    Execute a transition in your application.

    This is where you implement your actual automation logic.

    Args:
        from_state: Source state name
        to_state: Target state name

    Returns:
        Tuple of:
        - success (bool): Whether transition succeeded
        - duration_ms (float): Time taken in milliseconds
        - metadata (dict): Additional information
    """
    try:
        # Your automation code here
        # Example: Click button, navigate, etc.

        # Simulate transition
        success = my_automation.navigate(from_state, to_state)
        duration_ms = my_automation.get_last_duration()

        metadata = {
            "action": "navigation",
            "element": to_state,
            "timestamp": time.time()
        }

        return success, duration_ms, metadata

    except Exception as e:
        # Return failure on exception
        return False, 0.0, {"error": str(e)}
```

### Step 6: Run Exploration

```python
# Run exploration
report = explorer.explore(execute_transition)

# Access results
print(f"Coverage: {report['coverage']['transition_coverage_percent']:.1f}%")
print(f"Success Rate: {report['coverage']['success_rate_percent']:.1f}%")
print(f"Total Iterations: {report['summary']['iterations']}")
print(f"Deficiencies Found: {report['deficiencies']['total']}")
```

## Complete Example

```python
from app.testing.config import ExplorationConfig
from app.testing.exploration import PathExplorer
from app.testing.path_tracker import PathTracker
import time

# 1. Configure exploration
config = ExplorationConfig(
    strategy="hybrid",
    max_iterations=1000,
    coverage_target=0.95,
    enable_backtracking=True,
    enable_diversity=True,
    enable_failure_handling=True,
    export_on_completion=True,
    export_path="./results"
)

# 2. Create tracker (assumes you have state_graph)
tracker = PathTracker(state_graph)

# 3. Create explorer
explorer = PathExplorer(config, tracker, initial_state="login")

# 4. Define executor
def execute_transition(from_state, to_state):
    # Your automation logic
    success = perform_action(from_state, to_state)
    duration = measure_execution_time()
    metadata = {"action": "click", "element": to_state}

    return success, duration, metadata

# 5. Run exploration
report = explorer.explore(execute_transition)

# 6. View results
print("\n=== Exploration Results ===")
print(f"Strategy: {report['summary']['strategy']}")
print(f"Iterations: {report['summary']['iterations']}")
print(f"State Coverage: {report['coverage']['state_coverage_percent']:.1f}%")
print(f"Transition Coverage: {report['coverage']['transition_coverage_percent']:.1f}%")
print(f"Success Rate: {report['coverage']['success_rate_percent']:.1f}%")
print(f"Deficiencies: {report['deficiencies']['total']}")
```

## Common Patterns

### Pattern 1: Quick Coverage Check

```python
config = ExplorationConfig(
    strategy="greedy",
    max_iterations=200,
    coverage_target=0.70
)

explorer = PathExplorer(config, tracker)
report = explorer.explore(execute_transition)
```

### Pattern 2: Comprehensive Testing

```python
config = ExplorationConfig(
    strategy="hybrid",
    max_iterations=1000,
    coverage_target=0.95,
    enable_backtracking=True,
    enable_diversity=True,
    enable_failure_handling=True
)

explorer = PathExplorer(config, tracker)
report = explorer.explore(execute_transition)
```

### Pattern 3: Adaptive Learning

```python
config = ExplorationConfig(
    strategy="adaptive",
    max_iterations=2000,
    adaptive_learning_rate=0.1,
    adaptive_reward_new_state=50.0
)

explorer = PathExplorer(config, tracker)
report = explorer.explore(execute_transition)
```

### Pattern 4: With Coverage Callbacks

```python
# Setup callbacks
def on_milestone(metrics, milestone):
    print(f"{milestone}% coverage reached!")

tracker.on_coverage_milestone(on_milestone, 50.0)
tracker.on_coverage_milestone(on_milestone, 90.0)

# Run exploration
report = explorer.explore(execute_transition)
```

### Pattern 5: Navigate to Specific State

```python
# Start exploration
explorer = PathExplorer(config, tracker, initial_state="login")

# Navigate to specific state
success = explorer.explore_path("settings", execute_transition)

if success:
    print("Reached settings!")
    # Continue exploration from settings
    report = explorer.explore(execute_transition)
```

## Configuration Presets

### Preset 1: Fast Smoke Test (5 minutes)

```python
config = ExplorationConfig(
    strategy="greedy",
    max_iterations=200,
    coverage_target=0.70,
    enable_backtracking=True,
    enable_failure_handling=True
)
```

### Preset 2: Standard Test (30 minutes)

```python
config = ExplorationConfig(
    strategy="hybrid",
    max_iterations=1000,
    coverage_target=0.90,
    enable_backtracking=True,
    enable_diversity=True,
    enable_failure_handling=True,
    hybrid_phase_iterations=[100, 300, 600],
    hybrid_phase_strategies=["random_walk", "greedy", "adaptive"]
)
```

### Preset 3: Comprehensive Test (1-2 hours)

```python
config = ExplorationConfig(
    strategy="hybrid",
    max_iterations=3000,
    coverage_target=0.95,
    enable_backtracking=True,
    enable_diversity=True,
    enable_failure_handling=True,
    diversity_k_paths=5,
    failure_max_retries=5
)
```

### Preset 4: Nightly Full Coverage

```python
config = ExplorationConfig(
    strategy="adaptive",
    max_iterations=10000,
    coverage_target=0.99,
    adaptive_learning_rate=0.05,
    enable_backtracking=True,
    enable_diversity=True,
    enable_failure_handling=True
)
```

## Using YAML Configuration

### Create Config File

```yaml
# exploration_config.yaml
strategy: hybrid
max_iterations: 1000
coverage_target: 0.95
enable_backtracking: true
enable_diversity: true
enable_failure_handling: true

hybrid_phase_iterations: [100, 300, 600]
hybrid_phase_strategies: [random_walk, greedy, adaptive]

failure_max_retries: 3
failure_backoff_base_ms: 1000.0

export_on_completion: true
export_format: json
export_path: ./results
```

### Load and Use

```python
from app.testing.config import ExplorationConfig

# Load configuration
config = ExplorationConfig.from_yaml("exploration_config.yaml")

# Use normally
explorer = PathExplorer(config, tracker)
report = explorer.explore(execute_transition)
```

## Accessing Results

### Coverage Metrics

```python
metrics = tracker.get_coverage_metrics()

print(f"State Coverage: {metrics.state_coverage_percent:.1f}%")
print(f"Transition Coverage: {metrics.transition_coverage_percent:.1f}%")
print(f"Success Rate: {metrics.success_rate_percent:.1f}%")
print(f"Total Executions: {metrics.total_executions}")
print(f"Successful: {metrics.successful_executions}")
print(f"Failed: {metrics.failed_executions}")
print(f"Unique Paths: {metrics.unique_paths}")
```

### Deficiencies

```python
deficiencies = tracker.get_deficiencies()

print(f"Total Deficiencies: {len(deficiencies)}")

for deficiency in deficiencies:
    print(f"[{deficiency.severity.value}] {deficiency.title}")
    print(f"  {deficiency.description}")
    print(f"  Occurrences: {deficiency.occurrence_count}")
```

### Unstable Transitions

```python
unstable = tracker.get_unstable_transitions()

for stats in unstable:
    print(f"{stats.from_state} -> {stats.to_state}")
    print(f"  Success Rate: {stats.success_rate:.1f}%")
    print(f"  Attempts: {stats.total_attempts}")
```

### Export Results

```python
# Export as JSON
tracker.export_results(
    output_path="./results.json",
    format="json"
)

# Export as HTML report
tracker.export_results(
    output_path="./report.html",
    format="html"
)

# Export as CSV
tracker.export_results(
    output_path="./data.csv",
    format="csv"
)
```

## Troubleshooting

### Issue: Explorer gets stuck

**Solution:** Enable backtracking and restart on stuck
```python
config.enable_backtracking = True
config.restart_on_stuck = True
config.stuck_threshold = 20
```

### Issue: Low coverage

**Solution:** Use hybrid strategy with diversity
```python
config.strategy = "hybrid"
config.enable_diversity = True
config.greedy_unexplored_bonus = 3.0
```

### Issue: Many transition failures

**Solution:** Increase retry attempts and backoff
```python
config.failure_max_retries = 5
config.failure_backoff_base_ms = 2000.0
config.failure_skip_threshold = 10
```

### Issue: Exploration takes too long

**Solution:** Reduce iterations or increase coverage target
```python
config.max_iterations = 500  # Reduce from 1000
config.coverage_target = 0.85  # Lower from 0.95
config.early_stopping = True  # Stop when target reached
```

## Next Steps

1. **Read the full documentation**: `README.md`
2. **See examples**: `examples.py`
3. **Check implementation details**: `IMPLEMENTATION_SUMMARY.md`
4. **Customize configuration**: Adjust parameters in `ExplorationConfig`
5. **Implement your executor**: Create robust transition execution logic
6. **Monitor and iterate**: Use callbacks and logging to track progress

## Support

For issues or questions:
1. Check the main README.md
2. Review examples.py for usage patterns
3. See IMPLEMENTATION_SUMMARY.md for technical details
4. Review inline documentation (docstrings)

## Key Files

- `path_explorer.py` - Main orchestrator
- `strategies.py` - All 6 exploration strategies
- `config.py` - Configuration system
- `backtracking.py` - Intelligent navigation
- `diversity.py` - Path diversity
- `failure_handler.py` - Failure handling
- `examples.py` - Usage examples
- `README.md` - Full documentation
