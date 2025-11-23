# PathTracker Testing Module

## Overview

The PathTracker module provides comprehensive state/transition execution tracking for testing GUI applications with qontinui. It monitors which states and transitions have been visited, calculates coverage metrics, identifies deficiencies, and generates test reports.

## Features

- **Coverage Tracking**: Monitor state and transition coverage during test execution
- **Deficiency Detection**: Automatically detect bugs, performance issues, and unstable transitions
- **Performance Analysis**: Track execution times and identify slow transitions
- **Thread Safety**: All methods are thread-safe for concurrent test execution
- **Export Formats**: JSON, HTML, CSV, and Markdown report generation
- **Path History**: Track complete paths through the state graph
- **Callbacks**: Real-time notifications for deficiencies and coverage milestones

## Installation

The PathTracker module is included in qontinui-api. All dependencies are already installed via the main project dependencies.

## Quick Start

```python
from app.testing import PathTracker, ExecutionStatus, CoverageMetrics

# Initialize PathTracker with your state graph
tracker = PathTracker(
    state_graph=your_state_graph,
    enable_screenshots=True,
    screenshot_dir="./test_screenshots"
)

# Record transitions during test execution
tracker.record_transition(
    from_state="login",
    to_state="dashboard",
    success=True,
    duration_ms=1250.5
)

# Get coverage metrics
metrics = tracker.get_coverage_metrics()
print(f"State Coverage: {metrics.state_coverage_percent:.1f}%")
print(f"Transition Coverage: {metrics.transition_coverage_percent:.1f}%")
print(f"Success Rate: {metrics.success_rate_percent:.1f}%")

# Get deficiencies
deficiencies = tracker.get_deficiencies()
for deficiency in deficiencies:
    print(f"[{deficiency.severity.value}] {deficiency.title}")

# Export results
tracker.export_results("test_report.json", format="json")
tracker.export_results("test_report.html", format="html")
```

## File Structure

```
app/testing/
├── __init__.py                 # Public API exports
├── enums.py                    # Enumerations (ExecutionStatus, DeficiencySeverity, etc.)
├── models.py                   # Data models (TransitionExecution, CoverageMetrics, etc.)
├── deficiency_detector.py      # Deficiency detection logic
├── path_tracker.py             # Main PathTracker class
└── README.md                   # This file
```

## Core Components

### Enums

- **ExecutionStatus**: Status of a transition execution (SUCCESS, FAILURE, ERROR, SKIPPED, PARTIAL)
- **DeficiencySeverity**: Severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO)
- **DeficiencyCategory**: Type of deficiency (UNSTABLE_TRANSITION, SLOW_TRANSITION, TIMEOUT, etc.)

### Data Models

- **TransitionExecution**: Record of a single transition attempt
- **CoverageMetrics**: Complete coverage analysis with percentages
- **Deficiency**: Represents a bug or issue found during testing
- **PathHistory**: Sequential path through the state graph
- **TransitionStatistics**: Aggregated statistics for a transition

### Main Class

- **PathTracker**: Main class for tracking and analysis

## Usage Examples

### Example 1: Basic Tracking

```python
from app.testing import PathTracker

# Initialize
tracker = PathTracker(state_graph=graph)

# Record successful transition
tracker.record_transition(
    from_state="idle",
    to_state="processing",
    success=True,
    duration_ms=150.0
)

# Record failed transition
tracker.record_transition(
    from_state="processing",
    to_state="complete",
    success=False,
    error_message="Timeout waiting for completion",
    duration_ms=5000.0
)

# Get metrics
metrics = tracker.get_coverage_metrics()
print(f"Executed {metrics.total_executions} transitions")
print(f"Coverage: {metrics.transition_coverage_percent:.1f}%")
```

### Example 2: Deficiency Detection

```python
# Get all deficiencies
deficiencies = tracker.get_deficiencies()

# Filter by severity
critical_issues = tracker.get_deficiencies(severity=DeficiencySeverity.CRITICAL)

# Filter by category
slow_transitions = tracker.get_deficiencies(category=DeficiencyCategory.SLOW_TRANSITION)

# Print deficiency details
for deficiency in deficiencies:
    print(f"\n[{deficiency.severity.value}] {deficiency.title}")
    print(f"Category: {deficiency.category.value}")
    print(f"Description: {deficiency.description}")
    print(f"Occurrences: {deficiency.occurrence_count}")
    print(f"First observed: {deficiency.first_observed}")
```

### Example 3: Coverage-Guided Testing

```python
def run_coverage_guided_test(tracker, max_iterations=100):
    """Run test guided by coverage metrics."""
    current_state = tracker.state_graph.initial_state

    for i in range(max_iterations):
        # Check coverage
        metrics = tracker.get_coverage_metrics()
        if metrics.is_complete_coverage:
            print("Complete coverage achieved!")
            break

        # Get suggestions for next transition
        suggestions = tracker.suggest_next_transitions(
            current_state,
            prioritize_unexplored=True
        )

        if not suggestions:
            break

        # Try highest priority transition
        next_state, priority = suggestions[0]
        success = execute_transition(current_state, next_state)

        # Record result
        tracker.record_transition(
            from_state=current_state,
            to_state=next_state,
            success=success
        )

        if success:
            current_state = next_state

# Run the test
run_coverage_guided_test(tracker)
```

### Example 4: Real-time Monitoring

```python
# Register callbacks for real-time notifications
def on_critical_deficiency(deficiency):
    """Handle critical deficiencies immediately."""
    if deficiency.is_critical:
        send_alert(f"CRITICAL: {deficiency.title}")

def on_milestone(metrics, milestone):
    """Celebrate coverage milestones."""
    print(f"Achieved {milestone}% coverage!")

# Register callbacks
tracker.on_deficiency_detected(on_critical_deficiency)
tracker.on_coverage_milestone(on_milestone, 50.0)
tracker.on_coverage_milestone(on_milestone, 90.0)

# Callbacks fire automatically as tests run
run_tests(tracker)
```

### Example 5: Performance Analysis

```python
# Get transition statistics
stats = tracker.get_transition_statistics()

# Find slow transitions
slow_threshold_ms = 2000
slow_transitions = [
    s for s in stats
    if s.avg_duration_ms > slow_threshold_ms
]

for stat in slow_transitions:
    print(f"\n{stat.from_state} -> {stat.to_state}")
    print(f"  Average: {stat.avg_duration_ms:.0f}ms")
    print(f"  Min: {stat.min_duration_ms:.0f}ms")
    print(f"  Max: {stat.max_duration_ms:.0f}ms")
    print(f"  Success rate: {stat.success_rate:.1f}%")

# Find unstable transitions
unstable = tracker.get_unstable_transitions(min_threshold=0.95)
for stat in unstable:
    print(f"\n{stat.from_state} -> {stat.to_state}")
    print(f"  Success rate: {stat.success_rate:.1f}%")
    print(f"  Attempts: {stat.total_attempts}")
```

## API Reference

### PathTracker

#### Constructor

```python
PathTracker(
    state_graph: Any,
    enable_screenshots: bool = True,
    screenshot_dir: str = "./screenshots",
    max_history_size: int = 10000,
    performance_threshold_ms: float = 5000.0,
    stability_threshold: float = 0.95
)
```

#### Main Methods

- `record_transition()`: Record a transition execution
- `get_coverage_metrics()`: Get current coverage metrics
- `get_unexplored_transitions()`: Get unexecuted transitions
- `get_unstable_transitions()`: Get transitions with low success rates
- `get_deficiencies()`: Get detected deficiencies
- `get_path_history()`: Get path traversal history
- `get_transition_statistics()`: Get detailed transition statistics
- `export_results()`: Export results to file
- `reset()`: Clear all tracking data

#### Analysis Methods

- `analyze_reachability()`: Analyze state reachability
- `suggest_next_transitions()`: Get suggestions for coverage-guided testing
- `get_critical_path()`: Get critical path between states

#### Callback Methods

- `on_deficiency_detected()`: Register deficiency callback
- `on_coverage_milestone()`: Register coverage milestone callback

## Export Formats

### JSON Export

Complete data export with all fields:

```python
tracker.export_results("report.json", format="json")
```

### HTML Export

Interactive HTML report with visualizations:

```python
tracker.export_results("report.html", format="html")
```

### CSV Export

Tabular data for spreadsheet analysis:

```python
tracker.export_results("report.csv", format="csv")
```

### Markdown Export

Documentation-friendly format:

```python
tracker.export_results("report.md", format="markdown")
```

## Performance Considerations

### Memory Management

PathTracker uses a circular buffer with configurable `max_history_size` to prevent unbounded memory growth:

```python
tracker = PathTracker(
    state_graph=graph,
    max_history_size=5000  # Keep last 5000 executions
)
```

### Caching

Coverage metrics are cached and only recalculated when new transitions are recorded, making frequent calls to `get_coverage_metrics()` very efficient.

### Thread Safety

All public methods use `threading.RLock` for thread safety, allowing concurrent test execution.

### Screenshot Storage

To optimize disk usage:

1. Disable screenshots for non-critical tests
2. Capture only on failures
3. Use JPEG compression
4. Regularly clean up old screenshots

```python
# Disable screenshots
tracker = PathTracker(state_graph=graph, enable_screenshots=False)

# Or capture selectively
if not success:
    tracker.record_transition(..., screenshot=capture_screen())
```

## Integration

### With FastAPI Routes

```python
from fastapi import APIRouter
from app.testing import PathTracker

router = APIRouter()
tracker = None  # Initialize in startup event

@app.on_event("startup")
async def startup_event():
    global tracker
    tracker = PathTracker(state_graph=load_graph())

@router.post("/test/record_transition")
async def record_transition(data: dict):
    execution = tracker.record_transition(**data)
    return execution.to_dict()

@router.get("/test/metrics")
async def get_metrics():
    metrics = tracker.get_coverage_metrics()
    return metrics.to_dict()
```

## Testing

To test the PathTracker module:

```python
import pytest
from app.testing import PathTracker

def test_basic_tracking():
    # Create a mock state graph
    class MockStateGraph:
        states = {}

    tracker = PathTracker(state_graph=MockStateGraph())

    # Record a transition
    execution = tracker.record_transition(
        from_state="a",
        to_state="b",
        success=True,
        duration_ms=100.0
    )

    assert execution.is_successful
    assert execution.from_state == "a"
    assert execution.to_state == "b"

    # Check metrics
    metrics = tracker.get_coverage_metrics()
    assert metrics.total_executions == 1
    assert metrics.successful_executions == 1
```

## License

MIT License - See project root for details.

## Support

For issues or questions, please refer to the main qontinui-api documentation or create an issue in the repository.
