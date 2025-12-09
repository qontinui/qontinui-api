"""Testing utilities for qontinui-api.

This module provides comprehensive state/transition execution tracking for testing,
including coverage metrics, deficiency detection, and performance analysis.
"""

from app.testing.deficiency_detector import DeficiencyDetector
from app.testing.enums import (DeficiencyCategory, DeficiencySeverity,
                               ExecutionStatus)
from app.testing.models import (CoverageMetrics, Deficiency, PathHistory,
                                TransitionExecution, TransitionStatistics)
from app.testing.path_tracker import PathTracker

__all__ = [
    # Main class
    "PathTracker",
    # Enums
    "ExecutionStatus",
    "DeficiencySeverity",
    "DeficiencyCategory",
    # Data models
    "TransitionExecution",
    "CoverageMetrics",
    "Deficiency",
    "PathHistory",
    "TransitionStatistics",
    # Utilities
    "DeficiencyDetector",
]
