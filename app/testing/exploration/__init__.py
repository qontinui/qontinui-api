"""Exploration strategies for intelligent path traversal.

This package provides various exploration strategies for systematically
discovering and testing state transitions in GUI applications.
"""

from app.testing.exploration.backtracking import BacktrackingNavigator
from app.testing.exploration.diversity import PathDiversityEngine
from app.testing.exploration.failure_handler import FailureAwareExplorer
from app.testing.exploration.path_explorer import PathExplorer
from app.testing.exploration.strategies import (
                                                AdaptiveExplorer,
                                                BreadthFirstExplorer,
                                                DepthFirstExplorer,
                                                ExplorationStrategy,
                                                GreedyCoverageExplorer,
                                                HybridExplorer,
                                                RandomWalkExplorer,
)

__all__ = [
    "PathExplorer",
    "ExplorationStrategy",
    "RandomWalkExplorer",
    "GreedyCoverageExplorer",
    "DepthFirstExplorer",
    "BreadthFirstExplorer",
    "AdaptiveExplorer",
    "HybridExplorer",
    "BacktrackingNavigator",
    "PathDiversityEngine",
    "FailureAwareExplorer",
]
