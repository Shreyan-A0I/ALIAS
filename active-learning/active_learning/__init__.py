"""Lightweight, model-agnostic active learning utilities.

This package is intentionally small and composable so it is easy to plug in
future ERM / REx / GroupDRO / IRM models without rewriting the query loop.
"""

from .engine import ActiveLearningEngine, QueryResult
from .interfaces import PredictionModel
from .pool import PoolManager
from .scorers import (
    BaseScorer,
    CombinedScorer,
    DisagreementScorer,
    MetadataValueScorer,
    RandomScorer,
    ScoreBundle,
    UncertaintyScorer,
)
from .selectors import BalancedGroupSelector, BaseSelector, TopKSelector

__all__ = [
    "ActiveLearningEngine",
    "BalancedGroupSelector",
    "BaseScorer",
    "BaseSelector",
    "CombinedScorer",
    "DisagreementScorer",
    "MetadataValueScorer",
    "PoolManager",
    "PredictionModel",
    "QueryResult",
    "RandomScorer",
    "ScoreBundle",
    "TopKSelector",
    "UncertaintyScorer",
]
