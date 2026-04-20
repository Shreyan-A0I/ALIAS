"""Scoring functions used by the active learning engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .interfaces import PredictionModel


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    """Scale an array to the [0, 1] interval.

    Normalization keeps different score components numerically comparable when
    we combine them into one acquisition value.
    """

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


@dataclass
class ScoreBundle:
    """Container for acquisition scores and optional component breakdowns."""

    scores: np.ndarray
    components: Dict[str, np.ndarray] = field(default_factory=dict)

    def as_frame(self, candidates: pd.DataFrame, id_column: str) -> pd.DataFrame:
        """Return scores merged with candidate ids for easy inspection."""

        frame = pd.DataFrame(
            {
                id_column: candidates[id_column].to_numpy(),
                "score": self.scores,
            }
        )
        for name, values in self.components.items():
            frame[name] = values
        return frame


class BaseScorer(ABC):
    """Abstract scorer interface."""

    @abstractmethod
    def score(self, candidates: pd.DataFrame) -> ScoreBundle:
        """Return one scalar score per candidate."""


class RandomScorer(BaseScorer):
    """Assign a random score to each candidate.

    This is useful as a baseline and for smoke tests.
    """

    def __init__(self, random_state: int | None = None) -> None:
        self._rng = np.random.default_rng(random_state)

    def score(self, candidates: pd.DataFrame) -> ScoreBundle:
        scores = self._rng.random(len(candidates))
        return ScoreBundle(scores=scores, components={"random": scores})


class MetadataValueScorer(BaseScorer):
    """Use an existing metadata column as a score signal.

    This is handy for quick experiments, donor bonuses, or simulated scoring
    before a trained model is available.
    """

    def __init__(self, column: str, normalize: bool = True) -> None:
        self.column = column
        self.normalize = normalize

    def score(self, candidates: pd.DataFrame) -> ScoreBundle:
        if self.column not in candidates.columns:
            raise ValueError(f"Missing metadata column: {self.column}")
        values = candidates[self.column].to_numpy(dtype=float)
        scores = _normalize_scores(values) if self.normalize else values
        return ScoreBundle(scores=scores, components={self.column: scores})


class UncertaintyScorer(BaseScorer):
    """Scores candidates by model uncertainty."""

    def __init__(self, model: PredictionModel, normalize: bool = True) -> None:
        self.model = model
        self.normalize = normalize

    def score(self, candidates: pd.DataFrame) -> ScoreBundle:
        raw = np.asarray(self.model.predict_uncertainty(candidates), dtype=float)
        if raw.shape != (len(candidates),):
            raise ValueError("predict_uncertainty must return shape (n_candidates,)")
        scores = _normalize_scores(raw) if self.normalize else raw
        return ScoreBundle(scores=scores, components={"uncertainty": scores})


class DisagreementScorer(BaseScorer):
    """Score candidates by disagreement between two predictive models.

    This is the right abstraction for ERM-vs-REx, ERM-vs-GroupDRO, and
    ERM-vs-IRM scoring in the future.
    """

    def __init__(
        self,
        reference_model: PredictionModel,
        robust_model: PredictionModel,
        normalize: bool = True,
    ) -> None:
        self.reference_model = reference_model
        self.robust_model = robust_model
        self.normalize = normalize

    def score(self, candidates: pd.DataFrame) -> ScoreBundle:
        ref_pred = np.asarray(self.reference_model.predict(candidates), dtype=float)
        robust_pred = np.asarray(self.robust_model.predict(candidates), dtype=float)

        if ref_pred.shape != robust_pred.shape:
            raise ValueError("Reference and robust predictions must have the same shape.")
        if ref_pred.shape[0] != len(candidates):
            raise ValueError("Predictions must align with the number of candidates.")

        raw = np.mean((ref_pred - robust_pred) ** 2, axis=1)
        scores = _normalize_scores(raw) if self.normalize else raw
        return ScoreBundle(scores=scores, components={"disagreement": scores})


class CombinedScorer(BaseScorer):
    """Weighted combination of multiple scorers.

    Each component is normalized independently before weighting unless the
    underlying scorer already returns unnormalized scores by design.
    """

    def __init__(self, scorers: Mapping[str, BaseScorer], weights: Mapping[str, float]) -> None:
        if set(scorers) != set(weights):
            raise ValueError("Scorer names and weight names must match exactly.")
        self.scorers = dict(scorers)
        self.weights = dict(weights)

    def score(self, candidates: pd.DataFrame) -> ScoreBundle:
        total = np.zeros(len(candidates), dtype=float)
        components: Dict[str, np.ndarray] = {}

        for name, scorer in self.scorers.items():
            bundle = scorer.score(candidates)
            weighted = self.weights[name] * bundle.scores
            total += weighted
            components[name] = bundle.scores

            # Keep any nested details with a stable prefix for debugging.
            for component_name, values in bundle.components.items():
                components[f"{name}.{component_name}"] = values

        return ScoreBundle(scores=total, components=components)
