"""Interfaces for model adapters used by the active learning module."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class PredictionModel(Protocol):
    """Minimal model interface expected by the scorers.

    A future base model only needs to implement the methods used by the
    specific scorer. For example, disagreement scoring needs `predict`, while
    uncertainty scoring needs `predict_uncertainty`.
    """

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return predictions for each candidate.

        The expected shape is `(n_candidates, output_dim)`.
        """

    def predict_uncertainty(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return one scalar uncertainty value per candidate.

        The expected shape is `(n_candidates,)`.
        """
