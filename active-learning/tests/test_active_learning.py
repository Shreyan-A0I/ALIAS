"""Unit tests for the active learning package."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from active_learning import (  # noqa: E402
    ActiveLearningEngine,
    BalancedGroupSelector,
    CombinedScorer,
    DisagreementScorer,
    PoolManager,
    RandomScorer,
    TopKSelector,
    UncertaintyScorer,
)


class ToyModel:
    """Small deterministic model used for tests and examples."""

    def __init__(self, weight: float, bias: float, uncertainty_scale: float = 1.0) -> None:
        self.weight = weight
        self.bias = bias
        self.uncertainty_scale = uncertainty_scale

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        feature = candidates["feature"].to_numpy(dtype=float)
        return np.stack([self.weight * feature + self.bias], axis=1)

    def predict_uncertainty(self, candidates: pd.DataFrame) -> np.ndarray:
        feature = candidates["feature"].to_numpy(dtype=float)
        return np.abs(feature) * self.uncertainty_scale


class ActiveLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = pd.DataFrame(
            {
                "spot_id": [f"spot_{i}" for i in range(6)],
                "donor_id": ["A", "A", "B", "B", "C", "C"],
                "feature": [0.1, 0.4, 0.2, 0.9, 0.3, 0.7],
            }
        )

    def test_pool_manager_tracks_labeled_state(self) -> None:
        pool = PoolManager(self.candidates, id_column="spot_id", initial_labeled_ids=["spot_0"])
        self.assertEqual(len(pool.get_labeled()), 1)
        self.assertEqual(len(pool.get_unlabeled()), 5)

        revealed = pool.reveal(["spot_2", "spot_4"])
        self.assertEqual(set(revealed["spot_id"]), {"spot_2", "spot_4"})
        self.assertEqual(len(pool.get_labeled()), 3)

    def test_uncertainty_plus_disagreement_query(self) -> None:
        pool = PoolManager(self.candidates, id_column="spot_id", initial_labeled_ids=["spot_0"])
        reference_model = ToyModel(weight=1.0, bias=0.0, uncertainty_scale=1.0)
        robust_model = ToyModel(weight=0.5, bias=0.2, uncertainty_scale=0.5)

        scorer = CombinedScorer(
            scorers={
                "uncertainty": UncertaintyScorer(reference_model),
                "invariance": DisagreementScorer(reference_model, robust_model),
            },
            weights={"uncertainty": 0.7, "invariance": 0.3},
        )
        engine = ActiveLearningEngine(pool, scorer, TopKSelector())

        result = engine.query(k=2)
        self.assertEqual(len(result.selected), 2)
        self.assertIn("uncertainty.uncertainty", result.scored_pool.columns)
        self.assertIn("invariance.disagreement", result.scored_pool.columns)

        committed = engine.commit(result.selected)
        self.assertEqual(len(committed), 2)
        self.assertEqual(len(pool.get_labeled()), 3)

    def test_balanced_selector_spreads_across_groups(self) -> None:
        pool = PoolManager(self.candidates, id_column="spot_id")
        scorer = RandomScorer(random_state=0)
        engine = ActiveLearningEngine(pool, scorer, BalancedGroupSelector(group_column="donor_id"))

        result = engine.query(k=3)
        self.assertEqual(len(result.selected), 3)
        self.assertEqual(len(set(result.selected["donor_id"])), 3)


if __name__ == "__main__":
    unittest.main()
