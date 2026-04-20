"""Unit tests for the real Inv-SHAF acquisition functions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from active_learning.acquisition import (  # noqa: E402
    acquire_invariance,
    compute_invariance_violation,
)


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.features = torch.tensor(
            [
                [0.1, 0.2],
                [0.2, 0.1],
                [0.3, 0.4],
                [0.4, 0.3],
                [0.5, 0.6],
                [0.6, 0.5],
            ],
            dtype=torch.float32,
        )
        self.donor_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        self.donor_ids = ["A", "A", "B", "B", "C", "C"]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "features": self.features[idx],
            "donor_label": self.donor_labels[idx],
        }


class IdentityModel1:
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, features, return_domain=False):
        return features, None


class OffsetByDonorModel2:
    def __init__(self) -> None:
        self.offsets = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float32)

    def eval(self):
        return self

    def to(self, device):
        self.offsets = self.offsets.to(device)
        return self

    def __call__(self, features, donor_labels):
        offsets = self.offsets[donor_labels].unsqueeze(1)
        return features + offsets


class AcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = ToyDataset()
        self.model1 = IdentityModel1()
        self.model2 = OffsetByDonorModel2()
        self.pool_indices = np.arange(len(self.dataset))

    def test_compute_invariance_violation_matches_expected_mse(self) -> None:
        scores = compute_invariance_violation(
            self.model1,
            self.model2,
            self.dataset,
            self.pool_indices,
            device="cpu",
        )
        expected = np.array([0.01, 0.01, 0.25, 0.25, 1.0, 1.0], dtype=float)
        np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-6)

    def test_acquire_invariance_spreads_selection_across_donors(self) -> None:
        selected, scores = acquire_invariance(
            self.model1,
            self.model2,
            self.dataset,
            self.pool_indices,
            n_acquire=3,
            device="cpu",
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual({self.dataset.donor_ids[idx] for idx in selected}, {"A", "B", "C"})
        np.testing.assert_allclose(
            scores,
            np.array([0.01, 0.01, 0.25, 0.25, 1.0, 1.0], dtype=float),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
