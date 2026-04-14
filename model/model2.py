"""
Model 2: The Batch-Effect Cheater

8 separate linear heads (1024 → 100), one per donor.
Each head has full access to raw backbone features (bypasses Model 1's
bottleneck entirely) and will aggressively overfit to donor-specific shortcuts.

Purpose: Diagnostic tool for computing the Invariance Violation Score.
"""

import torch
import torch.nn as nn


class BatchEffectCheater(nn.Module):
    """
    Model 2: 8 independent linear layers, one per donor.
    A patch is routed only to the head matching its donor.
    No GRL, no bottleneck, no weight decay — we WANT overfitting.
    """

    def __init__(self, input_dim=1024, n_genes=100, n_donors=8):
        super().__init__()
        self.n_donors = n_donors

        # 8 separate linear heads
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, n_genes)
            for _ in range(n_donors)
        ])

    def forward(self, x, donor_labels):
        """
        Args:
            x: (B, 1024) cached feature vectors
            donor_labels: (B,) integer donor indices
        Returns:
            predictions: (B, n_genes)
        """
        B = x.shape[0]
        n_genes = self.heads[0].out_features
        predictions = torch.zeros(B, n_genes, device=x.device)

        for d in range(self.n_donors):
            mask = donor_labels == d
            if mask.any():
                predictions[mask] = self.heads[d](x[mask])

        return predictions

    def forward_single_donor(self, x, donor_idx):
        """Forward pass for patches from a single donor (used during training)."""
        return self.heads[donor_idx](x)
