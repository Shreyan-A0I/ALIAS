"""
Inv-SHAF Model 2: The Batch-Effect Cheater.
Diagnostic model with donor-specific heads. This model bypasses the invariant 
bottleneck to establish a baseline of what can be predicted using donor shortcuts.
"""

import torch
import torch.nn as nn


class BatchEffectCheater(nn.Module):
    """
    Independent linear heads for each donor.
    Used to calculate the Invariance Violation Score by comparing its 
    predictions with the invariant model.
    """

    def __init__(self, input_dim=1024, n_genes=100, n_donors=8):
        super().__init__()
        self.n_donors = n_donors

        # Each donor has its own linear projection
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, n_genes)
            for _ in range(n_donors)
        ])

    def forward(self, x, donor_labels):
        """
        Routes patches to their specific donor-head.
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
        """Used during isolated donor training rounds."""
        return self.heads[donor_idx](x)

