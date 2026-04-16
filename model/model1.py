"""
Model 1: The Invariant Learner

Architecture:
  Bottleneck (1024 → 128) with GELU, LayerNorm, Dropout(0.1)
  Head A: Cross-attention gene predictor (100 gene embeddings, 4 heads)
  Head C: Domain discriminator (128 → 64 → 8) with GRL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gradient_reversal import GradientReversalLayer


class Bottleneck(nn.Module):
    """Compress 1024D raw features into 128D invariant representation."""

    def __init__(self, input_dim=1024, bottleneck_dim=256, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(bottleneck_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class CrossAttentionGenePredictor(nn.Module):
    """
    Head A: Each of 100 gene embeddings independently attends to the
    bottleneck features via multi-head cross-attention, then projects
    to a scalar prediction per gene.
    """

    def __init__(self, n_genes=100, bottleneck_dim=256, n_heads=4, ffn_dim=128):
        super().__init__()
        self.n_genes = n_genes
        self.bottleneck_dim = bottleneck_dim

        # Learnable gene query embeddings
        self.gene_embeddings = nn.Parameter(
            torch.empty(n_genes, bottleneck_dim)
        )
        nn.init.xavier_uniform_(self.gene_embeddings)

        # Cross-attention projections
        self.attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Post-attention layer norm
        self.norm = nn.LayerNorm(bottleneck_dim)

        # Per-gene feed-forward network (shared architecture, applied per gene)
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, 1),  # scalar output per gene
        )

    def forward(self, h):
        """
        Args:
            h: (B, 128) bottleneck output
        Returns:
            (B, n_genes) predicted gene expression values
        """
        B = h.shape[0]

        # Reshape bottleneck to single-token sequence: (B, 1, 128)
        kv = h.unsqueeze(1)

        # Broadcast gene embeddings across batch: (B, 100, 128)
        queries = self.gene_embeddings.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: Q=(B,100,128), K=V=(B,1,128) -> (B,100,128)
        attended, _ = self.attn(queries, kv, kv)

        # Residual connection + LayerNorm
        attended = self.norm(attended + queries)

        # FFN: (B, 100, 128) -> (B, 100, 1) -> (B, 100)
        output = self.ffn(attended).squeeze(-1)

        return output


class DomainDiscriminator(nn.Module):
    """
    Head C: Classifies which of the 8 donors a patch came from.
    Receives input AFTER the GRL.
    """

    def __init__(self, bottleneck_dim=256, hidden_dim=128, n_donors=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, n_donors),
        )

    def forward(self, x):
        return self.net(x)


class InvariantLearner(nn.Module):
    """
    Model 1: Complete invariant learner combining bottleneck,
    gene predictor (Head A), and domain discriminator (Head C + GRL).
    """

    def __init__(self, input_dim=1024, bottleneck_dim=256, n_genes=100,
                 n_heads=4, ffn_dim=128, n_donors=8, dropout=0.1):
        super().__init__()

        self.bottleneck = Bottleneck(input_dim, bottleneck_dim, dropout)
        self.gene_predictor = CrossAttentionGenePredictor(
            n_genes, bottleneck_dim, n_heads, ffn_dim
        )
        self.grl = GradientReversalLayer()
        self.domain_discriminator = DomainDiscriminator(
            bottleneck_dim, 128, n_donors
        )

    def set_grl_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)

    def forward(self, x, return_domain=True):
        """
        Args:
            x: (B, 1024) cached feature vectors
            return_domain: whether to compute domain logits
        Returns:
            gene_preds: (B, n_genes)
            domain_logits: (B, n_donors) or None
        """
        h = self.bottleneck(x)  # (B, 128)
        gene_preds = self.gene_predictor(h)  # (B, n_genes)

        domain_logits = None
        if return_domain:
            h_reversed = self.grl(h)  # GRL applied
            domain_logits = self.domain_discriminator(h_reversed)  # (B, 8)

        return gene_preds, domain_logits
