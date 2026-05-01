"""
Inv-SHAF Model 1: The Invariant Learner

Architecture:
  Bottleneck (1024/1536 → 256) with GELU, LayerNorm, Dropout(0.3)
  Head A: Cross-attention gene predictor (100 gene embeddings, 4 heads)
  Head C: Domain discriminator (256 → 128 → 8) with GRL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gradient_reversal import GradientReversalLayer


class Bottleneck(nn.Module):
    """Compresses raw histology features into an invariant representation."""

    def __init__(self, input_dim=1024, bottleneck_dim=256, dropout=0.3):
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
    Predicts gene expression by allowing gene-specific embeddings to attend 
    to the shared visual bottleneck.
    """

    def __init__(self, n_genes=100, bottleneck_dim=256, n_heads=4, ffn_dim=128):
        super().__init__()
        self.n_genes = n_genes
        self.bottleneck_dim = bottleneck_dim

        # Gene-specific query vectors
        self.gene_embeddings = nn.Parameter(
            torch.empty(n_genes, bottleneck_dim)
        )
        nn.init.xavier_uniform_(self.gene_embeddings)

        self.attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(bottleneck_dim)

        # Output head shared across gene queries
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, 1),
        )

    def forward(self, h):
        """
        Args:
            h: (B, bottleneck_dim) feature tensor
        """
        B = h.shape[0]

        # Treat bottleneck as a single visual token
        kv = h.unsqueeze(1)

        # Cross-attention between genes (Q) and morphology (K, V)
        queries = self.gene_embeddings.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.attn(queries, kv, kv)

        attended = self.norm(attended + queries)
        output = self.ffn(attended).squeeze(-1)

        return output


class DomainDiscriminator(nn.Module):
    """Classifies donor identity to provide adversarial supervision via GRL."""

    def __init__(self, bottleneck_dim=256, hidden_dim=128, n_donors=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, n_donors),
        )

    def forward(self, x):
        return self.net(x)


class InvariantLearner(nn.Module):
    """Main model combining the bottleneck, predictor, and adversarial head."""

    def __init__(self, input_dim=1024, bottleneck_dim=256, n_genes=100,
                 n_heads=4, ffn_dim=128, n_donors=8, dropout=0.3):
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
        h = self.bottleneck(x)
        gene_preds = self.gene_predictor(h)

        domain_logits = None
        if return_domain:
            # Flip gradients to force the bottleneck to be domain-invariant
            h_reversed = self.grl(h)
            domain_logits = self.domain_discriminator(h_reversed)

        return gene_preds, domain_logits

