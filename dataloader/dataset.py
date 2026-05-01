"""
Dataset and data splitting utilities for Inv-SHAF.
Handles feature loading, stratified splitting, and gene standardization.
"""

import os
import json
import torch
import numpy as np
import anndata as ad
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset


# Pre-selected donors from the DLPFC anterior set
DONOR_IDS = [
    "Br2743_ant", "Br3942_ant", "Br6423_ant", "Br8492_ant",
    "Br6471_ant", "Br6522_ant", "Br8325_ant", "Br8667_ant",
]
DONOR_TO_IDX = {d: i for i, d in enumerate(DONOR_IDS)}


class InvSHAFDataset(Dataset):
    """
    Serves cached image features and expression targets for active learning.
    Aligns patch-level features with barcodes and donor metadata.
    """

    def __init__(
        self,
        cached_features_dir="data/cached_features",
        h5ad_path="data/spatialDLPFC.h5ad",
    ):
        super().__init__()
        self.cached_features_dir = cached_features_dir

        self.barcodes = []
        self.donor_ids = []
        self.features = []

        # Load pre-extracted backbone features
        for donor_id in DONOR_IDS:
            cache_path = os.path.join(self.cached_features_dir, f"{donor_id}.pt")
            cache = torch.load(cache_path, weights_only=False)

            for bc in cache["barcodes"]:
                self.barcodes.append(bc)
                self.donor_ids.append(donor_id)

            self.features.append(cache["features"])

        self.features = torch.cat(self.features, dim=0)
        if self.features.ndim > 2:
            self.features = self.features.view(self.features.shape[0], -1)
        
        self.n_samples = len(self.barcodes)

        # Align with gene expression targets
        print(f"Loading expression targets from {h5ad_path}...")
        adata = ad.read_h5ad(h5ad_path)
        adata.obs_names_make_unique()

        self.gene_names = list(adata.var_names)
        self.n_genes = len(self.gene_names)

        # Normalize Moran's I scores to use as strategy weights
        moran_df = adata.uns['moranI']
        moran_scores = [moran_df.loc[g, 'I'] if g in moran_df.index else 0.0 for g in self.gene_names]
        
        self.morans_i_weights = torch.tensor(moran_scores, dtype=torch.float32)
        min_w, max_w = self.morans_i_weights.min(), self.morans_i_weights.max()
        if max_w > min_w:
            self.morans_i_weights = (self.morans_i_weights - min_w) / (max_w - min_w)

        # Map barcodes to expression matrix rows
        import scipy.sparse as sp
        expression_matrix = []

        for bc, donor in zip(self.barcodes, self.donor_ids):
            mask = (adata.obs.index == bc) & (adata.obs["sample_id"] == donor)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                mask = adata.obs.index == bc
                idx = np.where(mask)[0]

            if len(idx) > 0:
                row = adata.X[idx[0]]
                if sp.issparse(row):
                    row = row.toarray().flatten()
                expression_matrix.append(row)
            else:
                expression_matrix.append(np.zeros(self.n_genes))

        self.targets = torch.tensor(np.array(expression_matrix), dtype=torch.float32)
        self.donor_labels = torch.tensor([DONOR_TO_IDX[d] for d in self.donor_ids], dtype=torch.long)

        # Global standardization parameters (calculated once to prevent drift)
        self.gene_means = self.targets.mean(dim=0)
        self.gene_stds = self.targets.std(dim=0)
        self.gene_stds[self.gene_stds < 1e-6] = 1.0

        print(f"Dataset ready: {self.n_samples} patches | {self.n_genes} genes")

    def update_standardization(self, labeled_indices):
        """Global standardization is used to keep round-to-round targets stable."""
        pass

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        target_std = (self.targets[idx] - self.gene_means) / self.gene_stds
        return {
            "features": self.features[idx],
            "targets": target_std,
            "targets_raw": self.targets[idx],
            "donor_label": self.donor_labels[idx],
            "index": idx,
        }


def create_splits(dataset, seed_pct=0.01, splits_dir="data/splits", random_seed=42):
    """Creates stratified test/pool/seed splits and persists barcodes for reproducibility."""
    os.makedirs(splits_dir, exist_ok=True)

    all_indices = np.arange(dataset.n_samples)
    donors = np.array(dataset.donor_ids)

    # 15% test set split
    pool_idx, test_idx = train_test_split(
        all_indices, test_size=0.15, stratify=donors, random_state=random_seed
    )

    # Initial labeled seed
    seed_size = int(seed_pct * dataset.n_samples)
    remaining_pool_idx, seed_idx = train_test_split(
        pool_idx, test_size=seed_size, stratify=donors[pool_idx], random_state=random_seed
    )

    # Persist barcodes to disk
    test_bc = [dataset.barcodes[i] for i in test_idx]
    seed_bc = [dataset.barcodes[i] for i in seed_idx]

    with open(os.path.join(splits_dir, f"seed_{random_seed}_test_barcodes.json"), "w") as f:
        json.dump(test_bc, f)
    with open(os.path.join(splits_dir, f"seed_{random_seed}_seed_barcodes.json"), "w") as f:
        json.dump(seed_bc, f)

    return test_idx, remaining_pool_idx, seed_idx


def make_dataloader(dataset, indices, batch_size=256, shuffle=True):
    """Utility to create a standard PyTorch DataLoader for a subset of the data."""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

