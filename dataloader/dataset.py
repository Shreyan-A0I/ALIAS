"""
Dataset and data splitting utilities for the Inv-SHAF active learning framework.

Handles:
- Loading cached 1024D ConvNeXt features + 100-gene expression targets
- Stratified train/test splitting (85/15)
- Initial seed creation (5% of total)
- Per-gene standardization (updated as labels are acquired)
- Barcode list persistence for reproducibility
"""

import os
import json
import torch
import numpy as np
import anndata as ad
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset


RANDOM_SEED = 57
DONOR_IDS = [
    "Br2743_ant", "Br3942_ant", "Br6423_ant", "Br8492_ant",
    "Br6471_ant", "Br6522_ant", "Br8325_ant", "Br8667_ant",
]
DONOR_TO_IDX = {d: i for i, d in enumerate(DONOR_IDS)}


class InvSHAFDataset(Dataset):
    """
    Dataset that serves cached 1024D feature vectors, 100-gene expression
    targets, and donor labels for any subset of barcodes.
    """

    def __init__(
        self,
        cached_features_dir="data/cached_features",
        h5ad_path="data/spatialDLPFC.h5ad",
    ):
        super().__init__()
        self.cached_features_dir = cached_features_dir

        # --- Load all cached features ---
        self.barcodes = []
        self.donor_ids = []
        self.features = []

        for donor_id in DONOR_IDS:
            cache_path = os.path.join(self.cached_features_dir, f"{donor_id}.pt")
            cache = torch.load(cache_path, weights_only=False)

            for bc in cache["barcodes"]:
                self.barcodes.append(bc)
                self.donor_ids.append(donor_id)

            self.features.append(cache["features"])

        self.features = torch.cat(self.features, dim=0)
        # Flatten in case cached features have extra spatial dims (N, 1024, 1, 1) → (N, 1024)
        if self.features.ndim > 2:
            self.features = self.features.view(self.features.shape[0], -1)
        self.n_samples = len(self.barcodes)

        # --- Load gene expression targets from h5ad ---
        print(f"Loading gene expression targets from {h5ad_path}...")
        adata = ad.read_h5ad(h5ad_path)
        adata.obs_names_make_unique()

        # Build a barcode→index mapping from the adata object
        # We need to handle that barcodes aren't unique across donors in adata
        # so we match by (sample_id, barcode)
        self.gene_names = list(adata.var_names)
        self.n_genes = len(self.gene_names)

        # Extract Moran's I scores for these specific genes and normalize them to sum to 1
        moran_df = adata.uns['moranI']
        moran_scores = []
        for g in self.gene_names:
            if g in moran_df.index:
                moran_scores.append(moran_df.loc[g, 'I'])
            else:
                moran_scores.append(0.0)
        
        self.morans_i_weights = torch.tensor(moran_scores, dtype=torch.float32)
        # Min-max scale weights to [0, 1] range to avoid negative penalization
        min_w = self.morans_i_weights.min()
        max_w = self.morans_i_weights.max()
        if max_w > min_w:
            self.morans_i_weights = (self.morans_i_weights - min_w) / (max_w - min_w)

        # Create expression matrix aligned with our barcode ordering
        import scipy.sparse as sp
        expression_matrix = []

        for i, (bc, donor) in enumerate(zip(self.barcodes, self.donor_ids)):
            # Find the matching row in adata
            mask = (adata.obs.index == bc) & (adata.obs["sample_id"] == donor)
            idx = np.where(mask)[0]

            if len(idx) == 0:
                # Fallback: try just barcode
                mask = adata.obs.index == bc
                idx = np.where(mask)[0]

            if len(idx) > 0:
                row = adata.X[idx[0]]
                if sp.issparse(row):
                    row = row.toarray().flatten()
                expression_matrix.append(row)
            else:
                # Should not happen if data is consistent
                expression_matrix.append(np.zeros(self.n_genes))

        self.targets = torch.tensor(
            np.array(expression_matrix), dtype=torch.float32
        )  # (N, 100)

        # Donor labels as integers
        self.donor_labels = torch.tensor(
            [DONOR_TO_IDX[d] for d in self.donor_ids], dtype=torch.long
        )

        # Global Standardization: Compute parameters from the ENTIRE pool once
        print("Computing Global gene standardization parameters...")
        self.gene_means = self.targets.mean(dim=0)
        self.gene_stds = self.targets.std(dim=0)
        # Prevent division by zero for low-variance genes
        self.gene_stds[self.gene_stds < 1e-6] = 1.0

        print(f"Dataset loaded: {self.n_samples} samples, {self.n_genes} genes, "
              f"{self.features.shape[1]}D features")

    def update_standardization(self, labeled_indices):
        """No-Op: Using Global Standardization for stability across rounds."""
        pass
        # (Old logic removed to prevent target drift)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        target_standardized = (
            (self.targets[idx] - self.gene_means) / self.gene_stds
        )

        return {
            "features": self.features[idx],       # (1024,)
            "targets": target_standardized,        # (n_genes,)
            "targets_raw": self.targets[idx],      # (n_genes,) unstandardized
            "donor_label": self.donor_labels[idx],  # scalar
            "index": idx,                           # for tracking
        }


def create_splits(dataset, seed_pct=0.01, splits_dir="data/splits", random_seed=42):
    """
    Create the fixed test set (15%), pool (85%), and initial seed (configurable).
    Saves barcode lists for reproducibility, appending the random_seed to filename.
    Returns: test_indices, pool_indices, seed_indices
    """
    os.makedirs(splits_dir, exist_ok=True)

    all_indices = np.arange(dataset.n_samples)
    donor_labels = np.array(dataset.donor_ids)

    # Stratified test/pool split (15% test)
    pool_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.15,
        stratify=donor_labels,
        random_state=random_seed,
    )

    # Seed from pool (stratified)
    seed_size = int(seed_pct * dataset.n_samples)
    pool_donor_labels = donor_labels[pool_indices]

    remaining_pool_indices, seed_indices = train_test_split(
        pool_indices,
        test_size=seed_size,
        stratify=pool_donor_labels,
        random_state=random_seed,
    )

    # Save barcode lists for reproducibility
    test_barcodes = [dataset.barcodes[i] for i in test_indices]
    seed_barcodes = [dataset.barcodes[i] for i in seed_indices]

    with open(os.path.join(splits_dir, f"seed_{random_seed}_test_barcodes.json"), "w") as f:
        json.dump(test_barcodes, f)
    with open(os.path.join(splits_dir, f"seed_{random_seed}_seed_barcodes.json"), "w") as f:
        json.dump(seed_barcodes, f)

    print(f"Splits created:")
    print(f"  Test set:       {len(test_indices)} ({len(test_indices)/dataset.n_samples*100:.1f}%)")
    print(f"  Initial seed:   {len(seed_indices)} ({len(seed_indices)/dataset.n_samples*100:.1f}%)")
    print(f"  Unlabeled pool: {len(remaining_pool_indices)} ({len(remaining_pool_indices)/dataset.n_samples*100:.1f}%)")

    return test_indices, remaining_pool_indices, seed_indices


def make_dataloader(dataset, indices, batch_size=256, shuffle=True):
    """Create a DataLoader for a subset of indices."""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
