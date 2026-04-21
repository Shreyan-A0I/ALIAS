"""
Acquisition functions for the active learning loop.

4 strategies:
  1. Random sampling (null hypothesis)
  2. MC Dropout uncertainty (U_MC)
  3. Invariance violation score (V_Inv)
  4. Spatial Structure Maximization (Moran's I weighted expression)
  5. Feature-Space Diversity (K-Center Greedy Coreset)
"""

import torch
import numpy as np
from dataloader.dataset import make_dataloader, DONOR_TO_IDX


def acquire_random(dataset, pool_indices, n_acquire, rng):
    """
    Strategy 1: Random acquisition, stratified by donor.
    """
    donor_pools = {}
    for idx in pool_indices:
        donor = dataset.donor_ids[idx]
        if donor not in donor_pools:
            donor_pools[donor] = []
        donor_pools[donor].append(idx)

    total_pool = len(pool_indices)
    selected = []

    for donor, indices in donor_pools.items():
        # Proportional allocation
        donor_share = int(n_acquire * len(indices) / total_pool)
        donor_share = min(donor_share, len(indices))
        chosen = rng.choice(indices, size=donor_share, replace=False)
        selected.extend(chosen.tolist())

    # If rounding left us short, fill randomly
    remaining = n_acquire - len(selected)
    if remaining > 0:
        leftover = list(set(pool_indices) - set(selected))
        extra = rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
        selected.extend(extra.tolist())

    return np.array(selected[:n_acquire])


@torch.no_grad()
def acquire_spatial_max(model, dataset, pool_indices, n_acquire, device):
    """
    Strategy 4: Spatial Structure Maximization.
    Computes a score for each patch = sum(PredictedExpr * Morans_I_Weights).
    Prioritizes patches predicted to express highly spatially-structured genes.
    """
    model.eval()
    model.to(device)
    
    loader = make_dataloader(dataset, pool_indices, batch_size=512, shuffle=False)
    all_scores = []
    
    weights = dataset.morans_i_weights.to(device)

    for batch in loader:
        features = batch["features"].to(device)
        
        # Deterministic prediction
        gene_preds, _ = model(features, return_domain=False)
        
        # Weighted sum: (B, 100) * (100,) -> sum over genes -> (B,)
        patch_spatial_scores = (gene_preds * weights).sum(dim=1)
        all_scores.append(patch_spatial_scores.cpu())
        
    scores = torch.cat(all_scores, dim=0).numpy()
    top_k = np.argsort(scores)[::-1][:n_acquire]
    return pool_indices[top_k], scores


@torch.no_grad()
def acquire_spatial_min(model, dataset, pool_indices, n_acquire, device):
    """
    Inverse of Strategy 4: Spatial Structure Minimization.
    Computes a score for each patch = sum(PredictedExpr * Morans_I_Weights).
    Prioritizes patches predicted to have the LOWEST expression of highly spatially-structured genes.
    """
    model.eval()
    model.to(device)
    
    loader = make_dataloader(dataset, pool_indices, batch_size=512, shuffle=False)
    all_scores = []
    
    weights = dataset.morans_i_weights.to(device)

    for batch in loader:
        features = batch["features"].to(device)
        gene_preds, _ = model(features, return_domain=False)
        patch_spatial_scores = (gene_preds * weights).sum(dim=1)
        all_scores.append(patch_spatial_scores.cpu())
        
    scores = torch.cat(all_scores, dim=0).numpy()
    # Pick the patches with the SMALLEST scores
    bottom_k = np.argsort(scores)[:n_acquire]
    return pool_indices[bottom_k], scores

def acquire_diversity(dataset, pool_indices, labeled_indices, n_acquire, device):
    """
    Strategy 5: Feature-Space Diversity (K-Center Greedy / Coreset).
    Selects unlabeled patches whose features are furthest from any currently labeled patch.
    Calculates Euclidean distance directly on the base features.
    """
    # Move features to GPU if possible to speed up distance calcs, or keep on CPU
    labeled_features = dataset.features[labeled_indices].to(device)
    pool_features = dataset.features[pool_indices].to(device)
    
    # We will incrementally pick the furthest point, add it to our "labeled" set, and repeat
    # For large pools, computing full pairwise D=O(N*M) is expensive.
    # Instead, we just maintain the minimum distance to the labeled set for each pool point.

    # 1. Compute initial min distances from each pool point to ANY labeled point
    # Since labeled_features can be ~3000, we batch the calculation over pool points
    n_pool = pool_features.shape[0]
    min_dist = torch.full((n_pool,), float('inf'), device=device)
    
    batch_size = 512
    for i in range(0, n_pool, batch_size):
        end = min(i + batch_size, n_pool)
        batch = pool_features[i:end]
        # Pairwise distance: (batch_size, 1, 1536) - (1, n_labeled, 1536) -> (batch_size, n_labeled)
        # Using torch.cdist for memory efficiency
        dists = torch.cdist(batch, labeled_features, p=2.0)
        min_dist[i:end] = dists.min(dim=1)[0]
        
    selected_pool_idx = []
    
    # 2. Greedily pick furthest
    for _ in range(n_acquire):
        # Find the point with the maximum minimum distance
        furthest_idx = torch.argmax(min_dist).item()
        selected_pool_idx.append(furthest_idx)
        
        # The new point is now "labeled". Update the min_dist for all remaining pool points.
        new_point = pool_features[furthest_idx:furthest_idx+1]
        
        new_dists = torch.cdist(pool_features, new_point, p=2.0).squeeze(1)
        # Update minimum distances efficiently
        min_dist = torch.minimum(min_dist, new_dists)
        
        # Don't pick this point again
        min_dist[furthest_idx] = -1.0
        
    acquired_actual_indices = pool_indices[selected_pool_idx]
    return acquired_actual_indices, None  # No 'score' array to return like the others


@torch.no_grad()
def compute_mc_uncertainty(model, dataset, pool_indices, device, T=10):
    """
    Compute MC Dropout uncertainty (U_MC) for each unlabeled patch.
    T stochastic forward passes through Model 1 with dropout ON.

    Returns: array of U_MC scores, shape (len(pool_indices),)
    """
    # Enable dropout for stochastic passes
    model.train()  # dropout ON
    model.to(device)

    loader = make_dataloader(dataset, pool_indices, batch_size=512, shuffle=False)

    all_uncertainties = []

    for batch in loader:
        features = batch["features"].to(device)
        B = features.shape[0]

        # T stochastic forward passes
        predictions = []
        for _ in range(T):
            gene_preds, _ = model(features, return_domain=False)
            predictions.append(gene_preds.cpu())

        # Stack: (T, B, n_genes)
        predictions = torch.stack(predictions, dim=0)

        # Per-gene variance across T passes: (B, n_genes)
        per_gene_var = predictions.var(dim=0)

        # Mean variance across genes: (B,)
        u_mc = per_gene_var.mean(dim=1)

        all_uncertainties.append(u_mc)

    return torch.cat(all_uncertainties, dim=0).numpy()


@torch.no_grad()
def compute_invariance_violation(model1, model2, dataset, pool_indices, device):
    """
    Compute invariance violation score (V_Inv) for each unlabeled patch.
    V_Inv = MSE between Model 1 (invariant) and Model 2 (cheater) predictions.

    Returns: array of V_Inv scores, shape (len(pool_indices),)
    """
    model1.eval()
    model2.eval()
    model1.to(device)
    model2.to(device)

    loader = make_dataloader(dataset, pool_indices, batch_size=512, shuffle=False)

    all_violations = []

    for batch in loader:
        features = batch["features"].to(device)
        donor_labels = batch["donor_label"].to(device)

        # Model 1 prediction (deterministic, dropout OFF)
        gene_preds_m1, _ = model1(features, return_domain=False)

        # Model 2 prediction
        gene_preds_m2 = model2(features, donor_labels)

        # MSE between the two predictions per patch
        v_inv = ((gene_preds_m1 - gene_preds_m2) ** 2).mean(dim=1)

        all_violations.append(v_inv.cpu())

    return torch.cat(all_violations, dim=0).numpy()


def acquire_uncertainty(model, dataset, pool_indices, n_acquire, device):
    """Strategy 2: Select patches with highest MC Dropout uncertainty."""
    u_mc = compute_mc_uncertainty(model, dataset, pool_indices, device)
    top_k = np.argsort(u_mc)[::-1][:n_acquire]
    return pool_indices[top_k], u_mc


def acquire_invariance(model1, model2, dataset, pool_indices, n_acquire, device):
    """Strategy 3: Select patches with highest invariance violation."""
    v_inv = compute_invariance_violation(model1, model2, dataset, pool_indices, device)
    top_k = np.argsort(v_inv)[::-1][:n_acquire]
    return pool_indices[top_k], v_inv
