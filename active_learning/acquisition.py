"""
Acquisition functions for the active learning loop.

4 strategies:
  1. Random sampling (null hypothesis)
  2. MC Dropout uncertainty (U_MC)
  3. Invariance violation score (V_Inv)
  4. Combined: U_MC + α·V_Inv
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


def acquire_combined(model1, model2, dataset, pool_indices, n_acquire, device, alpha=1.0):
    """
    Strategy 4: Combined score = U_MC_normalized + α·V_Inv_normalized.
    Both scores are z-score normalized before combining.
    """
    u_mc = compute_mc_uncertainty(model1, dataset, pool_indices, device)
    v_inv = compute_invariance_violation(model1, model2, dataset, pool_indices, device)

    # Z-score normalize both
    u_mc_norm = (u_mc - u_mc.mean()) / (u_mc.std() + 1e-8)
    v_inv_norm = (v_inv - v_inv.mean()) / (v_inv.std() + 1e-8)

    combined = u_mc_norm + alpha * v_inv_norm

    top_k = np.argsort(combined)[::-1][:n_acquire]
    return pool_indices[top_k], u_mc, v_inv
