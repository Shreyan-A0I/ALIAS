"""
Acquisition functions for the active learning loop.

6 strategies:
  1. Random sampling (null hypothesis)
  2. MC Dropout uncertainty (U_MC)
  3. Invariance violation score (V_Inv)
  4. Spatial Max (Moran's I)
  5. Feature-Space Cluster Centroids (K-Means Core)
  6. Adversarial Batch-Effect Hunting (Discriminator Entropy)
"""

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
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

def acquire_kmeans_core(dataset, pool_indices, n_acquire, device):
    """
    Strategy 5: Feature-Space Cluster Centroids.
    Clusters the raw 1536D features using MiniBatchKMeans and acquires the patch closest to each centroid.
    Ensures highly representative feature sampling while ignoring visual outliers.
    """
    pool_features = dataset.features[pool_indices].numpy()
    
    # Fast clustering
    kmeans = MiniBatchKMeans(
        n_clusters=n_acquire, 
        batch_size=1024, 
        n_init='auto', 
        random_state=42
    )
    kmeans.fit(pool_features)
    
    # K-means centroid feature vectors (n_acquire, 1536)
    centroids = torch.tensor(kmeans.cluster_centers_, device=device)
    pool_features_tensor = dataset.features[pool_indices].to(device)
    
    # Find the nearest pool point for each centroid
    # cdist shape: (n_acquire, n_pool)
    dists = torch.cdist(centroids, pool_features_tensor)
    
    # For each centroid, get the index of the closest pool patch
    closest_idxs = dists.argmin(dim=1).cpu().numpy()
    
    # Ensure uniqueness in case multiple centroids map to the same point (rare, but fallback)
    unique_idxs = list(set(closest_idxs))
    if len(unique_idxs) < n_acquire:
        # Fill randomly if duplicates happened
        remaining = n_acquire - len(unique_idxs)
        available = list(set(range(len(pool_indices))) - set(unique_idxs))
        fillers = np.random.choice(available, remaining, replace=False)
        unique_idxs.extend(fillers)
    
    return pool_indices[unique_idxs], None


@torch.no_grad()
def acquire_adversarial_batch(model, dataset, pool_indices, n_acquire, device):
    """
    Strategy 6: Adversarial Batch-Effect Hunting.
    Evaluates the unlabeled patches using the Domain Discriminator.
    Acquires patches with the lowest Entropy (i.e., highest Discriminator certainty of batch effect).
    """
    model.eval()
    model.to(device)
    
    loader = make_dataloader(dataset, pool_indices, batch_size=512, shuffle=False)
    all_entropies = []
    
    for batch in loader:
        features = batch["features"].to(device)
        
        # We MUST ensure GRL returns domain logits
        # return_domain=True allows the patch to pass through the GRL and into Head C
        _, domain_logits = model(features, return_domain=True)
        
        # Convert logits to probabilities
        probs = torch.softmax(domain_logits, dim=-1)
        
        # Calculate Entropy: -sum(p * log(p + epsilon))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        all_entropies.append(entropy.cpu())
        
    entropies = torch.cat(all_entropies, dim=0).numpy()
    
    # Lowest entropy = highest confidence in batch effect
    bottom_k = np.argsort(entropies)[:n_acquire]
    return pool_indices[bottom_k], entropies


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
