"""
Training routines for the Invariant Learner and the Batch-Effect Cheater.
Handles joint adversarial optimization, early stopping, and PCC evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from model.gradient_reversal import compute_grl_lambda
from dataloader.dataset import make_dataloader, DONOR_TO_IDX


def train_model1(model, dataset, labeled_indices, device,
                 epochs=50, batch_size=256, lr=5e-4, weight_decay=1e-4,
                 patience=12, alpha=1.0, beta=1.0):
    """
    Trains Model 1 using a joint loss: MSE for genes and CrossEntropy for donors.
    The GRL lambda scales linearly across epochs.
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    # 10% validation split for early stopping
    n_labeled = len(labeled_indices)
    perm = np.random.RandomState(42).permutation(n_labeled)
    val_size = max(1, int(0.1 * n_labeled))
    val_indices = labeled_indices[perm[:val_size]]
    train_indices = labeled_indices[perm[val_size:]]

    train_loader = make_dataloader(dataset, train_indices, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(dataset, val_indices, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        grl_lambda = compute_grl_lambda(epoch, epochs)
        model.set_grl_lambda(grl_lambda)

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            donor_labels = batch["donor_label"].to(device)

            gene_preds, domain_logits = model(features, return_domain=True)

            loss_pred = mse_loss_fn(gene_preds, targets)
            loss_domain = ce_loss_fn(domain_logits, donor_labels)
            loss_total = alpha * loss_pred + beta * loss_domain

            optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_total.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                donor_labels = batch["donor_label"].to(device)

                gene_preds, domain_logits = model(features, return_domain=True)
                loss_pred = mse_loss_fn(gene_preds, targets)
                loss_domain = ce_loss_fn(domain_logits, donor_labels)
                val_loss += (alpha * loss_pred + beta * loss_domain).item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model


def train_model2(model, dataset, labeled_indices, device,
                 epochs=100, batch_size=64, lr=1e-3, patience=15):
    """
    Trains the donor-specific heads of Model 2. 
    Intentionally overfits to batch shortcuts (no weight decay).
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    donor_groups = {}
    for idx in labeled_indices:
        donor = dataset.donor_ids[idx]
        donor_idx = DONOR_TO_IDX[donor]
        if donor_idx not in donor_groups:
            donor_groups[donor_idx] = []
        donor_groups[donor_idx].append(idx)

    best_state = None
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for donor_idx, indices in donor_groups.items():
            indices_arr = np.array(indices)
            loader = make_dataloader(dataset, indices_arr, batch_size=batch_size, shuffle=True)

            for batch in loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)

                preds = model.forward_single_donor(features, donor_idx)
                loss = mse_loss_fn(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        # Basic early stopping based on training convergence
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model


@torch.no_grad()
def evaluate_model1(model, dataset, test_indices, device):
    """Evaluates PCC on the test set, inverse-standardizing to original gene scales."""
    model.eval()
    model.to(device)

    loader = make_dataloader(dataset, test_indices, batch_size=512, shuffle=False)

    all_preds = []
    all_targets = []

    for batch in loader:
        features = batch["features"].to(device)
        targets_raw = batch["targets_raw"]

        gene_preds_std, _ = model(features, return_domain=False)
        gene_preds = (
            gene_preds_std.cpu() * (dataset.gene_stds + 1e-6) + dataset.gene_means
        )

        all_preds.append(gene_preds)
        all_targets.append(targets_raw)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Per-gene PCC calculation
    per_gene_pcc = []
    for g in range(all_preds.shape[1]):
        pred_g = all_preds[:, g]
        true_g = all_targets[:, g]

        if np.std(pred_g) < 1e-8 or np.std(true_g) < 1e-8:
            per_gene_pcc.append(0.0)
        else:
            r, _ = pearsonr(pred_g, true_g)
            per_gene_pcc.append(r if not np.isnan(r) else 0.0)

    per_gene_pcc = np.array(per_gene_pcc)
    mean_pcc = per_gene_pcc.mean()

    return mean_pcc, per_gene_pcc

