"""
Main experiment orchestrator for the Inv-SHAF active learning loop.

Runs:
  1. Baseline: Model 1 trained on full 85% pool
  2. Four AL strategies (Random, Uncertainty, Invariance, Combined)
     each from 5% → 30% in 1% increments (25 rounds)
"""

import os
import sys
import json
import time
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataset import InvSHAFDataset, create_splits, RANDOM_SEED
from model.model1 import InvariantLearner
from model.model2 import BatchEffectCheater
from active_learning.trainer import train_model1, train_model2, evaluate_model1
from active_learning.acquisition import (
    acquire_random, acquire_uncertainty, acquire_invariance, acquire_combined,
)


def run_experiment(
    results_dir="results",
    n_rounds=25,
    acquire_pct=0.01,  # 1% of total per round
    alpha=1.0,
    beta=1.0,
    lr=5e-4,
    features="convnext",
    device_str=None,
):
    """Run the full Inv-SHAF active learning experiment."""

    if device_str is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")
    os.makedirs(results_dir, exist_ok=True)

    # ========== LOAD DATA ==========
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    # Set feature directory based on selection
    feat_dir = "data/cached_features"
    if features == "uni":
        feat_dir = "data/cached_features_uni"

    dataset = InvSHAFDataset(cached_features_dir=feat_dir)
    test_indices, pool_indices, seed_indices = create_splits(dataset)

    n_total = dataset.n_samples
    n_acquire = int(acquire_pct * n_total)  # ~309 patches per round
    n_genes = dataset.n_genes
    input_dim = dataset.features.shape[1]

    print(f"Acquisition batch size: {n_acquire} patches per round")
    print(f"Feature dimension: {input_dim}")

    # ========== BASELINE ==========
    print("\n" + "=" * 60)
    print("TRAINING BASELINE (Model 1 on full pool)")
    print("=" * 60)

    # Update standardization with all pool data
    all_pool_and_seed = np.concatenate([pool_indices, seed_indices])
    dataset.update_standardization(all_pool_and_seed)

    model1_baseline = InvariantLearner(
        input_dim=input_dim,
        n_genes=n_genes,
        bottleneck_dim=256,
    ).to(device)
    model1_baseline = train_model1(
        model1_baseline, dataset, all_pool_and_seed, device,
        epochs=50, batch_size=256, lr=lr, alpha=alpha, beta=beta
    )

    baseline_pcc, baseline_per_gene = evaluate_model1(
        model1_baseline, dataset, test_indices, device
    )
    print(f"Baseline Mean PCC: {baseline_pcc:.4f}")

    # Save baseline
    baseline_results = {
        "mean_pcc": float(baseline_pcc),
        "per_gene_pcc": baseline_per_gene.tolist(),
    }
    with open(os.path.join(results_dir, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)

    # ========== ACTIVE LEARNING LOOP ==========
    strategies = ["random", "uncertainty", "invariance", "combined"]

    all_results = {"baseline": baseline_results, "strategies": {}}

    for strategy in strategies:
        print("\n" + "=" * 60)
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 60)

        # Reset to identical starting conditions
        rng = np.random.RandomState(RANDOM_SEED)
        labeled = seed_indices.copy()
        pool = pool_indices.copy()

        strategy_results = []
        strategy_acquisitions = []
        strategy_scores = []

        for round_num in range(1, n_rounds + 1):
            round_start = time.time()
            pct_labeled = (len(labeled) / n_total) * 100

            print(f"\n  Round {round_num}/{n_rounds} | "
                  f"Labeled: {len(labeled)} ({pct_labeled:.1f}%) | "
                  f"Pool: {len(pool)}")

            # --- TRAIN ---
            dataset.update_standardization(labeled)

            # Train Model 1 from scratch
            model1 = InvariantLearner(
                input_dim=input_dim,
                n_genes=n_genes,
                bottleneck_dim=256,
            ).to(device)
            model1 = train_model1(model1, dataset, labeled, device,
                                  epochs=50, batch_size=256, lr=lr,
                                  alpha=alpha, beta=beta)

            # Train Model 2 from scratch (needed for invariance & combined)
            model2 = None
            if strategy in ("invariance", "combined"):
                model2 = BatchEffectCheater(
                    input_dim=input_dim,
                    n_genes=n_genes,
                ).to(device)
                model2 = train_model2(model2, dataset, labeled, device,
                                      epochs=100, batch_size=64)

            # --- EVALUATE ---
            mean_pcc, per_gene_pcc = evaluate_model1(
                model1, dataset, test_indices, device
            )

            round_time = time.time() - round_start
            print(f"  Mean PCC: {mean_pcc:.4f} | Time: {round_time:.1f}s")

            strategy_results.append({
                "round": round_num,
                "pct_labeled": round(pct_labeled, 2),
                "n_labeled": int(len(labeled)),
                "mean_pcc": float(mean_pcc),
                "per_gene_pcc": per_gene_pcc.tolist(),
                "wall_time_seconds": round(round_time, 1),
            })

            # --- ACQUIRE ---
            if strategy == "random":
                acquired = acquire_random(dataset, pool, n_acquire, rng)
                strategy_scores.append({"u_mc": None, "v_inv": None})

            elif strategy == "uncertainty":
                acquired, u_mc = acquire_uncertainty(
                    model1, dataset, pool, n_acquire, device
                )
                strategy_scores.append({
                    "mean_u_mc_acquired": float(u_mc[np.isin(pool, acquired)].mean())
                    if len(acquired) > 0 else 0.0
                })

            elif strategy == "invariance":
                acquired, v_inv = acquire_invariance(
                    model1, model2, dataset, pool, n_acquire, device
                )
                strategy_scores.append({
                    "mean_v_inv_acquired": float(v_inv[np.isin(pool, acquired)].mean())
                    if len(acquired) > 0 else 0.0
                })

            elif strategy == "combined":
                acquired, u_mc, v_inv = acquire_combined(
                    model1, model2, dataset, pool, n_acquire, device, alpha=alpha
                )
                # Save score decomposition for the quaternary plot
                acq_mask = np.isin(pool, acquired)
                strategy_scores.append({
                    "mean_u_mc_acquired": float(u_mc[acq_mask].mean())
                    if acq_mask.any() else 0.0,
                    "mean_v_inv_acquired": float(v_inv[acq_mask].mean())
                    if acq_mask.any() else 0.0,
                })

            # Save acquired barcodes for reproducibility
            acquired_barcodes = [dataset.barcodes[i] for i in acquired]
            strategy_acquisitions.append(acquired_barcodes)

            # Move acquired patches: pool → labeled
            labeled = np.concatenate([labeled, acquired])
            pool = np.array([i for i in pool if i not in set(acquired)])

        # Save strategy results
        all_results["strategies"][strategy] = {
            "rounds": strategy_results,
            "acquisitions": strategy_acquisitions,
            "scores": strategy_scores,
        }

        # Save intermediate file
        with open(os.path.join(results_dir, f"{strategy}_results.json"), "w") as f:
            json.dump(all_results["strategies"][strategy], f, indent=2)

    # Save consolidated results
    with open(os.path.join(results_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to {results_dir}/")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="convnext", choices=["convnext", "uni"])
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    run_experiment(
        n_rounds=args.rounds,
        beta=args.beta,
        features=args.features
    )
