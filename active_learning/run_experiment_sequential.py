"""
Experiment orchestrator for Inv-SHAF.
Executes an ensemble of active learning strategies across multiple random seeds,
aggregating results to compute mean performance and standard error.
"""

import os
import sys
import json
import time
import torch
import numpy as np

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataset import InvSHAFDataset, create_splits
from model.model1 import InvariantLearner
from model.model2 import BatchEffectCheater
from active_learning.trainer import train_model1, train_model2, evaluate_model1
from active_learning.acquisition import (
    acquire_random, 
    acquire_uncertainty, 
    acquire_invariance, 
    acquire_spatial_min,
    acquire_kmeans_core, 
    acquire_adversarial_batch
)

def aggregate_results(results_dir, strategies, seeds, n_rounds):
    """Computes cross-seed averages for all strategy results."""
    print("\n" + "=" * 40)
    print("AGGREGATING ENSEMBLE RESULTS")
    print("=" * 40)
    
    aggregated = {}
    
    for strategy in strategies:
        strategy_data = {"rounds": []}
        
        for rnd in range(1, n_rounds + 1):
            rnd_pccs = []
            labeled_pct = 0.0
            
            for seed in seeds:
                path = os.path.join(results_dir, f"{strategy}_seed_{seed}.json")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        data = json.load(f)
                        for r_data in data["rounds"]:
                            if r_data["round"] == rnd:
                                rnd_pccs.append(r_data["mean_pcc"])
                                labeled_pct = r_data["pct_labeled"]
                                break
                                
            if rnd_pccs:
                strategy_data["rounds"].append({
                    "round": rnd,
                    "pct_labeled": labeled_pct,
                    "mean_pcc": float(np.mean(rnd_pccs)),
                    "std_pcc": float(np.std(rnd_pccs)),
                    "seeds_completed": len(rnd_pccs)
                })
                
        aggregated[strategy] = strategy_data
    
    out_path = os.path.join(results_dir, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregation complete: {out_path}")


def run_experiment(
    results_dir="results",
    n_rounds=45,
    acquire_pct=0.002,
    seeds=[13, 27, 56, 89, 104, 233, 401, 777, 892, 999],
    alpha=1.0,
    beta=1.0,
    lr=5e-4,
    features="uni",
    device_str=None,
):
    """Main AL loop across all strategies and seeds."""

    if device_str is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Device: {device}")
    os.makedirs(results_dir, exist_ok=True)

    # Load dataset features once to share across all seeds
    feat_dir = "data/cached_features_uni" if features == "uni" else "data/cached_features"
    dataset = InvSHAFDataset(cached_features_dir=feat_dir)
    
    n_total = dataset.n_samples
    n_acquire = int(acquire_pct * n_total)
    n_genes = dataset.n_genes
    input_dim = dataset.features.shape[1]

    strategies = [
        "random", 
        "uncertainty", 
        "invariance", 
        "spatial_min", 
        "kmeans_core", 
        "adversarial_batch"
    ]

    for strategy in strategies:
        print(f"\n>>> Starting Strategy: {strategy.upper()} <<<")

        for i, seed in enumerate(seeds):
            seed_out_path = os.path.join(results_dir, f"{strategy}_seed_{seed}.json")
            if os.path.exists(seed_out_path):
                continue

            print(f"Seed {seed} ({i+1}/{len(seeds)})")
            
            # Isolated split for this seed
            test_indices, pool_indices, seed_indices = create_splits(
                dataset, seed_pct=0.01, random_seed=seed
            )

            rng = np.random.RandomState(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            labeled = seed_indices.copy()
            pool = pool_indices.copy()

            strategy_results = []
            strategy_acquisitions = []

            for round_num in range(1, n_rounds + 1):
                round_start = time.time()
                pct_labeled = (len(labeled) / n_total) * 100

                # Normalizing pool data based on the current labeled set
                dataset.update_standardization(labeled)

                # Initialize and train Model 1
                model1 = InvariantLearner(
                    input_dim=input_dim, n_genes=n_genes, bottleneck_dim=256
                ).to(device)
                model1 = train_model1(model1, dataset, labeled, device,
                                      epochs=50, batch_size=256, lr=lr,
                                      alpha=alpha, beta=beta)

                # Train Model 2 only if the strategy depends on its predictions
                model2 = None
                if strategy in ("invariance", "adversarial_batch"):
                    model2 = BatchEffectCheater(
                        input_dim=input_dim, n_genes=n_genes
                    ).to(device)
                    model2 = train_model2(model2, dataset, labeled, device,
                                          epochs=100, batch_size=64)

                # Performance check
                mean_pcc, per_gene_pcc = evaluate_model1(
                    model1, dataset, test_indices, device
                )

                round_time = time.time() - round_start
                print(f"  R{round_num} | PCC: {mean_pcc:.4f} | {pct_labeled:.2f}% labels")

                strategy_results.append({
                    "round": round_num,
                    "pct_labeled": round(pct_labeled, 2),
                    "n_labeled": int(len(labeled)),
                    "mean_pcc": float(mean_pcc),
                    "per_gene_pcc": per_gene_pcc.tolist(),
                    "wall_time_seconds": round(round_time, 1),
                })

                if round_num == n_rounds:
                    break

                # Acquisition step
                if strategy == "random":
                    acquired = acquire_random(dataset, pool, n_acquire, rng)
                elif strategy == "uncertainty":
                    acquired, _ = acquire_uncertainty(model1, dataset, pool, n_acquire, device)
                elif strategy == "invariance":
                    acquired, _ = acquire_invariance(model1, model2, dataset, pool, n_acquire, device)
                elif strategy == "spatial_min":
                    acquired, _ = acquire_spatial_min(model1, dataset, pool, n_acquire, device)
                elif strategy == "kmeans_core":
                    acquired, _ = acquire_kmeans_core(dataset, pool, n_acquire, device)
                elif strategy == "adversarial_batch":
                    acquired, _ = acquire_adversarial_batch(model1, dataset, pool, n_acquire, device)

                # Update indices for the next round
                strategy_acquisitions.append([dataset.barcodes[i] for i in acquired])
                labeled = np.concatenate([labeled, acquired])
                pool = np.array([i for i in pool if i not in set(acquired)])

            # Serialize seed results
            with open(seed_out_path, "w") as f:
                json.dump({"rounds": strategy_results, "acquisitions": strategy_acquisitions}, f, indent=2)

    # Calculate final means across all seeds
    aggregate_results(results_dir, strategies, seeds, n_rounds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="uni", choices=["convnext", "uni"])
    parser.add_argument("--rounds", type=int, default=45)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    run_experiment(
        n_rounds=args.rounds,
        beta=args.beta,
        features=args.features,
        acquire_pct=0.0020
    )

