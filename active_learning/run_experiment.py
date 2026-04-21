"""
Main experiment orchestrator for the Inv-SHAF active learning loop.

Supports Memory-Safe Multiprocessing Ensemble Execution.
Loads dataset once in main process to share memory via fork,
reduces worker count for stability, and adds a live progress log.
"""

import os
# MUST SET THESE BEFORE ANY NUMPY OR PYTORCH IMPORTS TO PREVENT THREAD THRASHING
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Limit torch threads explicitly just to be absolutely certain
torch.set_num_threads(1)

# Add project root to path
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

# Global dataset placeholder for fork-sharing
_GLOBAL_DATASET = None

def log_progress(results_dir, message):
    """Appends a message to a shared live progress log."""
    path = os.path.join(results_dir, "live_progress.txt")
    with open(path, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")

def aggregate_results(results_dir, strategies, seeds, n_rounds):
    """Averages the results across all seeds for each strategy."""
    print("\n" + "=" * 60)
    print("AGGREGATING MULTI-SEED RESULTS")
    print("=" * 60)
    
    aggregated = {}
    
    for strategy in strategies:
        strategy_data = {"rounds": []}
        
        # We need to average PCC per round across all available seeds
        for rnd in range(1, n_rounds + 1):
            rnd_pccs = []
            labeled_pct = 0.0
            
            for seed in seeds:
                path = os.path.join(results_dir, f"{strategy}_seed_{seed}.json")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        data = json.load(f)
                        # Find the matching round
                        for r_data in data["rounds"]:
                            if r_data["round"] == rnd:
                                rnd_pccs.append(r_data["mean_pcc"])
                                labeled_pct = r_data["pct_labeled"]
                                break
                                
            if len(rnd_pccs) > 0:
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
    print(f"Aggregation complete. Saved to {out_path}")


def run_single_trial(args_tuple):
    """Atomic function to run a completely independent AL trial. Executed by the ProcessPool."""
    strategy, seed, results_dir, n_rounds, acquire_pct, alpha, beta, lr, device_str = args_tuple
    
    # Access the shared global dataset (Copy-on-Write)
    global _GLOBAL_DATASET
    dataset = _GLOBAL_DATASET
    
    device = torch.device(device_str)
    
    seed_out_path = os.path.join(results_dir, f"{strategy}_seed_{seed}.json")
    if os.path.exists(seed_out_path):
        return f"Skipped {strategy} Seed {seed} (Already Exists)"

    n_total = dataset.n_samples
    n_acquire = int(acquire_pct * n_total)
    n_genes = dataset.n_genes
    input_dim = dataset.features.shape[1]

    # Create strictly isolated splits for this particular seed
    test_indices, pool_indices, seed_indices = create_splits(
        dataset, seed_pct=0.01, random_seed=seed
    )

    # Reset random states and tracking arrays locally
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    labeled = seed_indices.copy()
    pool = pool_indices.copy()

    strategy_results = []
    strategy_acquisitions = []

    log_progress(results_dir, f"START: {strategy} | Seed {seed}")

    for round_num in range(1, n_rounds + 1):
        round_start = time.time()
        pct_labeled = (len(labeled) / n_total) * 100

        # --- TRAIN ---
        dataset.update_standardization(labeled)

        # Train Model 1
        model1 = InvariantLearner(
            input_dim=input_dim, n_genes=n_genes, bottleneck_dim=256
        ).to(device)
        model1 = train_model1(model1, dataset, labeled, device,
                              epochs=50, batch_size=256, lr=lr,
                              alpha=alpha, beta=beta)

        # Train Model 2 (only if strategy requires domain cheating)
        model2 = None
        if strategy in ("invariance", "adversarial_batch"):
            model2 = BatchEffectCheater(
                input_dim=input_dim, n_genes=n_genes
            ).to(device)
            model2 = train_model2(model2, dataset, labeled, device,
                                  epochs=100, batch_size=64)

        # --- EVALUATE ---
        mean_pcc, per_gene_pcc = evaluate_model1(
            model1, dataset, test_indices, device
        )

        round_time = time.time() - round_start

        strategy_results.append({
            "round": round_num,
            "pct_labeled": round(pct_labeled, 2),
            "n_labeled": int(len(labeled)),
            "mean_pcc": float(mean_pcc),
            "per_gene_pcc": per_gene_pcc.tolist(),
            "wall_time_seconds": round(round_time, 1),
        })

        # Occasional heart-beat logging
        if round_num % 5 == 0 or round_num == 1:
            log_progress(results_dir, f"Progress: {strategy} Seed {seed} | Round {round_num}/{n_rounds} | PCC: {mean_pcc:.4f}")

        # --- ACQUIRE ---
        if round_num == n_rounds:
            break

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
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Save acquired barcodes
        acquired_barcodes = [dataset.barcodes[i] for i in acquired]
        strategy_acquisitions.append(acquired_barcodes)

        # Move acquired patches from pool to labeled
        labeled = np.concatenate([labeled, acquired])
        pool = np.array([i for i in pool if i not in set(acquired)])

    # Save exactly this seed's specific file sequentially to avoid JSON corruption
    seed_data = {
        "rounds": strategy_results,
        "acquisitions": strategy_acquisitions,
    }
    with open(seed_out_path, "w") as f:
        json.dump(seed_data, f, indent=2)

    log_progress(results_dir, f"FINISH: {strategy} Seed {seed}")
    return f"Completed {strategy} Seed {seed} (Final PCC: {mean_pcc:.4f})"


def init_worker(dataset):
    """Initializer to set the global dataset in each worker process."""
    global _GLOBAL_DATASET
    _GLOBAL_DATASET = dataset


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
    """Run the multi-seed Inv-SHAF active learning experiment."""

    if device_str is None:
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"

    print(f"Using device class: {device_str}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the Progress Log
    log_path = os.path.join(results_dir, "live_progress.txt")
    with open(log_path, "w") as f:
        f.write(f"=== ALIAS PROJECT LIVE PROGRESS LOG ===\nStarted: {time.ctime()}\n\n")

    # ========== LOAD DATASET ONCE ==========
    print("\n" + "=" * 60)
    print("LOADING GLOBAL DATASET (Shared across workers)")
    print("=" * 60)

    feat_dir = "data/cached_features"
    if features == "uni":
        feat_dir = "data/cached_features_uni"

    dataset = InvSHAFDataset(cached_features_dir=feat_dir)
    input_dim = dataset.features.shape[1]

    print(f"Dataset ready. Dimensions: {input_dim}D. Sharing via fork...")

    strategies = [
        "random", 
        "uncertainty", 
        "invariance", 
        "spatial_min", 
        "kmeans_core", 
        "adversarial_batch"
    ]

    # ========== PREPARE TASK PAYLOADS ==========
    task_args = []
    for strategy in strategies:
        for seed in seeds:
            task_args.append((
                strategy, seed, results_dir, n_rounds, acquire_pct, 
                alpha, beta, lr, device_str
            ))

    print("\n" + "=" * 60)
    print(f"LAUNCHING {len(task_args)} MULTIPROCESSING TASKS")
    print(f"Worker count: 32 (optimized for 128GB RAM)")
    print("=" * 60)

    # ========== MASSIVE MULTIPROCESSING ==========
    start_time = time.time()
    
    # Scale down to 32 workers for memory stability on 128GB RAM
    max_workers = min(32, os.cpu_count() or 1)
    
    # Use 'fork' on Linux for efficient memory sharing of the global dataset
    with ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=init_worker, 
        initargs=(dataset,)
    ) as executor:
        futures = {executor.submit(run_single_trial, args): args for args in task_args}
        
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                print(f"[{i+1}/{len(task_args)}] {result}")
            except Exception as e:
                print(f"[{i+1}/{len(task_args)}] ERROR: {str(e)}")

    duration = (time.time() - start_time) / 60
    print(f"\nAll parallel executions finished in {duration:.1f} minutes!")

    # ========== AGGREGATION ==========
    aggregate_results(results_dir, strategies, seeds, n_rounds)

    print("\n" + "=" * 60)
    print("MULTISEED EXPERIMENT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="uni", choices=["convnext", "uni"])
    parser.add_argument("--rounds", type=int, default=45)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    # Prevent potential multiprocessing freeze on macOS
    if sys.platform == 'darwin':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    run_experiment(
        n_rounds=args.rounds,
        beta=args.beta,
        features=args.features,
        acquire_pct=0.0020
    )
