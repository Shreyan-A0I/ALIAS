"""
Visualization module for Inv-SHAF experiment results.

Generates:
  1. Primary: Learning curves (PCC vs % labeled, 5 lines)
  2. Secondary: Per-gene PCC box plots at 10%, 20%, 30%
  3. Quaternary: Score decomposition over rounds (Combined strategy)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Color scheme
COLORS = {
    "baseline": "#888888",
    "random": "#7f8c8d",
    "uncertainty": "#3498db",
    "invariance": "#e67e22",
    "combined": "#e74c3c",
}
LABELS = {
    "baseline": "Baseline (100% pool)",
    "random": "Random",
    "uncertainty": "Uncertainty (MC Dropout)",
    "invariance": "Invariance Violation",
    "combined": "Combined (Ours)",
}


def plot_learning_curves(results, output_dir="results"):
    """Primary plot: PCC vs % labeled with 5 lines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_pcc = results["baseline"]["mean_pcc"]

    # Baseline horizontal line
    ax.axhline(y=baseline_pcc, color=COLORS["baseline"], linestyle="--",
               linewidth=1.5, label=LABELS["baseline"], alpha=0.8)

    # Strategy curves
    for strategy in ["random", "uncertainty", "invariance", "combined"]:
        if strategy not in results["strategies"]:
            continue
        rounds = results["strategies"][strategy]["rounds"]
        pcts = [r["pct_labeled"] for r in rounds]
        pccs = [r["mean_pcc"] for r in rounds]

        lw = 2.5 if strategy == "combined" else 1.5
        ax.plot(pcts, pccs, color=COLORS[strategy], linewidth=lw,
                label=LABELS[strategy], marker="o", markersize=3)

    ax.set_xlabel("% of Total Data Labeled", fontsize=12)
    ax.set_ylabel("Mean PCC (100 Genes)", fontsize=12)
    ax.set_title("Inv-SHAF: Active Learning Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(5))

    plt.tight_layout()
    path = os.path.join(output_dir, "learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_gene_boxplots(results, output_dir="results"):
    """Secondary plot: Per-gene PCC distributions at 10%, 20%, 30%."""
    checkpoints = [10, 20, 30]

    fig, axes = plt.subplots(1, len(checkpoints), figsize=(15, 5), sharey=True)

    strategies = ["random", "uncertainty", "invariance", "combined"]

    for i, pct_target in enumerate(checkpoints):
        ax = axes[i]
        data_to_plot = []
        labels_to_plot = []

        for strategy in strategies:
            if strategy not in results["strategies"]:
                continue
            rounds = results["strategies"][strategy]["rounds"]

            # Find the round closest to target percentage
            closest = min(rounds, key=lambda r: abs(r["pct_labeled"] - pct_target))
            per_gene = closest["per_gene_pcc"]

            data_to_plot.append(per_gene)
            labels_to_plot.append(LABELS[strategy])

        bp = ax.boxplot(data_to_plot, labels=[l.split(" ")[0] for l in labels_to_plot],
                        patch_artist=True, widths=0.6)

        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[strategies[j]])
            patch.set_alpha(0.6)

        ax.set_title(f"{pct_target}% Labeled", fontsize=11, fontweight="bold")
        ax.set_ylabel("Per-Gene PCC" if i == 0 else "")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Per-Gene PCC Distribution at Key Checkpoints",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "per_gene_boxplots.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_score_decomposition(results, output_dir="results"):
    """Quaternary plot: How U_MC and V_Inv of acquired patches shift over rounds."""
    if "combined" not in results["strategies"]:
        return

    scores = results["strategies"]["combined"]["scores"]
    rounds_data = results["strategies"]["combined"]["rounds"]

    rounds = [r["round"] for r in rounds_data]
    u_mc_means = [s.get("mean_u_mc_acquired", 0.0) for s in scores]
    v_inv_means = [s.get("mean_v_inv_acquired", 0.0) for s in scores]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(rounds, u_mc_means, color=COLORS["uncertainty"], linewidth=2,
             marker="s", markersize=4, label="Mean U_MC (acquired)")
    ax1.set_xlabel("AL Round", fontsize=12)
    ax1.set_ylabel("Mean U_MC", fontsize=12, color=COLORS["uncertainty"])

    ax2 = ax1.twinx()
    ax2.plot(rounds, v_inv_means, color=COLORS["invariance"], linewidth=2,
             marker="^", markersize=4, label="Mean V_Inv (acquired)")
    ax2.set_ylabel("Mean V_Inv", fontsize=12, color=COLORS["invariance"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title("Score Decomposition Over AL Rounds (Combined Strategy)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "score_decomposition.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def generate_all_plots(results_path="results/all_results.json", output_dir="results"):
    """Load results and generate all plots."""
    with open(results_path) as f:
        results = json.load(f)

    plot_learning_curves(results, output_dir)
    plot_per_gene_boxplots(results, output_dir)
    plot_score_decomposition(results, output_dir)

    print("\nAll plots generated!")


if __name__ == "__main__":
    generate_all_plots()
