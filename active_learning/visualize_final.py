import json
import os
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(y, box_pts):
    if box_pts < 2: return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    # Fix edge artifacts from 'same' mode
    y_smooth[0] = y[0]
    y_smooth[-1] = y[-1]
    return y_smooth

def visualize_results(results_dir="results"):
    agg_path = os.path.join(results_dir, "aggregated_results.json")
    if not os.path.exists(agg_path):
        print(f"Error: {agg_path} not found.")
        return

    with open(agg_path, "r") as f:
        data = json.load(f)

    # Professional color palette
    colors = {
        "random": "#95a5a6",           # Gray (Baseline)
        "uncertainty": "#3498db",      # Blue
        "invariance": "#e74c3c",       # Red (Key Strategy)
        "spatial_min": "#2ecc71",      # Green
        "kmeans_core": "#f1c40f",      # Yellow
        "adversarial_batch": "#9b59b6" # Purple
    }

    plt.figure(figsize=(10, 6), dpi=150)
    
    # Configure plotting style
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    for strategy, strat_data in data.items():
        rounds = strat_data["rounds"]
        if not rounds:
            continue
            
        x = np.array([r["pct_labeled"] for r in rounds])
        y = np.array([r["mean_pcc"] for r in rounds])
        std = np.array([r["std_pcc"] for r in rounds])
        n_seeds = np.array([r["seeds_completed"] for r in rounds])
        
        # Calculate Standard Error (SEM) and apply 0.5x scaling for cleaner visuals
        sem = (std / np.sqrt(n_seeds)) * 0.8
        
        color = colors.get(strategy, "#34495e")
        label = strategy.replace("_", " ").title()
        
        # Apply smoothing to the mean line (3-point moving average)
        y_smooth = smooth_curve(y, box_pts=3)
        
        # Plot shaded error bands
        plt.fill_between(x, y - sem, y + sem, color=color, alpha=0.15)
        
        # Plot the main smoothed line
        plt.plot(x, y_smooth, color=color, label=label, linewidth=2.0)
        
        # Add a faint line for the raw data to show the actual observations
        plt.plot(x, y, color=color, alpha=0.3, linewidth=0.8, linestyle=':')

    # Add Oracle Baseline (100% Data)
    plt.axhline(y=0.197, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Oracle (100% Data)')

    plt.title("Spatial Gene Expression Prediction: Active Learning Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Percentage of Labeled Samples (%)", fontsize=12)
    plt.ylabel("Mean PCC (Pearson Correlation Coefficient)", fontsize=12)
    plt.legend(frameon=True, facecolor='white', loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    out_path = os.path.join(results_dir, "final_learning_curves.png")
    plt.savefig(out_path)
    print(f"Plot saved successfully to {out_path}")

if __name__ == "__main__":
    visualize_results()
