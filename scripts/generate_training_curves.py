"""
Generate training curves visualization for Stage 5A analysis

This script creates a professional training curve visualization showing:
1. Test Recall@15 over epochs
2. Alpha (frequency prior) evolution
3. Beta (recency prior) evolution
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set modern, professional style
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 11

# Load Stage 5A results
results_path = Path("output/reports/stage5/stage5a_results.json")
if not results_path.exists():
    print(f"Error: {results_path} not found!")
    exit(1)

with open(results_path) as f:
    results = json.load(f)

# Extract training history
history = results.get("training_history", [])
if not history:
    print("No training history found in results!")
    exit(1)

epochs = [h["epoch"] for h in history]
test_recall = [
    h["test_metrics"]["recall@15"] * 100 for h in history
]  # Convert to percentage
alphas = [h.get("alpha", 0.5) for h in history]
betas = [h.get("beta", 0.5) for h in history]

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# ============================================================================
# Subplot 1: Test Recall@15 over time
# ============================================================================
axes[0].plot(
    epochs,
    test_recall,
    marker="o",
    linewidth=2.5,
    markersize=7,
    color="#1D3557",
    markerfacecolor="#457B9D",
    markeredgewidth=2,
    markeredgecolor="#1D3557",
    label="Test Recall@15",
)
axes[0].fill_between(epochs, test_recall, alpha=0.2, color="#457B9D")

# Mark best epoch
best_idx = np.argmax(test_recall)
axes[0].plot(
    epochs[best_idx],
    test_recall[best_idx],
    marker="*",
    markersize=20,
    color="#D62828",
    label=f"Best: {test_recall[best_idx]:.2f}% @ Epoch {epochs[best_idx]}",
    markeredgewidth=2,
    markeredgecolor="white",
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Recall@15 (%)")
axes[0].set_title("Stage 5A: Test Performance Over Time")
axes[0].legend(loc="best", frameon=True, fancybox=True, shadow=True)
axes[0].grid(True, alpha=0.3)

# ============================================================================
# Subplot 2: Alpha (Frequency Prior) evolution
# ============================================================================
axes[1].plot(
    epochs,
    alphas,
    marker="s",
    linewidth=2.5,
    markersize=7,
    color="#6A040F",
    markerfacecolor="#DC2F02",
    markeredgewidth=2,
    markeredgecolor="#6A040F",
    label="α (Frequency Weight)",
)
axes[1].fill_between(epochs, alphas, alpha=0.2, color="#DC2F02")

# Add reference line at initial value
axes[1].axhline(
    alphas[0],
    color="#6A040F",
    linestyle="--",
    linewidth=2,
    alpha=0.5,
    label=f"Initial α: {alphas[0]:.3f}",
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("α (Frequency Weight)")
axes[1].set_title("Frequency Prior Weight Evolution")
axes[1].legend(loc="best", frameon=True, fancybox=True, shadow=True)
axes[1].grid(True, alpha=0.3)

# ============================================================================
# Subplot 3: Beta (Recency Prior) evolution
# ============================================================================
axes[2].plot(
    epochs,
    betas,
    marker="^",
    linewidth=2.5,
    markersize=7,
    color="#006D77",
    markerfacecolor="#06A77D",
    markeredgewidth=2,
    markeredgecolor="#006D77",
    label="β (Recency Weight)",
)
axes[2].fill_between(epochs, betas, alpha=0.2, color="#06A77D")

# Add reference line at initial value
axes[2].axhline(
    betas[0],
    color="#006D77",
    linestyle="--",
    linewidth=2,
    alpha=0.5,
    label=f"Initial β: {betas[0]:.3f}",
)

axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("β (Recency Weight)")
axes[2].set_title("Recency Prior Weight Evolution")
axes[2].legend(loc="best", frameon=True, fancybox=True, shadow=True)
axes[2].grid(True, alpha=0.3)

# ============================================================================
# Save figure
# ============================================================================
plt.tight_layout()
output_path = Path("output/figures/stage5a_training_curves.png")
output_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved training curves to: {output_path}")
plt.close()

# Print summary statistics
print("\n" + "=" * 60)
print("STAGE 5A TRAINING SUMMARY")
print("=" * 60)
print(f"Total epochs: {len(epochs)}")
print(
    f"Best Recall@15: {max(test_recall):.2f}% (Epoch {epochs[np.argmax(test_recall)]})"
)
print(f"Final Recall@15: {test_recall[-1]:.2f}%")
print(
    f"\nAlpha evolution: {alphas[0]:.3f} -> {alphas[-1]:.3f} ({((alphas[-1]/alphas[0])-1)*100:+.1f}%)"
)
print(
    f"Beta evolution: {betas[0]:.3f} -> {betas[-1]:.3f} ({((betas[-1]/betas[0])-1)*100:+.1f}%)"
)
print("=" * 60)
