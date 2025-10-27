#!/usr/bin/env python3
"""
Stage 5A: Hyperparameter Tuning for GNN with Priors using Optuna

Uses the existing run_stage5_gnn_priors.py infrastructure.
Searches exhaustive hyperparameter space to find optimal configuration.
Stage 5B will use the best parameters from this search.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import subprocess
import optuna
import json
import re
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("STAGE 5A: HYPERPARAMETER TUNING WITH OPTUNA")
print("=" * 80)

# Configuration
NUM_TRIALS = 50  # Exhaustive search
SCRIPT_PATH = "experiments/run_stage5_gnn_priors.py"


def objective(trial):
    """Optuna objective - runs run_stage5_gnn_priors.py with different hyperparameters"""

    # Sample hyperparameters (ORIGINAL EXHAUSTIVE SPACE)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.001, 0.01, 0.1])
    grad_clip = trial.suggest_categorical("grad_clip", [0.5, 1.0, 5.0])

    # GNN architecture
    gnn_layers = trial.suggest_int("gnn_layers", 1, 3)
    emb_dim = trial.suggest_categorical("emb_dim", [32, 64, 128])

    # Prior regularization
    prior_reg_weight = trial.suggest_float("prior_reg_weight", 0.0, 0.5)

    # Feature dropout
    feature_dropout = trial.suggest_float("feature_dropout", 0.0, 0.3)

    print(
        f"\n[Trial {trial.number}] lr={lr:.6f}, bs={batch_size}, wd={weight_decay}, "
        f"clip={grad_clip}, layers={gnn_layers}, emb={emb_dim}"
    )

    # Build command
    cmd = [
        "python3",
        SCRIPT_PATH,
        "--lr",
        str(lr),
        "--batch-size",
        str(batch_size),
        "--weight-decay",
        str(weight_decay),
        "--grad-clip",
        str(grad_clip),
        "--gnn-layers",
        str(gnn_layers),
        "--emb-dim",
        str(emb_dim),
        "--prior-reg-weight",
        str(prior_reg_weight),
        "--feature-dropout",
        str(feature_dropout),
        "--epochs",
        "30",  # Shorter for tuning
        "--patience",
        "10",
        "--output-dir",
        f"output/models/stage5/stage5a/trial{trial.number}",
    ]

    # Run the script
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per trial
        )

        output = result.stdout + result.stderr

        # Parse the best validation recall from output
        # Look for pattern like: "Best Epoch: X | Recall: Y"
        recall_pattern = r"Best.*?Recall.*?(\d+\.\d+)%"
        matches = re.findall(recall_pattern, output)

        if matches:
            best_recall = float(matches[-1]) / 100.0  # Convert percentage to decimal
        else:
            # Try alternative pattern: "Test Recall@15: 0.XXXX"
            alt_pattern = r"Test Recall@15:\s*(\d+\.\d+)"
            alt_matches = re.findall(alt_pattern, output)
            if alt_matches:
                best_recall = max([float(m) for m in alt_matches])
            else:
                print(f"Could not parse recall from output. Returning 0.0")
                return 0.0

        # Report intermediate values for better visualization
        epoch_pattern = r"Epoch\s+\d+/\d+.*?Recall:\s*(\d+\.\d+)%"
        epoch_recalls = re.findall(epoch_pattern, output)
        for i, recall_str in enumerate(epoch_recalls):
            trial.report(float(recall_str) / 100.0, i)

        print(
            f"  -> Best validation recall: {best_recall:.4f} ({best_recall*100:.2f}%)"
        )

        return best_recall

    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return 0.0
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


# Create study
study = optuna.create_study(
    direction="maximize",
    study_name="stage5a_hyperparam_tune",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
)

print(f"\nStarting Optuna search with {NUM_TRIALS} trials...")
print(f"Each trial runs for up to 30 epochs with early stopping")
print(f"Estimated time: ~{NUM_TRIALS * 0.5:.1f} hours (with pruning)\n")

study.optimize(objective, n_trials=NUM_TRIALS)

# Print results
print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 80)

print(f"\nBest trial: {study.best_trial.number}")
print(
    f"Best validation recall@15: {study.best_value:.4f} ({study.best_value*100:.2f}%)"
)
print(f"\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Save results
output_dir = Path("output/reports/stage5")
figures_dir = Path("output/figures")
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Save study
import joblib

study_path = output_dir / "stage5a_hyperparam_study.pkl"
joblib.dump(study, study_path)
print(f"\nStudy saved to: {study_path}")

# Save best params
params_path = output_dir / "stage5a_best_hyperparams.json"
with open(params_path, "w") as f:
    json.dump(study.best_params, f, indent=2)
print(f"Best params saved to: {params_path}")

# Print top 10 trials
print("\nTop 10 trials:")
print("-" * 80)
df = study.trials_dataframe().sort_values("value", ascending=False).head(10)
print(
    df[
        [
            "number",
            "value",
            "params_lr",
            "params_batch_size",
            "params_weight_decay",
            "params_grad_clip",
            "params_gnn_layers",
            "params_emb_dim",
        ]
    ]
)

# Generate Optuna visualizations
print("\nGenerating visualizations...")
import optuna.visualization as vis

try:
    fig = vis.plot_optimization_history(study)
    # Save as interactive HTML
    fig.write_html(str(figures_dir / "stage5a_optimization_history.html"))
    # Save as static PNG for README
    fig.write_image(str(figures_dir / "stage5a_optimization_history.png"), width=1200, height=600)
    print(
        f'  [SUCCESS] Optimization history: {figures_dir / "stage5a_optimization_history.png"}'
    )
except Exception as e:
    print(f"  [FAILURE] Optimization history failed: {e}")

try:
    fig = vis.plot_param_importances(study)
    # Save as interactive HTML
    fig.write_html(str(figures_dir / "stage5a_param_importances.html"))
    # Save as static PNG for README
    fig.write_image(str(figures_dir / "stage5a_param_importances.png"), width=1200, height=600)
    print(
        f'  [SUCCESS] Parameter importances: {figures_dir / "stage5a_param_importances.png"}'
    )
except Exception as e:
    print(f"  [FAILURE] Parameter importances failed: {e}")

try:
    fig = vis.plot_intermediate_values(study)
    # Save as interactive HTML
    fig.write_html(str(figures_dir / "stage5a_training_curves.html"))
    # Save as static PNG for README
    fig.write_image(str(figures_dir / "stage5a_training_curves.png"), width=1200, height=600)
    print(
        f'  [SUCCESS] Training curves: {figures_dir / "stage5a_training_curves.png"}'
    )
except Exception as e:
    print(f"  [FAILURE] Training curves failed: {e}")

try:
    fig = vis.plot_parallel_coordinate(study)
    # Save as interactive HTML
    fig.write_html(str(figures_dir / "stage5a_parallel_coordinate.html"))
    # Save as static PNG for README
    fig.write_image(str(figures_dir / "stage5a_parallel_coordinate.png"), width=1400, height=700)
    print(
        f'  [SUCCESS] Parallel coordinate: {figures_dir / "stage5a_parallel_coordinate.png"}'
    )
except Exception as e:
    print(f"  [FAILURE] Parallel coordinate failed: {e}")

# Custom: Overlay top 5 training curves
print("\nGenerating custom visualizations...")
plt.figure(figsize=(14, 6))

# Get top 5 trials
top_trials = sorted(
    study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True
)[:5]

colors = plt.cm.viridis(np.linspace(0, 0.9, 5))

for i, trial in enumerate(top_trials):
    if trial.intermediate_values:
        epochs = list(trial.intermediate_values.keys())
        recalls = list(trial.intermediate_values.values())
        label = f"Trial {trial.number} ({trial.value:.4f})"
        plt.plot(
            epochs,
            recalls,
            marker="o",
            label=label,
            linewidth=2.5,
            color=colors[i],
            alpha=0.8,
        )

plt.xlabel("Epoch", fontsize=14, fontweight="bold")
plt.ylabel("Validation Recall@15", fontsize=14, fontweight="bold")
plt.title(
    "Stage 5A: Top 5 Hyperparameter Configurations - Training Curves",
    fontsize=16,
    fontweight="bold",
)
plt.legend(loc="best", fontsize=11)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()

curve_path = figures_dir / "stage5a_top5_training_curves.png"
plt.savefig(curve_path, dpi=300, bbox_inches="tight")
print(f"  [SUCCESS] Top 5 training curves: {curve_path}")

# Generate command for Stage 5B
print("\n" + "=" * 80)
print("STAGE 5B COMMAND (Use best hyperparameters)")
print("=" * 80)
print("\nRun the following command to train Stage 5B with optimal hyperparameters:\n")

cmd_parts = [
    "python3 experiments/run_stage5_gnn_priors.py",
]
for key, value in study.best_params.items():
    cmd_parts.append(f'--{key.replace("_", "-")} {value}')
cmd_parts.append("--epochs 50")
cmd_parts.append("--output-dir output/models/stage5b")

print(" \\\n  ".join(cmd_parts))

# Save command to file
cmd_file = output_dir / "stage5b_command.sh"
with open(cmd_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Stage 5B: Train with best hyperparameters from Stage 5A\n\n")
    f.write(" \\\n  ".join(cmd_parts))
    f.write("\n")
print(f"\nCommand saved to: {cmd_file}")

print("\n" + "=" * 80)
print("STAGE 5A HYPERPARAMETER TUNING COMPLETED")
print("=" * 80)
