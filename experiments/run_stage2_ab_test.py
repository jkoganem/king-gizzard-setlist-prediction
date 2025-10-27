#!/usr/bin/env python3
"""
Stage 2: XGBoost A/B Test (Filtered vs Unfiltered)

Based on Stage 1 findings:
- Logistic: 21.91% (unfiltered) vs 18.31% (filtered)
- Need to test if XGBoost also benefits from unfiltered data

This script:
1. Runs Optuna optimization on BOTH filtered and unfiltered datasets
2. 1000 trials each for thorough exploration
3. Compares results to determine best configuration
4. Saves the better model
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.dataio import load_all_data, save_model
from src.features.engineer_features import engineer_all_features
from src.utils.constants import (
    SPECIALIZED_SHOW_DATES,
    ORCHESTRA_SHOW_DATES,
    RAVE_SHOW_DATES,
)
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("STAGE 2: XGBOOST A/B TEST (FILTERED vs UNFILTERED)")
print("=" * 80)
print("\nBased on Stage 1 results:")
print("  Logistic (unfiltered): 21.91% Recall@15")
print("  Logistic (filtered):   18.31% Recall@15")
print("  -> Unfiltered performs better with 70/15/15 split")
print("\nGoal: Test if XGBoost also benefits from unfiltered data")
print("  Running 1000 Optuna trials for each configuration")
print()

# Configuration
N_TRIALS = 1000
RANDOM_STATE = 42
N_JOBS = 4


def run_optimization(dataset_shows, filter_name, shows, songs, setlists, song_tags):
    """Run Optuna optimization for a dataset configuration."""

    print("\n" + "=" * 80)
    print(f"OPTIMIZING: {filter_name.upper()}")
    print("=" * 80)
    print(f"Shows: {len(dataset_shows)}")

    # Temporal split (70% train, 15% val, 15% test)
    n_train = int(len(dataset_shows) * 0.70)
    n_val = int(len(dataset_shows) * 0.85)

    train_shows = dataset_shows.iloc[:n_train]
    val_shows = dataset_shows.iloc[n_train:n_val]
    test_shows = dataset_shows.iloc[n_val:]
    train_show_ids = set(train_shows["show_id"].values)

    print(f"Train: {len(train_shows)} shows")
    print(f"Val: {len(val_shows)} shows")
    print(f"Test: {len(test_shows)} shows")

    # Engineer features
    print("\nEngineering features...")
    df, feature_cols = engineer_all_features(
        shows=dataset_shows,
        songs=songs,
        setlists=setlists,
        song_tags=song_tags,
        train_show_ids=train_show_ids,
        include_residency_features=True,
    )

    df_train = df[df["show_id"].isin(train_show_ids)]
    val_show_ids = set(val_shows["show_id"].values)
    test_show_ids = set(test_shows["show_id"].values)
    df_val = df[df["show_id"].isin(val_show_ids)]
    df_test = df[df["show_id"].isin(test_show_ids)]

    X_train, y_train = df_train[feature_cols].values, df_train["label"].values
    X_val, y_val = df_val[feature_cols].values, df_val["label"].values
    X_test, y_test = df_test[feature_cols].values, df_test["label"].values

    print(f"Train examples: {len(X_train):,}")
    print(f"Val examples: {len(X_val):,}")
    print(f"Test examples: {len(X_test):,}")

    # Optuna objective function
    def objective(trial):
        """Optuna objective function for XGBoost hyperparameter tuning."""
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "nthread": N_JOBS,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }

        # Train model on training set, validate on validation set
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Predict on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate Recall@K with marathon-aware K values
        val_df = df_val[["show_id", "song_id"]].copy()
        val_df["y_pred"] = y_pred_proba
        val_df["y_true"] = y_val

        # Identify marathon shows
        marathon_show_ids = set(
            dataset_shows[dataset_shows["is_marathon"] == 1]["show_id"].values
        )

        show_recalls = []
        for show_id in val_df["show_id"].unique():
            show_data = val_df[val_df["show_id"] == show_id]
            is_marathon = show_id in marathon_show_ids

            # Dynamic K: 24 for marathon, 15 for regular
            k = 24 if is_marathon else 15
            top_k = show_data.nlargest(k, "y_pred")["song_id"].values
            actual = show_data[show_data["y_true"] == 1]["song_id"].values

            if len(actual) > 0:
                hits = len(set(top_k) & set(actual))
                show_recalls.append(hits / len(actual))

        recall_at_k = np.mean(show_recalls) if show_recalls else 0.0
        return recall_at_k

    # Create Optuna study
    print(f"\nStarting Optuna optimization ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgboost_{filter_name}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    # Run optimization
    start_time = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    optimization_time = time.time() - start_time

    print(
        f"\nOptimization completed in {optimization_time:.1f}s ({optimization_time/60:.1f} min)"
    )
    print(f"Best trial: {study.best_trial.number}")
    print(
        f"Best validation Recall@15: {study.best_value:.4f} ({study.best_value*100:.2f}%)"
    )

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "nthread": N_JOBS,
        }
    )

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate on test set
    print("\nTest Set Evaluation:")
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    # Standard metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    # Recall@K with marathon-aware K values
    test_df = df_test[["show_id", "song_id"]].copy()
    test_df["y_pred"] = y_pred_proba
    test_df["y_true"] = y_test

    # Identify marathon shows
    marathon_show_ids = set(
        dataset_shows[dataset_shows["is_marathon"] == 1]["show_id"].values
    )

    recalls = {}
    recalls_regular = {}
    recalls_marathon = {}

    for k in [5, 10, 15, 20, 24]:
        show_recalls = []
        regular_recalls = []
        marathon_recalls = []

        for show_id in test_df["show_id"].unique():
            show_data = test_df[test_df["show_id"] == show_id]
            is_marathon = show_id in marathon_show_ids

            # Use dynamic K: 24 for marathon, 15 for regular (when computing @15)
            if k == 15:
                dynamic_k = 24 if is_marathon else 15
            else:
                dynamic_k = k

            top_k = show_data.nlargest(dynamic_k, "y_pred")["song_id"].values
            actual = show_data[show_data["y_true"] == 1]["song_id"].values

            if len(actual) > 0:
                hits = len(set(top_k) & set(actual))
                recall = hits / len(actual)
                show_recalls.append(recall)

                # Track separately for regular vs marathon
                if k == 15:
                    if is_marathon:
                        marathon_recalls.append(recall)
                    else:
                        regular_recalls.append(recall)

        recalls[k] = np.mean(show_recalls) if show_recalls else 0.0

        if k == 15:
            recalls_regular[15] = np.mean(regular_recalls) if regular_recalls else 0.0
            recalls_marathon[24] = (
                np.mean(marathon_recalls) if marathon_recalls else 0.0
            )

    print(f"  AUC: {auc:.4f}")
    print(f"  Overall Recall@K: {recalls[15]:.4f} ({recalls[15]*100:.2f}%)")
    print(
        f"    Regular shows (K=15): {recalls_regular[15]:.4f} ({recalls_regular[15]*100:.2f}%)"
    )
    print(
        f"    Marathon shows (K=24): {recalls_marathon[24]:.4f} ({recalls_marathon[24]*100:.2f}%)"
    )

    return {
        "filter": filter_name,
        "optimization_time": optimization_time,
        "n_trials": N_TRIALS,
        "best_trial": study.best_trial.number,
        "best_validation_recall_at_15": study.best_value,
        "best_params": study.best_params,
        "test_metrics": {
            "auc": auc,
            "logloss": logloss,
            "recall@5": recalls[5],
            "recall@10": recalls[10],
            "recall@15": recalls[15],
            "recall@20": recalls[20],
        },
        "model": final_model,
        "feature_cols": feature_cols,
        "study": study,
    }


# Load data
print("\nLoading data...")
shows, songs, setlists, song_tags = load_all_data()
shows["date"] = pd.to_datetime(shows["date"])
shows = shows.sort_values("date")

# Prepare datasets
recent_shows = shows[shows["date"] >= "2022-01-01"].copy()
specialized_dates = set(SPECIALIZED_SHOW_DATES + ORCHESTRA_SHOW_DATES + RAVE_SHOW_DATES)
specialized_dates = [pd.to_datetime(d) for d in specialized_dates]

# A: Filtered (remove specialized shows)
recent_filtered = recent_shows[~recent_shows["date"].isin(specialized_dates)].copy()

# B: Unfiltered (keep all shows)
recent_unfiltered = recent_shows.copy()

print(f"\nDataset sizes:")
print(f"  Unfiltered: {len(recent_unfiltered)} shows")
print(f"  Filtered:   {len(recent_filtered)} shows")

# Run both experiments
results = {}

# Experiment A: Unfiltered (based on Stage 1, this should be better)
results["unfiltered"] = run_optimization(
    recent_unfiltered, "unfiltered", shows, songs, setlists, song_tags
)

# Experiment B: Filtered
results["filtered"] = run_optimization(
    recent_filtered, "filtered", shows, songs, setlists, song_tags
)

# Compare and save best
print("\n" + "=" * 80)
print("A/B TEST RESULTS")
print("=" * 80)

print(f'\n{"Config":<15} {"Val Recall@15":<20} {"Test Recall@15":<20} {"Time":<15}')
print("-" * 70)

for config, res in results.items():
    val_recall = f"{res['best_validation_recall_at_15']*100:.2f}%"
    test_recall = f"{res['test_metrics']['recall@15']*100:.2f}%"
    opt_time = f"{res['optimization_time']/60:.1f} min"
    print(f"{config.upper():<15} {val_recall:<20} {test_recall:<20} {opt_time:<15}")

# Determine winner
if (
    results["unfiltered"]["test_metrics"]["recall@15"]
    > results["filtered"]["test_metrics"]["recall@15"]
):
    winner = "unfiltered"
    print(
        f'\nWINNER: UNFILTERED ({results["unfiltered"]["test_metrics"]["recall@15"]*100:.2f}%)'
    )
else:
    winner = "filtered"
    print(
        f'\nWINNER: FILTERED ({results["filtered"]["test_metrics"]["recall@15"]*100:.2f}%)'
    )

# Save best model
output_dir = Path("output")
models_dir = output_dir / "models" / "stage2"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "xgboost_tuned_stage2.pkl"
save_model(results[winner]["model"], model_path)
print(f"\nBest model saved to: {model_path}")

# Save results
reports_dir = output_dir / "reports" / "stage2"
reports_dir.mkdir(parents=True, exist_ok=True)

# Save detailed results
results_to_save = {
    "winner": winner,
    "unfiltered": {
        k: v
        for k, v in results["unfiltered"].items()
        if k not in ["model", "feature_cols", "study"]
    },
    "filtered": {
        k: v
        for k, v in results["filtered"].items()
        if k not in ["model", "feature_cols", "study"]
    },
}

results_path = reports_dir / "stage2_ab_test_results.json"
with open(results_path, "w") as f:
    json.dump(results_to_save, f, indent=2)

print(f"Results saved to: {results_path}")

# Generate visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

figures_dir = output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Use the winning configuration's study for visualizations
winner_study = results[winner]["study"]

# 1. Optuna optimization history
try:
    fig = plot_optimization_history(winner_study)
    fig.write_image(str(figures_dir / "stage2_optuna_history.png"))
    print(f'Saved: {figures_dir / "stage2_optuna_history.png"}')
except Exception as e:
    print(f"Warning: Could not generate optuna history plot: {e}")

# 2. Feature importance plots (gain, weight, cover) - Enhanced with cumulative plots
try:
    plt.style.use("seaborn-v0_8-paper")
    feature_cols = results[winner]["feature_cols"]

    # Generate 3 different importance types
    importance_types = [
        ("gain", "Gain", "Total gain of splits using this feature"),
        ("weight", "Weight", "Number of times feature is used in splits"),
        ("cover", "Cover", "Average coverage of splits using this feature"),
    ]

    for imp_type, imp_name, imp_desc in importance_types:
        importance = (
            results[winner]["model"].get_booster().get_score(importance_type=imp_type)
        )

        if importance:
            # Map feature names
            imp_df = pd.DataFrame(
                [
                    {
                        "feature": f"f{int(k[1:])}",
                        "feature_name": feature_cols[int(k[1:])],
                        "importance": v,
                    }
                    for k, v in importance.items()
                ]
            ).sort_values("importance", ascending=False)

            # Calculate cumulative importance
            imp_df["cumulative"] = (
                imp_df["importance"].cumsum() / imp_df["importance"].sum() * 100
            )

            # Find how many features for 90% and 95%
            n_90 = (imp_df["cumulative"] <= 90).sum() + 1
            n_95 = (imp_df["cumulative"] <= 95).sum() + 1

            if imp_type == "gain":
                print(f"\nFeature importance analysis ({imp_name}):")
                print(f"  Total features: {len(feature_cols)}")
                print(f"  Features for 90% importance: {n_90}")
                print(f"  Features for 95% importance: {n_95}")

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # LEFT: Top 20 features with colorful gradient
            top_n = 20
            top_features = imp_df.head(top_n)
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, top_n))

            ax1.barh(
                range(len(top_features)),
                top_features["importance"].values,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
            )
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features["feature_name"].values)
            ax1.invert_yaxis()
            ax1.set_xlabel(f"Importance ({imp_name})", fontsize=12, fontweight="bold")
            ax1.set_title(
                f"Top {top_n} Features - {imp_name} Importance\n(XGBoost Stage 2 - {len(feature_cols)} total features)",
                fontsize=13,
                fontweight="bold",
            )
            ax1.grid(axis="x", alpha=0.3, linestyle="--")

            # Add value labels on bars
            for i, (idx, row) in enumerate(top_features.iterrows()):
                ax1.text(
                    row["importance"],
                    i,
                    f" {row['importance']:.0f}",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            # RIGHT: Cumulative importance
            x = np.arange(len(imp_df))
            ax2.plot(
                x,
                imp_df["cumulative"].values,
                linewidth=2.5,
                color="steelblue",
                label="Cumulative Importance",
            )
            ax2.fill_between(
                x, 0, imp_df["cumulative"].values, alpha=0.3, color="steelblue"
            )

            # Add horizontal lines for 90% and 95%
            ax2.axhline(
                y=90,
                color="orange",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="90% threshold",
            )
            ax2.axhline(
                y=95,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="95% threshold",
            )

            # Add vertical lines
            ax2.axvline(
                x=n_90 - 1, color="orange", linestyle=":", linewidth=1.5, alpha=0.6
            )
            ax2.axvline(
                x=n_95 - 1, color="red", linestyle=":", linewidth=1.5, alpha=0.6
            )

            # Annotate
            ax2.text(
                n_90 - 1,
                92,
                f"{n_90} features",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="orange",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="orange",
                    alpha=0.8,
                ),
            )
            ax2.text(
                n_95 - 1,
                97,
                f"{n_95} features",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="red",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="red",
                    alpha=0.8,
                ),
            )

            ax2.set_xlabel(
                "Number of Features (ranked by importance)",
                fontsize=12,
                fontweight="bold",
            )
            ax2.set_ylabel("Cumulative Importance (%)", fontsize=12, fontweight="bold")
            ax2.set_title(
                f"Cumulative {imp_name} Importance\n({imp_desc})",
                fontsize=13,
                fontweight="bold",
            )
            ax2.grid(alpha=0.3, linestyle="--")
            ax2.legend(loc="lower right", fontsize=10)
            ax2.set_xlim(-1, len(imp_df))
            ax2.set_ylim(0, 105)

            # Add text showing total features
            ax2.text(
                0.98,
                0.02,
                f"Total: {len(feature_cols)} features",
                transform=ax2.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()

            fig_path = figures_dir / f"stage2_feature_importance_{imp_type}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {fig_path}")
except Exception as e:
    print(f"Warning: Could not generate feature importance plots: {e}")

print("\n" + "=" * 80)
print("STAGE 2 A/B TEST COMPLETED")
print("=" * 80)
print()
