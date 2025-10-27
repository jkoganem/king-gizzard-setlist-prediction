#!/usr/bin/env python3
"""Run Stage 1 factorial experiments"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.dataio import load_all_data
from src.features.engineer_features import engineer_all_features
from src.models.stage1.logistic import LogisticRegression
from src.models.stage1.random_forest import RandomForest
from src.models.stage1.xgboost import XGBoostModel
from src.utils.constants import (
    SPECIALIZED_SHOW_DATES,
    ORCHESTRA_SHOW_DATES,
    RAVE_SHOW_DATES,
)
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import time
import joblib

print("=" * 80)
print("STAGE 1: TRADITIONAL ML FACTORIAL EXPERIMENTS")
print("=" * 80)
print("\nFactorial Design: 3 models x 2 datasets x 2 filters = 12 experiments")
print("Models: Logistic, Random Forest, XGBoost")
print("Datasets: recent (2022+), full (all)")
print("Filters: no_filter, remove_specialized")
print()

# Load data once
print("Loading data...")
shows, songs, setlists, song_tags = load_all_data()
shows["date"] = pd.to_datetime(shows["date"])
shows = shows.sort_values("date")

# Specialized show dates
specialized_dates = set(SPECIALIZED_SHOW_DATES + ORCHESTRA_SHOW_DATES + RAVE_SHOW_DATES)

results = []

# Factorial loop
for dataset_name in ["recent", "full"]:
    for filter_specialized in [False, True]:
        # Filter dataset
        if dataset_name == "recent":
            dataset_shows = shows[shows["date"] >= "2022-01-01"].copy()
        else:
            dataset_shows = shows.copy()

        if filter_specialized:
            dataset_shows = dataset_shows[
                ~dataset_shows["date"].isin(specialized_dates)
            ]

        # Temporal split (70% train, 15% val, 15% test)
        n_train = int(len(dataset_shows) * 0.70)
        n_val = int(len(dataset_shows) * 0.85)

        train_shows = dataset_shows.iloc[:n_train]
        val_shows = dataset_shows.iloc[n_train:n_val]
        test_shows = dataset_shows.iloc[n_val:]
        train_show_ids = set(train_shows["show_id"].values)

        print(f'\n{"="*80}')
        print(
            f'Dataset: {dataset_name} | Filter: {"yes" if filter_specialized else "no"}'
        )
        print(
            f"Shows: {len(dataset_shows)} (train: {len(train_shows)}, val: {len(val_shows)}, test: {len(test_shows)})"
        )

        # Engineer features
        print("Engineering features...")
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

        # Create models directory
        models_dir = Path("output/models/stage1")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Train each model
        for model_name in ["logistic", "random_forest", "xgboost"]:
            print(f"\n  Training {model_name}...")
            start_time = time.time()

            if model_name == "logistic":
                model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=4)
            elif model_name == "random_forest":
                model = RandomForest(
                    n_estimators=200, max_depth=10, random_state=42, n_jobs=4
                )
            else:  # xgboost
                model = XGBoostModel(
                    n_estimators=300, max_depth=6, learning_rate=0.05, nthread=4
                )

            if model_name == "xgboost":
                model.fit(X_train, y_train, eval_set=(X_test, y_test))
            else:
                model.fit(X_train, y_train)

            train_time = time.time() - start_time

            # Predict
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Standard metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            logloss = log_loss(y_test, y_pred_proba)
            brier = brier_score_loss(y_test, y_pred_proba)

            # Recall@K with marathon-aware K values
            test_df = df_test[["show_id", "song_id"]].copy()
            test_df["y_pred"] = y_pred_proba
            test_df["y_true"] = y_test

            # Identify marathon shows in test set
            marathon_show_ids = set(
                dataset_shows[dataset_shows["is_marathon"] == 1]["show_id"].values
            )

            recalls = {}
            precisions = {}
            recalls_regular = {}
            recalls_marathon = {}

            for k in [5, 10, 15, 24]:
                show_recalls = []
                show_precisions = []
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
                        show_precisions.append(hits / dynamic_k)

                        # Track separately for regular vs marathon
                        if k == 15:
                            if is_marathon:
                                marathon_recalls.append(recall)
                            else:
                                regular_recalls.append(recall)

                recalls[k] = np.mean(show_recalls) if show_recalls else 0.0
                precisions[k] = np.mean(show_precisions) if show_precisions else 0.0

                if k == 15:
                    recalls_regular[15] = (
                        np.mean(regular_recalls) if regular_recalls else 0.0
                    )
                    recalls_marathon[24] = (
                        np.mean(marathon_recalls) if marathon_recalls else 0.0
                    )

            # F1@15 (using marathon-aware metrics)
            f1_15 = (
                2 * (precisions[15] * recalls[15]) / (precisions[15] + recalls[15])
                if (precisions[15] + recalls[15]) > 0
                else 0.0
            )

            # Save model
            filter_str = "filtered" if filter_specialized else "no_filter"
            model_filename = f"{model_name}_{dataset_name}_{filter_str}.pkl"
            model_path = models_dir / model_filename
            joblib.dump(model, model_path)
            print(f"    Model saved to: {model_path}")

            # Store results
            results.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "filter": filter_specialized,
                    "n_shows": len(dataset_shows),
                    "n_train": len(train_shows),
                    "n_test": len(test_shows),
                    "train_time": train_time,
                    "auc": auc,
                    "logloss": logloss,
                    "brier": brier,
                    "recall@5": recalls[5],
                    "recall@10": recalls[10],
                    "recall@15": recalls[15],
                    "precision@15": precisions[15],
                    "f1@15": f1_15,
                }
            )

            print(f"    Overall Recall@K: {recalls[15]:.4f} ({recalls[15]*100:.2f}%)")
            print(
                f"      Regular shows (K=15): {recalls_regular[15]:.4f} ({recalls_regular[15]*100:.2f}%)"
            )
            print(
                f"      Marathon shows (K=24): {recalls_marathon[24]:.4f} ({recalls_marathon[24]*100:.2f}%)"
            )
            print(f"    Train time: {train_time:.1f}s")

print(f'\n{"="*80}')
print("ALL EXPERIMENTS COMPLETED")
print("=" * 80)

# Save results
output_dir = Path("output/reports/stage1")
output_dir.mkdir(parents=True, exist_ok=True)

# Create results table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    ["dataset", "filter", "recall@15"], ascending=[True, True, False]
)

# Save to file
report_lines = []
report_lines.append("=" * 80)
report_lines.append("STAGE 1: TRADITIONAL ML FACTORIAL RESULTS")
report_lines.append("=" * 80)
report_lines.append("\n## Summary Table (sorted by Recall@15)\n")

for idx, row in results_df.iterrows():
    filter_str = "filtered" if row["filter"] else "no_filter"
    report_lines.append(
        f"\n{row['model']:15s} | {row['dataset']:6s} | {filter_str:9s} | "
        f"Recall@15: {row['recall@15']:.4f} ({row['recall@15']*100:5.2f}%) | "
        f"AUC: {row['auc']:.4f} | Time: {row['train_time']:5.1f}s"
    )

# Best model
best_idx = results_df["recall@15"].idxmax()
best = results_df.loc[best_idx]
report_lines.append(f"\n\n## Best Configuration:")
report_lines.append(f"Model: {best['model']}")
report_lines.append(f"Dataset: {best['dataset']}")
report_lines.append(f"Filter: {'yes' if best['filter'] else 'no'}")
report_lines.append(
    f"Recall@15: {best['recall@15']:.4f} ({best['recall@15']*100:.2f}%)"
)
report_lines.append(
    f"Recall@10: {best['recall@10']:.4f} ({best['recall@10']*100:.2f}%)"
)
report_lines.append(f"Recall@5: {best['recall@5']:.4f} ({best['recall@5']*100:.2f}%)")
report_lines.append(f"Precision@15: {best['precision@15']:.4f}")
report_lines.append(f"F1@15: {best['f1@15']:.4f}")
report_lines.append(f"AUC: {best['auc']:.4f}")

# Dataset comparison
report_lines.append("\n\n## Dataset Comparison (Recall@15):")
for model in ["logistic", "random_forest", "xgboost"]:
    report_lines.append(f"\n{model.upper()}:")
    model_results = results_df[results_df["model"] == model]
    for _, row in model_results.iterrows():
        filter_str = "filtered" if row["filter"] else "no_filter"
        report_lines.append(
            f"  {row['dataset']:6s} + {filter_str:9s}: {row['recall@15']:.4f} ({row['recall@15']*100:5.2f}%)"
        )

report_text = "\n".join(report_lines)
report_path = output_dir / "stage1_results.txt"
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\nResults saved to: {report_path}")
print("\n" + report_text)
