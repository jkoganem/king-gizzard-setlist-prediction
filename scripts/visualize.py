#!/usr/bin/env python3
"""
Visualization Script for King Gizzard Setlist Prediction

Generate publication-quality figures and plots:
- Feature importance (gain, cover, weight)
- Model performance comparisons
- Training curves and convergence
- Per-show analysis
- EDA visualizations

Usage:
    python scripts/visualize.py --results output/results --output-dir output/figures
    python scripts/visualize.py --feature-importance --model xgboost_full.pkl
    python scripts/visualize.py --model-comparison --results output/results

"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.utils.config import settings

# Configure plotting
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_feature_importance(
    model_path: Path,
    output_dir: Path,
    importance_types: List[str] = ["gain", "cover", "weight"],
    top_k: int = 20,
) -> None:
    """
    Plot feature importance for XGBoost model.

    Args:
        model_path: Path to XGBoost model
        output_dir: Output directory for plots
        importance_types: Types of importance to plot
        top_k: Number of top features to show
    """
    logger.info(f"Plotting feature importance for {model_path.name}")

    model = joblib.load(model_path)

    if not hasattr(model, "get_booster"):
        logger.warning(f"Model {model_path.name} does not support feature importance")
        return

    for imp_type in importance_types:
        # Get importance scores
        importance = model.get_booster().get_score(importance_type=imp_type)

        if not importance:
            continue

        # Create DataFrame
        df = (
            pd.DataFrame(
                {
                    "feature": list(importance.keys()),
                    "importance": list(importance.values()),
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_k)
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.barh(range(len(df)), df["importance"].values, color="steelblue")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel(f"Importance ({imp_type.capitalize()})")
        ax.set_title(f"Top {top_k} Features - {imp_type.capitalize()} Importance")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        output_path = (
            output_dir / f"feature_importance_{imp_type}_{model_path.stem}.pdf"
        )
        plt.savefig(output_path, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_path}")

        # Save CSV
        csv_path = output_dir / f"feature_importance_{imp_type}_{model_path.stem}.csv"
        df.to_csv(csv_path, index=False)


def plot_model_comparison(
    results_path: Path, output_dir: Path, metric: str = "recall@15"
) -> None:
    """
    Plot model performance comparison.

    Args:
        results_path: Path to results CSV
        output_dir: Output directory
        metric: Metric to compare
    """
    logger.info("Plotting model comparison")

    df = pd.read_csv(results_path)

    if metric not in df.columns:
        logger.error(f"Metric {metric} not found in results")
        return

    # Sort by metric
    df = df.sort_values(metric, ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(df))
    bars = ax.barh(range(len(df)), df[metric].values, color=colors)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"].values)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("_", " ").replace("@", " @ ").title())
    ax.set_title("Model Performance Comparison")
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row[metric] + 0.005, i, f"{row[metric]:.4f}", va="center", fontsize=9)

    plt.tight_layout()

    output_path = output_dir / f"model_comparison_{metric}.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_metrics_comparison(
    results_path: Path, output_dir: Path, k_values: List[int] = [10, 15, 20]
) -> None:
    """
    Plot comparison of Recall, Precision, and F1 across K values.

    Args:
        results_path: Path to results CSV
        output_dir: Output directory
        k_values: K values to compare
    """
    logger.info("Plotting metrics comparison")

    df = pd.read_csv(results_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["recall", "precision", "f1"]

    for ax, metric in zip(axes, metrics):
        data = []
        labels = []

        for k in k_values:
            col = f"{metric}@{k}"
            if col in df.columns:
                data.append(df[col].values)
                labels.append(f"K={k}")

        if not data:
            continue

        x = np.arange(len(df))
        width = 0.25

        for i, (values, label) in enumerate(zip(data, labels)):
            ax.bar(x + i * width, values, width, label=label)

        ax.set_xlabel("Model")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} @ K")
        ax.set_xticks(x + width)
        ax.set_xticklabels(df["model"].values, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "metrics_comparison.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_dataset_comparison(results_dir: Path, output_dir: Path) -> None:
    """
    Compare performance across different dataset configurations.

    Args:
        results_dir: Directory containing result files
        output_dir: Output directory
    """
    logger.info("Plotting dataset comparison")

    # Find all result files
    result_files = list(results_dir.glob("model_comparison*.csv"))

    if not result_files:
        logger.warning("No result files found")
        return

    # Combine results
    all_results = []

    for file in result_files:
        df = pd.read_csv(file)

        # Extract configuration from filename
        if "recent" in file.stem:
            dataset = "Recent"
        elif "full" in file.stem:
            dataset = "Full"
        else:
            dataset = "Unknown"

        if "filtered" in file.stem:
            filtering = "Filtered"
        else:
            filtering = "Unfiltered"

        df["dataset"] = dataset
        df["filtering"] = filtering
        df["config"] = f"{dataset}_{filtering}"

        all_results.append(df)

    if not all_results:
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Plot for each model
    models = combined["model"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models[:6]):  # Max 6 models
        ax = axes[idx]

        model_data = combined[combined["model"] == model]

        if "recall@15" in model_data.columns:
            x = np.arange(len(model_data))
            bars = ax.bar(x, model_data["recall@15"].values, color="steelblue")

            ax.set_xticks(x)
            ax.set_xticklabels(model_data["config"].values, rotation=45, ha="right")
            ax.set_ylabel("Recall@15")
            ax.set_title(model)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Remove empty subplots
    for idx in range(len(models), 6):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    output_path = output_dir / "dataset_comparison.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for setlist prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all visualizations
    python scripts/visualize.py --all --results output/results --output-dir output/figures

    # Feature importance for specific model
    python scripts/visualize.py --feature-importance --model output/models/xgboost_full.pkl

    # Model comparison
    python scripts/visualize.py --model-comparison --results output/results/model_comparison.csv
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Generate all visualizations"
    )
    parser.add_argument(
        "--feature-importance", action="store_true", help="Plot feature importance"
    )
    parser.add_argument(
        "--model-comparison", action="store_true", help="Plot model comparison"
    )
    parser.add_argument(
        "--metrics-comparison", action="store_true", help="Plot metrics comparison"
    )
    parser.add_argument(
        "--dataset-comparison", action="store_true", help="Plot dataset comparison"
    )

    parser.add_argument(
        "--model", type=Path, help="Path to model for feature importance"
    )
    parser.add_argument("--models-dir", type=Path, help="Directory with models")
    parser.add_argument(
        "--results", type=Path, help="Path to results file or directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/figures"),
        help="Output directory (default: output/figures)",
    )

    parser.add_argument(
        "--top-k", type=int, default=20, help="Top K features (default: 20)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf,png",
        help="Output formats (default: pdf,png)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("VISUALIZATION GENERATION")
    logger.info("=" * 70)

    # Feature importance
    if args.all or args.feature_importance:
        if args.model:
            plot_feature_importance(args.model, args.output_dir, top_k=args.top_k)
        elif args.models_dir:
            for model_path in args.models_dir.glob("xgboost*.pkl"):
                try:
                    plot_feature_importance(
                        model_path, args.output_dir, top_k=args.top_k
                    )
                except Exception as e:
                    logger.error(f"Failed for {model_path.name}: {e}")

    # Model comparison
    if args.all or args.model_comparison:
        if args.results:
            if args.results.is_file():
                plot_model_comparison(args.results, args.output_dir)
            elif args.results.is_dir():
                for result_file in args.results.glob("model_comparison*.csv"):
                    try:
                        plot_model_comparison(result_file, args.output_dir)
                    except Exception as e:
                        logger.error(f"Failed for {result_file.name}: {e}")

    # Metrics comparison
    if args.all or args.metrics_comparison:
        if args.results:
            if args.results.is_file():
                plot_metrics_comparison(args.results, args.output_dir)
            elif args.results.is_dir():
                for result_file in args.results.glob("model_comparison*.csv"):
                    try:
                        plot_metrics_comparison(result_file, args.output_dir)
                    except Exception as e:
                        logger.error(f"Failed for {result_file.name}: {e}")

    # Dataset comparison
    if args.all or args.dataset_comparison:
        if args.results and args.results.is_dir():
            plot_dataset_comparison(args.results, args.output_dir)

    logger.info("=" * 70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
