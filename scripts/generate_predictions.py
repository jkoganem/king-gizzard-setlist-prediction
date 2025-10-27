"""
Generate predictions from best model and compare to actual setlists.

Shows side-by-side comparison of:
- Top K predicted songs
- Actual setlist
- Which predictions were hits/misses
"""

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataio import load_all_data
from src.models.temporal_sets_gnn import TemporalSetsGNN
from src.models.stage5.dataset import TemporalSetsDataset


def load_best_model(device="cpu"):
    """Load the best performing model (Stage 5 GNN with priors)."""
    model_path = project_root / "output" / "models" / "stage5" / "stage5_gnn_priors.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. " "Please run Stage 5 training first."
        )

    checkpoint = torch.load(model_path, map_location=device)

    # Get model configuration from checkpoint
    config = checkpoint.get("config", {})

    # Initialize model
    model = TemporalSetsGNN(
        num_songs=config.get("num_songs", 203),
        num_venues=config.get("num_venues", 171),
        num_tours=config.get("num_tours", 11),
        num_countries=config.get("num_countries", 31),
        tabular_dim=config.get("tabular_dim", 34),
        embed_dim=config.get("embed_dim", 32),
        hidden_dim=config.get("hidden_dim", 64),
        num_graph_layers=config.get("num_graph_layers", 2),
        num_attention_heads=config.get("num_attention_heads", 4),
        dropout=config.get("dropout", 0.3),
        use_frequency_prior=config.get("use_frequency_prior", True),
        use_recency_prior=config.get("use_recency_prior", True),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def prepare_test_data():
    """Load and prepare test data."""
    print("Loading data...")
    shows, songs, setlists = load_all_data()

    # Apply same filtering as Stage 5
    # Recent data (2022+)
    shows = shows[shows["date"] >= "2022-01-01"].copy()

    # Remove specialized shows
    specialized_dates = [
        "2022-10-08",
        "2022-10-09",
        "2022-10-10",  # Red Rocks
        "2023-10-13",
        "2023-10-14",
        "2023-10-15",  # Hollywood Bowl
    ]
    shows = shows[~shows["date"].isin(specialized_dates)].copy()

    # Temporal split (70/15/15)
    shows = shows.sort_values("date").reset_index(drop=True)
    n_total = len(shows)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    test_shows = shows.iloc[n_train + n_val :].copy()

    # Filter setlists to test shows
    test_setlists = setlists[setlists["show_id"].isin(test_shows["show_id"])].copy()

    print(f"Test set: {len(test_shows)} shows")
    print(f"Date range: {test_shows['date'].min()} to " f"{test_shows['date'].max()}")

    return shows, songs, setlists, test_shows, test_setlists


def predict_setlist(model, show_data, songs_df, k=15):
    """
    Generate top-K predictions for a single show.

    Returns:
        pred_song_ids: List of predicted song IDs (sorted by score)
        pred_scores: Corresponding prediction scores
    """
    with torch.no_grad():
        # Prepare batch (single show)
        batch = {
            "song_ids": show_data["song_ids"].unsqueeze(0),
            "labels": show_data["labels"].unsqueeze(0),
            "tabular": show_data["tabular"].unsqueeze(0),
            "venue_id": show_data["venue_id"].unsqueeze(0),
            "tour_id": show_data["tour_id"].unsqueeze(0),
            "country_id": show_data["country_id"].unsqueeze(0),
        }

        # Add priors if available
        if "freq_prior" in show_data:
            batch["freq_prior"] = show_data["freq_prior"].unsqueeze(0)
        if "rec_prior" in show_data:
            batch["rec_prior"] = show_data["rec_prior"].unsqueeze(0)

        # Get predictions (logits for each song)
        logits = model(batch)  # Shape: [1, num_songs]
        scores = torch.sigmoid(logits).squeeze(0)  # Shape: [num_songs]

        # Get top-K predictions
        top_k_scores, top_k_indices = torch.topk(scores, k)

        pred_song_ids = show_data["song_ids"][top_k_indices].cpu().numpy()
        pred_scores = top_k_scores.cpu().numpy()

    return pred_song_ids, pred_scores


def format_prediction_comparison(
    show_info, predicted_songs, actual_songs, song_id_to_name, is_marathon=False
):
    """Format a nice comparison output."""
    output = []
    output.append("=" * 80)
    output.append(
        f"SHOW: {show_info['date']} - {show_info['venue_name']}, "
        f"{show_info['country']}"
    )
    output.append(f"Tour: {show_info.get('tour_name', 'N/A')}")

    if is_marathon:
        output.append("Type: MARATHON SHOW (26+ songs)")
        k = 24
    else:
        output.append("Type: Regular show (~15 songs)")
        k = 15

    output.append("=" * 80)
    output.append("")

    # Calculate metrics
    actual_set = set(actual_songs)
    predicted_set = set([p["song_id"] for p in predicted_songs[:k]])
    hits = actual_set & predicted_set

    recall = len(hits) / len(actual_set) if actual_set else 0
    precision = len(hits) / k if k > 0 else 0

    output.append(f"METRICS:")
    output.append(
        f"  Recall@{k}: {recall:.1%} "
        f"({len(hits)}/{len(actual_set)} songs predicted)"
    )
    output.append(f"  Precision@{k}: {precision:.1%}")
    output.append("")

    # Show predictions vs actual side by side
    output.append(f"{'TOP ' + str(k) + ' PREDICTIONS':<45} | {'ACTUAL SETLIST'}")
    output.append("-" * 80)

    max_len = max(len(predicted_songs[:k]), len(actual_songs))

    for i in range(max_len):
        # Predicted song
        if i < len(predicted_songs[:k]):
            pred = predicted_songs[i]
            song_name = song_id_to_name.get(
                pred["song_id"], f"Unknown ({pred['song_id']})"
            )
            score = pred["score"]

            # Mark if it's a hit
            marker = "[HIT]" if pred["song_id"] in actual_set else "[MISS]"
            pred_str = f"{i+1:2d}. {song_name[:35]:<35} ({score:.1%}) {marker}"
        else:
            pred_str = " " * 45

        # Actual song
        if i < len(actual_songs):
            actual_song_id = actual_songs[i]
            actual_name = song_id_to_name.get(
                actual_song_id, f"Unknown ({actual_song_id})"
            )

            # Mark if it was predicted
            marker = "[PRED]" if actual_song_id in predicted_set else ""
            actual_str = f"{i+1:2d}. {actual_name} {marker}"
        else:
            actual_str = ""

        output.append(f"{pred_str} | {actual_str}")

    output.append("")
    output.append(
        f"Summary: {len(hits)} correct predictions out of "
        f"{len(actual_set)} actual songs"
    )
    output.append("=" * 80)
    output.append("")

    return "\n".join(output)


def main():
    print("=" * 80)
    print("KING GIZZARD SETLIST PREDICTIONS")
    print("Using: Stage 5 GNN with Statistical Priors (47.03% Recall@15)")
    print("=" * 80)
    print()

    # Load model
    device = "cpu"
    print("Loading best model...")
    model, checkpoint = load_best_model(device)

    learned_alpha = checkpoint.get("alpha", 0.5)
    learned_beta = checkpoint.get("beta", 0.5)
    print(
        f"Model loaded. Learned priors: alpha={learned_alpha:.3f}, "
        f"beta={learned_beta:.3f}"
    )
    print()

    # Load data
    shows, songs, setlists, test_shows, test_setlists = prepare_test_data()

    # Create song_id to name mapping
    song_id_to_name = dict(zip(songs["song_id"], songs["song_name"]))

    # Create dataset to get preprocessed examples
    print("Preparing test dataset...")
    from src.features.feature_pipeline import engineer_all_features

    # Engineer features for all shows
    X = engineer_all_features(shows, setlists, songs["song_id"].tolist())

    # Get vocabularies (needed for dataset creation)
    song_vocab = {sid: idx for idx, sid in enumerate(sorted(songs["song_id"].unique()))}
    venue_vocab = {
        vid: idx for idx, vid in enumerate(sorted(shows["venue_id"].unique()))
    }
    tour_vocab = {
        t: idx for idx, t in enumerate(sorted(shows["tour_name"].dropna().unique()))
    }
    country_vocab = {c: idx for idx, c in enumerate(sorted(shows["country"].unique()))}

    # Create dataset for test shows
    test_dataset = TemporalSetsDataset(
        test_shows,
        setlists,
        X,
        song_vocab,
        venue_vocab,
        tour_vocab,
        country_vocab,
        feature_dropout_rate=0.0,
    )

    print(f"Test dataset created: {len(test_dataset)} examples")
    print()

    # Generate predictions for all test shows
    all_comparisons = []

    for idx in range(len(test_dataset)):
        show_data = test_dataset[idx]
        show_id = test_shows.iloc[idx]["show_id"]

        # Get show info
        show_info = test_shows[test_shows["show_id"] == show_id].iloc[0]
        is_marathon = show_info.get("is_marathon", 0) == 1
        k = 24 if is_marathon else 15

        # Generate predictions
        pred_song_ids, pred_scores = predict_setlist(model, show_data, songs, k=k)

        # Format predictions
        predicted_songs = [
            {"song_id": sid, "score": score}
            for sid, score in zip(pred_song_ids, pred_scores)
        ]

        # Get actual setlist
        actual_setlist = setlists[setlists["show_id"] == show_id].sort_values(
            "position"
        )
        actual_songs = actual_setlist["song_id"].tolist()

        # Format comparison
        comparison = format_prediction_comparison(
            show_info.to_dict(),
            predicted_songs,
            actual_songs,
            song_id_to_name,
            is_marathon,
        )

        all_comparisons.append(comparison)

    # Save to file
    output_dir = project_root / "output" / "reports" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "stage5_predictions_comparison.txt"

    with open(output_file, "w") as f:
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Stage 5 GNN with Statistical Priors\n")
        f.write(f"Test shows: {len(test_shows)}\n")
        f.write(
            f"Learned priors: alpha={learned_alpha:.4f}, " f"beta={learned_beta:.4f}\n"
        )
        f.write("\n\n")
        f.write("\n\n".join(all_comparisons))

    print(f"Predictions saved to: {output_file}")
    print()

    # Also print first 3 examples to console
    print("=" * 80)
    print("SAMPLE PREDICTIONS (first 3 shows)")
    print("=" * 80)
    print()

    for comparison in all_comparisons[:3]:
        print(comparison)

    print(f"... and {len(all_comparisons) - 3} more shows in output file.")
    print()
    print(f"Full predictions saved to: {output_file}")


if __name__ == "__main__":
    main()
