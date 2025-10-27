#!/usr/bin/env python3
"""
Production-ready setlist prediction script using Stage 5B GNN model.

The model uses ALL of the following inputs:
    - Song embeddings (learned representations)
    - Previous 5 setlists (GNN message passing over co-occurrence graph)
    - Venue embedding (venue-specific patterns)
    - Tour embedding (tour context)
    - Country embedding (regional preferences)
    - is_festival boolean flag
    - is_marathon boolean flag
    - Frequency priors (global song popularity, weighted by learned alpha)
    - Recency priors (exponential decay for recently played songs, weighted by learned beta)

Model Performance: 50.13% Recall@15 (correctly predicts ~7-8 out of 15 songs)

Required Arguments:
    --date          Show date in YYYY-MM-DD format (e.g., 2025-12-13)
    --country       Country code (e.g., AU for Australia, US for USA)

Optional Arguments:
    --venue         Venue name (e.g., "Sidney Myer Music Bowl")
    --city          City name (e.g., Melbourne)
    --festival      Add this flag if it's a festival show
    --marathon      Add this flag if it's a 3-hour marathon show (predicts 24 songs instead of 15)
    --output        Save predictions to JSON file (e.g., --output predictions.json)

Usage Examples:
    # Regular show (15 songs)
    python predict_setlist_full.py --date 2025-12-13 --venue "Sidney Myer Music Bowl" --city Melbourne --country AU

    # Marathon show (24 songs)
    python predict_setlist_full.py --date 2025-12-13 --venue "Sidney Myer Music Bowl" --country AU --marathon

    # Festival show
    python predict_setlist_full.py --date 2025-06-15 --venue "Red Rocks" --country US --festival

    # With output file
    python predict_setlist_full.py --date 2025-12-13 --country AU --city Melbourne --output melbourne_prediction.json
"""

import argparse
import json
from pathlib import Path
import sys

import torch
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path.cwd()))

from src.dataio import load_all_data
from src.models.stage5.temporal_sets_with_priors import TemporalSetsGNNWithPriors
from src.models.stage5.priors import compute_song_frequencies, compute_recency_scores


def load_gnn_model():
    """Load the trained Stage 5B GNN model."""
    model_path = Path("output/models/stage5/stage5b/stage5_gnn_priors.pt")

    if not model_path.exists():
        print(f"Error: Stage 5B model not found at {model_path}")
        print("Please train the model first by running:")
        print("  python experiments/run_stage5b_final_model.py")
        return None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Get model hyperparameters from checkpoint args
    args = checkpoint["args"]

    # Initialize model
    model = TemporalSetsGNNWithPriors(
        num_songs=len(checkpoint["song_to_idx"]),
        emb_dim=args["emb_dim"],
        gnn_layers=args["gnn_layers"],
        num_prev_shows=5,
        use_frequency_prior=args["use_frequency_prior"],
        use_recency_prior=args["use_recency_prior"],
        initial_alpha=args["initial_alpha"],
        initial_beta=args["initial_beta"],
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_setlist(user_input, model, checkpoint, shows, songs, setlists, top_k=None):
    """
    Predict setlist using the full GNN model.

    Args:
        user_input: Dict with keys {date, country, city, venue_name, is_festival, is_marathon}
        model: Loaded GNN model
        checkpoint: Model checkpoint with vocabularies
        shows, songs, setlists: DataFrames from load_all_data()

    Returns:
        List of predicted songs with scores
    """
    # Get vocabularies
    song_to_idx = checkpoint["song_to_idx"]
    venue_to_idx = checkpoint["venue_to_idx"]
    tour_to_idx = checkpoint["tour_to_idx"]
    country_to_idx = checkpoint["country_to_idx"]

    idx_to_song = {v: k for k, v in song_to_idx.items()}

    # Parse target date
    target_date = pd.to_datetime(user_input["date"])

    # Get venue ID
    venue_name = user_input.get("venue_name", "")
    if venue_name:
        venue_matches = shows[
            shows["venue_id"].str.contains(venue_name, case=False, na=False)
        ]
        if len(venue_matches) > 0:
            venue_id = venue_matches.iloc[0]["venue_id"]
        else:
            venue_id = "unknown"
    else:
        venue_id = "unknown"

    venue_idx = venue_to_idx.get(venue_id, 0)  # 0 = UNK

    # Get tour (use most recent tour before target date)
    recent_shows = shows[shows["date"] < target_date].sort_values(
        "date", ascending=False
    )
    if len(recent_shows) > 0:
        tour_id = recent_shows.iloc[0]["tour_name"]
    else:
        tour_id = "unknown"
    tour_idx = tour_to_idx.get(tour_id, 0)

    # Get country
    country = user_input.get("country", "US")
    country_idx = country_to_idx.get(country, 0)

    # Boolean flags
    is_festival = 1 if user_input.get("is_festival", False) else 0
    is_marathon = 1 if user_input.get("is_marathon", False) else 0

    # Get previous 5 shows (skip shows with 0 songs - incomplete/partial data)
    all_prev_shows = shows[shows["date"] < target_date].sort_values(
        "date", ascending=False
    )

    valid_prev_shows = []
    for _, show in all_prev_shows.iterrows():
        show_songs_count = len(setlists[setlists["show_id"] == show["show_id"]])
        if show_songs_count > 0:
            valid_prev_shows.append(show)
        if len(valid_prev_shows) >= 5:
            break

    prev_shows = pd.DataFrame(valid_prev_shows)

    print(f"\nUsing {len(prev_shows)} previous shows for context:")
    for _, show in prev_shows.iterrows():
        show_songs_count = len(setlists[setlists["show_id"] == show["show_id"]])
        print(f"  - {show['date'].date()}: {show_songs_count} songs")

    # Build previous setlists tensor
    max_songs_per_show = 30
    prev_setlists = torch.zeros((1, 5, max_songs_per_show), dtype=torch.long)

    for i, (_, show) in enumerate(prev_shows.iterrows()):
        show_songs = setlists[setlists["show_id"] == show["show_id"]]["song_id"].values
        for j, song_id in enumerate(show_songs[:max_songs_per_show]):
            song_idx_val = song_to_idx.get(song_id, 0)
            prev_setlists[0, i, j] = song_idx_val

    # Compute frequency prior
    train_show_ids = shows["show_id"].values
    freq_scores = compute_song_frequencies(setlists, songs, train_show_ids)

    # Ensure freq_scores has correct shape (num_songs + 1) including UNK token
    if freq_scores.shape[0] != len(song_to_idx) + 1:
        # Prepend 0 for UNK token (index 0)
        freq_scores = torch.cat([torch.tensor([0.0]), freq_scores])

    model.set_frequency_prior(freq_scores)

    # Compute recency scores for all songs
    recency_scores_dict = {}
    decay_factor = 0.5
    for i, (_, show) in enumerate(prev_shows.iterrows()):
        decay = decay_factor**i
        show_songs = setlists[setlists["show_id"] == show["show_id"]]["song_id"].values
        for song_id in show_songs:
            song_idx_val = song_to_idx.get(song_id, 0)
            recency_scores_dict[song_idx_val] = (
                recency_scores_dict.get(song_idx_val, 0) - decay
            )

    # Create recency tensor
    recency_tensor = torch.zeros((1, len(song_to_idx) + 1), dtype=torch.float32)
    for idx, score in recency_scores_dict.items():
        recency_tensor[0, idx] = score

    # Predict for all songs
    all_predictions = []

    with torch.no_grad():
        for song_id, song_idx_val in song_to_idx.items():
            if song_idx_val == 0:  # Skip UNK token
                continue

            song_ids = torch.tensor([song_idx_val], dtype=torch.long)
            venue_ids = torch.tensor([venue_idx], dtype=torch.long)
            tour_ids = torch.tensor([tour_idx], dtype=torch.long)
            country_ids = torch.tensor([country_idx], dtype=torch.long)
            is_festival_tensor = torch.tensor([is_festival], dtype=torch.long)
            is_marathon_tensor = torch.tensor([is_marathon], dtype=torch.long)

            # Forward pass
            logits = model(
                song_ids,
                prev_setlists,
                venue_ids,
                tour_ids,
                country_ids,
                is_festival_tensor,
                is_marathon_tensor,
                recency_tensor,
            )

            prob = torch.sigmoid(logits).item()
            all_predictions.append((song_id, prob))

    # Sort by probability
    all_predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top K (custom top_k overrides marathon flag)
    if top_k is not None:
        k = top_k
    else:
        k = 24 if is_marathon else 15
    top_k_predictions = all_predictions[:k]

    # Format output
    song_map = dict(zip(songs["song_id"], songs["title"]))
    predictions = []
    for song_id, prob in top_k_predictions:
        predictions.append(
            {
                "song_id": song_id,
                "song_name": song_map.get(song_id, song_id),
                "confidence": float(prob),
            }
        )

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Predict setlist for a King Gizzard show using Stage 5B GNN"
    )

    # Required arguments
    parser.add_argument("--date", required=True, help="Show date (YYYY-MM-DD)")
    parser.add_argument("--country", required=True, help="Country code (e.g., US, AU)")

    # Optional arguments
    parser.add_argument("--venue", help='Venue name (e.g., "Sidney Myer Music Bowl")')
    parser.add_argument("--city", help="City name")
    parser.add_argument("--festival", action="store_true", help="Is this a festival?")
    parser.add_argument(
        "--marathon", action="store_true", help="Is this a 3-hour marathon show?"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of songs to predict (overrides marathon flag). Example: --top-k 20",
    )
    parser.add_argument("--output", help="Save predictions to JSON file")

    args = parser.parse_args()

    print("=" * 80)
    print("KING GIZZARD SETLIST PREDICTOR - STAGE 5B GNN")
    print("=" * 80)
    print()

    # Display show details
    print("Show Details:")
    print(f"  Date: {args.date}")
    print(f"  Country: {args.country}")
    if args.city:
        print(f"  City: {args.city}")
    if args.venue:
        print(f"  Venue: {args.venue}")
    print(f"  Festival: {'Yes' if args.festival else 'No'}")
    print(f"  Marathon: {'Yes' if args.marathon else 'No'}")
    print()

    # Load model
    print("Loading Stage 5B GNN model...")
    result = load_gnn_model()
    if result is None:
        return

    model, checkpoint = result
    print(f"  Model loaded successfully!")
    print(f"  Test Recall@15: {checkpoint['test_recall_15']:.1%}")
    print(
        f"  Learned weights: alpha={checkpoint['prior_weights']['alpha']:.3f}, beta={checkpoint['prior_weights']['beta']:.3f}"
    )
    print()

    # Load data
    print("Loading data...")
    shows, songs, setlists, song_tags = load_all_data()
    shows["date"] = pd.to_datetime(shows["date"])
    print(f"  {len(shows)} shows, {len(songs)} songs loaded")
    print()

    # Prepare user input
    user_input = {
        "date": args.date,
        "country": args.country,
        "is_festival": args.festival,
        "is_marathon": args.marathon,
    }
    if args.venue:
        user_input["venue_name"] = args.venue
    if args.city:
        user_input["city"] = args.city

    # Predict
    print("Generating predictions...")
    predictions = predict_setlist(
        user_input, model, checkpoint, shows, songs, setlists, top_k=args.top_k
    )

    # Display results
    print()
    print("=" * 80)
    print("PREDICTED SETLIST")
    print("=" * 80)
    print()

    # Determine K for display (custom top_k > marathon flag > default)
    if args.top_k:
        k = args.top_k
    else:
        k = 24 if args.marathon else 15
    print(f"Top {k} songs (confidence):")
    print()

    for i, pred in enumerate(predictions, 1):
        print(f"  {i:2d}. {pred['song_name']:45s} ({pred['confidence']:.1%})")

    print()
    print("=" * 80)
    print(
        f"Model: Stage 5B GNN with Priors (Test Recall@15: {checkpoint['test_recall_15']:.1%})"
    )
    print(
        "This model achieves ~50% accuracy, meaning it correctly predicts 7-8 out of 15 songs."
    )
    print("=" * 80)
    print()

    # Save to file if requested
    if args.output:
        output_data = {
            "show_info": user_input,
            "model": "stage5b_gnn_priors",
            "predictions": predictions,
            "predicted_at": pd.Timestamp.now().isoformat(),
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved predictions to {args.output}")
        print()


if __name__ == "__main__":
    main()
