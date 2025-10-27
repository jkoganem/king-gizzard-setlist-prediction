#!/usr/bin/env python3
"""
Stage 5: Temporal Sets GNN with Frequency/Recency Priors

Extends Stage 4B GNN with SAFERec-style priors (2024):
- Frequency prior: Songs played often are more likely
- Recency prior: Songs played recently are penalized (avoid repeats)

Based on:
- DNNTSP (KDD 2020) for base architecture
- SAFERec (2024) for frequency/recency priors
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()))

from src.dataio import load_all_data, identify_marathon_shows
from src.models.stage5 import (
    TemporalSetsGNNWithPriors,
    compute_batch_recency_scores,
    compute_song_frequencies,
)
from src.utils.constants import (
    ORCHESTRA_SHOW_DATES,
    RAVE_SHOW_DATES,
    SPECIALIZED_SHOW_DATES,
)


class TemporalSetsDataset(Dataset):
    """Dataset that provides previous shows context for each prediction."""

    def __init__(
        self,
        shows,
        setlists,
        songs,
        song_to_idx,
        venue_to_idx,
        tour_to_idx,
        country_to_idx,
        marathon_show_ids,
        num_prev_shows=5,
        max_songs_per_show=20,
        feature_dropout=0.0,
        is_training=True,
        compute_recency=False,
        decay_factor=0.5,
        lookback_window=5,
    ):
        self.shows = shows.sort_values("date").reset_index(drop=True)
        self.setlists = setlists
        self.songs = songs
        self.song_to_idx = song_to_idx
        self.venue_to_idx = venue_to_idx
        self.tour_to_idx = tour_to_idx
        self.country_to_idx = country_to_idx
        self.marathon_show_ids = marathon_show_ids
        self.num_prev_shows = num_prev_shows
        self.max_songs_per_show = max_songs_per_show
        self.feature_dropout = feature_dropout
        self.is_training = is_training
        self.compute_recency = compute_recency
        self.decay_factor = decay_factor
        self.lookback_window = lookback_window

        # Vocab sizes for random sampling (dropout)
        self.vocab_sizes = {
            "venue": len(venue_to_idx),
            "tour": len(tour_to_idx),
            "country": len(country_to_idx),
        }

        # Build examples
        self.examples = []
        self._build_examples()

        # Precompute recency scores if needed
        self.recency_cache = {}
        if compute_recency:
            self._precompute_recency_scores()

    def _build_examples(self):
        """Build training examples with temporal context."""
        print(f"  Building examples from {len(self.shows)} shows...", flush=True)

        # Pre-compute all song IDs once
        all_song_ids = self.songs["song_id"].tolist()

        # Pre-group setlists by show for faster lookup
        setlists_by_show = {}
        for show_id, group in self.setlists.groupby("show_id"):
            setlists_by_show[show_id] = set(group["song_id"].tolist())

        print(f"  Pre-grouped {len(setlists_by_show)} shows", flush=True)

        for show_idx in range(self.num_prev_shows, len(self.shows)):
            if show_idx % 10 == 0:
                pct = (show_idx / len(self.shows)) * 100
                print(
                    f"    Processing show {show_idx}/{len(self.shows)} ({pct:.1f}%)...",
                    flush=True,
                )
                sys.stdout.flush()

            current_show = self.shows.iloc[show_idx]
            show_id = current_show["show_id"]

            # Get previous shows
            prev_show_ids = self.shows.iloc[show_idx - self.num_prev_shows : show_idx][
                "show_id"
            ].tolist()

            # Get songs in current show (positive examples)
            songs_in_show = setlists_by_show.get(show_id, set())

            if len(songs_in_show) == 0:
                continue

            # Sample negative examples (fewer for speed)
            negative_samples = [s for s in all_song_ids if s not in songs_in_show]
            num_negatives = min(len(songs_in_show) * 2, len(negative_samples), 30)
            negative_samples = np.random.choice(
                negative_samples, size=num_negatives, replace=False
            )

            # Create examples
            for song_id in songs_in_show:
                self.examples.append(
                    {
                        "song_id": song_id,
                        "show_id": show_id,
                        "prev_show_ids": prev_show_ids,
                        "venue_id": current_show["venue_id"],
                        "tour_name": current_show["tour_name"],
                        "country": current_show["country"],
                        "is_festival": int(current_show["is_festival"]),
                        "is_marathon": 1 if show_id in self.marathon_show_ids else 0,
                        "label": 1,
                    }
                )

            for song_id in negative_samples:
                self.examples.append(
                    {
                        "song_id": song_id,
                        "show_id": show_id,
                        "prev_show_ids": prev_show_ids,
                        "venue_id": current_show["venue_id"],
                        "tour_name": current_show["tour_name"],
                        "country": current_show["country"],
                        "is_festival": int(current_show["is_festival"]),
                        "is_marathon": 1 if show_id in self.marathon_show_ids else 0,
                        "label": 0,
                    }
                )

        print(f"  Built {len(self.examples)} examples", flush=True)
        sys.stdout.flush()

    def _precompute_recency_scores(self):
        """Precompute recency scores for all shows."""
        print("\n  Precomputing recency scores...", flush=True)

        unique_shows = self.shows["show_id"].unique()
        for i, show_id in enumerate(unique_shows):
            if i % 20 == 0:
                print(f"    {i}/{len(unique_shows)} shows processed...", flush=True)

            # Get recency scores for this show
            from src.models.stage5.priors import compute_recency_scores

            recency = compute_recency_scores(
                self.setlists,
                self.shows,
                show_id,
                self.songs,
                decay_factor=self.decay_factor,
                lookback_window=self.lookback_window,
            )

            # Convert to indices
            recency_indexed = torch.zeros(len(self.song_to_idx) + 1)
            for song_id, score in zip(self.songs["song_id"], recency):
                idx = self.song_to_idx.get(song_id, 0)
                recency_indexed[idx] = score

            self.recency_cache[show_id] = recency_indexed

        print("  Recency scores precomputed!", flush=True)

    def _get_prev_setlists(self, prev_show_ids):
        """Get setlists matrix for previous shows."""
        prev_setlists = torch.zeros(
            self.num_prev_shows, self.max_songs_per_show, dtype=torch.long
        )

        for i, show_id in enumerate(prev_show_ids):
            songs = self.setlists[self.setlists["show_id"] == show_id][
                "song_id"
            ].tolist()
            songs_idx = [
                self.song_to_idx.get(s, 0) for s in songs[: self.max_songs_per_show]
            ]
            prev_setlists[i, : len(songs_idx)] = torch.LongTensor(songs_idx)

        return prev_setlists

    def _apply_feature_dropout(self, venue_idx, tour_idx, country_idx):
        """Apply BERT-style dropout to categorical features."""
        if not self.is_training or self.feature_dropout == 0:
            return venue_idx, tour_idx, country_idx

        # Apply dropout to venue
        if np.random.rand() < self.feature_dropout:
            rand = np.random.rand()
            if rand < 0.8:
                venue_idx = 0  # UNK
            elif rand < 0.9:
                venue_idx = np.random.randint(1, self.vocab_sizes["venue"] + 1)

        # Apply dropout to tour
        if np.random.rand() < self.feature_dropout:
            rand = np.random.rand()
            if rand < 0.8:
                tour_idx = 0  # UNK
            elif rand < 0.9:
                tour_idx = np.random.randint(1, self.vocab_sizes["tour"] + 1)

        # Apply dropout to country
        if np.random.rand() < self.feature_dropout:
            rand = np.random.rand()
            if rand < 0.8:
                country_idx = 0  # UNK
            elif rand < 0.9:
                country_idx = np.random.randint(1, self.vocab_sizes["country"] + 1)

        return venue_idx, tour_idx, country_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        song_idx = self.song_to_idx.get(ex["song_id"], 0)
        prev_setlists = self._get_prev_setlists(ex["prev_show_ids"])
        venue_idx = self.venue_to_idx.get(ex["venue_id"], 0)
        tour_idx = self.tour_to_idx.get(ex["tour_name"], 0)
        country_idx = self.country_to_idx.get(ex["country"], 0)
        is_festival = ex["is_festival"]
        is_marathon = ex["is_marathon"]
        label = ex["label"]

        # Apply feature dropout during training
        venue_idx, tour_idx, country_idx = self._apply_feature_dropout(
            venue_idx, tour_idx, country_idx
        )

        # Get recency scores if available
        recency_scores = None
        if self.compute_recency and ex["show_id"] in self.recency_cache:
            recency_scores = self.recency_cache[ex["show_id"]]

        return (
            song_idx,
            prev_setlists,
            venue_idx,
            tour_idx,
            country_idx,
            is_festival,
            is_marathon,
            recency_scores,
            label,
        )


def custom_collate(batch):
    """Custom collate function to handle optional recency scores."""
    (
        song_ids,
        prev_setlists,
        venue_ids,
        tour_ids,
        country_ids,
        is_festival,
        is_marathon,
        recency_scores,
        labels,
    ) = zip(*batch)

    # Stack tensors
    song_ids = torch.LongTensor(song_ids)
    prev_setlists = torch.stack(prev_setlists)
    venue_ids = torch.LongTensor(venue_ids)
    tour_ids = torch.LongTensor(tour_ids)
    country_ids = torch.LongTensor(country_ids)
    is_festival = torch.LongTensor(is_festival)
    is_marathon = torch.LongTensor(is_marathon)
    labels = torch.FloatTensor(labels)

    # Handle recency scores (may be None)
    if recency_scores[0] is not None:
        recency_scores = torch.stack(recency_scores)
    else:
        recency_scores = None

    return (
        song_ids,
        prev_setlists,
        venue_ids,
        tour_ids,
        country_ids,
        is_festival,
        is_marathon,
        recency_scores,
        labels,
    )


def compute_recall_at_k(model, dataloader, song_to_idx, device, k=15):
    """Compute Recall@K on the dataset with marathon-aware K values."""
    model.eval()

    # Group predictions by show
    show_predictions = {}

    with torch.no_grad():
        for batch in dataloader:
            (
                song_ids,
                prev_setlists,
                venue_ids,
                tour_ids,
                country_ids,
                is_festival,
                is_marathon,
                recency_scores,
                labels,
            ) = batch

            song_ids = song_ids.to(device)
            prev_setlists = prev_setlists.to(device)
            venue_ids = venue_ids.to(device)
            tour_ids = tour_ids.to(device)
            country_ids = country_ids.to(device)
            is_festival = is_festival.to(device)
            is_marathon = is_marathon.to(device)

            if recency_scores is not None:
                recency_scores = recency_scores.to(device)

            preds = model(
                song_ids,
                prev_setlists,
                venue_ids,
                tour_ids,
                country_ids,
                is_festival,
                is_marathon,
                recency_scores,
            )

            # Group by show (same prev_setlists = same show)
            for i in range(len(song_ids)):
                # Use hash of prev_setlists as show identifier
                show_hash = tuple(prev_setlists[i].cpu().numpy().flatten())

                if show_hash not in show_predictions:
                    show_predictions[show_hash] = {
                        "preds": [],
                        "labels": [],
                        "song_ids": [],
                        "is_marathon": is_marathon[i].item(),
                    }

                show_predictions[show_hash]["preds"].append(preds[i].item())
                show_predictions[show_hash]["labels"].append(labels[i].item())
                show_predictions[show_hash]["song_ids"].append(song_ids[i].item())

    # Compute recall for each show with dynamic K
    recalls = []
    regular_recalls = []
    marathon_recalls = []

    for show_hash, data in show_predictions.items():
        preds = np.array(data["preds"])
        labels_arr = np.array(data["labels"])
        is_marathon = data["is_marathon"]

        # Dynamic K: 24 for marathon, 15 for regular (when computing @15)
        if k == 15:
            dynamic_k = 24 if is_marathon else 15
        else:
            dynamic_k = k

        # Sort by prediction
        sorted_idx = np.argsort(-preds)
        top_k_labels = labels_arr[sorted_idx[:dynamic_k]]

        # Compute recall
        num_correct = top_k_labels.sum()
        total_correct = labels_arr.sum()

        if total_correct > 0:
            recall = num_correct / total_correct
            recalls.append(recall)

            # Track separately for regular vs marathon (when k=15)
            if k == 15:
                if is_marathon:
                    marathon_recalls.append(recall)
                else:
                    regular_recalls.append(recall)

    overall_recall = np.mean(recalls) if recalls else 0.0

    # Return dict with breakdown when k=15
    if k == 15:
        return {
            "overall": overall_recall,
            "regular": np.mean(regular_recalls) if regular_recalls else 0.0,
            "marathon": np.mean(marathon_recalls) if marathon_recalls else 0.0,
        }
    else:
        return overall_recall


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Train Temporal GNN with Frequency/Recency Priors"
    )

    # Data arguments
    parser.add_argument(
        "--num-prev-shows", type=int, default=5, help="Number of previous shows to use"
    )

    # Prior arguments
    parser.add_argument(
        "--use-frequency-prior",
        action="store_true",
        default=True,
        help="Use frequency prior (default: True)",
    )
    parser.add_argument(
        "--use-recency-prior",
        action="store_true",
        default=True,
        help="Use recency prior (default: True)",
    )
    parser.add_argument(
        "--no-frequency-prior",
        dest="use_frequency_prior",
        action="store_false",
        help="Disable frequency prior",
    )
    parser.add_argument(
        "--no-recency-prior",
        dest="use_recency_prior",
        action="store_false",
        help="Disable recency prior",
    )
    parser.add_argument(
        "--initial-alpha", type=float, default=0.5, help="Initial frequency weight"
    )
    parser.add_argument(
        "--initial-beta", type=float, default=0.5, help="Initial recency weight"
    )
    parser.add_argument(
        "--decay-factor",
        type=float,
        default=0.5,
        help="Recency exponential decay factor",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=5,
        help="Number of previous shows for recency",
    )

    # Model arguments (OPTIMAL FROM STAGE 5A)
    parser.add_argument("--emb-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--gnn-layers", type=int, default=3, help="Number of GCN layers"
    )

    # Training arguments (OPTIMAL FROM STAGE 5A)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--feature-dropout",
        type=float,
        default=0.10455426356876447,
        help="Feature dropout rate (BERT-style)",
    )
    parser.add_argument(
        "--lr", type=float, default=5.466844213989016e-05, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=5.0, help="Gradient clipping max norm"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--scheduler-patience", type=int, default=5, help="LR scheduler patience"
    )

    # Regularization arguments (OPTIMAL FROM STAGE 5A)
    parser.add_argument(
        "--prior-reg-weight",
        type=float,
        default=0.4645137475998357,
        help="L2 regularization weight for alpha/beta (default: 0.465)",
    )
    parser.add_argument(
        "--clip-priors",
        action="store_true",
        default=False,
        help="Clip alpha/beta to prevent extreme values",
    )
    parser.add_argument(
        "--max-alpha",
        type=float,
        default=2.0,
        help="Maximum value for alpha (if clipping enabled)",
    )
    parser.add_argument(
        "--max-beta",
        type=float,
        default=2.0,
        help="Maximum value for beta (if clipping enabled)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/models/stage5/stage5b",
        help="Output directory for models",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="output/reports/stage5",
        help="Output directory for reports",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("STAGE 5B: FINAL MODEL WITH OPTIMAL HYPERPARAMETERS")
    print("=" * 80)
    print("\nOptimal configuration from Stage 5A hyperparameter search:")
    print("  - Learning rate: 5.47e-05")
    print("  - Batch size: 512")
    print("  - Weight decay: 0.01")
    print("  - Gradient clipping: 5.0")
    print("  - GNN layers: 3")
    print("  - Embedding dim: 64")
    print("  - Prior regularization: 0.465")
    print("  - Feature dropout: 0.105")
    print("\nBased on:")
    print("  - DNNTSP (KDD 2020) for base GNN architecture")
    print("  - SAFERec (2024) for frequency/recency priors")
    print("\nPrior Configuration:")
    print(f"  Frequency prior: {'Enabled' if args.use_frequency_prior else 'Disabled'}")
    print(f"  Recency prior:   {'Enabled' if args.use_recency_prior else 'Disabled'}")
    if args.use_frequency_prior:
        print(f"  Initial alpha (freq weight): {args.initial_alpha}")
    if args.use_recency_prior:
        print(f"  Initial beta (rec weight):  {args.initial_beta}")
        print(f"  Decay factor: {args.decay_factor}")
        print(f"  Lookback window: {args.lookback_window}")
    print(flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading data...")
    shows, songs, setlists, song_tags = load_all_data()

    # Apply best configuration from Stage 1/2
    shows["date"] = pd.to_datetime(shows["date"])
    shows = shows[shows["date"] >= "2022-01-01"]

    # Filter specialized shows
    specialized_dates = set(
        SPECIALIZED_SHOW_DATES + ORCHESTRA_SHOW_DATES + RAVE_SHOW_DATES
    )
    specialized_dates = [pd.to_datetime(d) for d in specialized_dates]
    shows = shows[~shows["date"].isin(specialized_dates)]

    setlists = setlists[setlists["show_id"].isin(shows["show_id"])]

    print(f"  {len(shows)} shows, {len(songs)} songs, {len(setlists)} setlist entries")

    # Temporal split (70% train, 15% val, 15% test)
    shows = shows.sort_values("date")
    n_train = int(len(shows) * 0.70)
    n_val = int(len(shows) * 0.85)

    train_shows = shows.iloc[:n_train]
    val_shows = shows.iloc[n_train:n_val]
    test_shows = shows.iloc[n_val:]

    print(
        f"\nTemporal split: {len(train_shows)} train, {len(val_shows)} val, {len(test_shows)} test"
    )

    # Identify marathon shows
    marathon_mask = identify_marathon_shows(shows, setlists)
    marathon_show_ids = set(shows[marathon_mask]["show_id"].values)
    print(f"\nIdentified {len(marathon_show_ids)} marathon shows")

    # Create vocabularies
    all_songs = songs["song_id"].unique().tolist()
    all_venues = shows["venue_id"].unique().tolist()
    all_tours = shows["tour_name"].unique().tolist()
    all_countries = shows["country"].unique().tolist()

    song_to_idx = {song: i + 1 for i, song in enumerate(all_songs)}
    venue_to_idx = {venue: i + 1 for i, venue in enumerate(all_venues)}
    tour_to_idx = {tour: i + 1 for i, tour in enumerate(all_tours)}
    country_to_idx = {country: i + 1 for i, country in enumerate(all_countries)}

    print(f"\nVocabularies:")
    print(f"  Songs: {len(song_to_idx)}")
    print(f"  Venues: {len(venue_to_idx)}")
    print(f"  Tours: {len(tour_to_idx)}")
    print(f"  Countries: {len(country_to_idx)}")

    # Compute frequency prior
    freq_scores = None
    if args.use_frequency_prior:
        print("\nComputing frequency priors...")
        train_show_ids = train_shows["show_id"].values
        freq_scores_raw = compute_song_frequencies(
            setlists, songs, train_show_ids, smoothing=1.0
        )

        # Convert to indexed tensor (includes padding idx 0)
        freq_scores = torch.zeros(len(song_to_idx) + 1)
        for song_id, score in zip(songs["song_id"], freq_scores_raw):
            idx = song_to_idx.get(song_id, 0)
            freq_scores[idx] = score

        print(
            f"  Frequency scores computed: min={freq_scores.min():.3f}, max={freq_scores.max():.3f}, mean={freq_scores.mean():.3f}"
        )

    # Create datasets
    print("\nCreating datasets...")
    print(
        f"Feature dropout rate: {args.feature_dropout} (BERT-style: 80% UNK, 10% random, 10% unchanged)"
    )

    train_dataset = TemporalSetsDataset(
        train_shows,
        setlists,
        songs,
        song_to_idx,
        venue_to_idx,
        tour_to_idx,
        country_to_idx,
        marathon_show_ids,
        num_prev_shows=args.num_prev_shows,
        feature_dropout=args.feature_dropout,
        is_training=True,
        compute_recency=args.use_recency_prior,
        decay_factor=args.decay_factor,
        lookback_window=args.lookback_window,
    )

    val_dataset = TemporalSetsDataset(
        val_shows,
        setlists,
        songs,
        song_to_idx,
        venue_to_idx,
        tour_to_idx,
        country_to_idx,
        marathon_show_ids,
        num_prev_shows=args.num_prev_shows,
        feature_dropout=0.0,
        is_training=False,
        compute_recency=args.use_recency_prior,
        decay_factor=args.decay_factor,
        lookback_window=args.lookback_window,
    )

    test_dataset = TemporalSetsDataset(
        test_shows,
        setlists,
        songs,
        song_to_idx,
        venue_to_idx,
        tour_to_idx,
        country_to_idx,
        marathon_show_ids,
        num_prev_shows=args.num_prev_shows,
        feature_dropout=0.0,
        is_training=False,
        compute_recency=args.use_recency_prior,
        decay_factor=args.decay_factor,
        lookback_window=args.lookback_window,
    )

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )

    # Initialize model
    print("\nInitializing Temporal Sets GNN with Priors...")
    model = TemporalSetsGNNWithPriors(
        num_songs=len(song_to_idx),
        emb_dim=args.emb_dim,
        gnn_layers=args.gnn_layers,
        num_prev_shows=args.num_prev_shows,
        use_frequency_prior=args.use_frequency_prior,
        use_recency_prior=args.use_recency_prior,
        initial_alpha=args.initial_alpha,
        initial_beta=args.initial_beta,
    ).to(device)

    # Set frequency prior
    if args.use_frequency_prior:
        model.set_frequency_prior(freq_scores.to(device))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Show prior weights
    prior_weights = model.get_prior_weights()
    print(
        f"  Prior weights: alpha={prior_weights['alpha']}, beta={prior_weights['beta']}"
    )

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # LR Scheduler (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # Maximize recall
        factor=0.5,  # Reduce LR by 50%
        patience=args.scheduler_patience,
        min_lr=1e-6,
    )
    print(
        f"LR Scheduler: ReduceLROnPlateau (patience={args.scheduler_patience}, factor=0.5)"
    )
    print(f"Early Stopping: patience={args.patience}")
    print()

    # Train
    print("\nTraining...")
    print(f"Early stopping patience: {args.patience} epochs")
    print(f"LR scheduler patience: {args.scheduler_patience} epochs")
    print()

    best_recall = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    # Track training history for visualization
    history = {"epoch": [], "loss": [], "recall": [], "alpha": [], "beta": []}

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch in train_loader:
            (
                song_ids,
                prev_setlists,
                venue_ids,
                tour_ids,
                country_ids,
                is_festival,
                is_marathon,
                recency_scores,
                labels,
            ) = batch

            song_ids = song_ids.to(device)
            prev_setlists = prev_setlists.to(device)
            venue_ids = venue_ids.to(device)
            tour_ids = tour_ids.to(device)
            country_ids = country_ids.to(device)
            is_festival = is_festival.to(device)
            is_marathon = is_marathon.to(device)
            labels = labels.to(device)

            if recency_scores is not None:
                recency_scores = recency_scores.to(device)

            optimizer.zero_grad()
            preds = model(
                song_ids,
                prev_setlists,
                venue_ids,
                tour_ids,
                country_ids,
                is_festival,
                is_marathon,
                recency_scores,
            )
            bce_loss = criterion(preds, labels)

            # Add L2 regularization on prior weights (alpha, beta)
            prior_reg_loss = 0.0
            if hasattr(model, "alpha") and hasattr(model, "beta"):
                prior_reg_loss = args.prior_reg_weight * (
                    model.alpha.pow(2) + model.beta.pow(2)
                )

            total_batch_loss = bce_loss + prior_reg_loss
            total_batch_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            # Optional: Clip alpha/beta to prevent extreme values
            if args.clip_priors:
                with torch.no_grad():
                    if hasattr(model, "alpha"):
                        model.alpha.clamp_(0.0, args.max_alpha)
                    if hasattr(model, "beta"):
                        model.beta.clamp_(-args.max_beta, args.max_beta)

            total_loss += bce_loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Evaluate on VALIDATION set for early stopping (NOT test set!)
        val_recall_result = compute_recall_at_k(
            model, val_loader, song_to_idx, device, k=15
        )
        val_recall = (
            val_recall_result["overall"]
            if isinstance(val_recall_result, dict)
            else val_recall_result
        )

        # Get current prior weights
        prior_weights = model.get_prior_weights()

        # Track history
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["recall"].append(val_recall)
        history["alpha"].append(prior_weights["alpha"])
        history["beta"].append(prior_weights["beta"])

        # Print detailed info every 5 epochs, brief otherwise
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                f"Test Recall@15: {val_recall:.4f} ({val_recall*100:.2f}%) | "
                f"alpha={prior_weights['alpha']:.3f}, beta={prior_weights['beta']:.3f} | "
                f"Time: {epoch_time:.1f}s",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {avg_loss:.4f} | Recall: {val_recall*100:.2f}% | Time: {epoch_time:.1f}s",
                flush=True,
            )
        sys.stdout.flush()

        # LR Scheduler step (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            scheduler.step(val_recall)

        # Early stopping (check every epoch)
        if val_recall > best_recall:
            best_recall = val_recall
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"\nEarly stopping at epoch {epoch+1} (best was epoch {best_epoch}: {best_recall*100:.2f}%)",
                flush=True,
            )
            # Load best model
            model.load_state_dict(best_model_state)
            break

    total_time = time.time() - start_time

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    test_recall_15_result = compute_recall_at_k(
        model, test_loader, song_to_idx, device, k=15
    )
    test_recall_10 = compute_recall_at_k(model, test_loader, song_to_idx, device, k=10)
    test_recall_5 = compute_recall_at_k(model, test_loader, song_to_idx, device, k=5)
    test_recall_20 = compute_recall_at_k(model, test_loader, song_to_idx, device, k=20)

    # Extract overall recall for backward compatibility
    test_recall_15 = (
        test_recall_15_result["overall"]
        if isinstance(test_recall_15_result, dict)
        else test_recall_15_result
    )

    print(f"\nTest Recall@5:  {test_recall_5:.4f} ({test_recall_5*100:.2f}%)")
    print(f"Test Recall@10: {test_recall_10:.4f} ({test_recall_10*100:.2f}%)")
    print(f"Overall Recall@K: {test_recall_15:.4f} ({test_recall_15*100:.2f}%)")
    if isinstance(test_recall_15_result, dict):
        print(
            f"  Regular shows (K=15): {test_recall_15_result['regular']:.4f} ({test_recall_15_result['regular']*100:.2f}%)"
        )
        print(
            f"  Marathon shows (K=24): {test_recall_15_result['marathon']:.4f} ({test_recall_15_result['marathon']*100:.2f}%)"
        )
    print(f"Test Recall@20: {test_recall_20:.4f} ({test_recall_20*100:.2f}%)")
    print(f"\nTraining time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Show final prior weights
    final_prior_weights = model.get_prior_weights()
    print(f"\nLearned prior weights:")
    print(f"  alpha (frequency): {final_prior_weights['alpha']:.4f}")
    print(f"  beta (recency):   {final_prior_weights['beta']:.4f}")

    # Save model
    output_path = Path(args.output_dir) / "stage5_gnn_priors.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "song_to_idx": song_to_idx,
            "venue_to_idx": venue_to_idx,
            "tour_to_idx": tour_to_idx,
            "country_to_idx": country_to_idx,
            "test_recall_15": test_recall_15,
            "test_recall_10": test_recall_10,
            "test_recall_5": test_recall_5,
            "test_recall_20": test_recall_20,
            "prior_weights": final_prior_weights,
            "history": history,
            "best_epoch": best_epoch,
            "args": vars(args),
        },
        output_path,
    )

    print(f"\nModel saved to: {output_path}")

    # Save results JSON
    results = {
        "model": "Temporal Sets GNN with Frequency/Recency Priors",
        "dataset": "recent (2022+), filtered",
        "recall@5": test_recall_5,
        "recall@10": test_recall_10,
        "recall@15": test_recall_15,
        "recall@20": test_recall_20,
        "training_time": total_time,
        "num_parameters": num_params,
        "priors": {
            "use_frequency": args.use_frequency_prior,
            "use_recency": args.use_recency_prior,
            "learned_alpha": final_prior_weights["alpha"],
            "learned_beta": final_prior_weights["beta"],
            "initial_alpha": args.initial_alpha,
            "initial_beta": args.initial_beta,
            "decay_factor": args.decay_factor,
            "lookback_window": args.lookback_window,
        },
        "architecture": {
            "emb_dim": args.emb_dim,
            "gnn_layers": args.gnn_layers,
            "num_prev_shows": args.num_prev_shows,
            "feature_dropout": args.feature_dropout,
        },
    }

    results_path = Path(args.report_dir) / "stage5_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    # Generate training curves visualization
    print("\n" + "=" * 80)
    print("GENERATING TRAINING CURVES")
    print("=" * 80)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Training Loss
        axes[0, 0].plot(
            history["epoch"], history["loss"], marker="o", linewidth=2, markersize=4
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: Validation Recall@15 (used for early stopping)
        axes[0, 1].plot(
            history["epoch"],
            [r * 100 for r in history["recall"]],
            marker="o",
            linewidth=2,
            markersize=4,
            color="green",
        )
        # Mark the best epoch
        if best_epoch in history["epoch"]:
            best_idx = history["epoch"].index(best_epoch)
            axes[0, 1].axvline(
                x=best_epoch,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Best (epoch {best_epoch})",
            )
            axes[0, 1].legend()
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Recall@15 (%)")
        axes[0, 1].set_title("Validation Recall@15 (for early stopping)")
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Alpha (Frequency Prior Weight)
        axes[1, 0].plot(
            history["epoch"],
            history["alpha"],
            marker="o",
            linewidth=2,
            markersize=4,
            color="blue",
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Alpha (Frequency Weight)")
        axes[1, 0].set_title("Learned Frequency Prior Weight")
        axes[1, 0].grid(alpha=0.3)

        # Plot 4: Beta (Recency Prior Weight)
        axes[1, 1].plot(
            history["epoch"],
            history["beta"],
            marker="o",
            linewidth=2,
            markersize=4,
            color="red",
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Beta (Recency Weight)")
        axes[1, 1].set_title("Learned Recency Prior Weight")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        figures_path = Path("output/figures/stage5b_training_curves.png")
        figures_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Training curves saved to: {figures_path}")
    except Exception as e:
        print(f"Warning: Could not generate training curves: {e}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON TO STAGE 4")
    print("=" * 80)
    print(f"\n{'Model':<40} {'Recall@15':<12}")
    print("-" * 60)
    print(f"{'Stage 4A (GNN, no dropout)':<40} {'39.55%':<12}")
    print(f"{'Stage 4B (GNN + dropout)':<40} {'41.18%':<12}")
    print(
        f"{'Stage 5 (GNN + dropout + priors)':<40} {f'{test_recall_15*100:.2f}%':<12}"
    )

    improvement = test_recall_15 - 0.4118
    print(f"\nImprovement over Stage 4B: {improvement*100:+.2f}%")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
