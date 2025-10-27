#!/usr/bin/env python3
"""
Stage 4B: Temporal Sets GNN with Feature Dropout (Production-Ready)

Based on "Predicting Temporal Sets with Deep Neural Networks" (KDD 2020)
Augmented with BERT-style feature dropout for cold start robustness.

Key Difference from Stage 4A:
- 15% feature dropout on venue/tour/country during training
- BERT strategy: 80% UNK, 10% random, 10% unchanged
- Handles unseen venues/tours in production

Purpose:
- Make GNN production-ready for cold start scenarios
- Quantify dropout cost vs Stage 4A (no dropout)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import time
import json

from src.dataio import load_all_data
from src.utils.constants import (
    SPECIALIZED_SHOW_DATES,
    ORCHESTRA_SHOW_DATES,
    RAVE_SHOW_DATES,
)
from src.models.stage3.temporal_sets import TemporalSetsGNN


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

        # Vocab sizes for random sampling
        self.vocab_sizes = {
            "venue": len(venue_to_idx),
            "tour": len(tour_to_idx),
            "country": len(country_to_idx),
        }

        # Build examples
        self.examples = []
        self._build_examples()

    def _build_examples(self):
        """Build training examples with temporal context."""
        import sys

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
            num_negatives = min(
                len(songs_in_show) * 2, len(negative_samples), 30
            )  # Cap at 30
            negative_samples = np.random.choice(
                negative_samples, size=num_negatives, replace=False
            )

            # Create examples
            for song_id in songs_in_show:
                self.examples.append(
                    {
                        "song_id": song_id,
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

    def __len__(self):
        return len(self.examples)

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
            # else: keep original (10%)

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

        return (
            song_idx,
            prev_setlists,
            venue_idx,
            tour_idx,
            country_idx,
            is_festival,
            is_marathon,
            label,
        )


def compute_recall_at_k(model, dataloader, song_to_idx, device, k=15):
    """Compute Recall@K on the dataset with marathon-aware K values."""
    model.eval()

    # Group predictions by show
    show_predictions = {}

    with torch.no_grad():
        for (
            song_ids,
            prev_setlists,
            venue_ids,
            tour_ids,
            country_ids,
            is_festival,
            is_marathon,
            labels,
        ) in dataloader:
            song_ids = song_ids.to(device)
            prev_setlists = prev_setlists.to(device)
            venue_ids = venue_ids.to(device)
            tour_ids = tour_ids.to(device)
            country_ids = country_ids.to(device)
            is_festival = is_festival.to(device)
            is_marathon = is_marathon.to(device)

            preds = model(
                song_ids,
                prev_setlists,
                venue_ids,
                tour_ids,
                country_ids,
                is_festival,
                is_marathon,
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
        labels = np.array(data["labels"])
        is_marathon = data["is_marathon"]

        # Dynamic K: 24 for marathon, 15 for regular (when computing @15)
        if k == 15:
            dynamic_k = 24 if is_marathon else 15
        else:
            dynamic_k = k

        # Sort by prediction
        sorted_idx = np.argsort(-preds)
        top_k_labels = labels[sorted_idx[:dynamic_k]]

        # Compute recall
        num_correct = top_k_labels.sum()
        total_correct = labels.sum()

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


def main():
    import sys

    print("\n" + "=" * 80, flush=True)
    print("STAGE 4B: TEMPORAL SETS GNN WITH FEATURE DROPOUT", flush=True)
    print("=" * 80, flush=True)
    print(
        "\nBased on: 'Predicting Temporal Sets with Deep Neural Networks' (KDD 2020)",
        flush=True,
    )
    print(
        "Augmented with: BERT-style feature dropout for cold start robustness",
        flush=True,
    )
    print("Dataset: recent (2022+), filtered (best from Stage 1)", flush=True)
    print(flush=True)
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # Temporal split (70% train, 15% val, 15% test - same as all stages)
    shows = shows.sort_values("date")
    n_train = int(len(shows) * 0.70)
    n_val = int(len(shows) * 0.85)

    train_shows = shows.iloc[:n_train]
    val_shows = shows.iloc[n_train:n_val]
    test_shows = shows.iloc[n_val:]

    print(
        f"\nTemporal split: {len(train_shows)} train, {len(val_shows)} val, {len(test_shows)} test"
    )

    # Identify marathon shows (now done in load_all_data)
    # Convert boolean Series to set of show IDs
    marathon_show_ids = set(shows[shows["is_marathon"] == 1]["show_id"].values)
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

    # Create datasets
    print("\nCreating datasets...")
    NUM_PREV_SHOWS = 5

    # Training with 15% feature dropout
    FEATURE_DROPOUT = 0.15
    print(
        f"Feature dropout rate: {FEATURE_DROPOUT} (BERT-style: 80% UNK, 10% random, 10% unchanged)"
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
        num_prev_shows=NUM_PREV_SHOWS,
        feature_dropout=FEATURE_DROPOUT,
        is_training=True,
    )
    # Test without dropout
    val_dataset = TemporalSetsDataset(
        val_shows,
        setlists,
        songs,
        song_to_idx,
        venue_to_idx,
        tour_to_idx,
        country_to_idx,
        marathon_show_ids,
        num_prev_shows=NUM_PREV_SHOWS,
        feature_dropout=0.0,
        is_training=False,
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
        num_prev_shows=NUM_PREV_SHOWS,
        feature_dropout=0.0,
        is_training=False,
    )

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")

    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    print("\nInitializing Temporal Sets GNN...")
    model = TemporalSetsGNN(
        num_songs=len(song_to_idx),
        emb_dim=64,
        gnn_layers=2,
        num_prev_shows=NUM_PREV_SHOWS,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("\nTraining...")
    num_epochs = 50
    best_recall = 0.0
    patience = 10
    patience_counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (
            song_ids,
            prev_setlists,
            venue_ids,
            tour_ids,
            country_ids,
            is_festival,
            is_marathon,
            labels,
        ) in enumerate(train_loader):
            song_ids = song_ids.to(device)
            prev_setlists = prev_setlists.to(device)
            venue_ids = venue_ids.to(device)
            tour_ids = tour_ids.to(device)
            country_ids = country_ids.to(device)
            is_festival = is_festival.to(device)
            is_marathon = is_marathon.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            preds = model(
                song_ids,
                prev_setlists,
                venue_ids,
                tour_ids,
                country_ids,
                is_festival,
                is_marathon,
            )
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_recall_result = compute_recall_at_k(
                model, val_loader, song_to_idx, device, k=15
            )
            test_recall = (
                test_recall_result["overall"]
                if isinstance(test_recall_result, dict)
                else test_recall_result
            )
            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f} | "
                f"Test Recall@15: {test_recall:.4f} ({test_recall*100:.2f}%) | "
                f"Time: {epoch_time:.1f}s",
                flush=True,
            )
            sys.stdout.flush()

            # Early stopping
            if test_recall > best_recall:
                best_recall = test_recall
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}", flush=True)
                # Load best model
                model.load_state_dict(best_model_state)
                break
        else:
            print(
                f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s",
                flush=True,
            )
            sys.stdout.flush()

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

    # Save model
    output_path = Path("output/models/stage4/stage4b_gnn_dropout.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "song_to_idx": song_to_idx,
            "venue_to_idx": venue_to_idx,
            "tour_to_idx": tour_to_idx,
            "test_recall_15": test_recall_15,
            "test_recall_10": test_recall_10,
            "test_recall_5": test_recall_5,
            "test_recall_20": test_recall_20,
        },
        output_path,
    )

    print(f"\nModel saved to: {output_path}")

    # Save results JSON
    results = {
        "model": "Temporal Sets GNN",
        "dataset": "recent (2022+), filtered",
        "recall@5": test_recall_5,
        "recall@10": test_recall_10,
        "recall@15": test_recall_15,
        "recall@20": test_recall_20,
        "training_time": total_time,
        "num_parameters": num_params,
        "num_prev_shows": NUM_PREV_SHOWS,
        "emb_dim": 64,
        "gnn_layers": 2,
    }

    results_path = Path("output/reports/stage4/stage4b_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON TO OTHER MODELS")
    print("=" * 80)
    print(f"\n{'Model':<30} {'Recall@15':<12}")
    print("-" * 50)
    print(f"{'XGBoost (Stage 2)':<30} {'23.52%':<12}")
    print(f"{'Logistic (Stage 1)':<30} {'21.97%':<12}")
    print(f"{'Temporal Sets GNN (Stage 3)':<30} {f'{test_recall_15*100:.2f}%':<12}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
