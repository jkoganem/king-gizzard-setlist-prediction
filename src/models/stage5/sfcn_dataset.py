"""Dataset builder for SFCN-TSP.

Creates efficient datasets with:
- Frequency features (global song popularity)
- Recency features (time-decay weighted recent plays)
- Lag features (binary indicators of appearance in last K shows)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class SFCNDataset(Dataset):
    """Dataset for SFCN-TSP training.

    Each example is (song_id, recency_score, lag_features, label) for a (show, song) pair.
    """

    def __init__(self, shows, songs, setlists, num_prev_shows=5, decay_factor=0.5):
        """Args:
        shows: DataFrame of shows (must be sorted by date)
        songs: DataFrame of songs
        setlists: DataFrame of setlist entries
        num_prev_shows: Number of previous shows for lag features (K)
        decay_factor: Exponential decay factor for recency scores.

        """
        self.shows = shows.sort_values("date").reset_index(drop=True)
        self.songs = songs
        self.setlists = setlists
        self.num_prev_shows = num_prev_shows
        self.decay_factor = decay_factor

        self.all_song_ids = songs["song_id"].unique()
        self.num_songs = len(self.all_song_ids)

        # Create song_id to index mapping
        self.song_to_idx = {
            song_id: idx for idx, song_id in enumerate(self.all_song_ids)
        }
        self.idx_to_song = {idx: song_id for song_id, idx in self.song_to_idx.items()}

        # Build examples
        self.examples = []
        self._build_examples()

    def _build_examples(self):
        """Build all (show, song) examples with features."""
        print(f"  Building SFCN examples from {len(self.shows)} shows...")

        # Pre-group setlists by show for faster lookup
        setlists_by_show = defaultdict(set)
        for _, row in self.setlists.iterrows():
            setlists_by_show[row["show_id"]].add(row["song_id"])

        # Convert shows to list for faster indexing
        shows_list = []
        for idx, row in self.shows.iterrows():
            shows_list.append(
                {"show_id": row["show_id"], "date": pd.to_datetime(row["date"])}
            )

        # Process each show
        for show_idx, show in enumerate(shows_list):
            if (show_idx + 1) % 20 == 0:
                print(f"    Processing show {show_idx + 1}/{len(shows_list)}...")

            show_id = show["show_id"]
            show_date = show["date"]
            played_songs = setlists_by_show[show_id]

            # Get previous K shows for lag features
            prev_shows = shows_list[max(0, show_idx - self.num_prev_shows) : show_idx]
            prev_shows.reverse()  # Most recent first

            # Build lag features for all songs
            lag_features_dict = self._build_lag_features_batch(
                prev_shows, setlists_by_show
            )

            # Build recency scores for all songs
            recency_dict = self._build_recency_scores(
                show_idx, shows_list, setlists_by_show
            )

            # Create examples for each song
            for song_id in self.all_song_ids:
                label = 1 if song_id in played_songs else 0
                song_idx = self.song_to_idx[song_id]

                self.examples.append(
                    {
                        "show_id": show_id,
                        "song_id": song_id,
                        "song_idx": song_idx,
                        "recency": recency_dict.get(song_id, 0.0),
                        "lag_features": lag_features_dict.get(
                            song_id, [0] * self.num_prev_shows
                        ),
                        "label": label,
                    }
                )

        print(f"  Built {len(self.examples)} examples")

    def _build_lag_features_batch(self, prev_shows, setlists_by_show):
        """Build lag features for all songs given previous K shows.

        Returns dict: song_id -> [lag_1, lag_2, ..., lag_K]
        """
        lag_dict = defaultdict(lambda: [0] * self.num_prev_shows)

        for lag_idx, prev_show in enumerate(prev_shows[: self.num_prev_shows]):
            prev_show_id = prev_show["show_id"]
            prev_songs = setlists_by_show[prev_show_id]

            for song_id in prev_songs:
                lag_dict[song_id][lag_idx] = 1

        return dict(lag_dict)

    def _build_recency_scores(self, current_show_idx, shows_list, setlists_by_show):
        """Build recency scores for all songs.

        Recency = -sum_{t=1}^{history} decay^t * appeared[t]
        (negative because recently played songs should be PENALIZED)

        Returns dict: song_id -> recency_score
        """
        recency_dict = defaultdict(float)

        current_date = shows_list[current_show_idx]["date"]

        # Look at all previous shows
        for prev_idx in range(current_show_idx):
            prev_show = shows_list[prev_idx]
            prev_date = prev_show["date"]
            days_ago = (current_date - prev_date).days

            if days_ago < 0:
                continue

            # Exponential decay based on days
            decay_weight = np.exp(
                -self.decay_factor * days_ago / 30.0
            )  # Normalize by 30 days

            # Penalize songs that appeared (negative score)
            for song_id in setlists_by_show[prev_show["show_id"]]:
                recency_dict[song_id] -= decay_weight

        return dict(recency_dict)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        return {
            "song_idx": torch.tensor(example["song_idx"], dtype=torch.long),
            "recency": torch.tensor(example["recency"], dtype=torch.float32),
            "lag_features": torch.tensor(example["lag_features"], dtype=torch.float32),
            "label": torch.tensor(example["label"], dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate batch of examples.

    Returns:
        dict with keys: song_indices, recency_scores, lag_features, labels

    """
    song_indices = torch.stack([b["song_idx"] for b in batch])
    recency_scores = torch.stack([b["recency"] for b in batch])
    lag_features = torch.stack([b["lag_features"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    return {
        "song_indices": song_indices,
        "recency_scores": recency_scores,
        "lag_features": lag_features,
        "labels": labels,
    }


def evaluate_sfcn(model, test_dataset, device="cpu", k=15):
    """Evaluate SFCN model on test set using Recall@K.

    Args:
        model: SFCN model
        test_dataset: SFCNDataset for testing
        device: Device to run on
        k: Top-K songs to consider

    Returns:
        float: Recall@K score

    """
    model.eval()

    recalls = []

    # Group examples by show
    show_examples = defaultdict(list)
    for example in test_dataset.examples:
        show_examples[example["show_id"]].append(example)

    with torch.no_grad():
        for show_id, examples in show_examples.items():
            # Get ground truth songs
            ground_truth = [ex["song_id"] for ex in examples if ex["label"] == 1]

            if len(ground_truth) == 0:
                continue

            # Prepare batch for all songs in this show
            song_indices = torch.tensor(
                [ex["song_idx"] for ex in examples], dtype=torch.long
            ).to(device)
            recency = torch.tensor(
                [ex["recency"] for ex in examples], dtype=torch.float32
            ).to(device)
            lag_features = torch.tensor(
                [ex["lag_features"] for ex in examples], dtype=torch.float32
            ).to(device)

            # Get predictions
            scores = model(song_indices, recency, lag_features)

            # Get top-K predictions
            _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
            top_k_song_ids = [
                examples[idx]["song_id"] for idx in top_k_indices.cpu().numpy()
            ]

            # Calculate recall
            hits = len(set(top_k_song_ids) & set(ground_truth))
            recall = hits / len(ground_truth)
            recalls.append(recall)

    return np.mean(recalls)
