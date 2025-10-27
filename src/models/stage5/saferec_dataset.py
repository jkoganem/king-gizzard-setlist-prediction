"""Dataset for SAFERec-TSP.

Different from SFCN: uses show sequences instead of individual examples.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class SAFERecDataset(Dataset):
    """Dataset for SAFERec-TSP.

    Each example is a (show_history, target_setlist) pair where:
    - show_history: [seq_len, max_songs_per_show] sequence of previous shows
    - target_setlist: [num_songs] binary labels for target show
    """

    def __init__(
        self,
        shows,
        songs,
        setlists,
        seq_len=10,
        max_songs_per_show=30,
        decay_factor=0.5,
    ):
        """Args:
        shows: DataFrame of shows (must be sorted by date)
        songs: DataFrame of songs
        setlists: DataFrame of setlist entries
        seq_len: Number of previous shows to use
        max_songs_per_show: Max songs per show (for padding)
        decay_factor: Decay for recency scores.

        """
        self.shows = shows.sort_values("date").reset_index(drop=True)
        self.songs = songs
        self.setlists = setlists
        self.seq_len = seq_len
        self.max_songs_per_show = max_songs_per_show
        self.decay_factor = decay_factor

        self.all_song_ids = sorted(songs["song_id"].unique())
        self.num_songs = len(self.all_song_ids)

        # Create mappings
        # +1 for padding
        self.song_to_idx = {
            song_id: idx + 1 for idx, song_id in enumerate(self.all_song_ids)
        }
        self.idx_to_song = {idx: song_id for song_id, idx in self.song_to_idx.items()}
        self.idx_to_song[0] = None  # Padding

        # Build examples
        self.examples = []
        self._build_examples()

    def _build_examples(self):
        """Build show sequence examples."""
        print(f"  Building SAFERec examples from {len(self.shows)} shows...")

        # Pre-group setlists by show
        setlists_by_show = defaultdict(list)
        for _, row in self.setlists.iterrows():
            song_idx = self.song_to_idx[row["song_id"]]
            setlists_by_show[row["show_id"]].append(song_idx)

        # Convert shows to list
        shows_list = []
        for _, row in self.shows.iterrows():
            shows_list.append(
                {
                    "show_id": row["show_id"],
                    "date": pd.to_datetime(row["date"]),
                    "songs": setlists_by_show[row["show_id"]],
                }
            )

        # Build examples
        for show_idx in range(len(shows_list)):
            if (show_idx + 1) % 50 == 0:
                print(f"    Processing show {show_idx + 1}/{len(shows_list)}...")

            target_show = shows_list[show_idx]

            # Get previous shows for history
            start_idx = max(0, show_idx - self.seq_len)
            prev_shows = shows_list[start_idx:show_idx]

            # Pad history if needed
            while len(prev_shows) < self.seq_len:
                prev_shows = [{"show_id": None, "songs": []}] + prev_shows

            # Build history tensor [seq_len, max_songs_per_show]
            history = []
            for prev_show in prev_shows:
                song_indices = prev_show["songs"][: self.max_songs_per_show]
                # Pad to max_songs_per_show
                song_indices += [0] * (self.max_songs_per_show - len(song_indices))
                history.append(song_indices)

            # Build target labels [num_songs]
            target_songs = set(target_show["songs"])
            labels = [
                1 if (i + 1) in target_songs else 0 for i in range(self.num_songs)
            ]

            # Build recency scores
            recency_scores = self._compute_recency(show_idx, shows_list)

            self.examples.append(
                {
                    "show_id": target_show["show_id"],
                    "history": history,  # [seq_len, max_songs_per_show]
                    "labels": labels,  # [num_songs]
                    "recency": recency_scores,  # [num_songs]
                }
            )

        print(f"  Built {len(self.examples)} examples")

    def _compute_recency(self, current_idx, shows_list):
        """Compute recency scores for all songs.

        Returns: [num_songs] array of recency penalties
        """
        recency = np.zeros(self.num_songs, dtype=np.float32)
        current_date = shows_list[current_idx]["date"]

        # Look at all previous shows
        for prev_idx in range(current_idx):
            prev_show = shows_list[prev_idx]
            prev_date = prev_show["date"]
            days_ago = (current_date - prev_date).days

            if days_ago < 0:
                continue

            # Exponential decay
            decay_weight = np.exp(-self.decay_factor * days_ago / 30.0)

            # Penalize songs that appeared (negative score)
            for song_idx in prev_show["songs"]:
                if 1 <= song_idx <= self.num_songs:
                    recency[song_idx - 1] -= decay_weight

        return recency

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        return {
            "history": torch.tensor(example["history"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.float32),
            "recency": torch.tensor(example["recency"], dtype=torch.float32),
        }


def collate_saferec(batch):
    """Collate batch for SAFERec."""
    histories = torch.stack([b["history"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    recencies = torch.stack([b["recency"] for b in batch])

    return {"histories": histories, "labels": labels, "recencies": recencies}


def evaluate_saferec(model, test_dataset, device="cpu", k=15):
    """Evaluate SAFERec on test set.

    Args:
        model: SAFERec model
        test_dataset: SAFERecDataset
        device: Device
        k: Top-K

    Returns:
        float: Recall@K

    """
    model.eval()
    recalls = []

    with torch.no_grad():
        for example in test_dataset:
            history = (
                example["history"].unsqueeze(0).to(device)
            )  # [1, seq_len, max_songs]
            labels = example["labels"].to(device)  # [num_songs]
            recency = example["recency"].unsqueeze(0).to(device)  # [1, num_songs]

            # Get ground truth
            ground_truth = torch.where(labels == 1)[0].cpu().numpy()
            if len(ground_truth) == 0:
                continue

            # Create song_ids tensor [1, num_songs] (all songs)
            num_songs = labels.size(0)
            song_ids = torch.arange(1, num_songs + 1, device=device).unsqueeze(
                0
            )  # [1, num_songs]

            # Get predictions
            scores = model(history, song_ids, recency).squeeze(0)  # [num_songs]

            # Get top-K
            _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
            top_k_indices = top_k_indices.cpu().numpy()

            # Calculate recall
            hits = len(set(top_k_indices) & set(ground_truth))
            recall = hits / len(ground_truth)
            recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0
