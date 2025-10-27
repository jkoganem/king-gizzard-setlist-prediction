"""Data loading utilities for SASRec model.

Converts setlist data into sequential format for next-song prediction.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SetlistSequenceDataset(Dataset):
    """Dataset for sequential setlist prediction.

    Converts setlists into sequences for SASRec training.
    Each sequence is (song_1, song_2, ..., song_t) and the target is song_{t+1}.
    """

    def __init__(self, shows, setlists, song_to_idx, max_seq_len=50, mode="train"):
        """Args:
        shows: DataFrame of shows
        setlists: DataFrame of setlist entries
        song_to_idx: Dict mapping song_id -> integer index
        max_seq_len: Maximum sequence length
        mode: 'train' or 'test'.

        """
        self.shows = shows.sort_values("date")
        self.setlists = setlists
        self.song_to_idx = song_to_idx
        self.max_seq_len = max_seq_len
        self.mode = mode

        self.sequences = []
        self.targets = []
        self.negatives = []  # For training only

        self._build_sequences()

    def _build_sequences(self):
        """Build sequences from setlists."""
        all_song_indices = list(self.song_to_idx.values())

        for show_id in self.shows["show_id"]:
            # Get songs for this show in order
            show_setlist = self.setlists[
                self.setlists["show_id"] == show_id
            ].sort_values("pos")
            song_ids = show_setlist["song_id"].tolist()

            # Convert to indices
            song_indices = [
                self.song_to_idx[sid] for sid in song_ids if sid in self.song_to_idx
            ]

            if len(song_indices) < 2:
                continue

            # Create training examples: predict each song given previous songs
            for i in range(1, len(song_indices)):
                # Input: first i songs
                seq = song_indices[:i]

                # Pad or truncate to max_seq_len
                if len(seq) > self.max_seq_len:
                    seq = seq[-self.max_seq_len :]
                else:
                    seq = [0] * (self.max_seq_len - len(seq)) + seq

                # Target: next song
                target = song_indices[i]

                self.sequences.append(seq)
                self.targets.append(target)

                # Sample negative (for BPR loss during training)
                if self.mode == "train":
                    neg = np.random.choice(all_song_indices)
                    while neg == target or neg == 0:  # Don't sample target or padding
                        neg = np.random.choice(all_song_indices)
                    self.negatives.append(neg)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.LongTensor(self.sequences[idx])
        target = torch.LongTensor([self.targets[idx]])

        if self.mode == "train":
            neg = torch.LongTensor([self.negatives[idx]])
            return seq, target, neg
        else:
            return seq, target


class SetlistSequenceBatchDataset(Dataset):
    """Dataset that creates batches where each example predicts the next song
    at each position in the sequence (for more efficient training).

    Example:
      Input:  [song1, song2, song3, song4, song5]
      Targets: [song2, song3, song4, song5, song6]
      (Predict song2 given song1, song3 given [song1, song2], etc.)

    """

    def __init__(self, shows, setlists, song_to_idx, max_seq_len=50):
        """Args:
        shows: DataFrame of shows
        setlists: DataFrame of setlist entries
        song_to_idx: Dict mapping song_id -> integer index
        max_seq_len: Maximum sequence length.

        """
        self.shows = shows.sort_values("date")
        self.setlists = setlists
        self.song_to_idx = song_to_idx
        self.max_seq_len = max_seq_len
        self.num_songs = len(song_to_idx)

        self.sequences = []
        self.positives = []
        self.negatives = []

        self._build_sequences()

    def _build_sequences(self):
        """Build sequences from setlists."""
        for show_id in self.shows["show_id"]:
            # Get songs for this show in order
            show_setlist = self.setlists[
                self.setlists["show_id"] == show_id
            ].sort_values("pos")
            song_ids = show_setlist["song_id"].tolist()

            # Convert to indices
            song_indices = [
                self.song_to_idx[sid] for sid in song_ids if sid in self.song_to_idx
            ]

            if len(song_indices) < 2:
                continue

            # Truncate if too long
            if len(song_indices) > self.max_seq_len + 1:
                song_indices = song_indices[: self.max_seq_len + 1]

            # Build input and target sequences
            seq_len = len(song_indices) - 1
            input_seq = song_indices[:-1]
            target_seq = song_indices[1:]

            # Pad to max_seq_len
            input_seq = input_seq + [0] * (self.max_seq_len - len(input_seq))
            target_seq = target_seq + [0] * (self.max_seq_len - len(target_seq))

            # Sample negatives for each position
            neg_seq = []
            for i in range(self.max_seq_len):
                if target_seq[i] == 0:
                    neg_seq.append(0)
                else:
                    neg = np.random.randint(1, self.num_songs + 1)
                    while neg == target_seq[i]:
                        neg = np.random.randint(1, self.num_songs + 1)
                    neg_seq.append(neg)

            self.sequences.append(input_seq)
            self.positives.append(target_seq)
            self.negatives.append(neg_seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.LongTensor(self.sequences[idx])
        pos = torch.LongTensor(self.positives[idx])
        neg = torch.LongTensor(self.negatives[idx])

        return seq, pos, neg


def create_sasrec_dataloaders(
    train_shows, val_shows, test_shows, setlists, songs, max_seq_len=50, batch_size=128
):
    """Create train/val/test dataloaders for SASRec.

    Args:
        train_shows: DataFrame of training shows
        val_shows: DataFrame of validation shows
        test_shows: DataFrame of test shows
        setlists: DataFrame of all setlist entries
        songs: DataFrame of songs
        max_seq_len: Maximum sequence length
        batch_size: Batch size for training

    Returns:
        train_loader, val_loader, test_loader, song_to_idx, idx_to_song

    """
    # Create song vocabulary
    all_song_ids = songs["song_id"].unique()
    song_to_idx = {
        sid: i + 1 for i, sid in enumerate(all_song_ids)
    }  # Start from 1 (0 is padding)
    idx_to_song = {i: sid for sid, i in song_to_idx.items()}
    idx_to_song[0] = "<PAD>"

    # Filter setlists to each split
    train_setlists = setlists[setlists["show_id"].isin(train_shows["show_id"])]
    val_setlists = setlists[setlists["show_id"].isin(val_shows["show_id"])]
    test_setlists = setlists[setlists["show_id"].isin(test_shows["show_id"])]

    # Create datasets
    train_dataset = SetlistSequenceBatchDataset(
        train_shows, train_setlists, song_to_idx, max_seq_len
    )
    val_dataset = SetlistSequenceDataset(
        val_shows, val_setlists, song_to_idx, max_seq_len, mode="test"
    )
    test_dataset = SetlistSequenceDataset(
        test_shows, test_setlists, song_to_idx, max_seq_len, mode="test"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker to avoid memory issues
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Created SASRec dataloaders:")
    print(f"  Train: {len(train_dataset)} sequences, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} sequences, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} sequences, {len(test_loader)} batches")
    print(f"  Vocabulary size: {len(song_to_idx)} songs")

    return train_loader, val_loader, test_loader, song_to_idx, idx_to_song
