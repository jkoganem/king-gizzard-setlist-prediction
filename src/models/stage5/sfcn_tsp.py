"""SFCN-TSP: Simplified Fully Connected Network for Temporal Set Prediction.

Based on "Predicting Temporal Sets with Simplified Fully Connected Networks" (AAAI 2023).

Key Innovation:
- Extremely simple architecture (just linear layers)
- Often BEATS complex GNN models in TSP tasks
- Uses frequency + recency + lag features

Architecture:
    Input: [freq, recency, lag_1, lag_2, ..., lag_K] (K previous shows)
    Hidden: Linear(input_dim, hidden_dim) -> ReLU -> Dropout
    Output: Linear(hidden_dim, 1) -> Sigmoid

Surprisingly, this simple model is competitive with DNNTSP (KDD 2020) and other
graph-based methods for temporal set prediction!
"""

import torch
import torch.nn as nn


class SFCN(nn.Module):
    """Simplified Fully Connected Network for Temporal Set Prediction.

    Args:
        num_songs: Number of songs in vocabulary
        num_prev_shows: Number of previous shows to use for lag features (K)
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate for regularization

    """

    def __init__(
        self,
        num_songs: int,
        num_prev_shows: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_songs = num_songs
        self.num_prev_shows = num_prev_shows

        # Input features: frequency (1) + recency (1) + lag indicators (K)
        input_dim = 1 + 1 + num_prev_shows

        # Simple 2-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Frequency bias (global song popularity)
        # This is a learnable embedding, one value per song
        self.freq_bias = nn.Embedding(num_songs + 1, 1)  # +1 for padding

    def forward(
        self,
        song_ids: torch.Tensor,  # [batch_size]
        recency_scores: torch.Tensor,  # [batch_size] - time-weighted recency
        lag_features: torch.Tensor,  # [batch_size, num_prev_shows] - binary indicators
    ):
        """Forward pass for SFCN.

        Args:
            song_ids: Song IDs to score [batch_size]
            recency_scores: Time-decay recency scores [batch_size]
            lag_features: Binary indicators if song appeared in last K shows [batch_size, K]

        Returns:
            Probabilities [batch_size] - probability of each song appearing in next show

        """
        batch_size = song_ids.size(0)

        # Get frequency bias for each song
        freq = self.freq_bias(song_ids).squeeze(-1)  # [batch_size]

        # Concatenate all features
        # Shape: [batch_size, 1 + 1 + K]
        features = torch.cat(
            [
                freq.unsqueeze(-1),  # [batch_size, 1]
                recency_scores.unsqueeze(-1),  # [batch_size, 1]
                lag_features,  # [batch_size, K]
            ],
            dim=-1,
        )

        # Two-layer MLP
        h = torch.relu(self.fc1(features))  # [batch_size, hidden_dim]
        h = self.dropout(h)
        logits = self.fc2(h).squeeze(-1)  # [batch_size]

        return torch.sigmoid(logits)


class SFCNBatch(nn.Module):
    """Batch version of SFCN that scores all songs at once.

    This is more efficient for evaluation where we need to score all 203 songs.
    """

    def __init__(
        self,
        num_songs: int,
        num_prev_shows: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_songs = num_songs
        self.num_prev_shows = num_prev_shows

        # Input: frequency (1) + recency (1) + lag indicators (K)
        input_dim = 1 + 1 + num_prev_shows

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Frequency bias
        self.freq_bias = nn.Embedding(num_songs + 1, 1)

    def forward(
        self,
        recency_scores: torch.Tensor,  # [num_songs] - recency for each song
        lag_features: torch.Tensor,  # [num_songs, num_prev_shows] - lag for each song
    ):
        """Score all songs at once for a single show.

        Args:
            recency_scores: Recency scores for all songs [num_songs]
            lag_features: Lag features for all songs [num_songs, K]

        Returns:
            Probabilities [num_songs] - probability for each song

        """
        num_songs = recency_scores.size(0)

        # Create song IDs [0, 1, 2, ..., num_songs-1]
        song_ids = torch.arange(num_songs, device=recency_scores.device)

        # Get frequency bias
        freq = self.freq_bias(song_ids).squeeze(-1)  # [num_songs]

        # Concatenate features
        features = torch.cat(
            [freq.unsqueeze(-1), recency_scores.unsqueeze(-1), lag_features], dim=-1
        )  # [num_songs, 1+1+K]

        # Forward pass
        h = torch.relu(self.fc1(features))
        h = self.dropout(h)
        logits = self.fc2(h).squeeze(-1)

        return torch.sigmoid(logits)


def build_lag_features(setlists, shows, song_id, current_show_id, num_prev_shows=5):
    """Build lag features for a song: binary indicators of appearance in last K shows.

    Args:
        setlists: DataFrame of setlist entries
        shows: DataFrame of shows (must be sorted by date)
        song_id: Song to build features for
        current_show_id: Current show ID
        num_prev_shows: Number of previous shows to look back (K)

    Returns:
        Binary array [K] indicating if song appeared in each of the last K shows
        (most recent first)

    """
    import pandas as pd

    # Get current show date
    current_show = shows[shows["show_id"] == current_show_id].iloc[0]
    current_date = pd.to_datetime(current_show["date"])

    # Get previous shows (before current show)
    prev_shows = (
        shows[pd.to_datetime(shows["date"]) < current_date]
        .sort_values("date", ascending=False)
        .head(num_prev_shows)
    )

    # Check if song appeared in each previous show
    lag_indicators = []
    for _, prev_show in prev_shows.iterrows():
        prev_setlist = setlists[setlists["show_id"] == prev_show["show_id"]]
        appeared = int(song_id in prev_setlist["song_id"].values)
        lag_indicators.append(appeared)

    # Pad with zeros if fewer than K previous shows
    while len(lag_indicators) < num_prev_shows:
        lag_indicators.append(0)

    return lag_indicators[:num_prev_shows]  # Ensure exactly K features
