"""Frequency and recency priors for temporal set prediction.

Based on SAFERec (2024): "Self-Attention and Frequency Enriched Model
for Next Basket Recommendation"

These priors capture two key patterns:
1. Frequency: Songs played often are more likely to appear
2. Recency: Songs played recently are less likely (avoid repeats)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def compute_song_frequencies(
    setlists: pd.DataFrame,
    songs: pd.DataFrame,
    train_show_ids: np.ndarray,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Compute global frequency prior for each song.

    Args:
        setlists: Setlist DataFrame with columns [show_id, song_id, position]
        songs: Songs DataFrame with column [song_id]
        train_show_ids: Array of training show IDs (to avoid leakage)
        smoothing: Laplace smoothing parameter (default: 1.0)

    Returns:
        Tensor of shape [num_songs] with log-frequency scores

    Formula:
        freq_score[i] = log(1 + count[i] / total_shows)

    Why log?
        - Diminishing returns (100->200 plays less important than 1->2)
        - Numerical stability
        - Matches SAFERec paper formulation

    """
    # Filter to training shows only
    train_setlists = setlists[setlists["show_id"].isin(train_show_ids)]

    # Count plays per song
    song_counts = (
        train_setlists.groupby("song_id").size().reindex(songs["song_id"], fill_value=0)
    )

    # Normalize by number of training shows
    n_shows = len(train_show_ids)
    play_rate = (song_counts + smoothing) / (n_shows + smoothing)

    # Log transform
    freq_scores = np.log1p(play_rate.values)

    return torch.tensor(freq_scores, dtype=torch.float32)


def compute_recency_scores(
    setlists: pd.DataFrame,
    shows: pd.DataFrame,
    current_show_id: str,
    songs: pd.DataFrame,
    decay_factor: float = 0.5,
    lookback_window: int = 5,
) -> torch.Tensor:
    """Compute recency penalty for each song based on recent plays.

    Args:
        setlists: Setlist DataFrame
        shows: Shows DataFrame with columns [show_id, date]
        current_show_id: ID of show we're predicting for
        songs: Songs DataFrame
        decay_factor: Exponential decay (0=no penalty, 1=full penalty)
        lookback_window: Number of previous shows to consider

    Returns:
        Tensor of shape [num_songs] with recency penalty scores
        (negative values = recently played -> penalize)

    Formula:
        recency_score[i] = -sum_{t in recent} decay^t * played[i, t]

    Why negative?
        - Recently played songs should be LESS likely
        - Avoids repeating songs from last night
        - Exponential decay: last show matters most

    """
    # Get current show date
    current_date = shows[shows["show_id"] == current_show_id]["date"].iloc[0]

    # Get previous shows (chronologically before current)
    prev_shows = shows[shows["date"] < current_date].sort_values(
        "date", ascending=False
    )
    prev_shows = prev_shows.head(lookback_window)

    # Initialize scores to zero
    recency_scores = np.zeros(len(songs))
    song_id_to_idx = {song_id: i for i, song_id in enumerate(songs["song_id"])}

    # Compute decay weights
    for t, show_id in enumerate(prev_shows["show_id"]):
        # Songs played in this show
        played_songs = setlists[setlists["show_id"] == show_id]["song_id"].values

        # Apply exponential decay
        weight = decay_factor**t

        for song_id in played_songs:
            if song_id in song_id_to_idx:
                idx = song_id_to_idx[song_id]
                recency_scores[idx] -= weight  # Negative = penalty

    return torch.tensor(recency_scores, dtype=torch.float32)


def compute_batch_recency_scores(
    setlists: pd.DataFrame,
    shows: pd.DataFrame,
    show_ids: List[str],
    songs: pd.DataFrame,
    decay_factor: float = 0.5,
    lookback_window: int = 5,
) -> torch.Tensor:
    """Compute recency scores for a batch of shows.

    Args:
        setlists: Setlist DataFrame
        shows: Shows DataFrame
        show_ids: List of show IDs to compute scores for
        songs: Songs DataFrame
        decay_factor: Exponential decay parameter
        lookback_window: Number of previous shows to consider

    Returns:
        Tensor of shape [batch_size, num_songs] with recency scores

    """
    batch_scores = []
    for show_id in show_ids:
        scores = compute_recency_scores(
            setlists,
            shows,
            show_id,
            songs,
            decay_factor=decay_factor,
            lookback_window=lookback_window,
        )
        batch_scores.append(scores)

    return torch.stack(batch_scores, dim=0)


class FrequencyRecencyPrior:
    """Learnable frequency/recency prior module.

    Usage:
        prior = FrequencyRecencyPrior(num_songs=203)
        prior.set_priors(freq_scores, recency_scores)

        # During forward pass:
        logits_with_prior = logits + prior(recency_scores)

    Attributes:
        alpha: Learnable weight for frequency prior
        beta: Learnable weight for recency prior
        freq_bias: Global frequency scores (fixed)
        use_frequency: Whether to use frequency prior
        use_recency: Whether to use recency prior

    """

    def __init__(
        self,
        num_songs: int,
        initial_alpha: float = 0.5,
        initial_beta: float = 0.5,
        use_frequency: bool = True,
        use_recency: bool = True,
    ):
        """Initialize prior module.

        Args:
            num_songs: Number of songs in vocabulary
            initial_alpha: Initial frequency weight
            initial_beta: Initial recency weight
            use_frequency: Enable frequency prior
            use_recency: Enable recency prior

        """
        self.num_songs = num_songs
        self.use_frequency = use_frequency
        self.use_recency = use_recency

        # Learnable parameters
        self.alpha = torch.nn.Parameter(torch.tensor(initial_alpha))
        self.beta = torch.nn.Parameter(torch.tensor(initial_beta))

        # Fixed frequency scores (set via set_freq_prior)
        self.register_buffer("freq_bias", torch.zeros(num_songs))

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Register buffer (for compatibility with nn.Module)."""
        setattr(self, name, tensor)

    def set_freq_prior(self, freq_scores: torch.Tensor) -> None:
        """Set global frequency prior (call once before training).

        Args:
            freq_scores: Tensor of shape [num_songs]

        """
        assert freq_scores.shape == (self.num_songs,)
        self.freq_bias = freq_scores

    def forward(self, recency_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute additive prior bias.

        Args:
            recency_scores: Tensor of shape [batch_size, num_songs] or [num_songs]
                           If None, only frequency prior is used

        Returns:
            prior_bias: Tensor to add to logits
                       Shape matches recency_scores if provided, else [num_songs]

        """
        bias = 0.0

        # Frequency prior (global)
        if self.use_frequency:
            bias = bias + self.alpha * self.freq_bias

        # Recency prior (show-specific)
        if self.use_recency and recency_scores is not None:
            bias = bias + self.beta * recency_scores

        return bias

    def __call__(self, recency_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass (callable interface)."""
        return self.forward(recency_scores)

    def parameters(self):
        """Return learnable parameters (for optimizer)."""
        return [self.alpha, self.beta]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "freq_bias": self.freq_bias,
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load state dictionary."""
        self.alpha = torch.nn.Parameter(state_dict["alpha"])
        self.beta = torch.nn.Parameter(state_dict["beta"])
        self.freq_bias = state_dict["freq_bias"]
