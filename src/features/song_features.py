"""Feature engineering for song-level features.

This module computes features related to individual songs including play rates,
recency statistics, and time-decay weighted features.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SongFeatures:
    """Features for a single song in prediction context.

    Attributes:
        song_id: Unique song identifier.
        global_play_rate: Frequency of song across all shows.
        tour_play_rate: Frequency of song on current tour.
        location_play_rate: Frequency of song in current location.
        recency_count_5: Number of times played in last 5 shows.
        recency_count_10: Number of times played in last 10 shows.
        time_decay_score: Weighted play score with exponential time decay.
        section_tags: Multi-hot encoding of section tags.

    """

    song_id: str
    global_play_rate: float
    tour_play_rate: float
    location_play_rate: float
    recency_count_5: int
    recency_count_10: int
    time_decay_score: float
    section_tags: Dict[str, bool]


def compute_song_features(
    songs_df: pd.DataFrame,
    setlists_df: pd.DataFrame,
    song_tags_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    halflife_days: int = 30,
) -> pd.DataFrame:
    """Compute song-level features for all songs.

    Args:
        songs_df: DataFrame with song information.
        setlists_df: DataFrame with setlist entries.
        song_tags_df: DataFrame with song tags.
        shows_df: DataFrame with show information.
        halflife_days: Half-life for time decay in days.

    Returns:
        DataFrame with computed song features.

    """
    # TODO: Implement song feature computation
    # - Calculate global play rates
    # - Calculate tour/location-specific rates
    # - Compute recency counts
    # - Apply time decay
    # - Create multi-hot tag encodings
    raise NotImplementedError("song_features.compute_song_features not yet implemented")


def compute_play_rate(
    song_id: str,
    setlists_df: pd.DataFrame,
    total_shows: int,
) -> float:
    """Compute play rate for a song.

    Args:
        song_id: Song identifier.
        setlists_df: DataFrame with setlist entries.
        total_shows: Total number of shows.

    Returns:
        Play rate as fraction of shows (0.0 to 1.0).

    """
    if total_shows == 0:
        return 0.0

    plays = (setlists_df["song_id"] == song_id).sum()
    return float(plays) / total_shows


def compute_time_decay_score(
    song_id: str,
    setlists_df: pd.DataFrame,
    shows_df: pd.DataFrame,
    reference_date: date,
    halflife_days: int,
) -> float:
    """Compute time-decayed play score for a song.

    Applies exponential decay based on how long ago each play occurred.

    Args:
        song_id: Song identifier.
        setlists_df: DataFrame with setlist entries.
        shows_df: DataFrame with show information including dates.
        reference_date: Reference date for computing decay.
        halflife_days: Half-life for exponential decay.

    Returns:
        Time-decayed score (higher = played more recently).

    """
    # Get all plays of this song
    plays_df = setlists_df[setlists_df["song_id"] == song_id].merge(
        shows_df[["show_id", "date"]], on="show_id"
    )

    if len(plays_df) == 0:
        return 0.0

    # Compute decay weights
    decay_constant = np.log(2) / halflife_days
    total_score = 0.0

    for _, play in plays_df.iterrows():
        play_date = pd.to_datetime(play["date"]).date()
        days_ago = (reference_date - play_date).days
        if days_ago < 0:
            continue  # Skip future shows
        weight = np.exp(-decay_constant * days_ago)
        total_score += weight

    return total_score


def create_section_multihot(
    song_id: str,
    song_tags_df: pd.DataFrame,
    all_sections: list[str],
) -> Dict[str, bool]:
    """Create multi-hot encoding of section tags for a song.

    Args:
        song_id: Song identifier.
        song_tags_df: DataFrame with song tags.
        all_sections: List of all possible section tags.

    Returns:
        Dictionary mapping section names to boolean presence flags.

    """
    song_tags = set(song_tags_df[song_tags_df["song_id"] == song_id]["tag"].tolist())

    return {section: (section in song_tags) for section in all_sections}
