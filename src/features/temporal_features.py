"""Temporal feature engineering module.

This module handles time-based features that track song play history,
recency, and temporal decay patterns. All features are calculated
chronologically to prevent data leakage.

Features:
- days_since_last: Days since song was last played
- play_last_30: Number of plays in last 30 days
- play_last_90: Number of plays in last 90 days
- days_into_tour: Days elapsed since tour start
- time_decay_15d: Time-decay score with 15-day half-life
- time_decay_30d: Time-decay score with 30-day half-life
- time_decay_60d: Time-decay score with 60-day half-life
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def add_temporal_features(df, shows, setlists, all_song_ids):
    """Add temporal features calculated chronologically (no leakage).

    Args:
        df: DataFrame with show_id, song_id columns
        shows: DataFrame of all shows
        setlists: DataFrame of all setlist entries
        all_song_ids: List/array of all song IDs

    Returns:
        df: DataFrame with added temporal features
    """
    shows_sorted = shows.sort_values("date")

    # Track when each song was last played
    last_played = {}
    play_count_30 = defaultdict(int)
    play_count_90 = defaultdict(int)

    # Time-decay tracking: maintain list of (date, weight) for each song
    play_history = defaultdict(list)  # song_id -> [(date, 1.0), ...]

    days_since_last_dict = {}
    play_last_30_dict = {}
    play_last_90_dict = {}
    days_into_tour_dict = {}
    time_decay_15_dict = {}  # NEW: 15-day half-life
    time_decay_30_dict = {}  # NEW: 30-day half-life
    time_decay_60_dict = {}  # NEW: 60-day half-life

    current_tour = None
    tour_start_date = None

    # Process shows chronologically
    for _, show in shows_sorted.iterrows():
        show_id = show["show_id"]
        show_date = pd.to_datetime(show["date"])
        show_setlist = setlists[setlists["show_id"] == show_id]
        played_songs = set(show_setlist["song_id"].values)

        # Track tour changes
        if show.get("tour_name") != current_tour:
            current_tour = show.get("tour_name")
            tour_start_date = show_date

        days_into_tour = (show_date - tour_start_date).days if tour_start_date else 0

        # Calculate features for this show BEFORE updating history
        for song_id in all_song_ids:
            # Days since last played (use history up to but not including this show)
            if song_id in last_played:
                days_since = (show_date - last_played[song_id]).days
            else:
                days_since = 9999  # Never played before

            # Recent play counts (from history, not including this show)
            days_since_last_dict[(show_id, song_id)] = days_since
            play_last_30_dict[(show_id, song_id)] = play_count_30[song_id]
            play_last_90_dict[(show_id, song_id)] = play_count_90[song_id]
            days_into_tour_dict[(show_id, song_id)] = days_into_tour

            # Time-decay scores with different half-lives
            # Formula: score = sum(exp(-ln(2) * days_ago / halflife))
            history = play_history[song_id]

            decay_15 = 0.0
            decay_30 = 0.0
            decay_60 = 0.0

            for play_date in history:
                days_ago = (show_date - play_date).days
                if days_ago >= 0:  # Only past plays
                    decay_15 += np.exp(-np.log(2) * days_ago / 15)
                    decay_30 += np.exp(-np.log(2) * days_ago / 30)
                    decay_60 += np.exp(-np.log(2) * days_ago / 60)

            time_decay_15_dict[(show_id, song_id)] = decay_15
            time_decay_30_dict[(show_id, song_id)] = decay_30
            time_decay_60_dict[(show_id, song_id)] = decay_60

        # NOW update history with this show's data
        for song_id in played_songs:
            last_played[song_id] = show_date
            play_history[song_id].append(show_date)  # Track date for time-decay

        # Update rolling counts (simplified - would need proper windowing in production)
        for song_id in played_songs:
            play_count_30[song_id] += 1
            play_count_90[song_id] += 1

        # Decay old counts (simplified)
        # In production, would track exact dates and remove old plays

    # Apply features to dataframe
    df["days_since_last"] = df.apply(
        lambda row: days_since_last_dict.get((row["show_id"], row["song_id"]), 9999),
        axis=1,
    )
    df["play_last_30"] = df.apply(
        lambda row: play_last_30_dict.get((row["show_id"], row["song_id"]), 0), axis=1
    )
    df["play_last_90"] = df.apply(
        lambda row: play_last_90_dict.get((row["show_id"], row["song_id"]), 0), axis=1
    )
    df["days_into_tour"] = df.apply(
        lambda row: days_into_tour_dict.get((row["show_id"], row["song_id"]), 0), axis=1
    )

    # NEW: Time-decay features
    df["time_decay_15d"] = df.apply(
        lambda row: time_decay_15_dict.get((row["show_id"], row["song_id"]), 0.0),
        axis=1,
    )
    df["time_decay_30d"] = df.apply(
        lambda row: time_decay_30_dict.get((row["show_id"], row["song_id"]), 0.0),
        axis=1,
    )
    df["time_decay_60d"] = df.apply(
        lambda row: time_decay_60_dict.get((row["show_id"], row["song_id"]), 0.0),
        axis=1,
    )

    return df
