"""Context feature engineering module.

This module handles tour, venue, proximity, and advanced contextual features.
All statistics are calculated from training data only to prevent leakage,
while temporal features are calculated chronologically.

Features:
Tour/Venue (train-only stats):
- played_this_tour: Binary flag if song played earlier in tour
- tour_play_rate: Song play rate on this tour
- venue_play_rate: Song play rate at this venue
- venue_affinity: Venue-specific preference (venue_rate / global_rate)

Proximity (temporal, chronological):
- days_since_prev: Days since previous show
- same_venue: Binary flag if same venue as previous show
- played_prev_night: Song played at previous show
- played_prev_venue: Song played at previous venue

Advanced (calendar/position):
- tour_position_norm: Normalized position within tour (0=start, 1=end)
- day_of_week: Day of week (0=Monday, 6=Sunday)
- is_weekend: Binary flag for Friday/Saturday/Sunday
- season: Quarter of year (0-3)
"""

import pandas as pd
from collections import defaultdict


def add_tour_venue_features(df, all_shows, train_shows, train_setlists, all_song_ids):
    """Add tour and venue features using TRAINING statistics only.

    CRITICAL: Statistics calculated from train_shows/train_setlists ONLY,
    then applied to ALL shows.

    Args:
        df: DataFrame with show_id, song_id columns
        all_shows: DataFrame of all shows
        train_shows: DataFrame of training shows only
        train_setlists: DataFrame of training setlist entries only
        all_song_ids: List/array of all song IDs

    Returns:
        df: DataFrame with added tour/venue features
    """
    # Calculate statistics from TRAINING data only
    tour_songs = defaultdict(set)
    venue_songs = defaultdict(set)
    tour_counts = defaultdict(lambda: defaultdict(int))
    venue_counts = defaultdict(lambda: defaultdict(int))
    tour_show_counts = defaultdict(int)
    venue_show_counts = defaultdict(int)

    for _, show in train_shows.iterrows():
        show_id = show["show_id"]
        tour_name = show.get("tour_name", "unknown")
        venue_id = show.get("venue_id", show.get("city", "unknown"))

        tour_show_counts[tour_name] += 1
        venue_show_counts[venue_id] += 1

        show_setlist = train_setlists[train_setlists["show_id"] == show_id]
        for song_id in show_setlist["song_id"].values:
            tour_songs[tour_name].add(song_id)
            venue_songs[venue_id].add(song_id)
            tour_counts[tour_name][song_id] += 1
            venue_counts[venue_id][song_id] += 1

    # Apply training statistics to ALL shows
    tour_features = {}
    for _, show in all_shows.iterrows():
        show_id = show["show_id"]
        tour_name = show.get("tour_name", "unknown")
        venue_id = show.get("venue_id", show.get("city", "unknown"))

        for song_id in all_song_ids:
            # Use training statistics
            played_this_tour = 1 if song_id in tour_songs[tour_name] else 0
            tour_play_rate = tour_counts[tour_name].get(song_id, 0) / max(
                tour_show_counts[tour_name], 1
            )
            venue_play_rate = venue_counts[venue_id].get(song_id, 0) / max(
                venue_show_counts[venue_id], 1
            )

            tour_features[(show_id, song_id)] = {
                "played_this_tour": played_this_tour,
                "tour_play_rate": tour_play_rate,
                "venue_play_rate": venue_play_rate,
            }

    df["played_this_tour"] = df.apply(
        lambda row: tour_features.get((row["show_id"], row["song_id"]), {}).get(
            "played_this_tour", 0
        ),
        axis=1,
    )
    df["tour_play_rate"] = df.apply(
        lambda row: tour_features.get((row["show_id"], row["song_id"]), {}).get(
            "tour_play_rate", 0
        ),
        axis=1,
    )
    df["venue_play_rate"] = df.apply(
        lambda row: tour_features.get((row["show_id"], row["song_id"]), {}).get(
            "venue_play_rate", 0
        ),
        axis=1,
    )

    return df


def add_proximity_features(df, shows, setlists):
    """Add proximity and no-repeat features (chronological, no leakage).

    Args:
        df: DataFrame with show_id, song_id columns
        shows: DataFrame of all shows
        setlists: DataFrame of all setlist entries

    Returns:
        df: DataFrame with added proximity features
    """
    from src.features.show_context import (
        compute_show_proximity,
        compute_no_repeat_signals,
    )

    # These are temporal features calculated chronologically
    proximity_df = compute_show_proximity(shows)
    df = df.merge(
        proximity_df[["show_id", "days_since_prev", "same_venue"]],
        on="show_id",
        how="left",
    )
    df["days_since_prev"] = df["days_since_prev"].fillna(999)
    df["same_venue"] = df["same_venue"].astype(int)

    # No-repeat signals
    no_repeat_df = compute_no_repeat_signals(shows, setlists)
    df = df.merge(
        no_repeat_df[["show_id", "song_id", "played_prev_night", "played_prev_venue"]],
        on=["show_id", "song_id"],
        how="left",
    )
    df["played_prev_night"] = df["played_prev_night"].fillna(0).astype(int)
    df["played_prev_venue"] = df["played_prev_venue"].fillna(0).astype(int)

    return df


def add_advanced_features(df, all_shows, train_shows, train_setlists, all_song_ids):
    """Add advanced features: venue affinity, tour position, calendar patterns.

    NEW HIGH-IMPACT FEATURES:
    1. venue_affinity: song_rate_at_venue / song_rate_global (captures venue preferences)
    2. tour_position_norm: normalized position within tour (0=start, 1=end)
    3. day_of_week: day of week (0=Monday, 6=Sunday)
    4. is_weekend: binary flag for Friday/Saturday/Sunday
    5. season: quarter of year (0-3)

    All computed with NO DATA LEAKAGE (train-only statistics).

    Args:
        df: DataFrame with show_id, song_id columns
        all_shows: DataFrame of all shows
        train_shows: DataFrame of training shows only
        train_setlists: DataFrame of training setlist entries only
        all_song_ids: List/array of all song IDs

    Returns:
        df: DataFrame with added advanced features
    """
    # Compute venue affinity from TRAINING data only
    # Formula: venue_affinity = (song plays at venue / venue shows) /
    #          (song plays global / all shows)

    train_setlists_with_venue = train_setlists.merge(
        train_shows[["show_id", "venue_id"]], on="show_id", how="left"
    )

    # Global rates from training
    train_song_global_rates = {}
    total_train_shows = len(train_shows)
    for song_id in all_song_ids:
        plays = (train_setlists["song_id"] == song_id).sum()
        train_song_global_rates[song_id] = (
            plays / total_train_shows if total_train_shows > 0 else 0
        )

    # Venue-specific rates from training
    venue_song_rates = {}
    venue_show_counts = train_shows["venue_id"].value_counts().to_dict()

    for venue_id in train_shows["venue_id"].unique():
        venue_shows = train_shows[train_shows["venue_id"] == venue_id]
        venue_setlists = train_setlists[
            train_setlists["show_id"].isin(venue_shows["show_id"])
        ]
        venue_total = len(venue_shows)

        for song_id in all_song_ids:
            plays_at_venue = (venue_setlists["song_id"] == song_id).sum()
            venue_rate = plays_at_venue / venue_total if venue_total > 0 else 0
            venue_song_rates[(venue_id, song_id)] = venue_rate

    # Compute venue affinity for each row
    venue_affinity_dict = {}
    for _, row in df.iterrows():
        show_id = row["show_id"]
        song_id = row["song_id"]

        show = all_shows[all_shows["show_id"] == show_id]
        if len(show) == 0:
            continue

        venue_id = show.iloc[0]["venue_id"]

        global_rate = train_song_global_rates.get(song_id, 0)
        venue_rate = venue_song_rates.get((venue_id, song_id), 0)

        # Affinity = venue_rate / global_rate (with smoothing)
        if global_rate > 0:
            # Add small constant to prevent division issues
            affinity = venue_rate / (global_rate + 0.01)
        else:
            affinity = 0

        venue_affinity_dict[(show_id, song_id)] = affinity

    df["venue_affinity"] = df.apply(
        lambda row: venue_affinity_dict.get((row["show_id"], row["song_id"]), 0), axis=1
    )

    # Tour position (normalized 0-1 within tour)
    tour_position_dict = {}
    all_shows_sorted = all_shows.sort_values("date")

    current_tour = None
    tour_shows = []

    for _, show in all_shows_sorted.iterrows():
        if show["tour_name"] != current_tour:
            # Process previous tour
            if len(tour_shows) > 0:
                for i, show_id in enumerate(tour_shows):
                    position = i / max(len(tour_shows) - 1, 1)  # Normalize to 0-1
                    tour_position_dict[show_id] = position

            current_tour = show["tour_name"]
            tour_shows = [show["show_id"]]
        else:
            tour_shows.append(show["show_id"])

    # Process last tour
    if len(tour_shows) > 0:
        for i, show_id in enumerate(tour_shows):
            position = i / max(len(tour_shows) - 1, 1)
            tour_position_dict[show_id] = position

    df["tour_position_norm"] = df["show_id"].map(tour_position_dict).fillna(0.5)

    # Calendar features
    show_dates = all_shows.set_index("show_id")["date"]
    show_dates = pd.to_datetime(show_dates)

    df["day_of_week"] = df["show_id"].map(
        lambda sid: (show_dates.get(sid).dayofweek if sid in show_dates.index else 0)
    )
    df["is_weekend"] = df["day_of_week"].isin([4, 5, 6]).astype(int)  # Fri, Sat, Sun
    df["season"] = df["show_id"].map(
        lambda sid: (
            (show_dates.get(sid).month - 1) // 3 if sid in show_dates.index else 0
        )
    )

    return df
