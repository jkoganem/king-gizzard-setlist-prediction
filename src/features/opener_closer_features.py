"""Feature engineering for opener/closer prediction.

This module creates rich features for predicting which song will open or close a show,
using show context, song attributes, and historical patterns.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np


def engineer_opener_closer_features(
    shows_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    setlists_df: pd.DataFrame,
    song_tags_df: pd.DataFrame,
    role: str,  # 'opener' or 'closer'
    train_show_ids: set = None,  # CRITICAL: Provide to prevent data leakage
) -> Tuple[pd.DataFrame, List[str]]:
    """Create rich features for opener/closer prediction.

    Instead of just using opener_rate/closer_rate (frequency), this creates
    features that include show context (tour, location, proximity) and song
    characteristics.

    **CRITICAL**: Provide train_show_ids to prevent data leakage. All statistics
    (global_role_rates, tour_role_rates, location_role_rates) are calculated
    from TRAINING shows only.

    Args:
        shows_df: DataFrame with show information
        songs_df: DataFrame with song information
        setlists_df: DataFrame with setlist entries
        song_tags_df: DataFrame with song tags (optional)
        role: Either 'opener' or 'closer'
        train_show_ids: Set of show_ids in training set (required to prevent leakage)

    Returns:
        Tuple of (features_df, feature_cols):
            - features_df: DataFrame with one row per (show, song) pair
            - feature_cols: List of feature column names

    """
    assert role in [
        "opener",
        "closer",
    ], f"role must be 'opener' or 'closer', got {role}"

    if train_show_ids is None:
        raise ValueError(
            "train_show_ids MUST be provided to prevent data leakage!\n"
            "Rates must be calculated from training data only."
        )

    print(f"\nEngineering {role} features (NO DATA LEAKAGE)...")

    # Sort shows by date
    shows_sorted = shows_df.sort_values("date").copy()
    shows_sorted["date"] = pd.to_datetime(shows_sorted["date"])

    all_songs = songs_df["song_id"].unique()

    # CRITICAL: Separate training shows for statistics calculation
    train_shows = shows_sorted[shows_sorted["show_id"].isin(train_show_ids)].copy()
    train_setlists = setlists_df[setlists_df["show_id"].isin(train_show_ids)].copy()

    print(f"  Using {len(train_shows)} TRAINING shows to calculate statistics")
    print(f"  Generating features for ALL {len(shows_sorted)} shows")

    # Build show_id -> opener/closer mapping (for ALL shows, for labeling)
    show_roles = {}
    for show_id, group in setlists_df.groupby("show_id"):
        setlist_sorted = group.sort_values("pos")
        if len(setlist_sorted) > 0:
            if role == "opener":
                show_roles[show_id] = setlist_sorted.iloc[0]["song_id"]
            else:  # closer
                show_roles[show_id] = setlist_sorted.iloc[-1]["song_id"]

    # CRITICAL: Compute statistics from TRAINING shows only
    train_show_roles = {
        sid: song for sid, song in show_roles.items() if sid in train_show_ids
    }

    # Compute global role rates (from TRAINING data only)
    role_counts = pd.Series(list(train_show_roles.values())).value_counts()
    total_train_shows = len(train_shows)
    global_role_rates = (role_counts / total_train_shows).to_dict()

    # Compute tour-specific rates (from TRAINING data only)
    tour_role_rates = {}
    for tour in train_shows["tour_name"].unique():
        if pd.isna(tour):
            continue
        tour_train_shows = train_shows[train_shows["tour_name"] == tour]
        tour_show_ids = set(tour_train_shows["show_id"])
        tour_roles = {
            sid: song for sid, song in train_show_roles.items() if sid in tour_show_ids
        }
        if len(tour_roles) > 0:
            tour_counts = pd.Series(list(tour_roles.values())).value_counts()
            tour_total = len(tour_train_shows)
            for song_id, count in tour_counts.items():
                tour_role_rates[(tour, song_id)] = count / tour_total

    # Compute location-specific rates (from TRAINING data only)
    location_role_rates = {}
    for country in train_shows["country"].unique():
        if pd.isna(country):
            continue
        country_train_shows = train_shows[train_shows["country"] == country]
        country_show_ids = set(country_train_shows["show_id"])
        country_roles = {
            sid: song
            for sid, song in train_show_roles.items()
            if sid in country_show_ids
        }
        if len(country_roles) > 0:
            country_counts = pd.Series(list(country_roles.values())).value_counts()
            country_total = len(country_train_shows)
            for song_id, count in country_counts.items():
                location_role_rates[(country, song_id)] = count / country_total

    # Create proximity features (from show_context module)
    from src.features.show_context import compute_show_proximity

    proximity_df = compute_show_proximity(shows_sorted)
    proximity_dict = proximity_df.set_index("show_id").to_dict("index")

    # Build examples
    examples = []
    for _, show in shows_sorted.iterrows():
        show_id = show["show_id"]
        tour = show.get("tour_name", "unknown")
        country = show.get("country", "unknown")
        is_festival = int(show.get("is_festival", 0))

        # Get proximity features
        proximity = proximity_dict.get(show_id, {})
        days_since_prev = proximity.get("days_since_prev", 999)
        same_venue = int(proximity.get("same_venue", False))

        # Find what was played prev night (for no-repeat signal)
        prev_show_id = proximity.get("prev_show_id")
        prev_role_song = show_roles.get(prev_show_id) if prev_show_id else None

        # Create example for each song
        for song_id in all_songs:
            # Label: 1 if this song was the opener/closer for this show
            label = 1 if show_roles.get(show_id) == song_id else 0

            # Feature 1: Global role rate
            global_rate = global_role_rates.get(song_id, 0.0)

            # Feature 2: Tour-specific rate
            tour_rate = tour_role_rates.get((tour, song_id), 0.0)

            # Feature 3: Location-specific rate
            location_rate = location_role_rates.get((country, song_id), 0.0)

            # Feature 4-5: Show proximity features
            # (days_since_prev, same_venue already computed)

            # Feature 6: No-repeat signal (was this song the opener/closer last night?)
            played_prev_night = 1 if song_id == prev_role_song else 0

            # Feature 7: Festival flag
            # (is_festival already defined)

            # Add example
            examples.append(
                {
                    "show_id": show_id,
                    "song_id": song_id,
                    "label": label,
                    f"{role}_rate_global": global_rate,
                    f"{role}_rate_tour": tour_rate,
                    f"{role}_rate_location": location_rate,
                    "days_since_prev": days_since_prev,
                    "same_venue": same_venue,
                    "played_prev_night": played_prev_night,
                    "is_festival": is_festival,
                }
            )

    df = pd.DataFrame(examples)

    # Define feature columns
    feature_cols = [
        f"{role}_rate_global",
        f"{role}_rate_tour",
        f"{role}_rate_location",
        "days_since_prev",
        "same_venue",
        "played_prev_night",
        "is_festival",
    ]

    print(
        f"  Created {len(df):,} examples ({len(shows_sorted)} shows x {len(all_songs)} songs)"
    )
    print(f"  Engineered {len(feature_cols)} features")

    return df, feature_cols
