"""Feature engineering pipeline with COMPLETE data leakage prevention.

This is the main orchestrator that coordinates all feature engineering modules.
It imports specialized feature functions from temporal, co-occurrence, context,
and residency modules to build a complete feature set.

Main Function:
- engineer_all_features: Coordinates entire feature engineering pipeline

The pipeline creates 27 baseline features or 34 features with residency support.
All features respect strict train/test separation to prevent data leakage.
"""

import pandas as pd

# Import feature engineering functions from specialized modules
from src.features.temporal_features import add_temporal_features
from src.features.cooccurrence_features import add_cooccurrence_features
from src.features.context_features import (
    add_tour_venue_features,
    add_proximity_features,
    add_advanced_features,
)
from src.features.residency_features import add_residency_features


def engineer_all_features(
    shows,
    songs,
    setlists,
    song_tags,
    train_show_ids=None,
    include_residency_features=True,
):
    """Engineer all features with COMPLETE data leakage prevention.

    CRITICAL PRINCIPLES:
    1. ALL statistics must be calculated from TRAINING data only
    2. NO test data should influence ANY feature calculation
    3. Temporal features must be calculated chronologically
    4. Features are applied to ALL shows but statistics from TRAIN only

    Args:
        shows: DataFrame of ALL shows (train + val + test)
        songs: DataFrame of songs
        setlists: DataFrame of setlist entries for ALL shows
        song_tags: DataFrame of song tags
        train_show_ids: Set of show_ids for training data. REQUIRED to prevent leakage!
        include_residency_features: If True, include 7 residency features (34 total).
            If False, use baseline 27 features.

    Returns:
        df: DataFrame with all features and labels for ALL shows
        feature_cols: List of feature column names
    """
    feature_count = 34 if include_residency_features else 27
    print(
        f"\nEngineering {feature_count} features "
        f"(STRICT NO-LEAKAGE VERSION, with ALL improvements + MARATHON)..."
    )

    if train_show_ids is None:
        raise ValueError("train_show_ids MUST be provided to prevent data leakage!")

    print(
        f"  Training on {len(train_show_ids)} shows, "
        f"creating features for {len(shows)} total shows"
    )

    # Count marathon/residency shows (already computed in shows DataFrame by dataio.load_all_data())
    if "is_marathon" in shows.columns:
        n_marathon = shows["is_marathon"].sum()
        print(f"  Found {n_marathon} marathon shows (computed in dataio)")
    if "is_residency" in shows.columns:
        n_residency = shows["is_residency"].sum()
        print(f"  Found {n_residency} residency shows (computed in dataio)")

    all_song_ids = songs["song_id"].unique()
    examples = []

    # Create base examples (show x song combinations) for ALL shows
    for _, show in shows.iterrows():
        show_setlist = setlists[setlists["show_id"] == show["show_id"]]
        played_songs = set(show_setlist["song_id"].values)

        for song_id in all_song_ids:
            examples.append(
                {
                    "show_id": show["show_id"],
                    "song_id": song_id,
                    "label": 1 if song_id in played_songs else 0,
                    "country": show["country"],
                    "is_festival": int(show["is_festival"]),
                    "is_marathon": int(
                        show.get("is_marathon", 0)
                    ),  # Read from shows DataFrame
                }
            )

    df = pd.DataFrame(examples)
    print(
        f"  Created {len(df)} examples ({len(shows)} shows x {len(all_song_ids)} songs)"
    )

    # ===================================================================
    # CRITICAL: Separate training data for statistics calculation
    # ===================================================================

    train_df = df[df["show_id"].isin(train_show_ids)]
    train_shows = shows[shows["show_id"].isin(train_show_ids)]
    train_setlists = setlists[setlists["show_id"].isin(train_show_ids)]

    # Feature 1: Global play rate (from TRAINING data only)
    print("  - play_rate (train-only statistics)")
    train_play_counts = train_df.groupby("song_id")["label"].sum()
    play_rates = (train_play_counts / len(train_shows)).to_dict()
    df["play_rate"] = df["song_id"].map(play_rates).fillna(0)

    # Feature 2: Country-specific play rate (from TRAINING data only)
    print("  - country_play_rate (train-only statistics)")
    train_country_play = train_df.groupby(["country", "song_id"])["label"].mean()
    # CRITICAL FIX: Use train_df mean, not df mean!
    train_global_mean = (
        train_df["play_rate"].mean() if "play_rate" in train_df.columns else 0.1
    )
    df["country_play_rate"] = df.apply(
        lambda row: train_country_play.get(
            (row["country"], row["song_id"]), train_global_mean
        ),
        axis=1,
    )

    # Temporal features (calculated chronologically - no leakage)
    print("  - temporal features (chronological, no leakage)")
    df = add_temporal_features(df, shows, setlists, all_song_ids)

    # Co-occurrence features (from TRAINING data only)
    print("  - co-occurrence features (train-only PMI)")
    df = add_cooccurrence_features(df, train_setlists, all_song_ids)

    # Tour/venue features (from TRAINING data only)
    print("  - tour/venue features (train-only statistics)")
    df = add_tour_venue_features(df, shows, train_shows, train_setlists, all_song_ids)

    # Core songs (based on TRAINING play rate only)
    core_song_threshold = 0.6
    # Use training play rates, not overall df play rates!
    train_song_rates = train_df.groupby("song_id")["label"].mean()
    core_songs = set(train_song_rates[train_song_rates > core_song_threshold].index)
    df["is_core_song"] = df["song_id"].apply(lambda x: 1 if x in core_songs else 0)

    # Proximity and no-repeat features (temporal, calculated chronologically)
    print("  - proximity and no-repeat features (chronological)")
    df = add_proximity_features(df, shows, setlists)

    # Residency features (optional - NEW, fixes venue_play_rate dilution)
    if include_residency_features:
        print("  - residency features (train-only statistics, temporal no-repeat)")
        df = add_residency_features(
            df, shows, train_shows, train_setlists, setlists, all_song_ids
        )

    # Advanced features (NEW: venue affinity, tour position, calendar)
    print("  - advanced features (venue affinity, tour position, calendar)")
    df = add_advanced_features(df, shows, train_shows, train_setlists, all_song_ids)

    # Define feature columns
    feature_cols = [
        "play_rate",
        "country_play_rate",
        "is_festival",
        "is_marathon",  # NEW: Marathon show flag
        "days_since_last",
        "play_last_30",
        "play_last_90",
        "days_into_tour",
        "time_decay_15d",
        "time_decay_30d",
        "time_decay_60d",
        "avg_cooccurrence",
        "max_cooccurrence",
        "num_common_cooccur",
        "played_this_tour",
        "tour_play_rate",
        "venue_play_rate",
        "venue_affinity",
        "tour_position_norm",
        "day_of_week",
        "is_weekend",
        "season",
        "is_core_song",
        "days_since_prev",
        "same_venue",
        "played_prev_night",
        "played_prev_venue",
    ]

    # Add residency features if requested
    if include_residency_features:
        feature_cols.extend(
            [
                "is_residency",
                "residency_night_number",
                "residency_total_nights",
                "played_earlier_in_residency",
                "nights_since_played_in_residency",
                "residency_night_1_rate",
                "residency_appearance_rate",
            ]
        )

    print(f"  Engineered {len(feature_cols)} features with NO DATA LEAKAGE")
    return df, feature_cols
