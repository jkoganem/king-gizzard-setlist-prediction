"""Residency feature engineering module.

This module handles detection and feature engineering for multi-night
residencies (consecutive shows at the same venue). Features help prevent
venue_play_rate dilution and capture residency-specific patterns.

Key Functions:
- identify_residencies: Detect multi-night residencies
- calculate_residency_rates: Compute statistics from training data
- add_residency_features: Add 7 residency features to dataframe

Features:
1. is_residency: Binary flag if show is part of multi-night residency
2. residency_night_number: Which night (1, 2, 3...)
3. residency_total_nights: Total nights in residency
4. played_earlier_in_residency: Was song played earlier in THIS residency?
5. nights_since_played_in_residency: How many nights ago in residency?
6. residency_night_1_rate: Rate on opening night (from training)
7. residency_appearance_rate: Rate across any night (from training)
"""

import pandas as pd
from collections import defaultdict


def identify_residencies(shows):
    """Identify multi-night residencies (consecutive shows at same venue).

    A residency is defined as 2+ consecutive shows at the same venue within 4 days.

    Args:
        shows: DataFrame of shows with venue_id, date columns

    Returns:
        List of dicts with residency metadata:
        {
            'venue_id': venue identifier,
            'show_ids': list of show_ids in order,
            'total_nights': number of nights,
            'start_date': first show date,
            'end_date': last show date
        }
    """
    shows = shows.sort_values("date").copy()
    shows["date"] = pd.to_datetime(shows["date"])

    residencies = []

    for venue_id in shows["venue_id"].unique():
        venue_shows = shows[shows["venue_id"] == venue_id].sort_values("date")

        if len(venue_shows) < 2:
            continue

        current_residency_shows = []
        prev_date = None

        for idx, (_, show) in enumerate(venue_shows.iterrows()):
            if prev_date is None:
                # First show at this venue
                current_residency_shows = [show]
            elif (show["date"] - prev_date).days <= 4:
                # Within 4 days - same residency
                current_residency_shows.append(show)
            else:
                # Gap > 4 days - new residency
                if len(current_residency_shows) >= 2:
                    # Previous group was a multi-night residency
                    residencies.append(
                        {
                            "venue_id": venue_id,
                            "show_ids": [s["show_id"] for s in current_residency_shows],
                            "total_nights": len(current_residency_shows),
                            "start_date": current_residency_shows[0]["date"],
                            "end_date": current_residency_shows[-1]["date"],
                        }
                    )
                current_residency_shows = [show]

            prev_date = show["date"]

        # Handle last group
        if len(current_residency_shows) >= 2:
            residencies.append(
                {
                    "venue_id": venue_id,
                    "show_ids": [s["show_id"] for s in current_residency_shows],
                    "total_nights": len(current_residency_shows),
                    "start_date": current_residency_shows[0]["date"],
                    "end_date": current_residency_shows[-1]["date"],
                }
            )

    return residencies


def calculate_residency_rates(residencies, setlists):
    """Calculate residency-specific song rates.

    Args:
        residencies: List of residency dicts from identify_residencies()
        setlists: DataFrame of setlist entries

    Returns:
        Two dicts:
        - residency_night_1_rate: (venue_id, song_id) -> rate on Night 1
        - residency_appearance_rate: (venue_id, song_id) -> rate across any night
    """
    residency_night_1_rates = defaultdict(lambda: defaultdict(int))
    residency_appearance_rates = defaultdict(lambda: defaultdict(int))
    residency_counts = defaultdict(int)

    for residency in residencies:
        venue_id = residency["venue_id"]
        residency_counts[venue_id] += 1

        # Night 1 songs
        night_1_show_id = residency["show_ids"][0]
        night_1_songs = set(setlists[setlists["show_id"] == night_1_show_id]["song_id"])
        for song_id in night_1_songs:
            residency_night_1_rates[venue_id][song_id] += 1

        # All songs across any night of residency
        all_residency_songs = set()
        for show_id in residency["show_ids"]:
            show_songs = set(setlists[setlists["show_id"] == show_id]["song_id"])
            all_residency_songs.update(show_songs)

        for song_id in all_residency_songs:
            residency_appearance_rates[venue_id][song_id] += 1

    # Convert counts to rates
    night_1_rate_dict = {}
    appearance_rate_dict = {}

    for venue_id in residency_counts:
        total = residency_counts[venue_id]
        for song_id, count in residency_night_1_rates[venue_id].items():
            night_1_rate_dict[(venue_id, song_id)] = count / total
        for song_id, count in residency_appearance_rates[venue_id].items():
            appearance_rate_dict[(venue_id, song_id)] = count / total

    return night_1_rate_dict, appearance_rate_dict


def add_residency_features(
    df, all_shows, train_shows, train_setlists, all_setlists, all_song_ids
):
    """Add 7 residency-aware features to prevent venue_play_rate dilution.

    Features:
    1. is_residency: Binary flag if show is part of multi-night residency
    2. residency_night_number: Which night (1, 2, 3...)
    3. residency_total_nights: Total nights in residency
    4. played_earlier_in_residency: Was song played earlier in THIS residency?
    5. nights_since_played_in_residency: How many nights ago in residency?
    6. residency_night_1_rate: Rate on opening night (from training)
    7. residency_appearance_rate: Rate across any night (from training)

    CRITICAL: Uses training data only for rates (#6, #7) to prevent leakage.

    Args:
        df: DataFrame with show_id, song_id columns
        all_shows: DataFrame of all shows
        train_shows: DataFrame of training shows only
        train_setlists: DataFrame of training setlist entries only
        all_setlists: DataFrame of all setlist entries
        all_song_ids: List/array of all song IDs

    Returns:
        df: DataFrame with added residency features
    """
    # Identify residencies in ALL shows (for context)
    all_residencies = identify_residencies(all_shows)

    # Calculate rates from TRAINING residencies only
    train_residencies = identify_residencies(train_shows)
    night_1_rates, appearance_rates = calculate_residency_rates(
        train_residencies, train_setlists
    )

    # Build residency lookup: show_id -> residency info
    show_to_residency = {}
    for residency in all_residencies:
        for idx, show_id in enumerate(residency["show_ids"]):
            show_to_residency[show_id] = {
                "venue_id": residency["venue_id"],
                "night_number": idx + 1,
                "total_nights": residency["total_nights"],
                "all_show_ids": residency["show_ids"],
            }

    # Build features for each (show, song) pair
    residency_features = {}

    for _, show in all_shows.iterrows():
        show_id = show["show_id"]
        venue_id = show.get("venue_id", show.get("city", "unknown"))

        residency_info = show_to_residency.get(show_id)

        if residency_info is None:
            # Not a residency - all features are 0
            for song_id in all_song_ids:
                residency_features[(show_id, song_id)] = {
                    "is_residency": 0,
                    "residency_night_number": 0,
                    "residency_total_nights": 0,
                    "played_earlier_in_residency": 0,
                    "nights_since_played_in_residency": 0,
                    "residency_night_1_rate": 0.0,
                    "residency_appearance_rate": 0.0,
                }
        else:
            # Is a residency
            night_number = residency_info["night_number"]
            total_nights = residency_info["total_nights"]

            # Get songs played in earlier nights of THIS residency
            earlier_songs = set()
            nights_since = {}

            # CRITICAL: Only look at earlier nights that are TEMPORAL (happened before)
            # This is NOT data leakage because these are previous nights of the residency
            # In production, we'd know what was played on Night 1 when predicting Night 2
            for i in range(night_number - 1):
                earlier_show_id = residency_info["all_show_ids"][i]
                # Use full setlists (temporal information - earlier nights happened first)
                earlier_setlist = all_setlists[
                    all_setlists["show_id"] == earlier_show_id
                ]

                for song_id in earlier_setlist["song_id"]:
                    earlier_songs.add(song_id)
                    nights_since[song_id] = night_number - (i + 1)

            # Create features for each song
            for song_id in all_song_ids:
                residency_features[(show_id, song_id)] = {
                    "is_residency": 1,
                    "residency_night_number": night_number,
                    "residency_total_nights": total_nights,
                    "played_earlier_in_residency": 1 if song_id in earlier_songs else 0,
                    "nights_since_played_in_residency": nights_since.get(song_id, 0),
                    "residency_night_1_rate": night_1_rates.get(
                        (venue_id, song_id), 0.0
                    ),
                    "residency_appearance_rate": appearance_rates.get(
                        (venue_id, song_id), 0.0
                    ),
                }

    # Add features to dataframe
    for feature_name in [
        "is_residency",
        "residency_night_number",
        "residency_total_nights",
        "played_earlier_in_residency",
        "nights_since_played_in_residency",
        "residency_night_1_rate",
        "residency_appearance_rate",
    ]:
        df[feature_name] = df.apply(
            lambda row: residency_features.get(
                (row["show_id"], row["song_id"]), {}
            ).get(feature_name, 0),
            axis=1,
        )

    return df
