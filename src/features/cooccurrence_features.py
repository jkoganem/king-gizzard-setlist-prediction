"""Co-occurrence feature engineering module.

This module calculates song co-occurrence patterns using Pointwise Mutual
Information (PMI) to measure how frequently songs appear together in setlists.
All statistics are calculated from training data only to prevent leakage.

Features:
- avg_cooccurrence: Average PMI with other songs in the show
- max_cooccurrence: Maximum PMI with any other song in the show
- num_common_cooccur: Number of songs with positive PMI in the show
"""

import numpy as np
from collections import Counter


def add_cooccurrence_features(df, train_setlists, all_song_ids):
    """Add PMI-based co-occurrence features using TRAINING data only.

    CRITICAL: PMI scores are calculated from train_setlists ONLY,
    then applied to all shows in df.

    Args:
        df: DataFrame with show_id, song_id, label columns
        train_setlists: DataFrame of setlist entries (training data only)
        all_song_ids: List/array of all song IDs

    Returns:
        df: DataFrame with added co-occurrence features
    """
    # Calculate PMI scores from TRAINING data only
    song_counts = Counter()
    pair_counts = Counter()
    total_train_shows = train_setlists["show_id"].nunique()

    for show_id, show_setlist in train_setlists.groupby("show_id"):
        songs_in_show = set(show_setlist["song_id"].values)

        for song in songs_in_show:
            song_counts[song] += 1

        for song1 in songs_in_show:
            for song2 in songs_in_show:
                if song1 < song2:
                    pair_counts[(song1, song2)] += 1

    # Calculate PMI from training statistics
    pmi_scores = {}
    for (song1, song2), pair_count in pair_counts.items():
        p_song1 = song_counts[song1] / total_train_shows
        p_song2 = song_counts[song2] / total_train_shows
        p_pair = pair_count / total_train_shows

        if p_song1 > 0 and p_song2 > 0 and p_pair > 0:
            pmi = np.log(p_pair / (p_song1 * p_song2))
            pmi_scores[(song1, song2)] = pmi
            pmi_scores[(song2, song1)] = pmi  # Symmetric

    # Apply PMI scores to ALL shows (but scores come from training only)
    cooccur_dict = {}

    # Need ALL setlists to know what songs are in each show
    # But PMI scores come from training only
    all_setlists = df.groupby("show_id")["song_id"].apply(
        lambda x: set(x[df.loc[x.index, "label"] == 1])
    )

    for show_id in df["show_id"].unique():
        songs_in_show = all_setlists.get(show_id, set())

        for song_id in all_song_ids:
            pmis = []
            for other_song in songs_in_show:
                if song_id != other_song:
                    # Use training PMI scores
                    pmi = pmi_scores.get((song_id, other_song), 0)
                    pmis.append(pmi)

            if pmis:
                avg_pmi = np.mean(pmis)
                max_pmi = max(pmis)
                num_common = sum(1 for p in pmis if p > 0)
            else:
                avg_pmi = 0
                max_pmi = 0
                num_common = 0

            cooccur_dict[(show_id, song_id)] = {
                "avg": avg_pmi,
                "max": max_pmi,
                "num": num_common,
            }

    df["avg_cooccurrence"] = df.apply(
        lambda row: cooccur_dict.get((row["show_id"], row["song_id"]), {}).get(
            "avg", 0
        ),
        axis=1,
    )
    df["max_cooccurrence"] = df.apply(
        lambda row: cooccur_dict.get((row["show_id"], row["song_id"]), {}).get(
            "max", 0
        ),
        axis=1,
    )
    df["num_common_cooccur"] = df.apply(
        lambda row: cooccur_dict.get((row["show_id"], row["song_id"]), {}).get(
            "num", 0
        ),
        axis=1,
    )

    return df
