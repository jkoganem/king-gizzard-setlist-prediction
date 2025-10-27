"""
Analyze co-occurrence patterns for a given song.

This script helps users understand which songs are likely to appear together
in a setlist by analyzing historical co-occurrence patterns using Pointwise
Mutual Information (PMI), a statistical measure that quantifies whether two
songs appear together more or less frequently than random chance would predict.

WHAT THIS TOOL DOES:
--------------------
If you're curious about questions like:
  - "If Rattlesnake is in the setlist, what other songs will likely appear?"
  - "Which songs does The River frequently appear with?"
  - "Are there songs that actively avoid each other?"

This tool analyzes the entire setlist database to find:
  1. POSITIVE ASSOCIATIONS: Songs that frequently appear together
  2. NEGATIVE ASSOCIATIONS: Songs that rarely/never appear together
  3. CO-OCCURRENCE RATES: % of times songs appear together

PMI SCORE INTERPRETATION:
------------------------
  • PMI > 2.0:  Very strong positive association (almost always played together)
  • PMI > 1.0:  Strong positive association (frequently played together)
  • PMI > 0.5:  Moderate positive association (often played together)
  • PMI ≈ 0.0:  No association (appears together by random chance)
  • PMI < -0.5: Moderate negative association (rarely together)
  • PMI < -1.0: Strong negative association (actively avoid each other)

REQUIRED ARGUMENTS:
------------------
  --song SONG_NAME
      The name of the song to analyze (case insensitive).
      Examples: "Rattlesnake", "The River", "Gamma Knife"

      The script will try to match partial names if exact match fails.
      If multiple songs match, you'll be prompted to be more specific.

OPTIONAL ARGUMENTS:
------------------
  --top-k K
      Number of top associations to display (default: 15).
      Use this to see more or fewer results.

      Examples:
        --top-k 10  (show top 10 associations)
        --top-k 20  (show top 20 associations)

  (Analysis defaults to 2022+ shows for more relevant patterns)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


def load_data():
    """Load curated data."""
    data_dir = Path("data/curated")

    songs = pd.read_parquet(data_dir / "songs.parquet")
    shows = pd.read_parquet(data_dir / "shows.parquet")
    setlists = pd.read_parquet(data_dir / "setlists.parquet")

    return songs, shows, setlists


def calculate_cooccurrence_matrix(setlists, song_to_idx):
    """Calculate raw co-occurrence counts."""
    num_songs = len(song_to_idx)
    cooccur_matrix = np.zeros((num_songs, num_songs), dtype=int)

    # Group by show_id
    show_groups = setlists.groupby("show_id")["song_id"].apply(list)

    for song_list in show_groups:
        # Count co-occurrences within each show
        for i, song_i in enumerate(song_list):
            if song_i not in song_to_idx:
                continue
            idx_i = song_to_idx[song_i]

            for song_j in song_list[i:]:  # Avoid double counting
                if song_j not in song_to_idx:
                    continue
                idx_j = song_to_idx[song_j]

                if idx_i == idx_j:
                    cooccur_matrix[idx_i, idx_j] += 1
                else:
                    cooccur_matrix[idx_i, idx_j] += 1
                    cooccur_matrix[idx_j, idx_i] += 1

    return cooccur_matrix


def calculate_pmi(cooccur_matrix, song_frequencies, total_shows):
    """Calculate Pointwise Mutual Information (PMI) matrix."""
    num_songs = cooccur_matrix.shape[0]
    pmi_matrix = np.zeros((num_songs, num_songs))

    # Calculate probabilities
    p_songs = song_frequencies / total_shows  # P(song_i)

    for i in range(num_songs):
        for j in range(num_songs):
            if cooccur_matrix[i, j] == 0:
                pmi_matrix[i, j] = -np.inf
                continue

            # P(song_i, song_j) = co-occurrence count / total shows
            p_joint = cooccur_matrix[i, j] / total_shows

            # P(song_i) * P(song_j)
            p_independent = p_songs[i] * p_songs[j]

            if p_independent > 0:
                # PMI = log(P(i,j) / (P(i) * P(j)))
                pmi_matrix[i, j] = np.log(p_joint / p_independent)
            else:
                pmi_matrix[i, j] = -np.inf

    return pmi_matrix


def analyze_song(title, songs, shows, setlists, top_k=15, recent_only=True):
    """Analyze co-occurrence patterns for a given song."""

    # Check if song exists
    matching_songs = songs[songs["title"].str.lower() == title.lower()]

    if len(matching_songs) == 0:
        # Try partial match
        matching_songs = songs[songs["title"].str.lower().str.contains(title.lower())]

        if len(matching_songs) == 0:
            print(f"\nERROR: Song '{title}' not found in database.")
            print("\nDid you mean one of these?")
            # Show similar songs (by first letter)
            first_letter = title[0].lower()
            similar = songs[songs["title"].str.lower().str.startswith(first_letter)]
            for _, row in similar.head(10).iterrows():
                print(f"  - {row['title']}")
            return

        if len(matching_songs) > 1:
            print(f"\nWARNING: Multiple songs match '{title}':")
            for _, row in matching_songs.iterrows():
                print(f"  - {row['title']}")
            print("\nPlease be more specific.")
            return

    target_song = matching_songs.iloc[0]
    target_song_id = target_song["song_id"]
    target_title = target_song["title"]

    print()
    print("=" * 80)
    print(f"CO-OCCURRENCE ANALYSIS: {target_title}")
    print("=" * 80)
    print()

    # Filter by recent shows if requested
    if recent_only:
        cutoff_date = "2022-01-01"
        shows = shows[shows["date"] >= cutoff_date]
        # Also filter setlists to only include recent shows
        setlists = setlists[setlists["show_id"].isin(shows["show_id"])]
        print(f"Analyzing recent shows only (2022+): {len(shows)} shows")
    else:
        print(f"Analyzing all shows: {len(shows)} shows")

    # Get shows where target song was played
    target_setlists = setlists[setlists["song_id"] == target_song_id]
    target_shows = shows[shows["show_id"].isin(target_setlists["show_id"])]

    num_appearances = len(target_shows)
    play_rate = num_appearances / len(shows) * 100

    print(
        f"Song appearances: {num_appearances} / {len(shows)} shows ({play_rate:.1f}%)"
    )
    print()

    if num_appearances == 0:
        print(
            f"WARNING: '{target_title}' has never been played in the analyzed period."
        )
        return

    # Calculate co-occurrence statistics
    song_to_idx = {song_id: idx for idx, song_id in enumerate(songs["song_id"])}
    idx_to_song = {idx: song_id for song_id, idx in song_to_idx.items()}

    # Get all songs that appeared with target song
    cooccurring_songs = defaultdict(int)

    for show_id in target_shows["show_id"]:
        show_songs = setlists[setlists["show_id"] == show_id]["song_id"].tolist()
        for other_song_id in show_songs:
            if other_song_id != target_song_id:
                cooccurring_songs[other_song_id] += 1

    # Calculate full co-occurrence matrix and PMI
    print("Calculating co-occurrence matrix and PMI scores...")
    cooccur_matrix = calculate_cooccurrence_matrix(setlists, song_to_idx)

    # Calculate song frequencies
    song_frequencies = (
        setlists.groupby("song_id")
        .size()
        .reindex(songs["song_id"], fill_value=0)
        .values
    )
    total_shows = len(shows)

    pmi_matrix = calculate_pmi(cooccur_matrix, song_frequencies, total_shows)

    target_idx = song_to_idx[target_song_id]

    # Get top co-occurring songs
    results = []
    for other_song_id, count in cooccurring_songs.items():
        other_idx = song_to_idx[other_song_id]
        other_title = songs[songs["song_id"] == other_song_id].iloc[0]["title"]

        cooccur_rate = count / num_appearances * 100
        pmi_score = pmi_matrix[target_idx, other_idx]
        raw_cooccur = int(cooccur_matrix[target_idx, other_idx])

        # Get frequency of other song
        other_appearances = song_frequencies[other_idx]
        other_play_rate = other_appearances / total_shows * 100

        results.append(
            {
                "title": other_title,
                "cooccur_count": count,
                "cooccur_rate": cooccur_rate,
                "pmi": pmi_score,
                "raw_cooccur": raw_cooccur,
                "other_play_rate": other_play_rate,
            }
        )

    # Sort by PMI (highest positive association first)
    results_df = pd.DataFrame(results).sort_values("pmi", ascending=False)

    print()
    print("=" * 80)
    print(f"TOP {top_k} SONGS BY CO-OCCURRENCE RATE")
    print("=" * 80)
    print()
    print("(Songs that most frequently appear together with the target song)")
    print()

    top_cooccur = results_df.sort_values("cooccur_rate", ascending=False).head(top_k)

    for i, row in enumerate(top_cooccur.itertuples(), 1):
        pmi_desc = (
            "[VERY STRONG]"
            if row.pmi > 2.0
            else (
                "[STRONG]"
                if row.pmi > 1.0
                else "[MODERATE]" if row.pmi > 0.5 else "[WEAK]"
            )
        )

        print(f"{i:2d}. {row.title}")
        print(
            f"    Co-occurrence: {row.cooccur_count}/{num_appearances} shows ({row.cooccur_rate:.1f}%)"
        )
        print(f"    Raw co-occurrence count: {row.raw_cooccur}")
        print(f"    PMI: {row.pmi:.2f} {pmi_desc}")
        print(f"    Baseline play rate: {row.other_play_rate:.1f}%")
        print()

    print()
    print("=" * 80)
    print(f"TOP {top_k} SONGS BY POSITIVE PMI")
    print("=" * 80)
    print()
    print("PMI Interpretation:")
    print(
        "  • PMI > 2.0: Very strong positive association (almost always played together)"
    )
    print("  • PMI > 1.0: Strong positive association (frequently played together)")
    print("  • PMI > 0.5: Moderate positive association (often played together)")
    print("  • PMI ≈ 0.0: No association (random chance)")
    print()

    top_positive = results_df.head(top_k)

    for i, row in enumerate(top_positive.itertuples(), 1):
        pmi_desc = (
            "[VERY STRONG]"
            if row.pmi > 2.0
            else (
                "[STRONG]"
                if row.pmi > 1.0
                else "[MODERATE]" if row.pmi > 0.5 else "[WEAK]"
            )
        )

        print(f"{i:2d}. {row.title}")
        print(
            f"    Co-occurrence: {row.cooccur_count}/{num_appearances} shows ({row.cooccur_rate:.1f}%)"
        )
        print(f"    Raw co-occurrence count: {row.raw_cooccur}")
        print(f"    PMI: {row.pmi:.2f} {pmi_desc}")
        print(f"    Baseline play rate: {row.other_play_rate:.1f}%")
        print()

    print()
    print("=" * 80)
    print(f"BOTTOM {top_k} SONGS BY CO-OCCURRENCE RATE")
    print("=" * 80)
    print()
    print("(Songs that rarely appear together with the target song)")
    print()

    bottom_cooccur = results_df.sort_values("cooccur_rate", ascending=True).head(top_k)

    for i, row in enumerate(bottom_cooccur.itertuples(), 1):
        print(f"{i:2d}. {row.title}")
        print(
            f"    Co-occurrence: {row.cooccur_count}/{num_appearances} shows ({row.cooccur_rate:.1f}%)"
        )
        print(f"    Raw co-occurrence count: {row.raw_cooccur}")
        print(f"    PMI: {row.pmi:.2f}")
        print(f"    Baseline play rate: {row.other_play_rate:.1f}%")
        print()

    print()
    print("=" * 80)
    print(f"TOP {top_k} SONGS BY NEGATIVE PMI (AVOIDANCE)")
    print("=" * 80)
    print()

    # Get songs with negative PMI, sort by PMI ascending (most negative first), then take top k
    negative_songs = (
        results_df[results_df["pmi"] < -0.5]
        .sort_values("pmi", ascending=True)
        .head(top_k)
    )

    if len(negative_songs) == 0:
        print("No significant negative associations found.")
    else:
        for i, row in enumerate(negative_songs.itertuples(), 1):
            print(f"{i:2d}. {row.title}")
            print(
                f"    Co-occurrence: {row.cooccur_count}/{num_appearances} shows ({row.cooccur_rate:.1f}%)"
            )
            print(f"    Raw co-occurrence count: {row.raw_cooccur}")
            print(f"    PMI: {row.pmi:.2f} (actively avoid each other)")
            print(f"    Baseline play rate: {row.other_play_rate:.1f}%")
            print()

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print(f"If '{target_title}' is in the setlist, you can expect:")
    print()

    # Get top 3 strongest associations
    top_3 = results_df.head(3)
    for i, row in enumerate(top_3.itertuples(), 1):
        likelihood = (
            "very likely"
            if row.cooccur_rate > 70
            else "likely" if row.cooccur_rate > 50 else "possibly"
        )
        print(
            f"  {i}. '{row.title}' is {likelihood} to appear ({row.cooccur_rate:.1f}% co-occurrence)"
        )

    print()
    print(f"Songs that rarely appear with '{target_title}':")
    print()

    # Get bottom 3 (most avoided)
    bottom_3 = results_df[results_df["pmi"] < 0].tail(3)
    if len(bottom_3) > 0:
        for i, row in enumerate(bottom_3.itertuples(), 1):
            print(
                f"  {i}. '{row.title}' (only {row.cooccur_rate:.1f}% co-occurrence, PMI={row.pmi:.2f})"
            )
    else:
        print("  No significant avoidance patterns detected.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze song co-occurrence patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze co-occurrence for "Rattlesnake"
  python analyze_song_cooccurrence.py --song "Rattlesnake"

  # Show top 20 associations
  python analyze_song_cooccurrence.py --song "The River" --top-k 20

  # Only analyze recent shows (2022+)
  python analyze_song_cooccurrence.py --song "Gamma Knife" --recent-only
""",
    )

    parser.add_argument(
        "--song",
        type=str,
        required=True,
        help="Song name to analyze (case insensitive)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top associations to show for both positive and negative (default: 3)",
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    songs, shows, setlists = load_data()

    # Analyze song (defaults to recent shows only - 2022+)
    analyze_song(args.song, songs, shows, setlists, top_k=args.top_k)


if __name__ == "__main__":
    main()
