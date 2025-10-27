"""
Generate PMI (Pointwise Mutual Information) matrix visualization.

This complements the raw co-occurrence matrix by showing which song pairs
have stronger-than-expected associations (accounting for individual song frequencies).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(".")

from src.dataio import load_all_data

# Set style
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.facecolor"] = "white"

# Output directory
output_dir = Path("output/figures")
output_dir.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("GENERATING PMI MATRIX VISUALIZATION")
print("=" * 80)

# Load data
print("\nLoading data...")
shows, songs, setlists, song_tags = load_all_data()

# Filter to 2022+ (same as training data)
shows = shows[shows["date"] >= "2022-01-01"].copy()
setlists = setlists[setlists["show_id"].isin(shows["show_id"])].copy()

print(f"  {len(shows)} shows, {len(songs)} songs, {len(setlists)} setlist entries")

# Get top 30 most played songs
song_counts = setlists.groupby("song_id").size().sort_values(ascending=False)
top30_songs = song_counts.head(30).index.tolist()

print(f"\nCalculating PMI for top 30 songs...")

# Build co-occurrence matrix
n_shows = len(shows)
n_songs = 30
cooccurrence = np.zeros((n_songs, n_songs))
individual_counts = {}

for song_id in top30_songs:
    individual_counts[song_id] = song_counts[song_id]

# Count co-occurrences
for show_id in shows["show_id"]:
    show_songs = setlists[setlists["show_id"] == show_id]["song_id"].values
    for i, song1 in enumerate(top30_songs):
        for j, song2 in enumerate(top30_songs):
            if song1 in show_songs and song2 in show_songs and i != j:
                cooccurrence[i, j] += 1

# Calculate PMI matrix
pmi_matrix = np.zeros((n_songs, n_songs))

for i in range(n_songs):
    for j in range(n_songs):
        if i == j:
            continue

        song_i = top30_songs[i]
        song_j = top30_songs[j]

        # P(i, j) = co-occurrence / n_shows
        p_ij = cooccurrence[i, j] / n_shows

        # P(i) = count(i) / n_shows
        p_i = individual_counts[song_i] / n_shows
        p_j = individual_counts[song_j] / n_shows

        # PMI = log(P(i,j) / (P(i) * P(j)))
        if p_ij > 0 and p_i > 0 and p_j > 0:
            pmi = np.log(p_ij / (p_i * p_j))
            pmi_matrix[i, j] = pmi

# Get song names (truncated)
song_names_short = []
for sid in top30_songs:
    name = (
        songs[songs["song_id"] == sid]["title"].values[0]
        if len(songs[songs["song_id"] == sid]) > 0
        else sid[:8]
    )
    if len(name) > 15:
        name = name[:12] + "..."
    song_names_short.append(name)

# Create visualization
print("\nGenerating PMI heatmap...")

fig, ax = plt.subplots(figsize=(14, 12))

# Create custom colormap: negative (blue) -> zero (white) -> positive (red)
# Use diverging colormap centered at 0
sns.heatmap(
    pmi_matrix,
    cmap="RdBu_r",  # Red for positive, Blue for negative, reversed
    center=0,  # Center colormap at 0
    annot=False,
    fmt=".2f",
    xticklabels=song_names_short,
    yticklabels=song_names_short,
    cbar_kws={"label": "PMI (Pointwise Mutual Information)"},
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    vmin=-2,  # Set bounds for better contrast
    vmax=2,
)

# Improve tick styling
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_title(
    "Song PMI Matrix (Top 30 Songs)\nRed = Appear together MORE than expected | Blue = LESS than expected",
    fontsize=14,
    pad=20,
)

plt.tight_layout()
plt.savefig(output_dir / "4b_pmi_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"  Saved: {output_dir / '4b_pmi_matrix.png'}")

# Print some statistics
print("\n" + "=" * 80)
print("PMI STATISTICS")
print("=" * 80)

# Find top positive PMI pairs (strong positive associations)
pmi_pairs = []
for i in range(n_songs):
    for j in range(i + 1, n_songs):
        if pmi_matrix[i, j] > 0:
            pmi_pairs.append(
                (
                    song_names_short[i],
                    song_names_short[j],
                    pmi_matrix[i, j],
                    int(cooccurrence[i, j]),
                )
            )

pmi_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nTop 10 song pairs by PMI (strongest positive associations):")
print("-" * 80)
print(f"{'Song 1':<20} {'Song 2':<20} {'PMI':>8} {'Co-occur':>10}")
print("-" * 80)
for song1, song2, pmi, count in pmi_pairs[:10]:
    print(f"{song1:<20} {song2:<20} {pmi:>8.2f} {count:>10}")

# Find negative PMI pairs (songs that avoid each other)
negative_pmi = [
    (
        song_names_short[i],
        song_names_short[j],
        pmi_matrix[i, j],
        int(cooccurrence[i, j]),
    )
    for i in range(n_songs)
    for j in range(i + 1, n_songs)
    if pmi_matrix[i, j] < -0.5
]
negative_pmi.sort(key=lambda x: x[2])

print("\n\nTop 10 song pairs with negative PMI (rarely appear together):")
print("-" * 80)
print(f"{'Song 1':<20} {'Song 2':<20} {'PMI':>8} {'Co-occur':>10}")
print("-" * 80)
for song1, song2, pmi, count in negative_pmi[:10]:
    print(f"{song1:<20} {song2:<20} {pmi:>8.2f} {count:>10}")

print("\n" + "=" * 80)
print("PMI MATRIX GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\nVisualization saved to: {output_dir / '4b_pmi_matrix.png'}")
print("\nInterpretation:")
print("  - Red cells: Songs appear together MORE than random chance")
print("  - White cells: Songs appear together as expected by chance")
print("  - Blue cells: Songs appear together LESS than random chance")
print("  - PMI > 2.0: Very strong positive association (likely a 'suite')")
print("  - PMI < -1.0: Songs tend to avoid each other")
