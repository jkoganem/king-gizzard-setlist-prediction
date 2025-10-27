"""
Generate comprehensive EDA visualizations for README.md

This script creates figures that demonstrate:
1. Problem difficulty (low setlist overlap)
2. Dataset structure (temporal patterns, frequency distributions)
3. Feature importance from XGBoost
4. Training curves from GNN models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.append(".")

from src.dataio import load_all_data

# Set modern, professional style
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 11

# Output directory
output_dir = Path("output/figures")
output_dir.mkdir(exist_ok=True, parents=True)

print("Loading data...")
shows, songs, setlists, song_tags = load_all_data()

# Filter to 2022+ (same as training data)
shows = shows[shows["date"] >= "2022-01-01"].copy()
setlists = setlists[setlists["show_id"].isin(shows["show_id"])].copy()

print(f"Loaded {len(shows)} shows, {len(songs)} songs, {len(setlists)} setlist entries")

# ============================================================================
# Figure 1: Setlist Overlap Analysis (Why This is Hard)
# ============================================================================

print("\nGenerating Figure 1: Setlist Overlap Analysis...")

# Calculate pairwise Jaccard similarity between consecutive shows
shows_sorted = shows.sort_values("date").reset_index(drop=True)
overlaps = []
setlist_sizes = []

for i in range(len(shows_sorted) - 1):
    show1_id = shows_sorted.iloc[i]["show_id"]
    show2_id = shows_sorted.iloc[i + 1]["show_id"]

    songs1 = set(setlists[setlists["show_id"] == show1_id]["song_id"])
    songs2 = set(setlists[setlists["show_id"] == show2_id]["song_id"])

    if len(songs1) > 0 and len(songs2) > 0:
        overlap = len(songs1 & songs2) / len(songs1 | songs2)
        overlaps.append(overlap)
        setlist_sizes.append((len(songs1), len(songs2)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Overlap distribution with seaborn
sns.histplot(
    overlaps,
    bins=30,
    kde=True,
    ax=axes[0],
    color="#2E86AB",
    edgecolor="black",
    alpha=0.7,
)
axes[0].axvline(
    np.mean(overlaps),
    color="#D62828",
    linestyle="--",
    linewidth=2.5,
    label=f"Mean: {np.mean(overlaps):.2%}",
)
axes[0].axvline(
    np.median(overlaps),
    color="#F77F00",
    linestyle=":",
    linewidth=2.5,
    label=f"Median: {np.median(overlaps):.2%}",
)
axes[0].set_xlabel("Jaccard Similarity (Consecutive Shows)")
axes[0].set_ylabel("Count")
axes[0].set_title("Setlist Overlap Between Consecutive Shows")
axes[0].legend(frameon=True, fancybox=True, shadow=True)

# Subplot 2: Setlist size distribution with seaborn
all_sizes = [size for pair in setlist_sizes for size in pair]
sns.histplot(
    all_sizes,
    bins=25,
    kde=True,
    ax=axes[1],
    color="#06A77D",
    edgecolor="black",
    alpha=0.7,
)
axes[1].axvline(
    np.mean(all_sizes),
    color="#D62828",
    linestyle="--",
    linewidth=2.5,
    label=f"Mean: {np.mean(all_sizes):.1f} songs",
)
axes[1].set_xlabel("Number of Songs per Setlist")
axes[1].set_ylabel("Count")
axes[1].set_title("Setlist Length Distribution")
axes[1].legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(output_dir / "1_setlist_overlap.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"  Mean overlap: {np.mean(overlaps):.2%}")
print(f"  Median overlap: {np.median(overlaps):.2%}")
print(f"  Mean setlist size: {np.mean(all_sizes):.1f} songs")

# ============================================================================
# Figure 2: Song Frequency Distribution (Dataset Structure)
# ============================================================================

print("\nGenerating Figure 2: Song Frequency Distribution...")

song_counts = setlists.groupby("song_id").size().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Top 20 most played songs with gradient colors
top20 = song_counts.head(20)
song_names = [
    (
        songs[songs["song_id"] == sid]["title"].values[0]
        if len(songs[songs["song_id"] == sid]) > 0
        else sid[:8]
    )
    for sid in top20.index
]

# Create gradient color palette
colors = sns.color_palette("viridis", len(top20))
bars = axes[0].barh(
    range(len(top20)), top20.values, color=colors, edgecolor="black", linewidth=0.5
)
axes[0].set_yticks(range(len(top20)))
axes[0].set_yticklabels(song_names, fontsize=9)
axes[0].set_xlabel("Number of Performances")
axes[0].set_title("Top 20 Most Played Songs")
axes[0].invert_yaxis()

# Subplot 2: Long tail distribution with better styling
sns.histplot(
    song_counts.values,
    bins=50,
    ax=axes[1],
    color="#E63946",
    edgecolor="black",
    alpha=0.7,
    kde=False,
)
axes[1].set_xlabel("Number of Performances")
axes[1].set_ylabel("Number of Songs (log scale)")
axes[1].set_title("Song Frequency Distribution (Long Tail)")
axes[1].set_yscale("log")
axes[1].text(
    0.95,
    0.95,
    f"Total: {len(song_counts)} songs",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig(output_dir / "2_song_frequency.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"  Most played song: {song_counts.index[0]} ({song_counts.values[0]} times)")
print(f"  Songs played once: {(song_counts == 1).sum()} / {len(song_counts)}")

# ============================================================================
# Figure 3: Temporal Patterns
# ============================================================================

print("\nGenerating Figure 3: Temporal Patterns...")

shows_sorted["month"] = pd.to_datetime(shows_sorted["date"]).dt.to_period("M")
monthly_shows = shows_sorted.groupby("month").size()

# Get average setlist size per month
monthly_avg_size = []
for month in monthly_shows.index:
    month_shows = shows_sorted[shows_sorted["month"] == month]
    month_setlists = setlists[setlists["show_id"].isin(month_shows["show_id"])]
    avg_size = month_setlists.groupby("show_id").size().mean()
    monthly_avg_size.append(avg_size)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Subplot 1: Number of shows per month with better styling
axes[0].plot(
    range(len(monthly_shows)),
    monthly_shows.values,
    marker="o",
    linewidth=2.5,
    markersize=8,
    color="#1D3557",
    markerfacecolor="#457B9D",
    markeredgewidth=2,
    markeredgecolor="#1D3557",
)
axes[0].fill_between(
    range(len(monthly_shows)), monthly_shows.values, alpha=0.25, color="#457B9D"
)
axes[0].set_xlabel("Time (Monthly)")
axes[0].set_ylabel("Number of Shows")
axes[0].set_title("Concert Activity Over Time")
axes[0].set_xticks(range(0, len(monthly_shows), max(1, len(monthly_shows) // 10)))

# Subplot 2: Average setlist size over time with better styling
axes[1].plot(
    range(len(monthly_avg_size)),
    monthly_avg_size,
    marker="s",
    linewidth=2.5,
    markersize=8,
    color="#6A040F",
    markerfacecolor="#DC2F02",
    markeredgewidth=2,
    markeredgecolor="#6A040F",
)
axes[1].fill_between(
    range(len(monthly_avg_size)), monthly_avg_size, alpha=0.25, color="#DC2F02"
)
axes[1].set_xlabel("Time (Monthly)")
axes[1].set_ylabel("Average Setlist Size")
axes[1].set_title("Average Setlist Length Over Time")
axes[1].set_xticks(range(0, len(monthly_avg_size), max(1, len(monthly_avg_size) // 10)))
axes[1].axhline(
    np.mean(monthly_avg_size),
    color="#6A040F",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Overall Mean: {np.mean(monthly_avg_size):.1f}",
)
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / "3_temporal_patterns.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"  Date range: {shows_sorted['date'].min()} to {shows_sorted['date'].max()}")
print(f"  Peak month: {monthly_shows.idxmax()} ({monthly_shows.max()} shows)")

# ============================================================================
# Figure 4: Co-occurrence Matrix (Top 30 Songs)
# ============================================================================

print("\nGenerating Figure 4: Song Co-occurrence Matrix...")

# Get top 30 most played songs
top30_songs = song_counts.head(30).index.tolist()

# Build co-occurrence matrix
cooccurrence = np.zeros((30, 30))
for show_id in shows["show_id"]:
    show_songs = setlists[setlists["show_id"] == show_id]["song_id"].values
    for i, song1 in enumerate(top30_songs):
        for j, song2 in enumerate(top30_songs):
            if song1 in show_songs and song2 in show_songs and i != j:
                cooccurrence[i, j] += 1

# Get song names
song_names_short = []
for sid in top30_songs:
    name = (
        songs[songs["song_id"] == sid]["title"].values[0]
        if len(songs[songs["song_id"] == sid]) > 0
        else sid[:8]
    )
    # Truncate long names
    if len(name) > 15:
        name = name[:12] + "..."
    song_names_short.append(name)

fig, ax = plt.subplots(figsize=(14, 12))

# Use seaborn heatmap for better styling
sns.heatmap(
    cooccurrence,
    cmap="rocket_r",
    annot=False,
    fmt="g",
    xticklabels=song_names_short,
    yticklabels=song_names_short,
    cbar_kws={"label": "Co-occurrence Count"},
    linewidths=0.5,
    linecolor="white",
    ax=ax,
)

# Improve tick styling
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_title("Song Co-occurrence Matrix (Top 30 Songs)")

plt.tight_layout()
plt.savefig(output_dir / "4_cooccurrence_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"  Matrix size: 30x30")
print(f"  Max co-occurrence: {cooccurrence.max():.0f}")

print("\nAll EDA figures generated successfully!")
print(f"Saved to: {output_dir}/")
