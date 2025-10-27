"""
Generate co-occurrence analysis and visualizations for README EDA section.

This script:
1. Analyzes song co-occurrence patterns in historical setlists
2. Identifies song "suites" and clusters
3. Generates co-occurrence matrix visualization
4. Computes PMI (Pointwise Mutual Information) to find strong associations
5. Creates network graph of top song relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import networkx as nx
from scipy.stats import chi2_contingency
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataio import load_gizzverse_data

# Create output directories
output_dir = Path("output/figures/eda")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CO-OCCURRENCE PATTERN ANALYSIS")
print("=" * 80)

# Load data
print("\nLoading data...")
df_shows, df_songs = load_gizzverse_data(
    dataset="recent", filter_specialized=True  # 2022+
)

print(f"  {len(df_shows)} shows")
print(f"  {len(df_songs)} unique songs")
print(f"  {len(df_shows) * df_shows.iloc[0]['setlist'].count(',')} total performances")

# Build co-occurrence matrix
print("\nBuilding co-occurrence matrix...")
song_names = sorted(df_songs.index.tolist())
n_songs = len(song_names)
song_to_idx = {song: i for i, song in enumerate(song_names)}

# Co-occurrence counts
cooccur_matrix = np.zeros((n_songs, n_songs))
song_counts = Counter()

for _, show in df_shows.iterrows():
    setlist = show["setlist"].split(", ")
    song_counts.update(setlist)

    # Count co-occurrences
    for i, song1 in enumerate(setlist):
        if song1 not in song_to_idx:
            continue
        idx1 = song_to_idx[song1]

        for song2 in setlist[i:]:  # Include self-loops
            if song2 not in song_to_idx:
                continue
            idx2 = song_to_idx[song2]

            cooccur_matrix[idx1, idx2] += 1
            if idx1 != idx2:
                cooccur_matrix[idx2, idx1] += 1

# Compute PMI (Pointwise Mutual Information)
print("\nComputing Pointwise Mutual Information (PMI)...")
n_shows = len(df_shows)
pmi_matrix = np.zeros((n_songs, n_songs))

for i in range(n_songs):
    for j in range(i, n_songs):
        if i == j:
            continue

        song_i = song_names[i]
        song_j = song_names[j]

        # P(i, j) = co-occurrence / n_shows
        p_ij = cooccur_matrix[i, j] / n_shows

        # P(i) = count(i) / n_shows, P(j) = count(j) / n_shows
        p_i = song_counts[song_i] / n_shows
        p_j = song_counts[song_j] / n_shows

        # PMI = log(P(i,j) / (P(i) * P(j)))
        if p_ij > 0 and p_i > 0 and p_j > 0:
            pmi = np.log(p_ij / (p_i * p_j))
            pmi_matrix[i, j] = pmi
            pmi_matrix[j, i] = pmi

# Find top song pairs by PMI
print("\nTop 20 song pairs by PMI (Pointwise Mutual Information):")
print("-" * 80)
print(f"{'Song 1':<35} {'Song 2':<35} {'PMI':>6} {'Co-occur':>8}")
print("-" * 80)

pmi_pairs = []
for i in range(n_songs):
    for j in range(i + 1, n_songs):
        if pmi_matrix[i, j] > 0:
            pmi_pairs.append(
                (
                    song_names[i],
                    song_names[j],
                    pmi_matrix[i, j],
                    int(cooccur_matrix[i, j]),
                )
            )

pmi_pairs.sort(key=lambda x: x[2], reverse=True)
for song1, song2, pmi, count in pmi_pairs[:20]:
    print(f"{song1:<35} {song2:<35} {pmi:>6.2f} {count:>8}")

# Identify suites (songs that always/almost always appear together)
print("\n\nIdentifying song suites (co-occurrence rate > 80%):")
print("-" * 80)
suites = []
for song1, song2, pmi, count in pmi_pairs:
    # Check if they appear together most of the time
    count1 = song_counts[song1]
    count2 = song_counts[song2]

    # Co-occurrence rate: how often they appear together when either is played
    co_rate = count / max(count1, count2)

    if co_rate > 0.8 and count >= 5:  # At least 5 co-occurrences
        suites.append((song1, song2, co_rate, count, count1, count2))

suites.sort(key=lambda x: x[2], reverse=True)
print(
    f"{'Song 1':<35} {'Song 2':<35} {'Co-rate':>8} {'Co-occur':>8} {'Count1':>7} {'Count2':>7}"
)
print("-" * 80)
for song1, song2, rate, co, c1, c2 in suites[:30]:
    print(f"{song1:<35} {song2:<35} {rate*100:>7.1f}% {co:>8} {c1:>7} {c2:>7}")

# === VISUALIZATION 1: Co-occurrence heatmap for top songs ===
print("\n\nGenerating co-occurrence heatmap...")

# Select top 30 most frequent songs
top_songs = [song for song, _ in song_counts.most_common(30)]
top_indices = [song_to_idx[song] for song in top_songs]

# Extract submatrix
top_cooccur = cooccur_matrix[np.ix_(top_indices, top_indices)]

# Normalize by geometric mean of marginal counts
normalized_cooccur = np.zeros_like(top_cooccur)
for i, idx_i in enumerate(top_indices):
    for j, idx_j in enumerate(top_indices):
        if i != j:
            song_i = song_names[idx_i]
            song_j = song_names[idx_j]
            geom_mean = np.sqrt(song_counts[song_i] * song_counts[song_j])
            if geom_mean > 0:
                normalized_cooccur[i, j] = top_cooccur[i, j] / geom_mean

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    normalized_cooccur,
    xticklabels=top_songs,
    yticklabels=top_songs,
    cmap="YlOrRd",
    cbar_kws={"label": "Normalized Co-occurrence"},
    square=True,
    linewidths=0.5,
    ax=ax,
)
plt.xticks(rotation=90, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.title(
    "Song Co-occurrence Patterns (Top 30 Songs)\nNormalized by √(count_i × count_j)",
    fontsize=14,
    pad=20,
)
plt.tight_layout()
plt.savefig(output_dir / "cooccurrence_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {output_dir / 'cooccurrence_heatmap.png'}")

# === VISUALIZATION 2: Network graph of strong associations ===
print("\nGenerating network graph of song associations...")

# Create graph with top PMI pairs
G = nx.Graph()

# Add nodes (top 40 songs)
top_40_songs = [song for song, _ in song_counts.most_common(40)]
for song in top_40_songs:
    G.add_node(song, count=song_counts[song])

# Add edges for strong associations (PMI > 2.0)
for song1, song2, pmi, count in pmi_pairs:
    if pmi > 2.0 and song1 in top_40_songs and song2 in top_40_songs:
        G.add_edge(song1, song2, weight=pmi, count=count)

print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Layout
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Draw
fig, ax = plt.subplots(figsize=(20, 16))

# Node sizes proportional to frequency
node_sizes = [song_counts[node] * 15 for node in G.nodes()]

# Edge widths proportional to PMI
edges = G.edges()
edge_weights = [G[u][v]["weight"] for u, v in edges]
edge_widths = [w * 0.8 for w in edge_weights]

# Draw network
nx.draw_networkx_nodes(
    G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7, ax=ax
)

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color="gray", ax=ax)

nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

plt.title(
    "Song Co-occurrence Network (PMI > 2.0)\nNode size = frequency, Edge width = PMI strength",
    fontsize=16,
    pad=20,
)
plt.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "cooccurrence_network.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {output_dir / 'cooccurrence_network.png'}")

# === VISUALIZATION 3: Example setlists showing suites ===
print("\nAnalyzing example setlists with suites...")

# Find setlists that contain multiple suite pairs
suite_pairs = [(s1, s2) for s1, s2, rate, co, c1, c2 in suites[:10]]

example_shows = []
for _, show in df_shows.iterrows():
    setlist = show["setlist"].split(", ")
    suite_count = sum(1 for s1, s2 in suite_pairs if s1 in setlist and s2 in setlist)
    if suite_count >= 2:
        example_shows.append((show["date"], show["venue"], setlist, suite_count))

example_shows.sort(key=lambda x: x[3], reverse=True)

print("\nExample setlists featuring multiple song suites:")
print("-" * 80)
for date, venue, setlist, count in example_shows[:5]:
    print(f"\n{date} - {venue}")
    print(f"  Suite pairs present: {count}")
    print(f"  Setlist: {', '.join(setlist[:10])}...")

# Save summary statistics
summary = {
    "total_songs": n_songs,
    "total_shows": n_shows,
    "avg_cooccur": float(np.mean(cooccur_matrix[cooccur_matrix > 0])),
    "max_pmi": float(pmi_pairs[0][2]) if pmi_pairs else 0,
    "n_strong_pairs": len([1 for _, _, pmi, _ in pmi_pairs if pmi > 2.0]),
    "n_suites": len(suites),
    "top_suites": [
        {"song1": s1, "song2": s2, "rate": float(rate), "count": int(co)}
        for s1, s2, rate, co, _, _ in suites[:10]
    ],
}

import json

with open(output_dir / "cooccurrence_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("CO-OCCURRENCE ANALYSIS COMPLETED")
print("=" * 80)
print(f"\nGenerated figures:")
print(f"  - {output_dir / 'cooccurrence_heatmap.png'}")
print(f"  - {output_dir / 'cooccurrence_network.png'}")
print(f"  - {output_dir / 'cooccurrence_summary.json'}")
