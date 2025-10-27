#!/usr/bin/env python3
"""
Genre Clustering Analysis - EDA

Analyze whether songs naturally cluster into genre groups based on co-occurrence patterns.
This validates that genres exist as meaningful structures in the data.

Methods:
1. Build co-occurrence matrix from setlists
2. Apply dimensionality reduction (t-SNE, UMAP)
3. Cluster songs using k-means, hierarchical clustering
4. Visualize clusters and compare to known genres
5. Measure cluster quality and genre alignment

Usage:
    python scripts/analyze_genre_clusters.py --visualize
    python scripts/analyze_genre_clusters.py --n-clusters 7 --method umap
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from src.dataio import load_all_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")


def build_cooccurrence_matrix(
    setlists: pd.DataFrame, songs: pd.DataFrame
) -> Tuple[np.ndarray, List[int]]:
    """
    Build song co-occurrence matrix from setlists.

    Args:
        setlists: Setlists DataFrame
        songs: Songs DataFrame

    Returns:
        Tuple of (co-occurrence matrix, list of song_ids)
    """
    logger.info("Building co-occurrence matrix...")

    # Get song IDs
    song_ids = sorted(songs["song_id"].unique())
    song_id_to_idx = {sid: idx for idx, sid in enumerate(song_ids)}

    n_songs = len(song_ids)
    cooccur = np.zeros((n_songs, n_songs))

    # Count co-occurrences
    for show_id, group in setlists.groupby("show_id"):
        show_songs = group["song_id"].values

        for i, song1 in enumerate(show_songs):
            for song2 in show_songs[i + 1 :]:
                idx1 = song_id_to_idx[song1]
                idx2 = song_id_to_idx[song2]
                cooccur[idx1, idx2] += 1
                cooccur[idx2, idx1] += 1  # Symmetric

    logger.info(f"Built {n_songs}x{n_songs} co-occurrence matrix")

    return cooccur, song_ids


def reduce_dimensions(
    cooccur: np.ndarray, method: str = "tsne", n_components: int = 2
) -> np.ndarray:
    """
    Reduce dimensionality of co-occurrence matrix for visualization.

    Args:
        cooccur: Co-occurrence matrix
        method: 'tsne', 'umap', or 'pca'
        n_components: Number of components (usually 2 for visualization)

    Returns:
        Reduced representation
    """
    logger.info(f"Reducing dimensions with {method}...")

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(cooccur)

    elif method == "tsne":
        # t-SNE on co-occurrence as similarity
        reducer = TSNE(
            n_components=n_components,
            metric="precomputed",
            init="random",
            random_state=42,
        )
        # Convert co-occurrence to distance
        max_cooccur = cooccur.max()
        distance = max_cooccur - cooccur
        np.fill_diagonal(distance, 0)
        embedding = reducer.fit_transform(distance)

    elif method == "umap":
        if not HAS_UMAP:
            logger.error("UMAP not installed. Install with: pip install umap-learn")
            return reduce_dimensions(cooccur, method="tsne", n_components=n_components)

        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(cooccur)

    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Embedding shape: {embedding.shape}")
    return embedding


def cluster_songs(
    cooccur: np.ndarray, n_clusters: int = 7, method: str = "kmeans"
) -> np.ndarray:
    """
    Cluster songs based on co-occurrence patterns.

    Args:
        cooccur: Co-occurrence matrix
        n_clusters: Number of clusters
        method: 'kmeans' or 'hierarchical'

    Returns:
        Cluster labels for each song
    """
    logger.info(f"Clustering with {method}, k={n_clusters}...")

    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(cooccur)

    elif method == "hierarchical":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = clusterer.fit_predict(cooccur)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute silhouette score
    silhouette = silhouette_score(cooccur, labels)
    logger.info(f"Silhouette score: {silhouette:.3f}")

    return labels


def visualize_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    songs: pd.DataFrame,
    song_ids: List[int],
    output_dir: Path,
    title: str = "Song Clusters",
) -> None:
    """Visualize song clusters in 2D."""
    logger.info("Creating visualization...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create color map
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))

    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[i]],
            label=f"Cluster {label}",
            alpha=0.6,
            s=50,
        )

    # Annotate some key songs
    song_id_to_idx = {sid: idx for idx, sid in enumerate(song_ids)}

    key_songs = [
        "Rattlesnake",
        "Nuclear Fusion",
        "Crumbling Castle",
        "The River",
        "Tezeta",
        "Boogieman Sam",
        "Planet B",
    ]

    for song_title in key_songs:
        song_match = songs[
            songs["title"].str.contains(song_title, case=False, na=False)
        ]
        if len(song_match) > 0:
            song_id = song_match.iloc[0]["song_id"]
            if song_id in song_id_to_idx:
                idx = song_id_to_idx[song_id]
                ax.annotate(
                    song_title,
                    (embedding[idx, 0], embedding[idx, 1]),
                    fontsize=8,
                    alpha=0.7,
                )

    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_path = output_dir / "genre_clusters.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"Saved: {output_path}")


def compare_to_genres(
    labels: np.ndarray, songs: pd.DataFrame, song_ids: List[int], output_dir: Path
) -> None:
    """Compare clusters to known genres (if available)."""
    logger.info("Comparing clusters to genres...")

    # Load genre mapping if available
    genre_file = PROJECT_ROOT / "data" / "genre_mapping.json"

    if not genre_file.exists():
        logger.warning("No genre mapping found. Run add_genre_tags.py first.")
        return

    with open(genre_file) as f:
        genre_data = json.load(f)

    # Create mapping
    title_to_genre = {
        title: data["primary_genre"] for title, data in genre_data.items()
    }

    # Assign genres to songs
    song_genres = []
    for song_id in song_ids:
        song = songs[songs["song_id"] == song_id].iloc[0]
        genre = title_to_genre.get(song["title"], "Unknown")
        song_genres.append(genre)

    song_genres = np.array(song_genres)

    # Compute adjusted Rand index
    ari = adjusted_rand_score(
        song_genres[song_genres != "Unknown"], labels[song_genres != "Unknown"]
    )

    logger.info(f"Adjusted Rand Index (cluster vs genre): {ari:.3f}")

    # Create confusion matrix
    unique_clusters = np.unique(labels)
    unique_genres = [g for g in np.unique(song_genres) if g != "Unknown"]

    confusion = np.zeros((len(unique_clusters), len(unique_genres)))

    for i, cluster in enumerate(unique_clusters):
        for j, genre in enumerate(unique_genres):
            count = ((labels == cluster) & (song_genres == genre)).sum()
            confusion[i, j] = count

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        confusion,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        xticklabels=unique_genres,
        yticklabels=[f"Cluster {i}" for i in unique_clusters],
        ax=ax,
    )

    ax.set_title(f"Cluster vs Genre Alignment (ARI={ari:.3f})", fontsize=12)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Cluster")

    plt.tight_layout()

    output_path = output_dir / "cluster_genre_confusion.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_dendrogram(
    cooccur: np.ndarray, songs: pd.DataFrame, song_ids: List[int], output_dir: Path
) -> None:
    """Plot hierarchical clustering dendrogram."""
    logger.info("Creating dendrogram...")

    # Use a subset for readability
    n_songs_plot = min(50, len(song_ids))
    subset_idx = np.random.choice(len(song_ids), n_songs_plot, replace=False)

    cooccur_subset = cooccur[subset_idx][:, subset_idx]
    song_ids_subset = [song_ids[i] for i in subset_idx]

    # Get song titles
    song_titles = []
    for song_id in song_ids_subset:
        title = songs[songs["song_id"] == song_id].iloc[0]["title"]
        song_titles.append(title[:30])  # Truncate long titles

    # Compute linkage
    linkage_matrix = linkage(cooccur_subset, method="ward")

    # Plot
    fig, ax = plt.subplots(figsize=(15, 10))

    dendrogram(
        linkage_matrix, labels=song_titles, leaf_rotation=90, leaf_font_size=8, ax=ax
    )

    ax.set_title("Hierarchical Clustering of Songs (by Co-occurrence)", fontsize=14)
    ax.set_xlabel("Song")
    ax.set_ylabel("Distance")

    plt.tight_layout()

    output_path = output_dir / "genre_dendrogram.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"Saved: {output_path}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze song clustering by co-occurrence patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=7, help="Number of clusters (default: 7)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tsne", "umap", "pca"],
        default="tsne",
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        choices=["kmeans", "hierarchical"],
        default="kmeans",
        help="Clustering method",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/eda"), help="Output directory"
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GENRE CLUSTERING ANALYSIS")
    logger.info("=" * 70)

    # Load data
    shows, songs, setlists, song_tags = load_all_data()
    logger.info(f"Loaded {len(songs)} songs from {len(shows)} shows")

    # Build co-occurrence matrix
    cooccur, song_ids = build_cooccurrence_matrix(setlists, songs)

    # Cluster songs
    labels = cluster_songs(
        cooccur, n_clusters=args.n_clusters, method=args.cluster_method
    )

    # Print cluster summaries
    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY")
    print("=" * 70)

    for cluster_id in range(args.n_clusters):
        songs_in_cluster = [
            song_ids[i] for i, label in enumerate(labels) if label == cluster_id
        ]
        print(f"\nCluster {cluster_id}: {len(songs_in_cluster)} songs")

        # Show top 5 songs
        for song_id in songs_in_cluster[:5]:
            song = songs[songs["song_id"] == song_id].iloc[0]
            print(f"  - {song['title']}")

    # Visualizations
    if args.visualize:
        logger.info("\nGenerating visualizations...")

        # 2D embedding
        embedding = reduce_dimensions(cooccur, method=args.method, n_components=2)

        # Cluster visualization
        visualize_clusters(
            embedding,
            labels,
            songs,
            song_ids,
            args.output_dir,
            title=f"Song Clusters ({args.method.upper()}, k={args.n_clusters})",
        )

        # Dendrogram
        plot_dendrogram(cooccur, songs, song_ids, args.output_dir)

        # Compare to genres if available
        compare_to_genres(labels, songs, song_ids, args.output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
