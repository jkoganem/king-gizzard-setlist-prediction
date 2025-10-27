"""Data I/O operations for the setlist predictor.

This module centralizes all filesystem operations including reading/writing
parquet files, JSON files, and model artifacts. No other modules should
perform direct filesystem I/O.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src.utils.config import settings


@dataclass(frozen=True)
class DataPaths:
    """Paths for data files.

    Attributes:
        venues: Path to venues parquet file.
        shows: Path to shows parquet file.
        songs: Path to songs parquet file.
        setlists: Path to setlists parquet file.
        song_tags: Path to song_tags parquet file.

    """

    venues: Path
    shows: Path
    songs: Path
    setlists: Path
    song_tags: Path

    @classmethod
    def from_curated_dir(cls, base_dir: Optional[Path] = None) -> "DataPaths":
        """Create DataPaths from curated data directory.

        Args:
            base_dir: Base directory for curated data. If None, uses settings.

        Returns:
            DataPaths instance with paths to all data files.

        """
        if base_dir is None:
            base_dir = settings.paths.data_curated_dir

        return cls(
            venues=base_dir / "venues.parquet",
            shows=base_dir / "shows.parquet",
            songs=base_dir / "songs.parquet",
            setlists=base_dir / "setlists.parquet",
            song_tags=base_dir / "song_tags.parquet",
        )


def load_parquet(file_path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame.

    Args:
        file_path: Path to parquet file.

    Returns:
        DataFrame containing the data.

    Raises:
        FileNotFoundError: If file does not exist.

    """
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    return pd.read_parquet(file_path)


def save_parquet(df: pd.DataFrame, file_path: Path) -> None:
    """Save a DataFrame to parquet format.

    Args:
        df: DataFrame to save.
        file_path: Output file path.

    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)


def identify_marathon_shows(shows: pd.DataFrame, setlists: pd.DataFrame) -> pd.Series:
    """Identify marathon shows (3-hour extended performances).

    Marathon shows are explicitly marketed as such and appear in show notes/descriptions.
    This uses keyword matching rather than length-based detection, as some regular shows
    in 2017-2019 had 22-23 songs without being marketed as "marathons".

    Args:
        shows: DataFrame of shows with 'notes' column.
        setlists: DataFrame of setlist entries (unused, kept for consistency).

    Returns:
        Boolean Series indicating marathon shows, indexed by show_id.

    """
    marathon_show_ids = set()

    # Only use explicit labeling from notes field
    marathon_keywords = ["marathon", "3 hour", "3hr", "three hour", "three-hour"]
    if "notes" in shows.columns:
        keyword_pattern = "|".join(marathon_keywords)
        marathon_from_notes = shows[
            shows["notes"].str.lower().str.contains(keyword_pattern, na=False)
        ]["show_id"].values
        marathon_show_ids.update(marathon_from_notes)

    # Return boolean Series indexed by show_id
    return shows["show_id"].isin(marathon_show_ids)


def identify_residency_shows(shows: pd.DataFrame) -> pd.Series:
    """Identify shows that are part of multi-night residencies.

    A residency is defined as 2+ consecutive shows at the same venue within 4 days.

    Args:
        shows: DataFrame of shows with 'venue_id' and 'date' columns.

    Returns:
        Boolean Series indicating residency shows, indexed by show_id.

    """
    from datetime import timedelta

    shows_sorted = shows.sort_values("date").copy()
    shows_sorted["date"] = pd.to_datetime(shows_sorted["date"])

    residency_show_ids = set()

    for venue_id in shows_sorted["venue_id"].unique():
        venue_shows = shows_sorted[shows_sorted["venue_id"] == venue_id].sort_values(
            "date"
        )

        if len(venue_shows) < 2:
            continue

        current_group = []
        prev_date = None

        for _, show in venue_shows.iterrows():
            if prev_date is None:
                current_group = [show["show_id"]]
            elif (show["date"] - prev_date).days <= 4:
                # Within 4 days - same residency
                current_group.append(show["show_id"])
            else:
                # Gap > 4 days - finalize previous group
                if len(current_group) >= 2:
                    residency_show_ids.update(current_group)
                current_group = [show["show_id"]]

            prev_date = show["date"]

        # Handle last group
        if len(current_group) >= 2:
            residency_show_ids.update(current_group)

    # Return boolean Series indexed by show_id
    return shows["show_id"].isin(residency_show_ids)


def load_all_data(
    paths: Optional[DataPaths] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all curated data files and enrich with computed show properties.

    Args:
        paths: DataPaths instance. If None, uses default curated directory.

    Returns:
        Tuple of (shows, songs, setlists, song_tags) DataFrames.

    Note:
        The shows DataFrame is enriched with computed boolean flags:
        - is_festival: Already in raw data (from setlist.fm API)
        - is_marathon: Computed from notes field (3-hour extended performances)
        - is_residency: Computed from venue/date patterns (multi-night residencies)

    """
    if paths is None:
        paths = DataPaths.from_curated_dir()

    shows = load_parquet(paths.shows)
    songs = load_parquet(paths.songs)
    setlists = load_parquet(paths.setlists)
    song_tags = (
        load_parquet(paths.song_tags) if paths.song_tags.exists() else pd.DataFrame()
    )

    # Enrich shows with computed boolean flags
    # Note: is_festival already exists in raw data
    shows["is_marathon"] = identify_marathon_shows(shows, setlists).astype(int)
    shows["is_residency"] = identify_residency_shows(shows).astype(int)

    return shows, songs, setlists, song_tags


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Dictionary containing JSON data.

    Raises:
        FileNotFoundError: If file does not exist.

    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Dictionary to save.
        file_path: Output file path.
        indent: JSON indentation level.

    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_model(model_path: Path) -> Any:
    """Load a trained model from disk.

    Args:
        model_path: Path to model file (.pkl or .joblib).

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If model file does not exist.

    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def save_model(model: Any, model_path: Path) -> None:
    """Save a trained model to disk.

    Args:
        model: Model object to save.
        model_path: Output file path.

    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

