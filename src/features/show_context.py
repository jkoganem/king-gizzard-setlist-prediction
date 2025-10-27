"""Feature engineering for show-level context.

This module computes features related to show context including tour information,
venue characteristics, temporal features, and proximity to previous shows.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic


@dataclass(frozen=True)
class ShowContextFeatures:
    """Show context features for prediction.

    Attributes:
        show_id: Unique show identifier.
        tour_name: Name of the tour.
        city: City where show takes place.
        country: Country code.
        venue_capacity_bucket: Discretized venue capacity (small/medium/large).
        is_festival: Whether show is part of a festival.
        is_weekend: Whether show is on weekend.
        is_holiday: Whether show is on a holiday.
        days_since_prev: Days since previous show.
        distance_km: Distance from previous show in kilometers.
        same_venue: Whether show is at same venue as previous.

    """

    show_id: str
    tour_name: str
    city: str
    country: str
    venue_capacity_bucket: str
    is_festival: bool
    is_weekend: bool
    is_holiday: bool
    days_since_prev: Optional[int]
    distance_km: Optional[float]
    same_venue: bool


def compute_show_proximity(shows_df: pd.DataFrame) -> pd.DataFrame:
    """Compute show-to-show proximity features.

    For each show, finds the previous show (by date) and computes:
    - days_since_prev: Days between this show and previous show
    - same_venue: Boolean indicating if venue is same as previous show
    - distance_km: Distance from previous show (if lat/lon available)

    Args:
        shows_df: DataFrame with columns [show_id, date, venue_id, city, state, country].
                  Optional: [lat, lon] for distance calculation.

    Returns:
        DataFrame with columns [show_id, prev_show_id, days_since_prev, same_venue, distance_km].

    """
    # Sort by date to establish chronological order
    shows_sorted = shows_df.sort_values("date").copy()
    shows_sorted["date"] = pd.to_datetime(shows_sorted["date"])

    proximity_data = []

    for i, (idx, show) in enumerate(shows_sorted.iterrows()):
        if i == 0:
            # First show - no previous
            proximity_data.append(
                {
                    "show_id": show["show_id"],
                    "prev_show_id": None,
                    "days_since_prev": None,
                    "same_venue": False,
                    "distance_km": None,
                }
            )
        else:
            # Get previous show
            prev_show = shows_sorted.iloc[i - 1]

            # Calculate days since previous show
            days_diff = (show["date"] - prev_show["date"]).days

            # Check if same venue
            same_venue = show.get("venue_id", show.get("city")) == prev_show.get(
                "venue_id", prev_show.get("city")
            )

            # Calculate distance if lat/lon available
            distance = None
            if (
                "lat" in show
                and "lon" in show
                and "lat" in prev_show
                and "lon" in prev_show
            ):
                if pd.notna(show["lat"]) and pd.notna(prev_show["lat"]):
                    distance = compute_haversine_distance(
                        prev_show["lat"], prev_show["lon"], show["lat"], show["lon"]
                    )

            proximity_data.append(
                {
                    "show_id": show["show_id"],
                    "prev_show_id": prev_show["show_id"],
                    "days_since_prev": days_diff,
                    "same_venue": same_venue,
                    "distance_km": distance,
                }
            )

    return pd.DataFrame(proximity_data)


def compute_no_repeat_signals(
    shows_df: pd.DataFrame, setlists_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute no-repeat signals for each (show, song) pair.

    For each show, computes how the candidate setlist overlaps with:
    1. Previous night's setlist (played_prev_night, overlap_prev)
    2. Last show at the same venue (played_prev_venue, overlap_prev_venue)

    Args:
        shows_df: DataFrame with show information [show_id, date, venue_id].
        setlists_df: DataFrame with setlist entries [show_id, song_id, pos].

    Returns:
        DataFrame with columns [show_id, song_id, played_prev_night, played_prev_venue,
                                 overlap_prev_jaccard, overlap_prev_venue_jaccard].

    """
    # Sort shows chronologically
    shows_sorted = shows_df.sort_values("date").copy()
    shows_sorted["date"] = pd.to_datetime(shows_sorted["date"])

    # Get all unique songs
    all_songs = set(setlists_df["song_id"].unique())

    # Build show_id -> setlist mapping
    show_setlists = {}
    for show_id, group in setlists_df.groupby("show_id"):
        show_setlists[show_id] = set(group["song_id"].values)

    # Build venue -> shows mapping (chronological)
    venue_shows = {}
    for _, show in shows_sorted.iterrows():
        venue = show.get("venue_id", show.get("city", "unknown"))
        if venue not in venue_shows:
            venue_shows[venue] = []
        venue_shows[venue].append(show["show_id"])

    no_repeat_data = []

    for i, (idx, show) in enumerate(shows_sorted.iterrows()):
        show_id = show["show_id"]
        venue = show.get("venue_id", show.get("city", "unknown"))

        # Find previous show (chronologically)
        prev_show_id = shows_sorted.iloc[i - 1]["show_id"] if i > 0 else None

        # Find previous show at this venue
        venue_show_list = venue_shows.get(venue, [])
        venue_idx = venue_show_list.index(show_id) if show_id in venue_show_list else -1
        prev_venue_show_id = venue_show_list[venue_idx - 1] if venue_idx > 0 else None

        # Get setlists
        prev_setlist = show_setlists.get(prev_show_id, set())
        prev_venue_setlist = show_setlists.get(prev_venue_show_id, set())

        # For each song, compute signals
        for song_id in all_songs:
            # Previous night signals
            played_prev_night = song_id in prev_setlist

            # Calculate Jaccard similarity with previous night
            # (How similar is the candidate song's context to last night?)
            # Note: This is a simplified version - in practice, you'd want to
            # compute overlap for the full predicted setlist, not per song
            if prev_setlist:
                overlap_prev_jaccard = 1.0 if played_prev_night else 0.0
            else:
                overlap_prev_jaccard = 0.0

            # Previous venue signals
            played_prev_venue = song_id in prev_venue_setlist

            # Calculate Jaccard with previous venue show
            if prev_venue_setlist:
                overlap_prev_venue_jaccard = 1.0 if played_prev_venue else 0.0
            else:
                overlap_prev_venue_jaccard = 0.0

            no_repeat_data.append(
                {
                    "show_id": show_id,
                    "song_id": song_id,
                    "played_prev_night": int(played_prev_night),
                    "played_prev_venue": int(played_prev_venue),
                    "overlap_prev_jaccard": overlap_prev_jaccard,
                    "overlap_prev_venue_jaccard": overlap_prev_venue_jaccard,
                }
            )

    return pd.DataFrame(no_repeat_data)


def discretize_capacity(capacity: Optional[int]) -> str:
    """Discretize venue capacity into buckets.

    Args:
        capacity: Venue capacity in number of people.

    Returns:
        Capacity bucket label ('small', 'medium', 'large', or 'unknown').

    """
    if capacity is None:
        return "unknown"
    if capacity < 2000:
        return "small"
    if capacity < 10000:
        return "medium"
    return "large"


def is_weekend(show_date: date) -> bool:
    """Check if date is a weekend.

    Args:
        show_date: Date to check.

    Returns:
        True if Saturday or Sunday, False otherwise.

    """
    return show_date.weekday() >= 5


def compute_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Compute haversine distance between two points.

    Args:
        lat1: Latitude of first point.
        lon1: Longitude of first point.
        lat2: Latitude of second point.
        lon2: Longitude of second point.

    Returns:
        Distance in kilometers.

    """
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except (ValueError, TypeError):
        return np.nan
