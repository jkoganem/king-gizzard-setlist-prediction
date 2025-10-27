"""Feature engineering modules.

This package provides feature computation for show context, song attributes,
and other derived features used in model training and prediction.
"""

from src.features.show_context import (
    ShowContextFeatures,
    compute_show_proximity,
    compute_no_repeat_signals,
    discretize_capacity,
    is_weekend,
    compute_haversine_distance,
)
from src.features.song_features import (
    SongFeatures,
    compute_song_features,
    compute_play_rate,
    compute_time_decay_score,
)

__all__ = [
    "ShowContextFeatures",
    "compute_show_proximity",
    "compute_no_repeat_signals",
    "discretize_capacity",
    "is_weekend",
    "compute_haversine_distance",
    "SongFeatures",
    "compute_song_features",
    "compute_play_rate",
    "compute_time_decay_score",
]
