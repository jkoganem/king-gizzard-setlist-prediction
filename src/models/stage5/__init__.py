"""Stage 5: Research Models and Improvements.

Models based on recent TSP literature (2023-2025):
- TemporalSetsGNNWithPriors: GNN + SAFERec-style frequency/recency priors
- (Future) SFCN-TSP: Simplified fully connected network (AAAI 2023)
- (Future) SAFERec-TSP: Transformer with frequency enrichment (2024)
"""

from src.models.stage5.priors import (
    FrequencyRecencyPrior,
    compute_batch_recency_scores,
    compute_recency_scores,
    compute_song_frequencies,
)
from src.models.stage5.temporal_sets_with_priors import TemporalSetsGNNWithPriors

__all__ = [
    "TemporalSetsGNNWithPriors",
    "FrequencyRecencyPrior",
    "compute_song_frequencies",
    "compute_recency_scores",
    "compute_batch_recency_scores",
]
