"""Stage 4 Temporal Sets GNN - imports from shared base implementation.

This module provides backward compatibility by importing from the shared
temporal_sets_gnn module. Stage 3 and Stage 4 use the same implementation.
"""

# Import all classes from shared implementation
from src.models.temporal_sets_gnn import (
    GraphConvLayer,
    TemporalAttention,
    TemporalSetsGNN,
)

__all__ = ["GraphConvLayer", "TemporalAttention", "TemporalSetsGNN"]
