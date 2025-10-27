"""Stage 4: Temporal Graph Neural Networks.

Models:
- TemporalSetsGNN: Graph convolution + temporal attention (KDD 2020)
"""

from src.models.stage4.temporal_sets import TemporalSetsGNN

__all__ = ["TemporalSetsGNN"]
