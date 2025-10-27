"""Stage 0: Sequential Models (For EDA and Understanding Setlist Structure).

Models in this stage analyze sequential patterns in setlists:
- SASRec: Self-Attentive Sequential Recommendation
- Sequence Generator: Helper utilities for sequential data
- Set Model: Experimental set-based approaches
"""

from src.models.stage0.sasrec import SASRec

__all__ = [
    "SASRec",
]
