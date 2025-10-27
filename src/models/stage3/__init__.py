"""Stage 3: Neural Models (Representation Learning Paradigm).

Models in this stage learn embeddings from categorical IDs:
- DeepFM: Factorization machines + deep network
- Temporal Sets GNN: Current best model (35.54% Recall@15)
- DCNV2: Deep & Cross Network V2 (optional)

Note: TabNet has been archived (incompatible with tabular-only approach)
"""

from src.models.stage3.deepfm import DeepFM
from src.models.stage3.temporal_sets import TemporalSetsGNN
from src.models.stage3.dcn import DCNV2

__all__ = [
    "DeepFM",
    "TemporalSetsGNN",
    "DCNV2",
]
