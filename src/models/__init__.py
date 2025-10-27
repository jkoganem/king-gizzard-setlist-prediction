"""Models for setlist prediction.

Organized by experimental stage:
- stage0/: Sequential models for EDA (SASRec)
- stage1/: Traditional ML models (Logistic, RF, XGBoost)
- stage3/: Neural models (DeepFM, Temporal Sets, DCNV2, TabNet)
"""

# Stage 1: Traditional ML (Feature Engineering)
from src.models.stage1 import LogisticRegression, RandomForest, XGBoostModel

# Stage 3: Neural Models (Representation Learning)
from src.models.stage3 import DeepFM, TemporalSetsGNN, DCNV2

# Stage 0: Sequential Models (For EDA)
from src.models.stage0 import SASRec

__all__ = [
    # Stage 1
    "LogisticRegression",
    "RandomForest",
    "XGBoostModel",
    # Stage 3
    "DeepFM",
    "TemporalSetsGNN",
    "DCNV2",
    # Stage 0
    "SASRec",
]
