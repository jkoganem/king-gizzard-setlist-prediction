"""Stage 1: Traditional ML Models (Feature Engineering Paradigm).

Models in this stage use pre-engineered tabular features:
- Logistic Regression: Linear baseline
- Random Forest: Non-linear ensemble baseline
- XGBoost: Gradient boosting (expected best traditional model)
"""

from src.models.stage1.logistic import LogisticRegression
from src.models.stage1.random_forest import RandomForest
from src.models.stage1.xgboost import XGBoostModel

__all__ = [
    "LogisticRegression",
    "RandomForest",
    "XGBoostModel",
]
