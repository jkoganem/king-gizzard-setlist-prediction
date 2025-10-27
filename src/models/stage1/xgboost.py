"""XGBoost model for setlist prediction.

Gradient boosted decision trees with optional hyperparameter tuning via Optuna.
Best-performing traditional ML model for this task.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier


class XGBoostModel:
    """XGBoost gradient boosting model for setlist prediction.

    This model uses gradient boosting with decision trees as base learners.
    It sequentially builds trees where each tree corrects the errors of the
    previous ensemble:

        F_m(x) = F_{m-1}(x) + eta * h_m(x)

    where h_m is the m-th tree and eta is the learning rate.

    XGBoost includes regularization (L1/L2) and advanced features like
    column subsampling and early stopping to prevent overfitting.

    Attributes:
        model: XGBoost XGBClassifier instance.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage (eta).
        subsample: Fraction of samples for each tree.
        colsample_bytree: Fraction of features for each tree.
        reg_alpha: L1 regularization on leaf weights.
        reg_lambda: L2 regularization on leaf weights.
        random_state: Random seed for reproducibility.
        nthread: Number of parallel threads.

    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        nthread: int = 4,
    ):
        """Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of trees.
            learning_rate: Learning rate (eta) for boosting.
            subsample: Row subsampling ratio.
            colsample_bytree: Column subsampling ratio.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            random_state: Random seed for reproducibility.
            nthread: Number of parallel threads.

        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.nthread = nthread

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            nthread=nthread,
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=20,
            verbose=0,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "XGBoostModel":
        """Fit XGBoost model with optional validation set for early stopping.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).
            eval_set: Optional (X_val, y_val) for early stopping.

        Returns:
            self: Fitted model.

        """
        if eval_set is not None:
            self.model.fit(X, y, eval_set=[eval_set], verbose=False)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for each class.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            Probabilities, shape (n_samples, 2) for [P(y=0), P(y=1)].

        """
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            Predicted labels, shape (n_samples,).

        """
        return self.model.predict(X)

    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> Dict[str, np.ndarray]:
        """Get feature importance scores from XGBoost.

        Args:
            importance_type: Type of importance to compute.
                'gain': Average gain across splits using the feature.
                'weight': Number of times feature is used in splits.
                'cover': Average coverage of splits using the feature.

        Returns:
            Dictionary mapping importance types to arrays of shape (n_features,).

        """
        booster = self.model.get_booster()
        importance_dict = {}

        for imp_type in ["gain", "weight", "cover"]:
            scores = booster.get_score(importance_type=imp_type)
            # Convert to array indexed by feature number
            n_features = len(scores)
            importance_array = np.zeros(n_features)
            for feat, score in scores.items():
                feat_idx = int(feat.replace("f", ""))
                importance_array[feat_idx] = score
            importance_dict[imp_type] = importance_array

        return importance_dict
