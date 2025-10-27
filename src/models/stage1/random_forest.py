"""Random Forest model for setlist prediction.

Ensemble baseline using sklearn's RandomForestClassifier with tuned hyperparameters.
Provides feature importance and handles non-linear relationships.
"""

from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier


class RandomForest:
    """Random Forest ensemble model for setlist prediction.

    This model uses an ensemble of decision trees to capture non-linear patterns
    in the feature space. Each tree is trained on a bootstrap sample and makes
    predictions by averaging:

        P(song_i | features) = (1/T) * sum_t P_t(song_i | features)

    where T is the number of trees and P_t is the prediction from tree t.

    Attributes:
        model: Sklearn RandomForestClassifier instance.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        min_samples_split: Minimum samples required to split a node.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs.

    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 20,
        random_state: int = 42,
        n_jobs: int = 4,
    ):
        """Initialize random forest model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees (prevents overfitting).
            min_samples_split: Minimum samples to split internal node.
            random_state: Random seed for reproducibility.
            n_jobs: Number of CPU cores to use.

        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = SKRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
            bootstrap=True,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """Fit random forest model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).

        Returns:
            self: Fitted model.

        """
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

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores from the random forest.

        Returns:
            Feature importances (Gini importance), shape (n_features,).
            Higher values indicate more important features.

        """
        return self.model.feature_importances_
