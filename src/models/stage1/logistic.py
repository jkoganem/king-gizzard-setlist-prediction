"""Logistic Regression model for setlist prediction.

Simple baseline using sklearn's LogisticRegression with default parameters.
Provides interpretable coefficients and fast training.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression as SKLogisticRegression


class LogisticRegression:
    """Logistic Regression baseline for setlist prediction.

    This model serves as a simple, interpretable baseline for the prediction task.
    It models the probability of each song being played as:

        P(song_i | features) = sigmoid(w^T x_i + b)

    where w are learned weights and x_i are the song features for show context.

    Attributes:
        model: Sklearn LogisticRegression instance.
        max_iter: Maximum iterations for optimization.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 for all cores).

    """

    def __init__(self, max_iter: int = 1000, random_state: int = 42, n_jobs: int = 4):
        """Initialize logistic regression model.

        Args:
            max_iter: Maximum iterations for solver convergence.
            random_state: Random seed for reproducibility.
            n_jobs: Number of CPU cores to use.

        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = SKLogisticRegression(
            max_iter=max_iter, random_state=random_state, n_jobs=n_jobs, solver="lbfgs"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit logistic regression model.

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

    def get_coefficients(self) -> Tuple[np.ndarray, float]:
        """Get learned model coefficients.

        Returns:
            Tuple of (weights, intercept).
            weights: Feature coefficients, shape (n_features,).
            intercept: Bias term.

        """
        return self.model.coef_[0], self.model.intercept_[0]
