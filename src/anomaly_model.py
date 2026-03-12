"""
anomaly_model.py — Isolation Forest anomaly detector (LightKubeGuard core).

Train on normal-only windows; score and predict on the full feature set.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple

from . import config


class IsolationForestDetector:
    """
    Wrapper around sklearn IsolationForest.

    Separation of fit (normal-only) and predict (full data) reflects
    the unsupervised, semi-supervised training protocol.
    """

    def __init__(
        self,
        n_estimators: int   = config.IF_N_ESTIMATORS,
        contamination: float = config.IF_CONTAMINATION,
        random_state: int   = config.IF_RANDOM_STATE,
    ) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X_normal: np.ndarray) -> "IsolationForestDetector":
        """Fit the model on normal (label==0) feature windows only."""
        self.model.fit(X_normal)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary predictions for all windows.
        IsolationForest returns -1 (anomaly) / +1 (normal).
        Converted to 1 (anomaly) / 0 (normal).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        raw = self.model.predict(X)
        return np.where(raw == -1, 1, 0).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores (lower = more anomalous).
        sklearn's decision_function output is used directly.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        return self.model.decision_function(X)


def train_and_predict(
    features_df: pd.DataFrame,
    labels_aligned: np.ndarray,
    n_estimators: int    = config.IF_N_ESTIMATORS,
    contamination: float = config.IF_CONTAMINATION,
    random_state: int    = config.IF_RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, IsolationForestDetector]:
    """
    Convenience function: train on normal windows, predict on all.

    Args:
        features_df     — normalised feature DataFrame (index aligned to df)
        labels_aligned  — ground-truth anomaly labels (same length)

    Returns:
        predictions     — binary array (1 = anomaly)
        scores          — anomaly scores (lower = more anomalous)
        detector        — fitted IsolationForestDetector instance
    """
    X = features_df.values
    normal_mask = labels_aligned == 0

    detector = IsolationForestDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    detector.fit(X[normal_mask])

    predictions = detector.predict(X)
    scores      = detector.score_samples(X)

    return predictions, scores, detector
