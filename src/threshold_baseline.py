"""
threshold_baseline.py — Rule-based threshold anomaly detector.

Flags a timestep as anomalous when any monitored metric exceeds
its configured threshold. This is the comparison baseline.
"""

import numpy as np
import pandas as pd

from . import config


def detect_threshold_anomalies(
    df: pd.DataFrame,
    thresholds: dict = config.THRESHOLDS,
) -> np.ndarray:
    """
    Apply static threshold rules to raw telemetry.

    Args:
        df          — raw telemetry DataFrame (must contain threshold metric cols)
        thresholds  — dict of {metric_name: threshold_value}
                      a timestep is flagged if metric > threshold

    Returns:
        predictions — binary array (1 = anomaly, 0 = normal), shape (N,)
    """
    n = len(df)
    flagged = np.zeros(n, dtype=int)

    for metric, threshold in thresholds.items():
        if metric not in df.columns:
            print(f"[threshold_baseline] Warning: column '{metric}' not found, skipping.")
            continue
        flagged = np.logical_or(flagged, df[metric].values > threshold).astype(int)

    return flagged


class ThresholdDetector:
    """
    Object-oriented wrapper around detect_threshold_anomalies.
    Makes the baseline interchangeable with IsolationForestDetector.
    """

    def __init__(self, thresholds: dict = config.THRESHOLDS) -> None:
        self.thresholds = thresholds

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions for full telemetry DataFrame."""
        return detect_threshold_anomalies(df, self.thresholds)
