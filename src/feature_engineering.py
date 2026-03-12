"""
feature_engineering.py — Sliding-window feature extraction.

For each of the 5 core metrics, computes:
    rolling mean, rolling std, rolling max, rate of change
→ 20-dimensional feature vector per timestep.

Features are StandardScaler-normalised before being returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

from . import config

# Metrics used in feature extraction (order is preserved)
FEATURE_METRICS = ["cpu", "memory", "net_in", "net_out", "latency"]


def _rolling_features(
    series: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Compute 4 rolling statistics for a single metric series."""
    name = series.name
    roll = series.rolling(window=window, min_periods=window)
    roc  = series.diff()                       # rate of change (lag-1 difference)

    return pd.DataFrame({
        f"{name}_mean": roll.mean(),
        f"{name}_std":  roll.std(),
        f"{name}_max":  roll.max(),
        f"{name}_roc":  roc,
    })


def extract_features(
    df: pd.DataFrame,
    window: int = config.WINDOW_SIZE,
    metrics: list = FEATURE_METRICS,
    fit_scaler: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler, pd.Index]:
    """
    Extract and normalise sliding-window features.

    Args:
        df          — raw telemetry DataFrame from data_generator
        window      — rolling window size
        metrics     — list of metric column names to featurise
        fit_scaler  — if True, fit a new StandardScaler; else use `scaler`
        scaler      — pre-fitted scaler (used when fit_scaler=False)

    Returns:
        features_df — normalised feature DataFrame (NaN rows dropped)
        scaler      — fitted StandardScaler
        valid_index — index of rows retained after NaN removal
    """
    parts = [_rolling_features(df[m], window) for m in metrics]
    feat_raw = pd.concat(parts, axis=1)

    # Rate-of-change for the first row is NaN; rolling windows add more NaNs
    feat_raw = feat_raw.dropna()
    valid_index = feat_raw.index

    if fit_scaler:
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_raw.values)
    else:
        if scaler is None:
            raise ValueError("Must supply a fitted scaler when fit_scaler=False.")
        feat_scaled = scaler.transform(feat_raw.values)

    features_df = pd.DataFrame(
        feat_scaled,
        columns=feat_raw.columns,
        index=valid_index,
    )

    return features_df, scaler, valid_index
