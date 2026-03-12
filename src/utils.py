"""
utils.py — Shared helper utilities for LightKubeGuard.
"""

import os
import numpy as np
from typing import List, Tuple


def ensure_output_dir(path: str) -> None:
    """Create directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def set_random_seed(seed: int) -> None:
    """Set numpy random seed for reproducibility."""
    np.random.seed(seed)


def anomaly_regions_from_labels(labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) index pairs for contiguous anomaly runs.
    Labels are expected to be 0 (normal) or 1 (anomaly).
    """
    regions = []
    in_region = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_region:
            start = i
            in_region = True
        elif v == 0 and in_region:
            regions.append((start, i - 1))
            in_region = False
    if in_region:
        regions.append((start, len(labels) - 1))
    return regions


def compute_detection_delay(
    labels: np.ndarray,
    predictions: np.ndarray,
    regions: List[Tuple[int, int]],
    max_delay: int = 9999,
) -> float:
    """
    Compute mean detection delay across anomaly regions.

    Delay for region (s, e):
      - number of steps from s to first prediction==1 within [s, e]
      - if never detected → max_delay (penalises missed regions)

    Returns the mean delay across all regions (float).
    """
    delays = []
    for (s, e) in regions:
        preds_in = predictions[s : e + 1]
        hits = np.where(preds_in == 1)[0]
        if len(hits) == 0:
            delays.append(max_delay)
        else:
            delays.append(int(hits[0]))
    return float(np.mean(delays)) if delays else 0.0


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` when denominator is zero."""
    return numerator / denominator if denominator != 0 else default
