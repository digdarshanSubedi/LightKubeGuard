"""
config.py — Centralized configuration for LightKubeGuard.
Edit values here to change experiment parameters globally.
"""

import os

# ─── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Simulation ────────────────────────────────────────────────────────────────
N_TIMESTEPS = 1000
WINDOW_SIZE = 20          # sliding-window size for feature engineering

# ─── Normal telemetry baseline ranges (min, max) ───────────────────────────────
BASELINE = {
    "cpu":      (20.0, 60.0),   # CPU utilisation %
    "memory":   (30.0, 70.0),   # Memory usage %
    "net_in":   (10.0, 80.0),   # Network ingress MB/s
    "net_out":  (5.0,  50.0),   # Network egress MB/s
    "latency":  (50.0, 200.0),  # Request latency ms
    "restarts": (0.0,  1.0),    # Container restart count
}

# ─── Anomaly scenario definitions ─────────────────────────────────────────────
# Each entry: (start_index, end_index, scenario_name, metric_overrides)
# metric_overrides: dict of metric -> (add_mean, add_std)
ANOMALY_SCENARIOS = [
    {
        "name":  "burst_hotspot",
        "start": 150,
        "end":   200,
        "overrides": {
            "cpu":     (45.0, 8.0),   # spike on top of baseline
            "latency": (200.0, 30.0),
        },
    },
    {
        "name":  "memory_stress",
        "start": 400,
        "end":   470,
        "overrides": {
            "memory":   (30.0, 5.0),
            "restarts": (3.0,  1.0),
        },
    },
    {
        "name":  "network_instability",
        "start": 700,
        "end":   780,
        "overrides": {
            "net_in":  (80.0, 20.0),
            "net_out": (60.0, 15.0),
            "latency": (150.0, 40.0),
        },
    },
]

# ─── Threshold-based baseline detector ────────────────────────────────────────
THRESHOLDS = {
    "cpu":      85.0,    # flag if CPU > this
    "memory":   90.0,    # flag if memory > this
    "latency":  300.0,   # flag if latency > this
}

# ─── Isolation Forest hyperparameters ─────────────────────────────────────────
IF_N_ESTIMATORS  = 200
IF_CONTAMINATION = 0.05
IF_RANDOM_STATE  = RANDOM_SEED

# ─── Output paths ─────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

OUTPUTS = {
    "dir":                   _BASE,
    "metrics_csv":           os.path.join(_BASE, "lightkubeguard_metrics.csv"),
    "results_csv":           os.path.join(_BASE, "lightkubeguard_results.csv"),
    "results_by_scenario":   os.path.join(_BASE, "lightkubeguard_results_by_scenario.csv"),
    "summary_txt":           os.path.join(_BASE, "paper_results_summary.txt"),
    "cpu_plot":              os.path.join(_BASE, "cpu_anomaly_plot.png"),
    "latency_plot":          os.path.join(_BASE, "latency_anomaly_plot.png"),
    "accuracy_plot":         os.path.join(_BASE, "accuracy_comparison.png"),
    "delay_plot":            os.path.join(_BASE, "detection_delay.png"),
    "roc_curve_plot":        os.path.join(_BASE, "roc_curve.png"),
}
