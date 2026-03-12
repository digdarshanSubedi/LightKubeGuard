"""
data_generator.py — Synthetic Kubernetes telemetry generator.

Generates 1 000 timesteps of realistic-looking metrics and injects
labelled anomaly regions according to config.ANOMALY_SCENARIOS.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from . import config
from .utils import set_random_seed, ensure_output_dir


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _normal_signal(
    n: int,
    low: float,
    high: float,
    noise_std: float = 3.0,
    trend_scale: float = 0.005,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a baseline signal uniformly drawn from [low, high],
    plus Gaussian noise and a mild time-varying trend.
    """
    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)
    mid = (low + high) / 2.0
    base = mid + rng.uniform(low - mid, high - mid, size=n)
    noise = rng.normal(0, noise_std, size=n)
    trend = trend_scale * np.arange(n)
    signal = base + noise + trend
    # soft-clip to keep values physically plausible (±50% beyond range)
    return np.clip(signal, low * 0.5, high * 1.8)


def _inject_anomalies(
    df: pd.DataFrame,
    scenarios: list,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Inject additive spikes into specified metric columns for each scenario.

    Returns:
        df        — modified DataFrame
        labels    — binary anomaly label per timestep (0/1)
        scenarios_arr — integer scenario ID per timestep (0 = normal)
    """
    labels = np.zeros(len(df), dtype=int)
    scenario_ids = np.zeros(len(df), dtype=int)   # 0 = normal

    for idx, scenario in enumerate(scenarios, start=1):
        s = scenario["start"]
        e = scenario["end"]
        labels[s:e] = 1
        scenario_ids[s:e] = idx

        for metric, (mean_add, std_add) in scenario["overrides"].items():
            if metric not in df.columns:
                continue
            length = e - s
            spike = rng.normal(mean_add, std_add, size=length)
            df.loc[df.index[s:e], metric] = df[metric].iloc[s:e].values + spike

    return df, labels, scenario_ids


# ─── Public API ────────────────────────────────────────────────────────────────

def generate_telemetry(
    n_timesteps: int = config.N_TIMESTEPS,
    scenarios: list = config.ANOMALY_SCENARIOS,
    seed: int = config.RANDOM_SEED,
    save_path: Optional[str] = config.OUTPUTS["metrics_csv"],
) -> pd.DataFrame:
    """
    Generate synthetic Kubernetes-style telemetry.

    Returns a DataFrame with columns:
        timestep, cpu, memory, net_in, net_out, latency, restarts,
        anomaly_label, scenario_id, scenario_name
    """
    set_random_seed(seed)
    rng = np.random.default_rng(seed)

    timesteps = np.arange(n_timesteps)

    # --- baseline signals -------------------------------------------------------
    cpu      = _normal_signal(n_timesteps, *config.BASELINE["cpu"],     noise_std=4.0,  rng=rng)
    memory   = _normal_signal(n_timesteps, *config.BASELINE["memory"],  noise_std=3.0,  rng=rng)
    net_in   = _normal_signal(n_timesteps, *config.BASELINE["net_in"],  noise_std=6.0,  rng=rng)
    net_out  = _normal_signal(n_timesteps, *config.BASELINE["net_out"], noise_std=4.0,  rng=rng)
    latency  = _normal_signal(n_timesteps, *config.BASELINE["latency"], noise_std=15.0, rng=rng)
    restarts = np.clip(rng.poisson(0.3, size=n_timesteps).astype(float), 0, 3)

    # mild correlation: high CPU tends to raise latency slightly
    latency += 0.5 * (cpu - cpu.mean())

    df = pd.DataFrame({
        "timestep": timesteps,
        "cpu":      cpu,
        "memory":   memory,
        "net_in":   net_in,
        "net_out":  net_out,
        "latency":  latency,
        "restarts": restarts,
    })

    # --- inject labelled anomalies ----------------------------------------------
    df, labels, scenario_ids = _inject_anomalies(df, scenarios, rng)

    df["anomaly_label"] = labels

    # store numeric + readable scenario id
    df["scenario_id"]   = scenario_ids
    scenario_name_map   = {0: "normal"}
    scenario_name_map.update({i + 1: s["name"] for i, s in enumerate(scenarios)})
    df["scenario_name"] = df["scenario_id"].map(scenario_name_map)

    if save_path is not None:
        ensure_output_dir(config.OUTPUTS["dir"])
        df.to_csv(save_path, index=False)
        print(f"[data_generator] Saved telemetry → {save_path}")

    return df
