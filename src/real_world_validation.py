"""
real_world_validation.py — Real-world validation using NAB dataset.

Loads the CPU utilization anomaly detection benchmark, extracts anomaly windows,
runs Isolation Forest, and computes metrics on real-world data.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
import matplotlib.pyplot as plt

from . import config


def load_nab_data(csv_path, labels_path):
    """
    Load NAB CSV and extract anomaly windows for the specific file.
    
    Args:
        csv_path: Path to cpu_utilization_asg_misconfiguration.csv
        labels_path: Path to combined_windows.json
    
    Returns:
        df: DataFrame with timestamp and value
        anomaly_windows: List of (start_idx, end_idx) tuples for this file
    """
    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Load labels
    with open(labels_path, "r") as f:
        labels = json.load(f)
    
    # Extract windows for this specific file
    # Try multiple key formats
    possible_keys = [
        "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
        "cpu_utilization_asg_misconfiguration.csv",
    ]
    anomaly_windows = []
    key_found = None
    
    for key in possible_keys:
        if key in labels:
            key_found = key
            break
    
    if key_found:
        for window in labels[key_found]:
            # window is [start_time_str, end_time_str]
            start_dt = pd.to_datetime(window[0])
            end_dt = pd.to_datetime(window[1])
            
            # Find indices by timestamp matching
            # Use idxmax to find first match, then work backwards/forwards
            mask_start = df["timestamp"] >= start_dt
            mask_end = df["timestamp"] <= end_dt
            
            if mask_start.any() and mask_end.any():
                start_idx = df[mask_start].index[0]
                end_idx = df[mask_end].index[-1]
                
                if end_idx > start_idx:
                    anomaly_windows.append((start_idx, end_idx))
    
    return df, anomaly_windows


def create_nab_features(values, window_size=20):
    """
    Create sliding-window features from univariate NAB time series.
    
    Args:
        values: 1D array of CPU utilization values
        window_size: Size of rolling window
    
    Returns:
        features: (n_samples, n_features) array
        valid_idx: Indices of non-NaN rows
    """
    n = len(values)
    
    # Rolling statistics
    rolling_mean = pd.Series(values).rolling(window_size, center=False).mean().values
    rolling_std = pd.Series(values).rolling(window_size, center=False).std().values
    rolling_max = pd.Series(values).rolling(window_size, center=False).max().values
    
    # Rate of change
    diff = np.diff(values, prepend=values[0])
    
    # Stack into feature matrix (5 features)
    features = np.column_stack([
        rolling_mean,
        rolling_std,
        rolling_max,
        diff,
        values  # raw value
    ])
    
    # Find valid (non-NaN) indices
    valid_idx = ~np.isnan(features).any(axis=1)
    
    return features[valid_idx], np.where(valid_idx)[0]


def create_ground_truth_labels(n_samples, anomaly_windows, valid_idx):
    """
    Create binary labels from anomaly windows, aligned to valid indices.
    
    Args:
        n_samples: Total number of samples in original data
        anomaly_windows: List of (start_idx, end_idx)
        valid_idx: Indices of valid (non-NaN) rows
    
    Returns:
        labels: Binary array where 1 = anomaly, 0 = normal (aligned to valid_idx)
    """
    labels = np.zeros(n_samples, dtype=int)
    
    for start, end in anomaly_windows:
        labels[start:end+1] = 1
    
    return labels[valid_idx]


def run_nab_validation():
    """
    Run full real-world validation pipeline on NAB dataset.
    """
    print("\n[Real-World Validation]")
    
    csv_path = config.REAL_WORLD_DATA["csv_path"]
    labels_path = config.REAL_WORLD_DATA["labels_path"]
    
    # Check files exist
    if not os.path.exists(csv_path):
        print(f"[!] CSV not found: {csv_path}")
        return None
    if not os.path.exists(labels_path):
        print(f"[!] Labels not found: {labels_path}")
        return None
    
    print(f"Loading NAB data from {csv_path}")
    df, anomaly_windows = load_nab_data(csv_path, labels_path)
    
    if not anomaly_windows:
        print("[!] No anomaly windows found for this file.")
        return None
    
    print(f"Found {len(anomaly_windows)} anomaly window(s)")
    
    # Create features
    print("Creating sliding-window features...")
    values = df["value"].values
    features, valid_idx = create_nab_features(values, window_size=config.WINDOW_SIZE)
    
    # Ground truth labels
    labels = create_ground_truth_labels(len(values), anomaly_windows, valid_idx)
    
    print(f"Features shape: {features.shape}")
    print(f"Anomaly prevalence: {labels.sum() / len(labels) * 100:.1f}%")
    
    # Train Isolation Forest on normal data only
    print("Training Isolation Forest on normal samples...")
    normal_mask = labels == 0
    if normal_mask.sum() < 10:
        print("[!] Not enough normal samples to train.")
        return None
    
    if_model = IsolationForest(
        n_estimators=config.IF_N_ESTIMATORS,
        contamination=config.IF_CONTAMINATION,
        random_state=config.RANDOM_SEED
    )
    if_model.fit(features[normal_mask])
    
    # Predict on all data
    print("Running predictions...")
    predictions = if_model.predict(features)
    predictions = (predictions == -1).astype(int)  # Convert to binary
    scores = -if_model.score_samples(features)  # Anomaly scores (higher = more anomalous)
    
    # Compute metrics
    print("Computing metrics...")
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # ROC-AUC
    fpr_roc, tpr_roc, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr_roc, tpr_roc)
    
    # Detection delay (if anomaly windows exist)
    detection_delay = None
    if anomaly_windows:
        delays = []
        for start_win, end_win in anomaly_windows:
            # Check if model detected anomaly within this window
            window_preds = predictions[start_win:end_win+1]
            if window_preds.any():
                first_detection = start_win + np.argmax(window_preds)
                delay = first_detection - start_win
                delays.append(delay)
        
        if delays:
            detection_delay = np.mean(delays)
    
    # Build results dict
    results = {
        "Dataset": config.REAL_WORLD_DATASET_NAME,
        "Samples": len(labels),
        "Anomaly Prevalence (%)": labels.sum() / len(labels) * 100,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "FPR": fpr,
        "ROC-AUC": roc_auc,
        "Detection Delay (timesteps)": detection_delay if detection_delay else "N/A",
    }
    
    print("\n" + "=" * 50)
    print("NAB VALIDATION RESULTS")
    print("=" * 50)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"{key:.<35} {val:.4f}")
        else:
            print(f"{key:.<35} {val}")
    
    # Generate visualization
    print("\nGenerating visualization...")
    plot_path = config.REAL_WORLD_DATA["output_plot"]
    plot_nab_results(df, predictions, valid_idx, anomaly_windows, roc_auc, plot_path)
    print(f"[visualization] Saved → {plot_path}")
    
    # Save results CSV
    csv_out = config.REAL_WORLD_DATA["output_results"]
    results_df = pd.DataFrame([results])
    results_df.to_csv(csv_out, index=False)
    print(f"[results] Saved → {csv_out}")
    
    # Save summary
    summary_path = config.REAL_WORLD_DATA["output_summary"]
    save_nab_summary(results, summary_path)
    print(f"[summary] Saved → {summary_path}")
    
    return results


def plot_nab_results(df, predictions, valid_idx, anomaly_windows, roc_auc, save_path):
    """
    Generate publication-ready plot for NAB results.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=180)
    
    # Plot 1: Time series with anomalies
    ax = axes[0]
    ax.plot(df["timestamp"], df["value"], "b-", linewidth=1.5, label="CPU Utilization", alpha=0.8)
    
    # Shade true anomaly windows
    for start, end in anomaly_windows:
        ax.axvspan(df["timestamp"].iloc[start], df["timestamp"].iloc[end], 
                   alpha=0.2, color="red", label="True Anomaly" if start == anomaly_windows[0][0] else "")
    
    # Mark detected anomalies
    detected_timestamps = df["timestamp"].iloc[valid_idx][predictions == 1]
    ax.scatter(detected_timestamps, df["value"].iloc[valid_idx][predictions == 1], 
               color="orange", s=30, alpha=0.7, label="Detected Anomaly", zorder=5)
    
    ax.set_ylabel("CPU Utilization (%)", fontsize=10)
    ax.set_title("NAB Real-World Validation: CPU Utilization Anomaly Detection", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Scores over time
    ax = axes[1]
    scores = np.zeros(len(df))
    scores[valid_idx] = predictions  # Use binary predictions as simple scores
    
    ax.plot(df["timestamp"], scores, "g-", linewidth=1, label="Detected (Binary)", alpha=0.7)
    
    for start, end in anomaly_windows:
        ax.axvspan(df["timestamp"].iloc[start], df["timestamp"].iloc[end], 
                   alpha=0.2, color="red")
    
    ax.set_ylabel("Anomaly Flag", fontsize=10)
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylim([-0.1, 1.1])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_nab_summary(results, output_path):
    """
    Save human-readable summary of NAB validation results.
    """
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NAB REAL-WORLD VALIDATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {results['Dataset']}\n")
        f.write(f"Total Samples Analyzed: {results['Samples']}\n")
        f.write(f"Anomaly Prevalence: {results['Anomaly Prevalence (%)']:.2f}%\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Precision: {results['Precision']:.4f}\n")
        f.write(f"Recall: {results['Recall']:.4f}\n")
        f.write(f"F1-Score: {results['F1-Score']:.4f}\n")
        f.write(f"False Positive Rate: {results['FPR']:.4f}\n")
        f.write(f"ROC-AUC: {results['ROC-AUC']:.4f}\n")
        
        if results['Detection Delay (timesteps)'] != "N/A":
            f.write(f"Average Detection Delay: {results['Detection Delay (timesteps)']:.2f} timesteps\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            "This validation demonstrates LightKubeGuard's effectiveness on real-world "
            "data from the Numenta Anomaly Benchmark (NAB). The model was trained on "
            "normal-only samples and evaluated on the full dataset including labeled "
            "anomaly windows.\n"
        )


if __name__ == "__main__":
    results = run_nab_validation()
    if results:
        print("\n✓ Real-world validation complete.")
    else:
        print("\n[!] Real-world validation skipped (files missing or error).")
