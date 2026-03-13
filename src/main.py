"""
main.py — LightKubeGuard end-to-end pipeline entry point.

Run from the project root:
    python -m src.main
or:
    python src/main.py
"""

import sys
import os

# Allow running as a script (python src/main.py) without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src import config
from src.utils import ensure_output_dir, anomaly_regions_from_labels
from src.data_generator import generate_telemetry
from src.feature_engineering import extract_features
from src.anomaly_model import train_and_predict
from src.threshold_baseline import ThresholdDetector
from src.evaluation import (
    compute_metrics,
    evaluate_by_scenario,
    save_results,
    save_scenario_results,
    write_paper_summary,
)
from src.visualization import generate_all_figures
from src.real_world_validation import run_nab_validation


def run_pipeline() -> None:
    ensure_output_dir(config.OUTPUTS["dir"])

    print("\n" + "=" * 60)
    print("  LightKubeGuard — Anomaly Detection Pipeline")
    print("=" * 60)

    # ── STEP 1: Generate telemetry ────────────────────────────────
    print("\n[1/7] Generating synthetic Kubernetes telemetry …")
    df = generate_telemetry()
    print(f"      {len(df)} timesteps generated. "
          f"Anomaly rate: {df['anomaly_label'].mean()*100:.1f}%")

    # ── STEP 2: Feature engineering ───────────────────────────────
    print("\n[2/7] Extracting sliding-window features …")
    features_df, scaler, valid_index = extract_features(df)
    print(f"      Feature matrix: {features_df.shape[0]} rows × {features_df.shape[1]} cols")

    # Align ground-truth labels to the valid (post-NaN-drop) index
    labels_aligned = df["anomaly_label"].values[valid_index]

    # ── STEP 3: Train Isolation Forest ───────────────────────────
    print("\n[3/7] Training Isolation Forest (normal windows only) …")
    if_predictions, if_scores, detector = train_and_predict(features_df, labels_aligned)
    print(f"      IF detections: {if_predictions.sum()} flagged out of {len(if_predictions)} windows")

    # Map IF predictions back to full df timeline (undetected rows = 0)
    if_predictions_full = np.zeros(len(df), dtype=int)
    for j, orig_idx in enumerate(valid_index):
        if_predictions_full[orig_idx] = if_predictions[j]

    # ── STEP 4: Threshold baseline ────────────────────────────────
    print("\n[4/7] Running threshold-based baseline …")
    thresh_detector  = ThresholdDetector()
    thresh_preds_full = thresh_detector.predict(df)
    print(f"      Threshold detections: {thresh_preds_full.sum()} flagged out of {len(df)} timesteps")

    # ── STEP 5: Evaluate both methods ────────────────────────────
    print("\n[5/7] Evaluating detection performance …")
    y_true_full = df["anomaly_label"].values
    regions     = anomaly_regions_from_labels(y_true_full)

    result_if = compute_metrics(
        y_true_full, if_predictions_full,
        y_score=None,   # scores aligned to valid_index; skip global AUC for simplicity
        regions=regions,
        method_name="LightKubeGuard",
    )

    # For IF, compute AUC using aligned labels/scores
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels_aligned, -if_scores)
        result_if["roc_auc"] = round(auc, 4)
        roc_computed = True
    except Exception:
        roc_computed = False

    result_thresh = compute_metrics(
        y_true_full, thresh_preds_full,
        regions=regions,
        method_name="Threshold",
    )

    results = [result_if, result_thresh]
    save_results(results)

    # Per-scenario evaluation
    df_scenario = evaluate_by_scenario(df, if_predictions, thresh_preds_full, valid_index)
    save_scenario_results(df_scenario)
    scenario_computed = True

    write_paper_summary(results, scenario_computed, roc_computed)

    # ── STEP 6: Generate figures ──────────────────────────────────
    print("\n[6/7] Generating publication-ready figures …")
    generate_all_figures(df, if_predictions_full, results,
                         thresh_predictions_full=thresh_preds_full,
                         if_scores_aligned=if_scores,
                         labels_aligned=labels_aligned)

    # ── STEP 7: Run summary ───────────────────────────────────────
    print("\n[7/7] Pipeline complete.")
    print("\n" + "─" * 60)
    print("  RESULTS SUMMARY")
    print("─" * 60)
    _fmt = "{:<28} {:>12} {:>12}"
    print(_fmt.format("Metric", "LightKubeGuard", "Threshold"))
    print("─" * 60)
    for key, label in [
        ("precision",          "Precision"),
        ("recall",             "Recall"),
        ("f1_score",           "F1-Score"),
        ("fpr",                "False Positive Rate"),
        ("avg_detection_delay","Avg. Detect. Delay"),
    ]:
        print(_fmt.format(label, str(result_if[key]), str(result_thresh[key])))
    if roc_computed:
        print(_fmt.format("ROC-AUC", str(result_if.get("roc_auc", "N/A")), "N/A"))
    print("─" * 60)

    print("\n  Saved outputs:")
    for key, path in config.OUTPUTS.items():
        if key == "dir":
            continue
        if os.path.isfile(path):
            print(f"    ✓ {path}")

    print("\n" + "=" * 60 + "\n")

    # ── Optional: Real-World Validation ───────────────────────────
    if config.ENABLE_REAL_WORLD_VALIDATION:
        print("\n" + "=" * 60)
        print("  Real-World Validation (NAB Dataset)")
        print("=" * 60)
        nab_results = run_nab_validation()
        if nab_results:
            print("\n✓ Real-world validation complete.")
        else:
            print("\n[!] Real-world validation incomplete (check file paths).")
    else:
        print("\n[Real-World Validation] Disabled in config.ENABLE_REAL_WORLD_VALIDATION")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()
