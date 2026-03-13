"""
visualization.py — Clean, presentation-ready figure generation.

Design principles (IEEE/conference presentation):
  • White background — readable when projected
  • Minimal clutter — signal is always readable
  • Clear anomaly shading — soft red fill with a crisp border
  • Detection shown as compact rug / tick marks — NOT full-height bars
  • Consistent, accessible colour palette (blue / orange colour-blind safe)
  • Larger fonts — legible from the back of the room

Figures generated:
  1. cpu_anomaly_plot.png
  2. latency_anomaly_plot.png
  3. accuracy_comparison.png
  4. detection_delay.png
  5. roc_curve.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve
from typing import Dict, List, Optional

from . import config
from .utils import anomaly_regions_from_labels, ensure_output_dir


# ──────────────────────────────────────────────────────────────────────────────
# PALETTE  (colour-blind safe, IEEE-friendly, projects well)
# ──────────────────────────────────────────────────────────────────────────────

_C = {
    "signal":       "#1565C0",   # strong blue — main time-series line
    "anom_fill":    "#FFCDD2",   # light red fill — true anomaly region
    "anom_edge":    "#C62828",   # dark red edge — anomaly boundary
    "lkg_tick":     "#1565C0",   # IF detection ticks — same blue as signal
    "thresh_tick":  "#E65100",   # threshold ticks — burnt orange
    "bar_lkg":      "#1976D2",   # bar for LightKubeGuard
    "bar_thresh":   "#EF6C00",   # bar for Threshold
    "best_border":  "#2E7D32",   # green highlight for "better" bar
    "grid":         "#EEEEEE",
    "text":         "#212121",
    "subtext":      "#616161",
    "white":        "#FFFFFF",
}

_DPI        = 180
_FONT_TITLE = 14
_FONT_LABEL = 12
_FONT_TICK  = 10
_FONT_ANNOT = 9.5
_LW_SIGNAL  = 1.0
_ALPHA_ANOM = 0.35        # anomaly fill alpha

# Apply globally
_RC = {
    "figure.facecolor":     _C["white"],
    "axes.facecolor":       _C["white"],
    "axes.edgecolor":       "#BDBDBD",
    "axes.labelcolor":      _C["text"],
    "axes.titlecolor":      _C["text"],
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.color":          _C["subtext"],
    "ytick.color":          _C["subtext"],
    "xtick.labelsize":      _FONT_TICK,
    "ytick.labelsize":      _FONT_TICK,
    "legend.facecolor":     _C["white"],
    "legend.edgecolor":     "#BDBDBD",
    "legend.labelcolor":    _C["text"],
    "grid.color":           _C["grid"],
    "grid.linewidth":       0.6,
    "text.color":           _C["text"],
    "font.family":          "DejaVu Sans",
    "font.size":            10,
}


def _apply_style() -> None:
    plt.rcParams.update(_RC)


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    ensure_output_dir(config.OUTPUTS["dir"])
    fig.savefig(
        path,
        dpi=_DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"[visualization] Saved → {path}")


def _shade_anomaly_regions(
    ax: plt.Axes,
    regions: list,
    df: pd.DataFrame,
) -> list:
    """
    Draw a soft filled rectangle for every true anomaly region.
    Returns a list of patch handles for the legend (one entry only).
    """
    scenario_arr = (
        df["scenario_name"].values
        if "scenario_name" in df.columns else None
    )
    handles = []

    for i, (s, e) in enumerate(regions):
        patch = ax.axvspan(
            s,
            e,
            facecolor=_C["anom_fill"],
            edgecolor=_C["anom_edge"],
            linewidth=0.9,
            alpha=_ALPHA_ANOM,
            label="True Anomaly" if i == 0 else "_nolegend_",
            zorder=1,
        )
        if i == 0:
            handles.append(patch)

        # Scenario label — small, centred, non-intrusive
        if scenario_arr is not None:
            mid = (s + e) / 2
            name = str(scenario_arr[min(s + 5, len(scenario_arr) - 1)])
            name_clean = name.replace("_", " ").title()
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            ax.text(
                mid,
                ymax - span * 0.04,
                name_clean,
                ha="center",
                va="top",
                fontsize=8,
                color=_C["anom_edge"],
                style="italic",
                zorder=5,
            )
    return handles


def _rug_detections(
    ax: plt.Axes,
    ts: np.ndarray,
    signal: np.ndarray,
    predictions: np.ndarray,
    color: str,
    label: str,
    row: float,          # y-position in normalised axes coords (for rug)
    height: float = 0.05,
) -> mpatches.Patch:
    """
    Draw compact rug marks (short tick lines) at each detected timestep
    at the bottom of the main axis — NOT full-height bars.
    Returns a proxy artist for the legend.
    """
    det_idx = np.where(predictions == 1)[0]
    if len(det_idx) == 0:
        return mpatches.Patch(color=color, label=label)

    ymin, ymax = signal.min(), signal.max()
    span = ymax - ymin

    # rug at row_frac from bottom
    y0 = ymin + span * row
    y1 = y0 + span * height

    ax.vlines(
        ts[det_idx],
        ymin=y0,
        ymax=y1,
        color=color,
        linewidth=0.9,
        alpha=0.85,
        label=label,
        zorder=4,
    )
    return mpatches.Patch(color=color, label=label)


def _legend(ax: plt.Axes, handles, **kwargs) -> None:
    leg = ax.legend(
        handles=handles,
        fontsize=_FONT_ANNOT,
        framealpha=0.9,
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=1.6,
        **kwargs,
    )
    leg.get_frame().set_linewidth(0.8)


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 & 2 — Time-series with anomaly regions
# ──────────────────────────────────────────────────────────────────────────────

def plot_signal_anomalies(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    ml_predictions_full: np.ndarray,
    save_path: str,
    thresh_predictions_full: Optional[np.ndarray] = None,
) -> None:
    """
    Single-panel time-series figure:
      • Signal line (blue)
      • True anomaly regions (light red shading)
      • LightKubeGuard detections as compact rug marks near the bottom
      • Threshold detections (if provided) as a second rug row below
    """
    _apply_style()

    ts = df["timestep"].values
    signal = df[metric].values
    labels = df["anomaly_label"].values
    regions = anomaly_regions_from_labels(labels)

    has_thresh = thresh_predictions_full is not None

    fig, ax = plt.subplots(figsize=(11, 4.2))

    # ── signal line ─────────────────────────────────────────────────────────
    signal_handle, = ax.plot(
        ts,
        signal,
        color=_C["signal"],
        linewidth=_LW_SIGNAL,
        label=ylabel,
        zorder=3,
        solid_capstyle="round",
    )

    # ── anomaly shading (needs y-limits first) ──────────────────────────────
    ax.autoscale(enable=True, axis="y")
    ax.set_xlim(ts[0], ts[-1])
    anom_handles = _shade_anomaly_regions(ax, regions, df)

    # ── detection rug marks ─────────────────────────────────────────────────
    # Reserve bottom 16 % for rug marks (split between two methods if both)
    if has_thresh:
        lkg_handle = _rug_detections(
            ax,
            ts,
            signal,
            ml_predictions_full,
            color=_C["lkg_tick"],
            label="LightKubeGuard (IF) detections",
            row=0.09,
            height=0.04,
        )
        thr_handle = _rug_detections(
            ax,
            ts,
            signal,
            thresh_predictions_full,
            color=_C["thresh_tick"],
            label="Threshold detections",
            row=0.02,
            height=0.04,
        )
        legend_handles = [signal_handle] + anom_handles + [lkg_handle, thr_handle]
    else:
        lkg_handle = _rug_detections(
            ax,
            ts,
            signal,
            ml_predictions_full,
            color=_C["lkg_tick"],
            label="LightKubeGuard (IF) detections",
            row=0.03,
            height=0.05,
        )
        legend_handles = [signal_handle] + anom_handles + [lkg_handle]

    # ── labels & formatting ─────────────────────────────────────────────────
    ax.set_xlabel("Timestep", fontsize=_FONT_LABEL, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=_FONT_LABEL, labelpad=6)
    ax.set_title(title, fontsize=_FONT_TITLE, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    _legend(ax, legend_handles, loc="upper left")

    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Detection accuracy bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(
    results: List[Dict],
    save_path: str = config.OUTPUTS["accuracy_plot"],
) -> None:
    """
    Clean grouped vertical bar chart: Precision, Recall, F1.
    Each method gets its own colour; value labels sit above bars.
    """
    _apply_style()

    metrics = ["precision", "recall", "f1_score"]
    metric_labels = ["Precision", "Recall", "F1-Score"]
    colors = [_C["bar_lkg"], _C["bar_thresh"]]
    n_metrics = len(metrics)
    n_methods = len(results)

    x = np.arange(n_metrics)
    bar_w = 0.30
    offsets = np.linspace(
        -(n_methods - 1) * bar_w / 2,
        (n_methods - 1) * bar_w / 2,
        n_methods,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (result, color) in enumerate(zip(results, colors)):
        vals = [result[m] for m in metrics]
        bars = ax.bar(
            x + offsets[i],
            vals,
            bar_w,
            color=color,
            alpha=0.88,
            label=result["method"],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.018,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=_FONT_ANNOT,
                color=_C["text"],
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=_FONT_LABEL)
    ax.set_ylabel("Score  (0 – 1)", fontsize=_FONT_LABEL, labelpad=6)
    ax.set_ylim(0, 1.22)
    ax.axhline(1.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax.set_title(
        "Detection Accuracy Comparison",
        fontsize=_FONT_TITLE,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=_FONT_ANNOT, framealpha=0.9, loc="upper right")

    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Detection delay & FPR
# ──────────────────────────────────────────────────────────────────────────────

def plot_detection_delay(
    results: List[Dict],
    save_path: str = config.OUTPUTS["delay_plot"],
) -> None:
    """
    Two side-by-side vertical bar charts:
      Left  — Average detection delay (timesteps) — lower is better
      Right — False Positive Rate — lower is better
    The better bar gets a green border annotation.
    """
    _apply_style()

    sub_metrics = ["avg_detection_delay", "fpr"]
    sub_labels = ["Avg. Detection Delay\n(timesteps)", "False Positive\nRate"]
    colors = [_C["bar_lkg"], _C["bar_thresh"]]
    methods = [r["method"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.suptitle(
        "Operational Efficiency: LightKubeGuard vs. Threshold Baseline",
        fontsize=_FONT_TITLE,
        fontweight="bold",
        y=1.02,
    )

    for col, (sm, sl) in enumerate(zip(sub_metrics, sub_labels)):
        ax = axes[col]
        vals = [r[sm] for r in results]

        bars = ax.bar(
            methods,
            vals,
            color=colors,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
            width=0.45,
            zorder=3,
        )

        # value labels
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(vals) * 0.04,
                f"{val:.3f}" if sm == "fpr" else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=_FONT_LABEL,
                color=_C["text"],
                fontweight="bold",
            )

        # highlight the better (lower) bar
        better_idx = int(np.argmin(vals))
        bars[better_idx].set_edgecolor(_C["best_border"])
        bars[better_idx].set_linewidth(2.2)
        bx = bars[better_idx].get_x()
        bw = bars[better_idx].get_width()
        ax.text(
            bx + bw / 2,
            max(vals) * 1.25,
            "★ Better",
            ha="center",
            va="bottom",
            fontsize=9,
            color=_C["best_border"],
            fontweight="bold",
        )

        ax.set_ylabel(sl, fontsize=_FONT_LABEL, labelpad=6)
        ax.set_ylim(0, max(vals) * 1.55 + 0.02)
        ax.set_title(
            sl.replace("\n", " "),
            fontsize=_FONT_TITLE - 1,
            fontweight="bold",
            pad=10,
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.tick_params(axis="x", labelsize=_FONT_LABEL)
        ax.annotate(
            "↓ Lower is better",
            xy=(0.98, 0.96),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8,
            color=_C["subtext"],
            style="italic",
        )

    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — ROC Curve for Isolation Forest
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc_value: float,
    save_path: str,
) -> None:
    """
    Generate a publication-ready ROC curve for the Isolation Forest model.
    """
    _apply_style()

    # Higher score should mean more likely anomaly for ROC
    fpr, tpr, _ = roc_curve(y_true, -y_score)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)

    # Random baseline
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1.5,
        color=_C["subtext"],
        alpha=0.6,
        label="Random Classifier",
        zorder=1,
    )

    # ROC curve
    ax.plot(
        fpr,
        tpr,
        linewidth=2.8,
        color=_C["bar_lkg"],
        label=f"LightKubeGuard (AUC = {auc_value:.4f})",
        zorder=3,
    )

    ax.set_xlabel("False Positive Rate", fontsize=_FONT_LABEL, labelpad=6)
    ax.set_ylabel("True Positive Rate", fontsize=_FONT_LABEL, labelpad=6)

    # No big title; caption in paper explains the figure
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)

    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.5, zorder=0)

    leg = ax.legend(
        loc="lower right",
        fontsize=_FONT_ANNOT,
        framealpha=0.92,
        edgecolor="#BDBDBD",
        borderpad=0.7,
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    _save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_figures(
    df: pd.DataFrame,
    ml_predictions_full: np.ndarray,
    results: List[Dict],
    thresh_predictions_full: Optional[np.ndarray] = None,
    if_scores_aligned: Optional[np.ndarray] = None,
    labels_aligned: Optional[np.ndarray] = None,
) -> None:
    """Generate and save all presentation-ready figures."""
    plot_signal_anomalies(
        df,
        "cpu",
        "CPU Utilisation (%)",
        "CPU Utilisation - Anomaly Detection",
        ml_predictions_full,
        config.OUTPUTS["cpu_plot"],
        thresh_predictions_full=thresh_predictions_full,
    )

    plot_signal_anomalies(
        df,
        "latency",
        "Request Latency (ms)",
        "Request Latency - Anomaly Detection",
        ml_predictions_full,
        config.OUTPUTS["latency_plot"],
        thresh_predictions_full=thresh_predictions_full,
    )

    plot_accuracy_comparison(results)
    plot_detection_delay(results)

    # Plot ROC curve if scores and labels are provided
    if if_scores_aligned is not None and labels_aligned is not None:
        auc_value = 0.0
        for r in results:
            if r.get("method", "").lower().startswith("lightkubeguard"):
                auc_value = r.get("roc_auc", 0.0)
                break

        if auc_value:
            plot_roc_curve(
                labels_aligned,
                if_scores_aligned,
                auc_value,
                config.OUTPUTS["roc_curve_plot"],
            )