"""
architecture_diagram.py
-----------------------
Generates a colorful, color-coordinated, publication-ready system architecture
diagram for:

  "LightKubeGuard: A Lightweight Machine Learning Framework for
   Detecting Resource Hotspots in Kubernetes-Based Cloud Systems"

Color scheme (each layer has its own hue family):
  ┌─ Green   ─┐  Kubernetes Environment  (infrastructure / source)
  ├─ Blue Teal─┤  Telemetry Monitoring   (data collection)
  ├─ Purple  ─┤  Anomaly Detection (IF)  (ML model)
  ├─ Orange  ─┤  Threshold Baseline      (rule-based comparison)
  └─ Dark Navy┘  Monitoring Output       (decision / result)

Output files (saved to outputs/):
  architecture.png   — 300 DPI raster
  architecture.pdf   — vector for LaTeX

Usage:
  python src/architecture_diagram.py

Customisation:
  • Change per-node colors in the NODES list  (fill / edge / text keys)
  • Change layout in NODES  (cx, cy, w, h)
  • Change connections in EDGES
  • Change DPI / FIG_SIZE at the top
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "architecture.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "architecture.pdf")

FIG_SIZE = (7.0, 7.5)   # inches — wide enough for two-column IEEE
DPI      = 300

# Global background
BG_COLOR    = "#F8F9FA"   # very light warm gray — easier on the eye than white

# Arrow style
ARROW_COLOR = "#37474F"   # dark blue-gray for all connectors
ARROW_LW    = 1.3

# Box dimensions (data coords; canvas is 0-10 × 0-10)
BOX_W  = 4.8    # full-width box
BOX_W2 = 2.15   # half-width box (parallel pair)
BOX_H  = 0.88   # all boxes same height

# ── Per-node color definitions ────────────────────────────────────────────────
#  fill   : box background
#  edge   : border colour (slightly darker shade of fill hue)
#  tcolor : title text colour
#  scolor : subtitle text colour

_THEME = {
    "green": {
        "fill":   "#E8F5E9",   # very light green
        "edge":   "#2E7D32",   # dark forest green
        "tcolor": "#1B5E20",
        "scolor": "#388E3C",
    },
    "teal": {
        "fill":   "#E0F2F1",   # very light teal
        "edge":   "#00695C",   # dark teal
        "tcolor": "#004D40",
        "scolor": "#00796B",
    },
    "purple": {
        "fill":   "#EDE7F6",   # very light purple
        "edge":   "#512DA8",   # deep purple
        "tcolor": "#311B92",
        "scolor": "#673AB7",
    },
    "orange": {
        "fill":   "#FFF3E0",   # very light orange
        "edge":   "#E65100",   # burnt orange
        "tcolor": "#BF360C",
        "scolor": "#F4511E",
    },
    "navy": {
        "fill":   "#E3F2FD",   # very light blue
        "edge":   "#1565C0",   # dark blue
        "tcolor": "#0D47A1",
        "scolor": "#1976D2",
    },
}

# ── Node layout ───────────────────────────────────────────────────────────────
NODES = [
    {
        "id":      "k8s",
        "label":   "Kubernetes Environment",
        "subtitle":"Containerized Workloads",
        "cx": 5.0, "cy": 8.8,
        "w": BOX_W, "h": BOX_H,
        **_THEME["green"],
    },
    {
        "id":      "telemetry",
        "label":   "Telemetry Monitoring",
        "subtitle":"CPU · Memory · Network · Latency · Restarts",
        "cx": 5.0, "cy": 6.75,
        "w": BOX_W, "h": BOX_H,
        **_THEME["teal"],
    },
    {
        "id":      "iforest",
        "label":   "Anomaly Detection",
        "subtitle":"Isolation Forest (ML)",
        "cx": 3.15, "cy": 4.6,
        "w": BOX_W2, "h": BOX_H,
        **_THEME["purple"],
    },
    {
        "id":      "threshold",
        "label":   "Threshold Baseline",
        "subtitle":"Static Rule-Based Alerts",
        "cx": 6.85, "cy": 4.6,
        "w": BOX_W2, "h": BOX_H,
        **_THEME["orange"],
    },
    {
        "id":      "output",
        "label":   "Monitoring Output",
        "subtitle":"Alerts and Detection Results",
        "cx": 5.0, "cy": 2.5,
        "w": BOX_W, "h": BOX_H,
        **_THEME["navy"],
    },
]

# Directed edges — (from_id, to_id)
EDGES = [
    ("k8s",       "telemetry"),
    ("telemetry", "iforest"),
    ("telemetry", "threshold"),
    ("iforest",   "output"),
    ("threshold",  "output"),
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_box(ax: plt.Axes, node: dict) -> None:
    """
    Draw a rounded rectangle box with a colored left accent bar,
    a bold title, and an italic subtitle.
    """
    cx, cy = node["cx"], node["cy"]
    w,  h  = node["w"],  node["h"]
    x, y   = cx - w / 2, cy - h / 2

    # ── shadow (faint, offset rectangle behind the box) ──────────────────────
    shadow = mpatches.FancyBboxPatch(
        (x + 0.04, y - 0.04), w, h,
        boxstyle="round,pad=0.07",
        linewidth=0,
        facecolor="#CFD8DC",
        alpha=0.45,
        zorder=2,
    )
    ax.add_patch(shadow)

    # ── main box ─────────────────────────────────────────────────────────────
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.07",
        linewidth=1.6,
        edgecolor=node["edge"],
        facecolor=node["fill"],
        zorder=3,
    )
    ax.add_patch(box)

    # ── left accent bar (solid strip in edge colour) ──────────────────────────
    bar_w = 0.09
    accent = mpatches.FancyBboxPatch(
        (x, y), bar_w, h,
        boxstyle="round,pad=0.02",
        linewidth=0,
        facecolor=node["edge"],
        alpha=0.85,
        zorder=4,
        clip_on=True,
    )
    ax.add_patch(accent)

    # ── bold title ────────────────────────────────────────────────────────────
    ax.text(
        cx + bar_w * 0.3, cy + h * 0.14,
        node["label"],
        ha="center", va="center",
        fontsize=9.0, fontweight="bold",
        color=node["tcolor"],
        zorder=5,
    )
    # ── italic subtitle ───────────────────────────────────────────────────────
    ax.text(
        cx + bar_w * 0.3, cy - h * 0.20,
        node["subtitle"],
        ha="center", va="center",
        fontsize=7.4, style="italic",
        color=node["scolor"],
        zorder=5,
    )


def draw_arrow(
    ax:       plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
    color:    str  = ARROW_COLOR,
    lw:       float = ARROW_LW,
) -> None:
    """
    Draw a straight directed arrow from (x1,y1) → (x2,y2).
    """
    arrow = FancyArrowPatch(
        posA=(x1, y1), posB=(x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        zorder=2,
        shrinkA=5, shrinkB=5,
    )
    ax.add_patch(arrow)


def _get_node(node_id: str) -> dict:
    for n in NODES:
        if n["id"] == node_id:
            return n
    raise ValueError(f"Node '{node_id}' not found.")


def _connection_points(src: dict, dst: dict):
    """Exit from bottom-centre of src, enter top-centre of dst."""
    sx, sy = src["cx"], src["cy"]
    dx, dy = dst["cx"], dst["cy"]
    sh, dh = src["h"] / 2, dst["h"] / 2

    if abs(sx - dx) < 0.05:           # vertically aligned
        if sy > dy:
            return sx, sy - sh,  dx, dy + dh
        else:
            return sx, sy + sh,  dx, dy - dh
    else:                              # diagonal (fan-out / fan-in)
        return sx, sy - sh,  dx, dy + dh


def draw_legend(ax: plt.Axes) -> None:
    """
    Draw a small color-keyed legend at the bottom of the figure.
    """
    legend_items = [
        ("Kubernetes Infra",  _THEME["green"]["edge"],  _THEME["green"]["fill"]),
        ("Data Collection",   _THEME["teal"]["edge"],   _THEME["teal"]["fill"]),
        ("ML Model",          _THEME["purple"]["edge"], _THEME["purple"]["fill"]),
        ("Rule-Based",        _THEME["orange"]["edge"], _THEME["orange"]["fill"]),
        ("Output / Results",  _THEME["navy"]["edge"],   _THEME["navy"]["fill"]),
    ]

    total_w = len(legend_items) * 1.45
    start_x = 5.0 - total_w / 2

    for i, (label, edge, fill) in enumerate(legend_items):
        lx = start_x + i * 1.45
        ly = 1.55

        swatch = mpatches.FancyBboxPatch(
            (lx - 0.08, ly - 0.13), 0.18, 0.26,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=fill,
            zorder=5,
        )
        ax.add_patch(swatch)

        ax.text(
            lx + 0.15, ly,
            label,
            ha="left", va="center",
            fontsize=6.2, color="#37474F",
            zorder=6,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def generate_architecture_diagram() -> None:
    """Build and save the color-coordinated architecture diagram."""
    _ensure_output_dir(OUTPUT_DIR)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.set_xlim(1.0, 9.0)
    ax.set_ylim(1.2, 10.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── header ───────────────────────────────────────────────────────────────
    ax.text(
        5.0, 10.05,
        "LightKubeGuard - System Architecture",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color="#1A237E",
    )
    ax.text(
        5.0, 9.68,
        "Lightweight ML-Based Anomaly Detection for Kubernetes-Based Cloud Systems",
        ha="center", va="center",
        fontsize=7.8, style="italic",
        color="#455A64",
    )

    # ── dashed grouping box around the two detection boxes ──────────────────
    lf = _get_node("iforest")
    rt = _get_node("threshold")
    pad = 0.22
    gx = lf["cx"] - lf["w"] / 2 - pad
    gy = lf["cy"] - lf["h"] / 2 - pad
    gw = (rt["cx"] + rt["w"] / 2 + pad) - gx
    gh = lf["h"] + 2 * pad

    group_box = mpatches.FancyBboxPatch(
        (gx, gy), gw, gh,
        boxstyle="round,pad=0.05",
        linewidth=1.0, linestyle="dashed",
        edgecolor="#90A4AE",
        facecolor="#ECEFF1",
        alpha=0.55,
        zorder=1,
    )
    ax.add_patch(group_box)

    ax.text(
        (gx + gx + gw) / 2, gy + gh + 0.07,
        "Detection Engine",
        ha="center", va="bottom",
        fontsize=7.2, style="italic",
        color="#607D8B", fontweight="bold",
    )

    # ── draw all nodes ───────────────────────────────────────────────────────
    for node in NODES:
        draw_box(ax, node)

    # ── draw all arrows ──────────────────────────────────────────────────────
    for (src_id, dst_id) in EDGES:
        src = _get_node(src_id)
        dst = _get_node(dst_id)
        x1, y1, x2, y2 = _connection_points(src, dst)

        # Colour the arrow using the destination node's edge colour for clarity
        arrow_col = dst["edge"]
        draw_arrow(ax, x1, y1, x2, y2, color=arrow_col)

    # ── legend ───────────────────────────────────────────────────────────────
    draw_legend(ax)

    # ── thin horizontal rule above legend ────────────────────────────────────
    ax.axhline(y=1.88, xmin=0.05, xmax=0.95,
               color="#B0BEC5", linewidth=0.6, linestyle="--")

    fig.tight_layout(pad=0.5)

    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[architecture] Saved PNG → {OUTPUT_PNG}")

    fig.savefig(OUTPUT_PDF, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[architecture] Saved PDF → {OUTPUT_PDF}")

    plt.close(fig)


if __name__ == "__main__":
    generate_architecture_diagram()
