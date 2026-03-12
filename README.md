# LightKubeGuard

**LightKubeGuard** is a lightweight, modular Python research framework that detects
resource hotspots in Kubernetes-style cloud telemetry using an Isolation Forest anomaly
detector — compared against a static threshold-based baseline.

The project is designed to be fully reproducible and publication-ready, targeting an
IEEE-style research paper.

---

## Project Structure

```
LightKubeGuard/
│
├── notebooks/
│   └── experiment_pipeline.ipynb   # notebook demo of the full pipeline
│
├── src/
│   ├── config.py                   # centralised configuration
│   ├── data_generator.py           # synthetic Kubernetes telemetry
│   ├── feature_engineering.py      # sliding-window feature extraction
│   ├── anomaly_model.py            # Isolation Forest detector
│   ├── threshold_baseline.py       # rule-based threshold detector
│   ├── evaluation.py               # metrics, per-scenario, paper summary
│   ├── visualization.py            # publication-ready figures
│   ├── utils.py                    # shared helpers
│   └── main.py                     # end-to-end pipeline entry point
│
├── outputs/                        # generated automatically on first run
│   ├── cpu_anomaly_plot.png
│   ├── latency_anomaly_plot.png
│   ├── accuracy_comparison.png
│   ├── detection_delay.png
│   ├── lightkubeguard_metrics.csv
│   ├── lightkubeguard_results.csv
│   ├── lightkubeguard_results_by_scenario.csv
│   └── paper_results_summary.txt
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone / download the project

```bash
cd LightKubeGuard
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Pipeline

From the `LightKubeGuard/` directory:

```bash
python -m src.main
```

This single command will:
1. Generate 1 000 timesteps of synthetic Kubernetes telemetry
2. Extract 20-dimensional sliding-window features
3. Train an Isolation Forest on normal windows only
4. Run the threshold-based baseline on the full dataset
5. Evaluate both methods (Precision, Recall, F1, FPR, Detection Delay, ROC-AUC)
6. Generate four publication-ready PNG figures
7. Save all CSV results and a paper summary text file

---

## Run the Notebook

```bash
jupyter notebook notebooks/experiment_pipeline.ipynb
```

Run all cells top-to-bottom. The notebook calls the same `src/` modules as the script
and displays all figures inline.

To run headlessly (e.g. for CI):

```bash
jupyter nbconvert --to notebook --execute notebooks/experiment_pipeline.ipynb \
    --output notebooks/experiment_pipeline_executed.ipynb
```

---

## Expected Outputs

| File | Description |
|------|-------------|
| `outputs/lightkubeguard_metrics.csv` | Raw simulated telemetry with labels |
| `outputs/lightkubeguard_results.csv` | Aggregate metrics for both methods |
| `outputs/lightkubeguard_results_by_scenario.csv` | Per-scenario metrics |
| `outputs/paper_results_summary.txt` | IEEE-paper-ready summary |
| `outputs/cpu_anomaly_plot.png` | CPU signal with detections overlaid |
| `outputs/latency_anomaly_plot.png` | Latency signal with detections overlaid |
| `outputs/accuracy_comparison.png` | Precision/Recall/F1 bar chart |
| `outputs/detection_delay.png` | Detection delay bar chart |
| `outputs/roc_curve.png` | ROC curve for Isolation Forest |

---

## Modifying the Pipeline

All key parameters live in **`src/config.py`**. Common adjustments:

| What to change | Where |
|----------------|-------|
| Number of timesteps | `N_TIMESTEPS` |
| Sliding window size | `WINDOW_SIZE` |
| Anomaly scenarios | `ANOMALY_SCENARIOS` list |
| Threshold rules | `THRESHOLDS` dict |
| Isolation Forest params | `IF_N_ESTIMATORS`, `IF_CONTAMINATION` |
| Output locations | `OUTPUTS` dict |

After editing `config.py`, re-run `python -m src.main` — no other files need changing.

### Swapping the ML model

`anomaly_model.py` exposes `IsolationForestDetector` with `.fit()` / `.predict()` / `.score_samples()` methods. Implement the same interface to plug in another detector (e.g. One-Class SVM, LOF).

### Adding new metrics or features

1. Add the metric to `data_generator.py` baseline signals and `BASELINE` in `config.py`.  
2. Add its name to `FEATURE_METRICS` in `feature_engineering.py`.  
3. Re-run — features and downstream evaluation update automatically.

---

## Reproducibility

- Global `RANDOM_SEED = 42` controls all stochastic elements.
- All reported metrics are computed from simulated data — no values are fabricated.

---

## Dependencies

```
numpy, pandas, scikit-learn, matplotlib, jupyter
```

No seaborn; no notebook-only logic.
