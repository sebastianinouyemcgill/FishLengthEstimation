# Architecture (presentation-ready)

Fish length (SL, mm) from RGB images + YOLO fish/marker polygons.

## Recommended pipeline

```
image + polygons → mask → marker calibration → skeleton length (mm)
                              ↓ optional
                    feature vector → sklearn regression → corrected length (mm)
```

**Production default:** `skeleton` + `marker` calibration.  
**Optional:** regression correction trained on manual `validation_lengths*.csv`.  
**Experimental (OFF by default):** grid auto-calibration, monocular depth, 3D skeleton.

## Module layout

| Package | Role |
|---------|------|
| `src/dataset.py`, `src/masks.py` | Core I/O, polygon → mask |
| `src/methods/geometric/` | bbox, PCA, **skeleton** (baseline) |
| `src/methods/calibration/` | marker (default), grid (experimental) |
| `src/methods/regression/` | sklearn model + mask features |
| `src/methods/depth/` | depth + 3D (experimental, isolated) |
| `src/methods/estimator.py` | `FishLengthEstimator` unified API |
| `src/evaluation.py` | MAE, RMSE, comparison CSV |
| `src/experiments/` | `run_experiment`, notebooks, 2×2 grids |
| `src/pipelines/` | Legacy runners (unchanged numerics) |

Legacy import paths (`src.measurement`, `src.calibration`, `src.models`) remain valid.

## Configuration

- **`config.yaml`** — human-readable defaults (method, calibration, regression, experimental flags)
- **`ProjectConfig`** / **`RunExperimentsConfig`** — runtime (notebooks, CLI)
- Environment: `FISHNET_METHOD`, `FISHNET_GRID_AUTO`, `FISHNET_ALLOW_EXPERIMENTAL`, etc.

## Experiment outputs

Each run: `outputs/runs/<run_name>/`

- `predictions.csv`, `comparison.csv`, `metrics.json`
- Regression runs: `regression_model.joblib`, `train_features.csv`
- Summary: `outputs/runs/comparison_grid_summary.csv` (2×2 GT × method grid)

## Notebooks

| Notebook | Purpose |
|----------|---------|
| 01 | Dataset exploration (unchanged imports) |
| 02 | Validation length labels → `data/annotations/*.csv` |
| 03 | `run_experiment` / `run_configured_experiments` / optional 4-way grid |
| 04 | MAE/RMSE plots, per-image debug |

## Metric regression tests

`tests/test_metric_reproduction.py` locks MAE for the 2×2 benchmark (~22.8 / ~23.9 mm regression, ~31.2 mm skeleton on `validation_lengths2`).
