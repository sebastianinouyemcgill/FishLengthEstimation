# FishNet CV — Fish Length Estimation

Estimates **fish standard length (SL, mm)** from RGB images and **YOLO polygon** annotations (fish + blue/yellow 100 mm calibration markers).

## Pipeline

```
polygons → fish mask → marker calibration → skeleton length (mm)
                              ↓ optional
                    sklearn regression → corrected length (mm)
```

| Stage | Method | Status |
|-------|--------|--------|
| Geometry | **Skeleton** (also bbox, PCA) | Production baseline |
| Scale | Marker rectangles (100 mm) | Default |
| Correction | RandomForest on mask features | Optional |
| Grid / depth / 3D | — | Experimental, **off** by default |

**Class IDs:** 0 = blue marker, 1 = yellow marker, 2 = fish.

Further detail: [`docs/architecture.md`](docs/architecture.md) · defaults: [`config.yaml`](config.yaml)

---

## Results (validation set, n=30)

| Ground truth CSV | Skeleton MAE | Regression MAE |
|------------------|--------------|----------------|
| `validation_lengths` | 51.7 mm | 22.8 mm |
| `validation_lengths2` | 31.2 mm | 23.9 mm |

`validation_lengths2` replaced an outlier (image 2572). Regression is trained on marker-calibrated skeleton features.

---

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"   # optional: pytest, ruff
pytest tests/ -v
```

**Python 3.11+** · core deps: numpy, opencv, scikit-image, scikit-learn, pandas, jupyter.

### Dataset (local, gitignored)

```
data/fishnet/
├── images/{train,valid,test}/
└── labels/{train,valid,test}/

data/annotations/
├── validation_lengths.csv    # manual SL lengths (notebook 02)
└── validation_lengths2.csv   # corrected labels
```

YOLO labels: `class_id x1 y1 x2 y2 ...` (normalized polygon vertices).

---

## Workflow (notebooks)

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_dataset_exploration.ipynb` | Verify paths, masks, polygons |
| 02 | `02_annotation_helper.ipynb` | Export manual length CSVs |
| 03 | `03_run_experiments.ipynb` | **Configure → run → verify** |
| 04 | `04_analyze_results.ipynb` | MAE/RMSE tables and plots |

Each notebook uses `setup_notebook_environment()` from `src.colab_bootstrap` (works locally and on Colab/Drive).

### Notebook 03 — key settings (`RUN_CFG`)

```python
pipelines=["baseline"]
methods=["skeleton"]
splits=["valid"]
ground_truth_source="validation_lengths2"   # or validation_lengths

# Train new regression model:
run_regression_calibration=True

# Or apply saved model (baseline only):
use_regression_model=True
regression_model_path=REPO / "outputs/runs/regression_skeleton_validation_lengths2/regression_model.joblib"
```

### Test submission (`predictions.csv`)

Final format — `image_id` **with extension**, `predicted_length_mm`:

```bash
python scripts/export_test_predictions.py \
  --model outputs/runs/regression_skeleton_validation_lengths2/regression_model.joblib \
  --output predictions.csv
```

---

## Project structure

```
fishnet_cv/
├── config.yaml              # method, calibration, regression, experimental flags
├── predictions.csv          # test-set submission (committed)
├── notebooks/               # 01–04 workflow
├── src/
│   ├── dataset.py, masks.py, evaluation.py, submission.py
│   ├── methods/
│   │   ├── geometric/       # bbox, PCA, skeleton
│   │   ├── calibration/   # marker (default), grid (experimental)
│   │   ├── regression/      # sklearn model + features
│   │   └── estimator.py     # FishLengthEstimator
│   ├── pipelines/           # experiment runners
│   └── experiments/         # run_experiment API
├── scripts/export_test_predictions.py
└── tests/                   # metric reproduction guards
```

### `FishLengthEstimator` (programmatic API)

```python
from src.config import get_config
from src.methods import FishLengthEstimator

cfg = get_config()
cfg.measurement_method = "skeleton"
cfg.use_regression_model = True
cfg.regression_model_path = "outputs/runs/.../regression_model.joblib"

df = FishLengthEstimator(cfg).run_batch(split="valid")
```

---

## Experiment outputs

Each run: `outputs/runs/<run_name>/`

| File | Content |
|------|---------|
| `predictions.csv` | Lengths per image |
| `comparison.csv` | Per-image error vs ground truth |
| `metrics.json` | MAE, RMSE |
| `regression_model.joblib` | Trained model (regression runs only) |

Registry: `outputs/runs/experiments.csv`

---

## Configuration

| Source | Use |
|--------|-----|
| `config.yaml` | Defaults (`method: skeleton`, experimental off) |
| `ProjectConfig` / `RunExperimentsConfig` | Notebooks and scripts |
| `ground_truth_source` | `validation_lengths` \| `validation_lengths2` \| `lengths_mm` |

Experimental modules (`grid_auto`, `depth/`, `advanced` pipeline) remain in the repo for research but are disabled in `config.yaml` and not part of the report pipeline.

---

## Google Colab

Mount Drive, then run notebook cell 1 — paths resolve to `UH_CV/` on Drive via `src/paths.py`. Same notebooks and `RUN_CFG` as local.

---

## Legacy note

An earlier flat layout (`outputs/predictions/`, `python main.py` without run directories) is still supported in code for compatibility. The presentation workflow uses **notebooks 01–04** and `outputs/runs/` only.
