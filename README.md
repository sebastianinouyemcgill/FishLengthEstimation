# FishNet CV — Fish Length Estimation

Research-oriented Python scaffold for estimating **physical fish length (mm)** from RGB images and **YOLO polygon** annotations (fish, blue calibration rectangles, yellow calibration rectangles). Geometry and calibration follow the course assignment; see **`assignment.pdf`** (local only, gitignored) for full requirements.

## Goals

| Input | Output |
|--------|--------|
| RGB fish images | `predictions.csv`: `image_id`, `predicted_length_mm` |
| YOLO seg polygons: fish, blue/yellow 100 mm markers | Optional figures, metrics (MAE, RMSE) |

**Class IDs** (from `data/fishnet.yaml` when using the FishNet archive):

| ID | Name |
|----|------|
| 0 | blue calibration rectangle |
| 1 | yellow calibration rectangle |
| 2 | fish |

Calibration rectangles are assumed to be **100 mm** long (`CALIBRATION_RECT_MM` in `src/config.py`).

---

## Requirements

- **Python 3.11+**
- See `requirements.txt` or install with `pip install -e .`

Core libraries: `numpy`, `opencv-python`, `matplotlib`, `pandas`, `scikit-image`, `scikit-learn`, `scipy`, `tqdm`, `PyYAML`, `jupyter`.

Optional (commented in `requirements.txt`): `ultralytics`, `torch`, `transformers`.

---

## Environment setup

```bash
cd fishnet_cv
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# Or editable install:
pip install -e ".[dev]"
```

Register the Jupyter kernel (optional):

```bash
python -m ipykernel install --user --name fishnet-cv --display-name "FishNet CV"
```

Run tests:

```bash
pytest tests/ -v
```

---

## Dataset placement

The repo may contain data locally, but **`data/` and `assignment.pdf` are gitignored**.

### Recommended layout

```
data/
├── raw/
│   └── fishnet/              # extracted dataset
│       ├── images/
│       │   ├── train/
│       │   ├── valid/        # or val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── valid/
│           └── test/
├── processed/                # cached masks, rectified images
└── annotations/              # ground-truth CSV for evaluation
    └── lengths_mm.csv        # columns: image_id, length_mm
```

### If you already have `data/fishnet/`

`src/config.py` auto-detects `data/fishnet` when `data/raw/fishnet` is absent (current clone layout).

Extract the archive if needed:

```bash
tar -xzf data/fishnet_dataset.tar.gz -C data/raw/
# Ensure images/ and labels/ exist under data/raw/fishnet/
```

Label format: one line per instance — `class_id x1 y1 x2 y2 ...` with **normalized** polygon coordinates (YOLO segmentation).

---

## Example commands

### Legacy mode (unchanged)

Writes to `outputs/predictions/predictions.csv` — existing notebooks still work:

```bash
python main.py --split test --method pca
python main.py --split test --method skeleton --limit 5 --visualize
```

### Managed experiment runs (recommended)

Each run gets an isolated directory under `outputs/runs/<run_name>/`:

```bash
python main.py --pipeline baseline --method bbox --run-name baseline_bbox_v1 --split valid
python main.py --pipeline advanced --method pca --perspective --run-name advanced_pca_rect_v1 --split valid
python main.py --pipeline baseline --method pca --run-name baseline_pca_v1 --overwrite  # replace existing
```

Auto-generated run name when omitted:

```bash
python main.py --pipeline baseline --method pca --split valid
# → outputs/runs/baseline_pca_2026-05-30_143500/
```

### Notebook-first API (primary workflow)

```python
from src.experiments import run_experiment, run_experiments

run = run_experiment(
    pipeline="baseline",
    method="bbox",
    split="valid",
    run_name="baseline_bbox_v1",
)

results_df = run_experiments([
    {"pipeline": "baseline", "method": "bbox", "run_name": "baseline_bbox_v1", "split": "valid"},
    {"pipeline": "baseline", "method": "pca", "run_name": "baseline_pca_v1", "split": "valid"},
    {"pipeline": "advanced", "method": "pca", "perspective": True, "run_name": "advanced_pca_rect_v1"},
])
```

See `notebooks/03_run_experiments.ipynb` and `notebooks/04_analyze_results.ipynb`.

Environment overrides (legacy mode):

```bash
export FISHNET_SPLIT=test
export FISHNET_METHOD=pca
export FISHNET_PERSPECTIVE=1
python main.py
```

**Evaluation** (legacy path — still supported):

```python
from pathlib import Path
from src.evaluation import evaluate_run

evaluate_run(
    ground_truth_path=Path("data/annotations/validation_lengths.csv"),
    predictions_path=Path("outputs/predictions/predictions.csv"),
    method="pca",
    split="valid",
)
```

**Evaluation** (run directory):

```python
evaluate_run(
    ground_truth_path=Path("data/annotations/validation_lengths.csv"),
    predictions_path=Path("outputs/runs/baseline_bbox_v1/predictions.csv"),
    output_dir=Path("outputs/runs/baseline_bbox_v1"),
)
```

---

See [`docs/repository_overview.md`](docs/repository_overview.md) for a concise file-by-file map and storage layout.

## Repository architecture

```
fishnet_cv/
├── data/                 # gitignored — raw, processed, annotations
├── notebooks/            # exploratory + experiment workflows
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── masks.py
│   ├── calibration.py
│   ├── measurement.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── utils.py
│   ├── pipelines/        # baseline vs advanced runners
│   │   ├── base.py       # shared inference loop
│   │   ├── baseline.py
│   │   ├── advanced.py
│   │   └── registry.py
│   └── experiments/      # RunManager + notebook API
│       ├── run_manager.py
│       ├── results.py
│       └── __init__.py   # run_experiment, run_experiments
├── outputs/
│   ├── runs/             # isolated experiment runs
│   │   ├── experiments.csv   # master registry (append-only)
│   │   └── <run_name>/
│   ├── predictions/      # legacy flat output
│   ├── figures/
│   └── metrics/
├── tests/
├── main.py               # CLI (legacy + managed runs)
├── requirements.txt
└── pyproject.toml
```

### Module responsibilities

| Module | Role |
|--------|------|
| `dataset.py` | Recursive image/label discovery, `DatasetSample`, splits |
| `masks.py` | Rasterization, morphology, skeleton, overlays |
| `calibration.py` | Marker scale (px/mm), optional `findHomography` + warp |
| `measurement.py` | Pluggable length methods; `register_measurement_method()` |
| `evaluation.py` | CSV merge, MAE/RMSE; optional `output_dir` per run |
| `visualization.py` | Overlays, PCA axis, homography plots |
| `pipelines/` | `BaselinePipeline` (geometry only) vs `AdvancedPipeline` (perspective, etc.) |
| `experiments/` | `RunManager`, `run_experiment()`, registry CSV |

---

## Experiment workflow (notebook-first)

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `dataset_exploration.ipynb` | Verify data, masks, polygons |
| 2 | `annotation_helper.ipynb` | Manual lengths → `validation_lengths.csv` |
| 3 | **`03_run_experiments.ipynb`** | Run baseline + advanced experiments |
| 4 | **`04_analyze_results.ipynb`** | Compare MAE/RMSE, per-image errors |

Legacy notebooks (`baseline_measurement.ipynb`, `perspective_correction.ipynb`, `evaluation.ipynb`) still work unchanged.

### Baseline vs advanced

| | Baseline | Advanced (legacy) | Advanced (modern) |
|---|----------|-------------------|-------------------|
| **Pipeline** | `pipeline="baseline"` | `pipeline="advanced"` | `pipeline="advanced"` + flags |
| **Methods** | bbox, pca, skeleton | bbox, pca, skeleton | skeleton recommended for 3D |
| **Calibration** | blue/yellow markers | markers (+ optional perspective) | grid auto (no markers) |
| **Perspective** | always off | optional `perspective=True` | disabled when grid/depth/3D on |
| **Depth / 3D** | off | off | optional Depth Anything V3 + 3D arc length |

Baseline always uses `src/pipelines/base.py` (`run_inference`). Advanced uses the same loop only when **all** of `use_grid_auto_calibration`, `use_depth_estimation`, and `use_3d_measurement` are false; otherwise `src/pipelines/advanced_inference.py`.

---

## Advanced pipeline (grid + depth + 3D)

Bypasses fiduciary markers when grid auto-calibration is enabled. Stages:

1. Fish mask from YOLO polygons (`src/masks.py`)
2. 2D skeleton (`skeletonize_mask`)
3. Grid spacing via Hough lines + clustering (`src/calibration/grid_auto.py`)
4. Depth Anything V3 with disk cache (`src/depth/`)
5. 3D skeleton arc length (`src/measurement/skeleton3d.py`)

Config flags (all default **False** — baseline unchanged):

| Flag | Env var | Meaning |
|------|---------|---------|
| `use_grid_auto_calibration` | `FISHNET_GRID_AUTO=1` | `pixels_per_mm` from tank grid |
| `use_depth_estimation` | `FISHNET_DEPTH=1` | Cached depth maps under `data/processed/depth/` |
| `use_3d_measurement` | `FISHNET_3D=1` | 3D arc length (requires depth) |

Set physical grid cell size with `grid_square_mm` (default 10 mm) or `FISHNET_GRID_SQUARE_MM`.

### Manual install (depth)

**macOS note:** DA3 lists `xformers` as a dependency, but it is **not required** (DA3 falls back to pure PyTorch SwiGLU). Do **not** install xformers on Apple Silicon unless you use a prebuilt wheel — building from source fails with `clang++: unsupported option '-fopenmp'`.

Use the install script (recommended):

```bash
chmod +x scripts/install_advanced_deps.sh
./scripts/install_advanced_deps.sh
export KMP_DUPLICATE_LIB_OK=TRUE   # avoids OpenMP duplicate-runtime abort on macOS
```

Or manually:

```bash
pip install torch>=2.1 torchvision
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git --no-deps
pip install "numpy<2" einops huggingface_hub imageio opencv-python pillow omegaconf \
  safetensors typer requests trimesh e3nn addict evo "moviepy==1.0.3" plyfile pycolmap
```

First run downloads Hugging Face weights (default: `depth-anything/DA3-SMALL` ~80M params). Set `FISHNET_DEPTH_MODEL=depth-anything/DA3-BASE` if you have enough disk space.

### Example commands (compare ablations)

```bash
# Baseline (unchanged)
python main.py --pipeline baseline --method skeleton --run-name baseline_skeleton_v1 --split valid

# Grid auto only (2D skeleton, no markers for scale)
python main.py --pipeline advanced --method skeleton --grid-auto \
  --run-name advanced_grid_only_v1 --split valid

# Grid + depth cache, still 2D skeleton
python main.py --pipeline advanced --method skeleton --grid-auto --depth \
  --run-name advanced_grid_depth_v1 --split valid

# Full advanced: grid + depth + 3D arc length
python main.py --pipeline advanced --method skeleton --grid-auto --depth --3d \
  --run-name advanced_full_v1 --split valid
```

Notebook API:

```python
from src.experiments import run_experiments

run_experiments([
    {"pipeline": "baseline", "method": "skeleton", "run_name": "baseline_skeleton_v1", "split": "valid"},
    {"pipeline": "advanced", "method": "skeleton", "use_grid_auto_calibration": True,
     "run_name": "advanced_grid_only_v1", "split": "valid"},
    {"pipeline": "advanced", "method": "skeleton", "use_grid_auto_calibration": True,
     "use_depth_estimation": True, "use_3d_measurement": True,
     "run_name": "advanced_full_v1", "split": "valid"},
])
```

Evaluation and CSV formats are unchanged; compare runs in `04_analyze_results.ipynb` via `experiments.csv` and per-run `comparison.csv`.

**Calibration guardrails (advanced):** Grid scale is used only when `grid_ppm / marker_ppm` is within `[0.85, 1.15]` (see `grid_ppm_ratio_min/max` in config); otherwise marker scale is used. Relative depth does not add a Z term unless `use_depth_metric_scale=True` (default **False**); with the default, 3D mode uses the same 2D skeleton length as baseline while still caching depth maps.

### Run directory layout

```
outputs/runs/baseline_bbox_v1/
├── predictions.csv
├── comparison.csv      # if ground truth available
├── metrics.json
├── config.json
├── figures/            # optional (--visualize)
└── debug/
```

Master registry: `outputs/runs/experiments.csv` (append-only, never deletes).

---

## Migration summary

### Old workflow

```bash
python main.py --split valid --method bbox
# → outputs/predictions/predictions.csv (overwrites)
evaluate_run(gt, predictions)  # → outputs/metrics/comparison.csv (overwrites)
```

### New workflow

```python
from src.experiments import run_experiment
run_experiment(pipeline="baseline", method="bbox", run_name="baseline_bbox_v1", split="valid")
# → outputs/runs/baseline_bbox_v1/  (isolated, no overwrite unless overwrite=True)
# → appends row to outputs/runs/experiments.csv
```

### Files added

- `src/pipelines/` — baseline, advanced, registry, shared base
- `src/experiments/` — RunManager, results, `run_experiment` API
- `notebooks/03_run_experiments.ipynb`, `04_analyze_results.ipynb`
- `outputs/runs/.gitkeep`

### Files modified

- `main.py` — thin CLI; legacy mode when no `--pipeline`/`--run-name`
- `src/evaluation.py` — optional `output_dir`; still writes legacy paths when omitted
- `README.md` — this section

### Backwards compatibility

- `python main.py --split test --method pca` — unchanged
- `main.run_pipeline()` — preserved, delegates to shared inference
- `evaluate_run()` without `output_dir` — still writes `outputs/metrics/comparison.csv`
- `experiments.jsonl` — still appended on every evaluation
- All existing notebooks run without modification

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `dataset_exploration.ipynb` | Load samples, visualize polygons and masks |
| `annotation_helper.ipynb` | Manual validation lengths CSV |
| **`03_run_experiments.ipynb`** | Run and compare experiments (primary) |
| **`04_analyze_results.ipynb`** | Registry tables, plots, per-image errors |
| `baseline_measurement.ipynb` | Legacy exploratory baseline |
| `perspective_correction.ipynb` | Homography experiments |
| `evaluation.ipynb` | Legacy flat-path evaluation |

Open from repo root so `import src` resolves, or run:

```bash
cd fishnet_cv && jupyter lab notebooks/
```

---

## Google Colab + Drive

Persistent data and experiment outputs live on Google Drive under **`UH_CV/`** when running in Colab. Local runs keep the existing `data/` and `outputs/` layout unchanged.

### Drive folder layout

```
UH_CV/
├── data/
│   ├── fishnet/          # images/ + labels/
│   ├── images/           # optional extras
│   └── labels/
├── runs/                 # experiment runs (predictions, metrics, config)
├── figures/
│   ├── debug/
│   └── analysis/
├── cache/
│   ├── depth_maps/
│   └── intermediate_outputs/
├── logs/
└── exports/
    ├── predictions/
    └── final_results/
```

### Colab startup (first cell)

```python
from src.colab_bootstrap import mount_google_drive, setup_notebook_environment
from src.config import get_config
from src.experiments import run_experiment
from src.utils import setup_logging

mount_google_drive()
REPO, STORAGE = setup_notebook_environment()
setup_logging()
cfg = get_config()

run_experiment(
    pipeline="baseline",
    method="bbox",
    split="valid",
    run_name="baseline_bbox_v1",
    validation_set_only=True,
    cfg=cfg,
)
```

Clone or upload the repo to Colab, then `pip install -r requirements.txt` (and advanced deps if needed).

### Path configuration

All paths resolve via `src/paths.py` and `ProjectConfig`:

| Variable / field | Local default | Colab (Drive) |
|------------------|---------------|---------------|
| `PROJECT_ROOT` | repo root | repo root (code) |
| `storage_root` | `PROJECT_ROOT` | `/content/drive/MyDrive/UH_CV` |
| `runs_root` | `outputs/runs` | `UH_CV/runs` |
| `figures_root` | `outputs/figures` | `UH_CV/figures` |
| depth cache | `data/processed/depth` | `UH_CV/cache/depth_maps` |
| `outputs_metrics` / logs | `outputs/metrics` | `UH_CV/logs` |

Override storage anywhere with:

```bash
export FISHNET_DRIVE_ROOT=/path/to/UH_CV
```

Notebook experiment cell flags (`RunExperimentsConfig`): `cache_results`, `save_figures_to_drive`, `save_metrics_to_drive`, `save_predictions_to_drive` (default **True**). Set to `False` on Colab to use ephemeral `/tmp` for that artifact class.

---

## Extending measurement methods

1. Implement `def my_method(mask: np.ndarray) -> float` returning **pixels**.
2. Register: `register_measurement_method("my_method", my_method)`.
3. Add CLI choice in `main.py` and document in this README.

Use `pixels_to_mm()` with `CalibrationResult.pixels_per_mm` for metric output.

---

## Outputs

| Path | Content |
|------|---------|
| `outputs/runs/<run_name>/` | Isolated run: predictions, metrics, comparison, config |
| `outputs/runs/experiments.csv` | Master registry (append-only) |
| `outputs/predictions/predictions.csv` | Legacy flat predictions |
| `outputs/figures/` | Legacy figures (or `<run>/figures/` with `--visualize`) |
| `outputs/metrics/comparison.csv` | Legacy evaluation output |
| `outputs/metrics/experiments.jsonl` | Legacy JSONL log (still appended) |

---

## Development notes

- Starter code includes **`TODO`** markers for assignment-specific tuning (homography corners, skeleton geodesic, GT CSV schema).
- Type hints and docstrings are used throughout; logging is enabled via `src/utils.setup_logging()`.
- VS Code / Cursor: select the `.venv` interpreter and use the integrated test explorer with `pytest`.

---

## License

MIT (adjust as needed for your course submission).
