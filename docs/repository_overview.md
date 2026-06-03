# Repository overview

Fish length estimation from RGB images and YOLO polygon labels. Two pipelines: **baseline** (markers + 2D geometry) and **advanced** (optional grid auto-calibration, depth, 3D skeleton).

## 1. Repository structure

```
fishnet_cv/
├── main.py                 # CLI entry (legacy + managed runs)
├── pyproject.toml          # Package metadata and dependencies
├── requirements.txt        # Core pip install list
├── requirements-advanced.txt  # Core + torch (advanced pipeline)
├── data/                   # Gitignored content; scaffolding tracked
├── docs/                   # COLAB.md, this file
├── notebooks/              # Jupyter workflows
├── outputs/                # Gitignored experiment artifacts (local)
├── scripts/                # Install and audit helpers
├── src/                    # Application code
└── tests/                  # pytest suite
```

## 2. Execution flow

```
main.py / notebooks
    → get_config() / ProjectConfig
    → run_experiment() or run_inference()
        → BaselinePipeline → pipelines/base.py
        → AdvancedPipeline → base.py OR advanced_inference.py
    → evaluate_run() (optional)
    → outputs/runs/<run_name>/
```

**Legacy path:** `python main.py --split valid --method pca` → `outputs/predictions/predictions.csv`.

**Managed path:** `python main.py --pipeline baseline --run-name ...` → `outputs/runs/<run_name>/`.

## 3. Experiment flow

| Step | Notebook | Action |
|------|----------|--------|
| Explore data | `01_dataset_exploration.ipynb` | Verify paths, masks, polygons |
| Annotations | `02_annotation_helper.ipynb` | Build `validation_lengths.csv` |
| Run grid | `03_run_experiments.ipynb` | `run_experiments()` via `RunExperimentsConfig` |
| Analyze | `04_analyze_results.ipynb` | Registry, MAE/RMSE plots |
| Debug viz | `05_visualization_debug.ipynb` | Per-image inspection panels |
| Colab | `06_colab_main.ipynb` | Single-run Colab entry |

Registry: append-only `outputs/runs/experiments.csv`.

## 4. Storage locations

| Artifact | Local path | Colab / Drive (`UH_CV/`) |
|----------|------------|---------------------------|
| Dataset | `data/fishnet/` | `data/fishnet/` |
| Annotations | `data/annotations/` | `data/annotations/` |
| Runs | `outputs/runs/<name>/` | `runs/<name>/` |
| Depth cache | `data/processed/depth/` | `cache/depth_maps/` |
| Legacy predictions | `outputs/predictions/` | `exports/predictions/` |
| Metrics log | `outputs/metrics/experiments.jsonl` | `logs/` |

Resolved by `src/paths.py` and `src/config.py`.

## 5. File summaries

### Entry points

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `main.py` | CLI for legacy and managed runs | Args, env vars | `predictions.csv`, run dirs | shared |
| `src/experiments/__init__.py` | `run_experiment`, `run_experiments` | Config dicts | `ExperimentResult`, registry row | shared |

### Configuration

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `src/config.py` | `ProjectConfig`, class IDs, env overrides | Env, paths | Config object | shared |
| `src/paths.py` | Local vs Colab storage roots | Runtime env | `StoragePaths` | shared |
| `src/colab_bootstrap.py` | Mount Drive, `sys.path`, mkdirs | Colab runtime | Repo + storage paths | shared |
| `data/fishnet.yaml` | YOLO class name reference (not loaded by code) | — | — | shared |
| `pyproject.toml` | Editable install, optional `[advanced]` | — | — | shared |

### Data interface

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `src/dataset.py` | Image/label discovery, `DatasetSample` | Paths, split | Samples, polygons | shared |
| `src/masks.py` | Rasterize polygons, skeleton | Polygons, masks | `ndarray` masks | shared |

### Calibration & measurement

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `src/calibration/marker.py` | Blue/yellow marker scale, homography | Sample, config | `CalibrationResult` | baseline + advanced |
| `src/calibration/grid_auto.py` | Tank grid spacing (Hough) | Image, mask | `GridCalibrationResult` | advanced |
| `src/measurement/core.py` | bbox / pca / skeleton 2D length | Mask, calibration | px, mm | shared |
| `src/measurement/skeleton3d.py` | 3D arc length with depth | Mask, depth, calibration | mm | advanced |

### Depth (advanced)

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `src/depth/depth_anything.py` | Depth Anything V3 wrapper | RGB image | Depth map | advanced |
| `src/depth/cache.py` | Load/save `.npy` depth cache | image_id, split | `ndarray` | advanced |

### Pipelines

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `src/pipelines/base.py` | Shared inference loop | Dataset, method | `predictions.csv` | baseline; advanced fallback |
| `src/pipelines/baseline.py` | No perspective wrapper | Config | CSV | baseline |
| `src/pipelines/advanced.py` | Routes to base or advanced_inference | Flags | CSV | advanced |
| `src/pipelines/advanced_inference.py` | Grid + depth + 3D path | Config, flags | CSV | advanced |
| `src/pipelines/registry.py` | `get_pipeline(name)` | Name | Pipeline instance | shared |

### Experiments & evaluation

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `src/experiments/run_manager.py` | Run dirs, config.json, registry | Run name | Paths | shared |
| `src/experiments/results.py` | Load registry / scan runs | `runs_root` | DataFrame | shared |
| `src/experiments/notebook_helpers.py` | Notebook configs and plots | DataFrames | Figures | shared |
| `src/evaluation.py` | MAE, RMSE, comparison CSV | GT + pred CSV | metrics | shared |

### Visualization

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `src/visualization/legacy.py` | Masks, PCA overlays (inference) | Sample | PNG | baseline |
| `src/visualization/framework.py` | Multi-panel inspection | `image_id`, run_dir | Figure | shared |
| `src/visualization/context.py` | Build inspection context | Sample, run config | `ImageInspectionContext` | shared |
| `src/visualization/panels.py` | Panel renderers | Context | Axes | shared |
| `src/visualization/debug.py` | Debug text panel | Context | Text | shared |
| `src/visualization/_common.py` | Save paths, colors | Config | PNG paths | shared |

### Utilities

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `src/utils.py` | Logging, polygon helpers | — | — | shared |

### Scripts

| File | Purpose | Inputs | Outputs | Pipeline |
|------|---------|--------|---------|----------|
| `scripts/install_advanced_deps.sh` | Install torch + DA3 | Shell | venv packages | advanced |
| `scripts/audit_advanced_pipeline.py` | Debug intermediate CSV | Validation IDs | Audit CSV | advanced |

### Notebooks

| File | Purpose | Pipeline |
|------|---------|----------|
| `01_dataset_exploration.ipynb` | Data QA | shared |
| `02_annotation_helper.ipynb` | Manual lengths CSV | shared |
| `03_run_experiments.ipynb` | Batch experiments | shared |
| `04_analyze_results.ipynb` | Compare runs | shared |
| `05_visualization_debug.ipynb` | Per-image debug | shared |
| `06_colab_main.ipynb` | Minimal Colab runner | shared |

### Tests

| File | Covers |
|------|--------|
| `tests/test_*.py` | Unit tests for dataset, measurement, experiments, paths, visualization, advanced |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Setup, commands, architecture |
| `docs/COLAB.md` | Colab + Drive workflow |
| `docs/repository_overview.md` | This file |

### Generated / gitignored (do not commit)

- `outputs/` — runs, figures, metrics, legacy predictions
- `data/fishnet/`, `data/processed/depth/` — dataset and cache
- `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`
- `assignment.pdf` — local assignment spec
