# Google Colab workflow

**Main notebook:** upload [`notebooks/colab_main.ipynb`](../notebooks/colab_main.ipynb) to `MyDrive/UH_CV/` and open in Colab.

## 1. Mount Drive and configure paths

```python
from src.colab_bootstrap import mount_google_drive, setup_notebook_environment
from src.config import get_config
from src.utils import setup_logging

mount_google_drive()
REPO, STORAGE = setup_notebook_environment()
setup_logging()
cfg = get_config()
print(cfg.is_colab, STORAGE, cfg.runs_root)
```

## 2. Place data on Drive

Upload or copy the FishNet extract to:

`MyDrive/UH_CV/data/fishnet/images/` and `.../labels/`

Annotations: `MyDrive/UH_CV/data/annotations/validation_lengths.csv`

## 3. Run a baseline experiment

```python
from src.experiments import run_experiment

run_experiment(
    pipeline="baseline",
    method="bbox",
    split="valid",
    run_name="baseline_bbox_v1",
    validation_set_only=True,
    cfg=cfg,
)
```

Outputs: `UH_CV/runs/baseline_bbox_v1/predictions.csv`, `metrics.json`, `comparison.csv`.

## 4. Notebook storage flags

In `03_run_experiments.ipynb`, `RunExperimentsConfig` supports:

| Flag | Default | Meaning |
|------|---------|---------|
| `cache_results` | True | Depth maps under `cache/depth_maps` |
| `save_predictions_to_drive` | True | Runs under `runs/` (else `/tmp`) |
| `save_metrics_to_drive` | True | Metrics in run dir / `logs` |
| `save_figures_to_drive` | True | Figures under run dir / `figures` |

Use `project_config_for_experiments(RUN_CFG)` before `run_experiments()`.

## 5. Local override

```bash
export FISHNET_DRIVE_ROOT="$HOME/Google Drive/UH_CV"
python main.py --pipeline baseline --method bbox --run-name test_v1 --split valid
```
