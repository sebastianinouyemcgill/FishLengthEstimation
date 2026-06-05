# Pipeline architecture (refactor)

Production benchmark paths are separated from archived experimental code.

## Directory layout

| Package | Role | Default in experiments |
|---------|------|------------------------|
| `src/baseline/` | Bbox / PCA / skeleton measurement, `run_inference` | **Active** |
| `src/calibration/` | Marker + grid auto calibration | **Active** (grid on advanced) |
| `src/keypoints/` | Pseudo-labels, dataset manifest, eval hooks (HRNet later) | Scaffold only |
| `src/experimental/` | Depth, 3D skeleton, perspective homography | **Inactive** |
| `src/pipelines/` | `BaselinePipeline`, `AdvancedPipeline`, routing | **Active** |

Implementation files remain in `measurement/`, `depth/`, etc.; logical packages re-export APIs.

## Config flags

| Flag | Default | Notes |
|------|---------|--------|
| `use_grid_auto_calibration` | `False` | Primary advanced benchmark |
| `use_depth_estimation` | `False` | Experimental |
| `use_3d_measurement` | `False` | Experimental |
| `use_perspective` | `False` | Experimental (alias: `apply_perspective_correction`) |
| `use_hrnet_keypoints` | `False` | Future |
| `use_pseudo_label_training` | `False` | Future |

Experimental flags are **clamped off** in `run_experiment` unless `FISHNET_ALLOW_EXPERIMENTAL=1`.

## Experiment routing

```
pipeline == "baseline"
  → BaselinePipeline → run_inference (markers, 2D methods)
  → perspective / depth / 3D / grid forced off

pipeline == "advanced"
  → resolve_run_config (clamp experimental)
  → if grid or depth or 3D:
        AdvancedPipeline → run_advanced_inference
     else:
        AdvancedPipeline → run_inference (e.g. perspective-only legacy)
```

Grid + skeleton uses **2D** `estimate_length_mm` only (no depth/3D calls when flags are false).

## Official experiment tree

Default notebook grid (`RunExperimentsConfig`):

- `baseline` × `bbox`, `pca`, `skeleton`
- `advanced` × same methods with `use_grid_auto_calibration=True`

Use `benchmark_experiment_specs()` for an explicit benchmark list without depth/3D/perspective.

## Disabled modules (by default)

- `src/depth/` — depth maps
- `src/measurement/skeleton3d.py` — 3D arc length
- Perspective branch in `run_inference` — homography rectification

Enable for research: `export FISHNET_ALLOW_EXPERIMENTAL=1` then pass CLI flags or config.

## Data flow

```
RGB image + YOLO polygons
  → masks (fish)
  → calibration (marker and/or grid) → pixels_per_mm
  → length (bbox | pca | skeleton) → predictions.csv
```

Future (no shared mutable state):

```
  → pseudo keypoints JSONL (keypoints/)
  → HRNet inference (not implemented)
```
