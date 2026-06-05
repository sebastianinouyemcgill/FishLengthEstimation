"""
Notebook-first experiment API.

Primary entry points::

    from src.experiments import run_experiment, run_experiments

    run = run_experiment(pipeline="baseline", method="bbox", split="valid")
    results_df = run_experiments([{...}, {...}])
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ProjectConfig, get_config
from src.evaluation import MetricResult, evaluate_run, load_predictions_csv
from src.experiments.ground_truth import default_ground_truth_path, load_validation_image_ids
from src.experiments.regression import run_regression_experiment
from src.experiments.results import ExperimentResult, results_to_dataframe
from src.experiments.run_manager import RunExistsError, RunManager
from src.pipelines.advanced import AdvancedPipeline
from src.pipelines.baseline import BaselinePipeline
from src.pipelines.registry import get_pipeline
from src.pipelines.routing import resolve_run_config

logger = logging.getLogger(__name__)

__all__ = [
    "RunExistsError",
    "RunManager",
    "ExperimentResult",
    "run_experiment",
    "run_experiments",
    "run_regression_experiment",
    "results_to_dataframe",
    "default_ground_truth_path",
    "load_validation_image_ids",
]


def run_experiment(
    *,
    pipeline: str = "baseline",
    method: str = "bbox",
    split: str | None = None,
    run_name: str | None = None,
    perspective: bool | None = None,
    ground_truth_path: Path | str | None = None,
    limit: int | None = None,
    image_ids: list[str] | None = None,
    validation_set_only: bool = False,
    visualize: bool = False,
    overwrite: bool = False,
    evaluate: bool = True,
    cfg: ProjectConfig | None = None,
    use_grid_auto_calibration: bool | None = None,
    use_depth_estimation: bool | None = None,
    use_3d_measurement: bool | None = None,
    use_regression_model: bool | None = None,
    regression_model_path: Path | str | None = None,
) -> ExperimentResult:
    """
    Run one experiment into ``outputs/runs/<run_name>/``.

    Parameters
    ----------
    pipeline:
        ``baseline`` (no perspective) or ``advanced`` (optional perspective).
    perspective:
        Only applies to ``advanced`` pipeline when no grid/depth/3D flags are set.
    use_grid_auto_calibration, use_depth_estimation, use_3d_measurement:
        Advanced pipeline only. All default to False (unchanged baseline behavior).
    ground_truth_path:
        CSV for evaluation. Defaults to validation_lengths.csv or lengths_mm.csv.
    image_ids:
        Explicit list of ``image_id`` values to predict. Ignores ``limit``.
    validation_set_only:
        If True, load IDs from ``validation_lengths.csv`` / ``validation_images.csv``.
    evaluate:
        If True and ground truth exists, compute MAE/RMSE into the run dir.
    overwrite:
        Replace an existing run directory when True.

    Returns
    -------
    ExperimentResult
        Structured summary including paths and metrics.
    """
    cfg = cfg or get_config()
    split = split or cfg.default_split

    if validation_set_only:
        image_ids = load_validation_image_ids(cfg)
    elif image_ids is not None:
        image_ids = [str(i) for i in image_ids]

    pipe = get_pipeline(pipeline)

    cfg, use_perspective, adv_grid, adv_depth, adv_3d, _adv_hrnet = resolve_run_config(
        cfg,
        pipeline=pipeline,
        perspective=perspective,
        use_grid_auto_calibration=use_grid_auto_calibration,
        use_depth_estimation=use_depth_estimation,
        use_3d_measurement=use_3d_measurement,
    )

    if use_regression_model is not None:
        cfg = replace(cfg, use_regression_model=use_regression_model)
    if regression_model_path is not None:
        cfg = replace(cfg, regression_model_path=Path(regression_model_path))

    manager = RunManager(cfg)
    resolved_name = manager.resolve_run_name(run_name, pipeline=pipeline, method=method)
    run_dir = manager.prepare_run(resolved_name, overwrite=overwrite)

    config_path = manager.save_config(
        run_dir,
        run_name=resolved_name,
        pipeline=pipeline,
        method=method,
        split=split,
        perspective=use_perspective,
        limit=limit,
        ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
        n_image_ids=len(image_ids) if image_ids else None,
        extra={
            "use_grid_auto_calibration": adv_grid,
            "use_depth_estimation": adv_depth,
            "use_3d_measurement": adv_3d,
            "use_hrnet_keypoints": cfg.use_hrnet_keypoints,
            "experimental_disabled_by_policy": not adv_depth and not adv_3d and not use_perspective,
        }
        if pipeline == "advanced"
        else None,
    )

    predictions_path = run_dir / "predictions.csv"
    figures_dir = run_dir / "figures" if visualize else None

    if isinstance(pipe, BaselinePipeline):
        pipe.run(
            cfg=cfg,
            method=method,
            split=split,
            predictions_path=predictions_path,
            limit=limit,
            image_ids=image_ids,
            visualize=visualize,
            figures_dir=figures_dir,
        )
    else:
        assert isinstance(pipe, AdvancedPipeline)
        pipe.run(
            cfg=cfg,
            method=method,
            split=split,
            perspective=use_perspective,
            predictions_path=predictions_path,
            limit=limit,
            image_ids=image_ids,
            visualize=visualize,
            figures_dir=figures_dir,
            use_grid_auto_calibration=adv_grid,
            use_depth_estimation=adv_depth,
            use_3d_measurement=adv_3d,
        )

    n_predictions = len(load_predictions_csv(predictions_path))
    gt_path = Path(ground_truth_path) if ground_truth_path else default_ground_truth_path(cfg)

    metrics: MetricResult | None = None
    comparison_path: Path | None = None
    metrics_path: Path | None = None
    evaluated = False

    if evaluate and gt_path is not None and gt_path.is_file():
        metrics, _ = evaluate_run(
            gt_path,
            predictions_path,
            cfg=cfg,
            method=method,
            split=split,
            output_dir=run_dir,
            pipeline=pipeline,
            perspective=use_perspective,
            run_name=resolved_name,
        )
        comparison_path = run_dir / "comparison.csv"
        metrics_path = run_dir / "metrics.json"
        evaluated = True
    elif evaluate:
        logger.warning(
            "No ground truth found; skipping evaluation. "
            "Add data/annotations/validation_lengths.csv or pass ground_truth_path."
        )

    result = ExperimentResult(
        run_name=resolved_name,
        run_dir=run_dir,
        pipeline=pipeline,
        method=method,
        split=split,
        perspective=use_perspective,
        predictions_path=predictions_path,
        comparison_path=comparison_path if evaluated else None,
        metrics_path=metrics_path,
        config_path=config_path,
        metrics=metrics,
        n_predictions=n_predictions,
        evaluated=evaluated,
    )

    manager.append_registry(
        run_name=resolved_name,
        pipeline=pipeline,
        method=method,
        split=split,
        perspective=use_perspective,
        predictions_path=predictions_path,
        comparison_path=comparison_path,
        metrics=metrics,
        timestamp=result.timestamp,
    )

    return result


def run_experiments(
    configs: list[dict[str, Any]],
    *,
    cfg: ProjectConfig | None = None,
    stop_on_error: bool = False,
) -> pd.DataFrame:
    """
    Run multiple experiments sequentially; return a summary DataFrame.

    Each dict is passed as kwargs to ``run_experiment`` (except ``cfg``).
    """
    cfg = cfg or get_config()
    results: list[ExperimentResult] = []
    for i, spec in enumerate(configs):
        spec = dict(spec)
        logger.info("Running experiment %d/%d: %s", i + 1, len(configs), spec)
        try:
            results.append(run_experiment(cfg=cfg, **spec))
        except RunExistsError:
            if stop_on_error:
                raise
            logger.error("Skipping existing run (pass overwrite=True): %s", spec.get("run_name"))
        except Exception:
            if stop_on_error:
                raise
            logger.exception("Experiment failed: %s", spec)
    return results_to_dataframe(results)
