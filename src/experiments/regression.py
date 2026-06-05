"""
Regression calibration experiments (train features → predict SH length).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.calibration import calibrate_sample, rectify_image
from src.config import ProjectConfig, get_config
from src.dataset import iterate_image_ids
from src.evaluation import MetricResult, evaluate_run, load_ground_truth_csv, mae, rmse
from src.experiments.ground_truth import default_ground_truth_path, load_validation_image_ids
from src.experiments.results import ExperimentResult
from src.experiments.run_manager import RunManager
from src.masks import mask_from_class
from src.measurement.features import FEATURE_COLUMNS, extract_length_features
from src.models.length_regression import TARGET_COLUMN, LengthRegressionModel
from src.pipelines.regression_inference import run_regression_inference

logger = logging.getLogger(__name__)


def build_feature_table(
    cfg: ProjectConfig,
    split: str,
    image_ids: list[str],
    ground_truth_path: Path,
) -> pd.DataFrame:
    """Extract per-image features merged with manual SH length labels."""
    gt = load_ground_truth_csv(ground_truth_path)
    if "length_mm" not in gt.columns:
        raise ValueError(f"Ground truth CSV needs length_mm; got {list(gt.columns)}")
    gt = gt.rename(columns={"length_mm": TARGET_COLUMN})
    gt_ids = set(gt["image_id"].astype(str))
    ids = [str(i) for i in image_ids if str(i) in gt_ids]
    if not ids:
        raise ValueError("No overlap between image_ids and ground-truth image_id values")

    rows: list[dict] = []
    for sample in tqdm(
        iterate_image_ids(cfg, split=split, image_ids=ids, load_images=True),
        desc=f"Features ({split})",
        unit="img",
    ):
        if sample.image is None:
            continue
        image = sample.image
        if cfg.use_perspective or cfg.apply_perspective_correction:
            calib = calibrate_sample(sample, cfg=cfg)
            if calib.homography is not None:
                image = rectify_image(image, calib.homography)
        else:
            calib = calibrate_sample(sample, cfg=cfg)

        fish_mask = mask_from_class(
            sample.annotations,
            class_name="fish",
            height=image.shape[0],
            width=image.shape[1],
        )
        feat = extract_length_features(fish_mask, calib)
        row = {"image_id": sample.image_id, **feat.as_dict()}
        rows.append(row)

    features_df = pd.DataFrame(rows)
    merged = features_df.merge(
        gt[["image_id", TARGET_COLUMN]],
        on="image_id",
        how="inner",
    )
    logger.info("Built feature table: %d rows, split=%s", len(merged), split)
    return merged


def run_regression_experiment(
    cfg: ProjectConfig | None = None,
    *,
    train_split: str = "valid",
    eval_split: str | None = None,
    run_name: str | None = None,
    ground_truth_path: Path | str | None = None,
    validation_set_only: bool = True,
    image_ids: list[str] | None = None,
    model_type: str = "random_forest",
    overwrite: bool = False,
    evaluate: bool = True,
) -> ExperimentResult:
    """
    Train ``LengthRegressionModel`` on labeled images, then predict on an eval split.

    Mirrors ``run_experiment``: writes under ``outputs/runs/<run_name>/`` with
    ``predictions.csv``, ``regression_model.joblib``, and optional metrics.
    """
    cfg = cfg or get_config()
    eval_split = eval_split or cfg.default_split
    gt_path = Path(ground_truth_path) if ground_truth_path else default_ground_truth_path(cfg)
    if gt_path is None or not gt_path.is_file():
        raise FileNotFoundError(
            "Regression training requires labeled lengths. "
            "Add data/annotations/validation_lengths.csv or pass ground_truth_path."
        )

    if validation_set_only and image_ids is None:
        train_ids = load_validation_image_ids(cfg)
        eval_ids = train_ids
    elif image_ids is not None:
        train_ids = [str(i) for i in image_ids]
        eval_ids = train_ids
    else:
        raise ValueError("Pass validation_set_only=True or explicit image_ids for regression")

    manager = RunManager(cfg)
    resolved_name = manager.resolve_run_name(
        run_name,
        pipeline="regression",
        method="calibrated",
    )
    run_dir = manager.prepare_run(resolved_name, overwrite=overwrite)

    train_df = build_feature_table(cfg, train_split, train_ids, gt_path)
    model = LengthRegressionModel(model_type=model_type)
    model.fit(train_df[list(FEATURE_COLUMNS)], train_df[TARGET_COLUMN])
    model_path = run_dir / "regression_model.joblib"
    model.save(model_path)
    train_df.to_csv(run_dir / "train_features.csv", index=False)

    predictions_path = run_dir / "predictions.csv"
    run_regression_inference(
        cfg,
        eval_split,
        predictions_path,
        model,
        image_ids=eval_ids,
    )

    config_path = manager.save_config(
        run_dir,
        run_name=resolved_name,
        pipeline="regression",
        method="calibrated",
        split=eval_split,
        perspective=False,
        limit=None,
        ground_truth_path=str(gt_path),
        n_image_ids=len(eval_ids),
        extra={
            "train_split": train_split,
            "model_type": model_type,
            "use_regression_model": True,
        },
    )

    metrics: MetricResult | None = None
    comparison_path: Path | None = None
    metrics_path: Path | None = None
    evaluated = False

    if evaluate:
        metrics, comparison = evaluate_run(
            gt_path,
            predictions_path,
            cfg=cfg,
            method="regression",
            split=eval_split,
            output_dir=run_dir,
            pipeline="regression",
            perspective=False,
            run_name=resolved_name,
        )
        comparison_path = run_dir / "comparison.csv"
        metrics_path = run_dir / "metrics.json"
        evaluated = True

        if metrics_path and "skeleton_length_mm" in comparison.columns:
            import json
            from dataclasses import asdict

            baseline_mae = mae(
                comparison["length_mm"].to_numpy(),
                comparison["skeleton_length_mm"].to_numpy(),
            )
            baseline_rmse = rmse(
                comparison["length_mm"].to_numpy(),
                comparison["skeleton_length_mm"].to_numpy(),
            )
            payload = asdict(metrics)
            payload.update(
                {
                    "skeleton_baseline_mae_mm": baseline_mae,
                    "skeleton_baseline_rmse_mm": baseline_rmse,
                    "regression_mae_mm": metrics.mae_mm,
                    "regression_rmse_mm": metrics.rmse_mm,
                }
            )
            metrics_path.write_text(json.dumps(payload, indent=2))
            logger.info(
                "MAE skeleton=%.3f mm regression=%.3f mm",
                baseline_mae,
                metrics.mae_mm,
            )

    n_predictions = len(pd.read_csv(predictions_path))
    result = ExperimentResult(
        run_name=resolved_name,
        run_dir=run_dir,
        pipeline="regression",
        method="calibrated",
        split=eval_split,
        perspective=False,
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
        pipeline="regression",
        method="calibrated",
        split=eval_split,
        perspective=False,
        predictions_path=predictions_path,
        comparison_path=comparison_path,
        metrics=metrics,
        timestamp=result.timestamp,
    )
    return result
