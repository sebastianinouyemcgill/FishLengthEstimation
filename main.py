#!/usr/bin/env python3
"""
End-to-end inference: load dataset → calibrate → measure → write predictions.

Legacy mode (unchanged)::

    python main.py --split test --method pca

Managed experiment runs::

    python main.py --pipeline baseline --method bbox --run-name baseline_bbox_v1
    python main.py --pipeline advanced --method skeleton --grid-auto --depth --3d \\
        --run-name advanced_full_v1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import get_config
from src.experiments import RunExistsError, run_experiment
from src.pipelines.base import run_inference
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def run_pipeline(
    cfg,
    split: str,
    method: str,
    limit: int | None = None,
    visualize: bool = False,
    output_name: str = "predictions.csv",
) -> Path:
    """
    Legacy pipeline: write predictions to ``outputs/predictions/<output_name>``.

    Preserved for backwards compatibility with existing notebooks and scripts.
    """
    cfg.ensure_output_dirs()
    predictions_path = cfg.outputs_predictions / output_name
    return run_inference(
        cfg,
        split=split,
        method=method,
        predictions_path=predictions_path,
        limit=limit,
        visualize=visualize,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fish length estimation pipeline (FishNet CV)",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        choices=("baseline", "advanced"),
        help="Pipeline type (enables managed run when combined with --run-name or auto name)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Experiment run name under outputs/runs/ (auto-generated if omitted with --pipeline)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing run directory",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation even if ground truth exists",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to ground-truth CSV for evaluation",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=("train", "valid", "test"),
        help="Dataset split (default: config default_split)",
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=("bbox", "pca", "skeleton"),
        help="Measurement method (default: config measurement_method)",
    )
    parser.add_argument(
        "--perspective",
        action="store_true",
        help="Homography rectification (advanced only, legacy path without grid/depth/3D)",
    )
    parser.add_argument(
        "--grid-auto",
        action="store_true",
        help="Automatic grid calibration without fiduciary markers (advanced)",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Depth Anything V3 depth maps with caching (advanced)",
    )
    parser.add_argument(
        "--3d",
        dest="use_3d",
        action="store_true",
        help="3D skeleton arc-length measurement (advanced; requires --depth)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (debug)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save diagnostic figures (run dir or outputs/figures/)",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Legacy: output CSV filename under outputs/predictions/",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level))
    cfg = get_config()

    split = args.split or cfg.default_split
    method = args.method or cfg.measurement_method

    # Managed experiment mode when --pipeline or --run-name is set
    if args.pipeline or args.run_name:
        pipeline = args.pipeline or "baseline"
        if (args.depth or args.use_3d or args.perspective) and pipeline == "advanced":
            import os

            if os.environ.get("FISHNET_ALLOW_EXPERIMENTAL", "").lower() not in ("1", "true", "yes"):
                logger.warning(
                    "Depth/3D/perspective flags are ignored unless FISHNET_ALLOW_EXPERIMENTAL=1 "
                    "(archived experimental paths)"
                )
        try:
            result = run_experiment(
                pipeline=pipeline,
                method=method,
                split=split,
                run_name=args.run_name,
                perspective=args.perspective if pipeline == "advanced" else False,
                ground_truth_path=Path(args.ground_truth) if args.ground_truth else None,
                limit=args.limit,
                visualize=args.visualize,
                overwrite=args.overwrite,
                evaluate=not args.no_evaluate,
                cfg=cfg,
                use_grid_auto_calibration=args.grid_auto if pipeline == "advanced" else None,
                use_depth_estimation=args.depth if pipeline == "advanced" else None,
                use_3d_measurement=args.use_3d if pipeline == "advanced" else None,
            )
        except RunExistsError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
        logger.info("Run complete: %s", result.run_dir)
        if result.metrics:
            logger.info("MAE=%.3f mm RMSE=%.3f mm", result.metrics.mae_mm, result.metrics.rmse_mm)
        return

    # Legacy mode
    if args.split:
        cfg.default_split = split
    if args.method:
        cfg.measurement_method = method
    if args.perspective:
        cfg.apply_perspective_correction = True

    logger.info(
        "Legacy pipeline: split=%s method=%s perspective=%s",
        cfg.default_split,
        cfg.measurement_method,
        cfg.apply_perspective_correction,
    )
    logger.info("Dataset root: %s", cfg.resolve_dataset_root())

    run_pipeline(
        cfg,
        split=cfg.default_split,
        method=cfg.measurement_method,
        limit=args.limit,
        visualize=args.visualize,
        output_name=args.output,
    )


if __name__ == "__main__":
    main()
