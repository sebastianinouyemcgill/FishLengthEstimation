"""
Advanced measurement loop: grid calibration, depth, and 3D skeleton.

Does not modify ``run_inference`` in ``base.py`` (baseline path unchanged).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.calibration import CalibrationResult, calibrate_sample
from src.calibration.grid_auto import (
    GridCalibrationResult,
    estimate_depth_metric_scale,
    estimate_grid_calibration,
    grid_result_to_marker_calibration,
)
from src.config import ProjectConfig
from src.dataset import iterate_dataset, iterate_image_ids
from src.depth import get_depth_estimator
from src.masks import mask_from_class
from src.measurement import estimate_length_mm, estimate_skeleton_3d_length_mm
from src.utils import get_logger

logger = get_logger(__name__)

_DEBUG_LOG = Path(__file__).resolve().parents[2] / ".cursor" / "debug-95b075.log"


def uses_advanced_features(cfg: ProjectConfig) -> bool:
    """True when any advanced-only stage is enabled."""
    return (
        cfg.use_grid_auto_calibration
        or cfg.use_depth_estimation
        or cfg.use_3d_measurement
    )


def _agent_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    *,
    run_id: str = "fix-verify",
) -> None:
    # #region agent log
    payload = {
        "sessionId": "95b075",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
    }
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # #endregion


def _choose_calibration(
    sample,
    cfg: ProjectConfig,
    image,
    grid: GridCalibrationResult | None = None,
) -> tuple[CalibrationResult, str, float, float, GridCalibrationResult | None]:
    """
    Pick scale for metric conversion.

    Returns (calibration, scale_source, marker_ppm, grid_ppm, grid_result).
    scale_source is ``grid``, ``marker``, or ``marker_fallback``.
    """
    marker_calib = calibrate_sample(sample, cfg=cfg)
    marker_ppm = marker_calib.pixels_per_mm
    grid_ppm = 0.0

    if not cfg.use_grid_auto_calibration:
        return marker_calib, "marker", marker_ppm, grid_ppm, None

    if grid is None:
        grid = estimate_grid_calibration(image, cfg=cfg)
    grid_ppm = grid.pixels_per_mm if grid.success else 0.0

    if grid.success and marker_ppm > 0:
        ratio = grid_ppm / marker_ppm
        if cfg.grid_ppm_ratio_min <= ratio <= cfg.grid_ppm_ratio_max:
            _agent_log(
                "H1",
                "advanced_inference._choose_calibration",
                "grid_accepted",
                {
                    "image_id": sample.image_id,
                    "marker_ppm": marker_ppm,
                    "grid_ppm": grid_ppm,
                    "ratio": ratio,
                },
            )
            return grid_result_to_marker_calibration(grid), "grid", marker_ppm, grid_ppm, grid

        logger.warning(
            "%s: grid ppm ratio %.2f outside [%.2f, %.2f]; using marker scale",
            sample.image_id,
            ratio,
            cfg.grid_ppm_ratio_min,
            cfg.grid_ppm_ratio_max,
        )
        _agent_log(
            "H1",
            "advanced_inference._choose_calibration",
            "grid_rejected_marker_fallback",
            {
                "image_id": sample.image_id,
                "marker_ppm": marker_ppm,
                "grid_ppm": grid_ppm,
                "ratio": ratio,
            },
        )
        return marker_calib, "marker_fallback", marker_ppm, grid_ppm, grid

    if grid.success:
        _agent_log(
            "H1",
            "advanced_inference._choose_calibration",
            "grid_no_markers",
            {"image_id": sample.image_id, "grid_ppm": grid_ppm},
        )
        return grid_result_to_marker_calibration(grid), "grid", marker_ppm, grid_ppm, grid

    logger.warning(
        "%s: grid calibration failed; falling back to marker scale",
        sample.image_id,
    )
    return marker_calib, "marker_fallback", marker_ppm, grid_ppm, grid


def _depth_scale_for_sample(
    cfg: ProjectConfig,
    depth_map,
    grid: GridCalibrationResult | None,
) -> float:
    """Millimeters per depth unit; 0 disables the Z contribution in 3D arc length."""
    if not cfg.use_depth_metric_scale or depth_map is None:
        return 0.0
    if grid is None or not grid.success:
        return 0.0
    return estimate_depth_metric_scale(
        depth_map,
        grid.pixels_per_grid_square,
        grid.grid_square_mm,
    )


def _measure_sample(
    sample,
    cfg: ProjectConfig,
    fish_mask,
    calib: CalibrationResult,
    depth_scale: float,
    method: str,
    depth_map=None,
) -> float:
    """Single-image length in mm."""
    if cfg.use_3d_measurement and cfg.use_depth_estimation:
        if depth_map is None:
            logger.warning("%s: missing depth for 3D measurement", sample.image_id)
            return float("nan")
        # Without metric Z calibration, 3D arc uses relative depth and diverges from
        # baseline skeleton; use the same 2D skeleton + scale path until Z is valid.
        if depth_scale <= 0:
            length = estimate_length_mm(fish_mask, calib, method="skeleton")
            _agent_log(
                "H3",
                "advanced_inference._measure_sample",
                "2d_skeleton_marker_scale",
                {
                    "image_id": sample.image_id,
                    "length_mm": length,
                    "pixels_per_mm": calib.pixels_per_mm,
                },
            )
            return length
        length = estimate_skeleton_3d_length_mm(
            fish_mask,
            depth_map,
            calib,
            depth_scale_mm_per_unit=depth_scale,
        )
        _agent_log(
            "H3",
            "advanced_inference._measure_sample",
            "3d_length",
            {
                "image_id": sample.image_id,
                "length_mm": length,
                "pixels_per_mm": calib.pixels_per_mm,
                "depth_scale": depth_scale,
            },
        )
        return length

    use_method = method
    if cfg.use_3d_measurement and not cfg.use_depth_estimation:
        logger.warning(
            "%s: use_3d_measurement requires use_depth_estimation; using skeleton 2D",
            sample.image_id,
        )
        use_method = "skeleton"

    return estimate_length_mm(fish_mask, calib, method=use_method)


def run_advanced_inference(
    cfg: ProjectConfig,
    split: str,
    method: str,
    predictions_path: Path,
    *,
    limit: int | None = None,
    image_ids: list[str] | None = None,
    visualize: bool = False,
    figures_dir: Path | None = None,
) -> Path:
    """
    Run advanced pipeline stages and write ``predictions.csv``.

    Same CSV schema as baseline: ``image_id``, ``predicted_length_mm``.
    """
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []

    depth_estimator = None
    if cfg.use_depth_estimation:
        depth_estimator = get_depth_estimator(cfg)

    if image_ids is not None:
        iterator = iterate_image_ids(cfg, split=split, image_ids=image_ids, load_images=True)
        desc = f"Advanced ({split}, n={len(image_ids)})"
    else:
        iterator = iterate_dataset(cfg, split=split, load_images=True, limit=limit)
        desc = f"Advanced ({split})"

    for sample in tqdm(iterator, desc=desc, unit="img"):
        if sample.image is None:
            logger.warning("Skipping %s: no image", sample.image_id)
            continue

        image = sample.image
        fish_mask = mask_from_class(
            sample.annotations,
            class_name="fish",
            height=image.shape[0],
            width=image.shape[1],
        )

        grid_debug_dir = None
        if cfg.use_grid_auto_calibration and getattr(cfg, "visualize_grid_debug", False):
            grid_debug_dir = predictions_path.parent / "grid_debug"
        grid_result = (
            estimate_grid_calibration(
                image,
                cfg=cfg,
                debug_output_dir=grid_debug_dir,
                image_id=sample.image_id,
            )
            if cfg.use_grid_auto_calibration
            else None
        )
        calib, _scale_source, _marker_ppm, _grid_ppm, grid_result = _choose_calibration(
            sample, cfg, image, grid=grid_result
        )

        depth_map = None
        depth_scale = 0.0
        if cfg.use_depth_estimation and depth_estimator is not None:
            depth_map = depth_estimator.predict_depth(
                image,
                image_id=sample.image_id,
                split=split,
                use_cache=True,
            )
            depth_scale = _depth_scale_for_sample(cfg, depth_map, grid_result)

        length_mm = _measure_sample(
            sample,
            cfg,
            fish_mask,
            calib,
            depth_scale,
            method,
            depth_map=depth_map,
        )
        rows.append({"image_id": sample.image_id, "predicted_length_mm": length_mm})
        logger.debug("%s: %.2f mm (advanced)", sample.image_id, length_mm)

        if visualize:
            from src.visualization import visualize_image

            visualize_image(
                sample.image_id,
                cfg=cfg,
                run_dir=predictions_path.parent,
                debug=True,
                save=True,
                show=False,
                split=split,
                method=method,
                output_dir=figures_dir,
            )

    df = pd.DataFrame(rows)
    df.to_csv(predictions_path, index=False)
    logger.info("Wrote %d advanced predictions to %s", len(df), predictions_path)
    return predictions_path
