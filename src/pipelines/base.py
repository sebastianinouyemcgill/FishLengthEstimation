"""
Shared inference loop used by baseline and advanced pipelines.

Extracted from ``main.run_pipeline`` so CLI and notebooks share one code path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.calibration import calibrate_sample, rectify_image
from src.config import ProjectConfig
from src.dataset import iterate_dataset, iterate_image_ids
from src.masks import mask_from_class
from src.measurement import estimate_length_mm
from src.visualization.legacy import visualize_sample

logger = logging.getLogger(__name__)


def run_inference(
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
    Run measurement on images in a split and write ``predictions.csv``.

    Parameters
    ----------
    predictions_path:
        Full path for the output CSV (typically inside a run directory).
    image_ids:
        If set, only these ``image_id`` values are processed (ignores ``limit``).
    figures_dir:
        When ``visualize=True``, save figures here instead of ``cfg.outputs_figures``.
    """
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []

    if image_ids is not None:
        iterator = iterate_image_ids(cfg, split=split, image_ids=image_ids, load_images=True)
        desc = f"Measuring ({split}, n={len(image_ids)})"
    else:
        iterator = iterate_dataset(cfg, split=split, load_images=True, limit=limit)
        desc = f"Measuring ({split})"

    for sample in tqdm(iterator, desc=desc, unit="img"):
        if sample.image is None:
            logger.warning("Skipping %s: no image", sample.image_id)
            continue

        image = sample.image
        if cfg.use_perspective or cfg.apply_perspective_correction:
            calib = calibrate_sample(sample, cfg=cfg)
            if calib.homography is not None:
                image = rectify_image(image, calib.homography)
                # TODO: warp fish/marker polygons with the same homography
        else:
            calib = calibrate_sample(sample, cfg=cfg)

        fish_mask = mask_from_class(
            sample.annotations,
            class_name="fish",
            height=image.shape[0],
            width=image.shape[1],
        )

        length_mm = estimate_length_mm(fish_mask, calib, method=method)
        rows.append({"image_id": sample.image_id, "predicted_length_mm": length_mm})
        logger.debug("%s: %.2f mm", sample.image_id, length_mm)

        if visualize:
            if figures_dir is not None:
                cfg = _with_figures_dir(cfg, figures_dir)
            visualize_sample(
                image,
                fish_mask,
                calibration=calib,
                cfg=cfg,
                image_id=sample.image_id,
                save=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(predictions_path, index=False)
    logger.info("Wrote %d predictions to %s", len(df), predictions_path)
    return predictions_path


def _with_figures_dir(cfg: ProjectConfig, figures_dir: Path) -> ProjectConfig:
    """Return a shallow copy of cfg with outputs_figures redirected."""
    import copy

    out = copy.copy(cfg)
    out.outputs_figures = figures_dir
    return out
