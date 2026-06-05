"""
Optional regression stage: baseline features → calibrated SH length (mm).

Does not modify ``pipelines.base.run_inference``; baseline experiments stay identical
when ``cfg.use_regression_model`` is False.
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
from src.measurement.features import extract_length_features
from src.models.length_regression import LengthRegressionModel

logger = logging.getLogger(__name__)


def run_regression_inference(
    cfg: ProjectConfig,
    split: str,
    predictions_path: Path,
    model: LengthRegressionModel,
    *,
    method: str = "skeleton",
    limit: int | None = None,
    image_ids: list[str] | None = None,
) -> Path:
    """
    Run mask → features → regression and write ``predictions.csv``.

    Also stores ``skeleton_length_mm`` (baseline) for MAE comparison.
    """
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []

    if image_ids is not None:
        iterator = iterate_image_ids(cfg, split=split, image_ids=image_ids, load_images=True)
        desc = f"Regression ({split}, n={len(image_ids)})"
    else:
        iterator = iterate_dataset(cfg, split=split, load_images=True, limit=limit)
        desc = f"Regression ({split})"

    for sample in tqdm(iterator, desc=desc, unit="img"):
        if sample.image is None:
            logger.warning("Skipping %s: no image", sample.image_id)
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

        features = extract_length_features(fish_mask, calib)
        skeleton_mm = estimate_length_mm(fish_mask, calib, method="skeleton")
        corrected_mm = float(model.predict(features.as_array().reshape(1, -1))[0])

        row: dict[str, str | float] = {
            "image_id": sample.image_id,
            "predicted_length_mm": corrected_mm,
            "skeleton_length_mm": skeleton_mm,
        }
        row.update(features.as_dict())
        rows.append(row)
        logger.debug(
            "%s: skeleton=%.2f mm regression=%.2f mm",
            sample.image_id,
            skeleton_mm,
            corrected_mm,
        )

    df = pd.DataFrame(rows)
    df.to_csv(predictions_path, index=False)
    logger.info("Wrote %d regression predictions to %s", len(df), predictions_path)
    return predictions_path
