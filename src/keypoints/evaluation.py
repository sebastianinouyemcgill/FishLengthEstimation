"""
Evaluation hooks for future HRNet keypoints (pixel error + length proxy).

No model weights — compares predictions to pseudo labels or GT length when available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.calibration import CalibrationResult


@dataclass
class KeypointEvalSummary:
    """Aggregate metrics for a keypoint prediction table."""

    n_images: int
    mean_pixel_error: float
    mean_length_error_mm: float


def mean_endpoint_pixel_error(
    pred_xy: np.ndarray,
    ref_xy: np.ndarray,
) -> float:
    """
  Mean distance over matched endpoints (same count, paired by index).

  ``pred_xy`` and ``ref_xy`` shape ``(N, 2)``.
    """
    if pred_xy.shape != ref_xy.shape or len(pred_xy) == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(pred_xy - ref_xy, axis=1)))


def length_mm_from_keypoints(
    keypoints_xy: list[tuple[float, float]],
    pixels_per_mm: float,
) -> float:
    """Euclidean span between first and last keypoint in millimeters."""
    if len(keypoints_xy) < 2 or pixels_per_mm <= 0:
        return float("nan")
    (x0, y0), (x1, y1) = keypoints_xy[0], keypoints_xy[-1]
    span_px = float(np.hypot(x1 - x0, y1 - y0))
    return span_px / pixels_per_mm


def evaluate_keypoint_predictions(
    predictions: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    calib: CalibrationResult,
    image_id_col: str = "image_id",
    pred_cols: tuple[str, str, str, str] = ("x0", "y0", "x1", "y1"),
    ref_cols: tuple[str, str, str, str] = ("x0", "y0", "x1", "y1"),
    gt_length_col: str | None = "length_mm",
) -> KeypointEvalSummary:
    """
    Compare predicted vs reference keypoints and optional GT length.

    ``predictions`` and ``reference`` are merged on ``image_id_col``.
    """
    merged = predictions.merge(
        reference,
        on=image_id_col,
        suffixes=("_pred", "_ref"),
    )
    if merged.empty:
        return KeypointEvalSummary(0, float("nan"), float("nan"))

    pixel_errors: list[float] = []
    length_errors: list[float] = []
    ppm = calib.pixels_per_mm

    def _col(name: str, suffix: str) -> str:
        key = f"{name}{suffix}"
        return key if key in merged.columns else name

    for _, row in merged.iterrows():
        pred = np.array(
            [
                [row[_col(pred_cols[0], "_pred")], row[_col(pred_cols[1], "_pred")]],
                [row[_col(pred_cols[2], "_pred")], row[_col(pred_cols[3], "_pred")]],
            ],
            dtype=np.float64,
        )
        ref = np.array(
            [
                [row[_col(ref_cols[0], "_ref")], row[_col(ref_cols[1], "_ref")]],
                [row[_col(ref_cols[2], "_ref")], row[_col(ref_cols[3], "_ref")]],
            ],
            dtype=np.float64,
        )
        pixel_errors.append(mean_endpoint_pixel_error(pred, ref))
        if ppm > 0:
            pred_len = length_mm_from_keypoints(
                [(pred[0, 0], pred[0, 1]), (pred[1, 0], pred[1, 1])],
                ppm,
            )
            if gt_length_col and f"{gt_length_col}" in row.index and np.isfinite(row[gt_length_col]):
                length_errors.append(abs(pred_len - float(row[gt_length_col])))

    return KeypointEvalSummary(
        n_images=len(merged),
        mean_pixel_error=float(np.nanmean(pixel_errors)),
        mean_length_error_mm=float(np.nanmean(length_errors)) if length_errors else float("nan"),
    )
