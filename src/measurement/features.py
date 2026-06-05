"""
Geometric features for length regression calibration.

Mask-derived measurements (skeleton, PCA, bbox diagonal) are converted to mm using
marker calibration. Shape descriptors (area, perimeter, aspect ratio) stay in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.calibration import CalibrationResult
from src.masks import cleanup_mask
from src.measurement.core import (
    measure_bbox_length,
    measure_pca_length,
    measure_skeleton_length,
    pixels_to_mm,
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "skeleton_length",
    "pca_length",
    "bounding_box_diagonal_length",
    "mask_area",
    "mask_perimeter",
    "aspect_ratio",
)


@dataclass(frozen=True)
class LengthFeatures:
    """Per-image feature vector for regression."""

    skeleton_length: float
    pca_length: float
    bounding_box_diagonal_length: float
    mask_area: float
    mask_perimeter: float
    aspect_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {
            "skeleton_length": self.skeleton_length,
            "pca_length": self.pca_length,
            "bounding_box_diagonal_length": self.bounding_box_diagonal_length,
            "mask_area": self.mask_area,
            "mask_perimeter": self.mask_perimeter,
            "aspect_ratio": self.aspect_ratio,
        }

    def as_array(self) -> np.ndarray:
        return np.array([self.as_dict()[c] for c in FEATURE_COLUMNS], dtype=np.float64)


def _mask_bbox_aspect_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    if height <= 0:
        return 0.0
    return width / height


def _mask_area_and_perimeter(mask: np.ndarray) -> tuple[float, float]:
    binary = (mask > 0).astype(np.uint8)
    area = float(binary.sum())
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return area, 0.0
    perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), closed=True))
    return area, perimeter


def extract_length_features(
    fish_mask: np.ndarray,
    calibration: CalibrationResult,
    *,
    preprocess: bool = True,
) -> LengthFeatures:
    """
    Compute regression inputs from a fish mask and marker calibration.

    Length features are in millimeters; ``mask_area`` / ``mask_perimeter`` are in pixels.
    """
    mask = cleanup_mask(fish_mask) if preprocess else fish_mask
    ppm = calibration.pixels_per_mm
    skeleton_mm = pixels_to_mm(measure_skeleton_length(mask), ppm)
    pca_mm = pixels_to_mm(measure_pca_length(mask)[0], ppm)
    bbox_mm = pixels_to_mm(measure_bbox_length(mask), ppm)
    area, perimeter = _mask_area_and_perimeter(mask)
    aspect = _mask_bbox_aspect_ratio(mask)
    return LengthFeatures(
        skeleton_length=skeleton_mm,
        pca_length=pca_mm,
        bounding_box_diagonal_length=bbox_mm,
        mask_area=area,
        mask_perimeter=perimeter,
        aspect_ratio=aspect,
    )
