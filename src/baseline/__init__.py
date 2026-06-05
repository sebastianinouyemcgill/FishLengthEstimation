"""
Official benchmark: 2D fish measurement (bbox, PCA, skeleton).

Re-exports stable APIs; implementation lives in ``measurement``, ``masks``, and
``pipelines.base`` / ``pipelines.baseline``.
"""

from src.measurement import (
    MeasurementMethod,
    estimate_length_mm,
    measure_bbox_length,
    measure_fish_length,
    measure_pca_length,
    measure_skeleton_length,
    pixels_to_mm,
)
from src.masks import cleanup_mask, extract_contours, mask_from_class, skeletonize_mask
from src.pipelines.base import run_inference
from src.pipelines.baseline import BaselinePipeline

__all__ = [
    "BaselinePipeline",
    "MeasurementMethod",
    "cleanup_mask",
    "estimate_length_mm",
    "extract_contours",
    "mask_from_class",
    "measure_bbox_length",
    "measure_fish_length",
    "measure_pca_length",
    "measure_skeleton_length",
    "pixels_to_mm",
    "run_inference",
    "skeletonize_mask",
]
