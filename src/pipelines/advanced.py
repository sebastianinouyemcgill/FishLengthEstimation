"""
Advanced pipeline: grid calibration, depth, 3D skeleton, optional perspective.

When no advanced feature flags are set, delegates to the shared baseline inference
loop (marker calibration + 2D methods) for backwards-compatible behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import ProjectConfig, get_config
from src.pipelines.advanced_inference import run_advanced_inference
from src.pipelines.routing import uses_advanced_inference_path, uses_production_advanced_features
from src.pipelines.base import run_inference

logger = logging.getLogger(__name__)


@dataclass
class AdvancedPipeline:
    """
    Extensions beyond the assignment baseline.

    Feature flags (all default False) control grid calibration, depth, and 3D length.
    """

    name: str = "advanced"

    def configure(
        self,
        cfg: ProjectConfig,
        *,
        method: str,
        split: str,
        perspective: bool = False,
        use_grid_auto_calibration: bool | None = None,
        use_depth_estimation: bool | None = None,
        use_3d_measurement: bool | None = None,
    ) -> ProjectConfig:
        cfg.default_split = split
        cfg.measurement_method = method
        if use_grid_auto_calibration is not None:
            cfg.use_grid_auto_calibration = use_grid_auto_calibration
        if use_depth_estimation is not None:
            cfg.use_depth_estimation = use_depth_estimation
        if use_3d_measurement is not None:
            cfg.use_3d_measurement = use_3d_measurement

        cfg.use_perspective = perspective
        cfg.apply_perspective_correction = perspective

        if uses_production_advanced_features(cfg) and perspective:
            logger.info(
                "Ignoring perspective=True: grid/HRNet advanced path uses raw image space"
            )
            cfg.use_perspective = False
            cfg.apply_perspective_correction = False
        return cfg

    def run(
        self,
        cfg: ProjectConfig | None = None,
        *,
        method: str = "pca",
        split: str = "test",
        perspective: bool = False,
        predictions_path: Path,
        limit: int | None = None,
        image_ids: list[str] | None = None,
        visualize: bool = False,
        figures_dir: Path | None = None,
        use_grid_auto_calibration: bool | None = None,
        use_depth_estimation: bool | None = None,
        use_3d_measurement: bool | None = None,
    ) -> Path:
        cfg = self.configure(
            cfg or get_config(),
            method=method,
            split=split,
            perspective=perspective,
            use_grid_auto_calibration=use_grid_auto_calibration,
            use_depth_estimation=use_depth_estimation,
            use_3d_measurement=use_3d_measurement,
        )

        if cfg.use_3d_measurement and method != "skeleton":
            logger.info(
                "Advanced 3D measurement uses skeleton; overriding method %s -> skeleton",
                method,
            )
            method = "skeleton"

        logger.info(
            "AdvancedPipeline: split=%s method=%s grid=%s depth=%s 3d=%s perspective=%s",
            split,
            method,
            cfg.use_grid_auto_calibration,
            cfg.use_depth_estimation,
            cfg.use_3d_measurement,
            cfg.use_perspective,
        )

        if uses_advanced_inference_path(cfg):
            return run_advanced_inference(
                cfg,
                split=split,
                method=method,
                predictions_path=predictions_path,
                limit=limit,
                image_ids=image_ids,
                visualize=visualize,
                figures_dir=figures_dir,
            )

        return run_inference(
            cfg,
            split=split,
            method=method,
            predictions_path=predictions_path,
            limit=limit,
            image_ids=image_ids,
            visualize=visualize,
            figures_dir=figures_dir,
        )
