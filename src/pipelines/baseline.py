"""
Baseline pipeline: geometric measurement only (no perspective correction).

Uses provided fish and marker polygons, calibration, and bbox/pca/skeleton methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import ProjectConfig, get_config
from src.pipelines.base import run_inference

logger = logging.getLogger(__name__)


@dataclass
class BaselinePipeline:
    """
    Assignment baseline — no ML, no homography rectification.

    Methods: ``bbox``, ``pca``, ``skeleton``.
    """

    name: str = "baseline"

    def configure(
        self,
        cfg: ProjectConfig,
        *,
        method: str,
        split: str,
    ) -> ProjectConfig:
        cfg.default_split = split
        cfg.measurement_method = method
        cfg.apply_perspective_correction = False
        return cfg

    def run(
        self,
        cfg: ProjectConfig | None = None,
        *,
        method: str = "bbox",
        split: str = "test",
        predictions_path: Path,
        limit: int | None = None,
        image_ids: list[str] | None = None,
        visualize: bool = False,
        figures_dir: Path | None = None,
    ) -> Path:
        cfg = self.configure(cfg or get_config(), method=method, split=split)
        logger.info(
            "BaselinePipeline: split=%s method=%s perspective=False n_images=%s",
            split,
            method,
            len(image_ids) if image_ids else f"all(limit={limit})",
        )
        if cfg.use_regression_model:
            if not cfg.regression_model_path or not Path(cfg.regression_model_path).is_file():
                raise FileNotFoundError(
                    "use_regression_model=True requires cfg.regression_model_path "
                    "to an existing model file (run run_regression_experiment first)."
                )
            from src.models.length_regression import LengthRegressionModel
            from src.pipelines.regression_inference import run_regression_inference

            model = LengthRegressionModel.load(cfg.regression_model_path)
            return run_regression_inference(
                cfg,
                split=split,
                predictions_path=predictions_path,
                model=model,
                method=method,
                limit=limit,
                image_ids=image_ids,
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
