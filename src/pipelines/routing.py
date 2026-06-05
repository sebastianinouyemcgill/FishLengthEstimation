"""
Pipeline routing and feature-flag policy.

Separates production paths (baseline measurement, grid calibration) from
archived experimental paths (depth, 3D, perspective).
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace

from src.config import ProjectConfig

logger = logging.getLogger(__name__)


def experimental_allowed() -> bool:
    """When False (default), depth / 3D / perspective are clamped off for runs."""
    return os.environ.get("FISHNET_ALLOW_EXPERIMENTAL", "").lower() in ("1", "true", "yes")


def uses_experimental_features(cfg: ProjectConfig) -> bool:
    """True when any archived experimental stage is enabled on config."""
    return bool(
        cfg.use_depth_estimation
        or cfg.use_3d_measurement
        or cfg.use_perspective
        or cfg.apply_perspective_correction
    )


def uses_production_advanced_features(cfg: ProjectConfig) -> bool:
    """Grid calibration and future HRNet keypoints (official advanced benchmark)."""
    return bool(cfg.use_grid_auto_calibration or cfg.use_hrnet_keypoints)


def uses_advanced_inference_path(cfg: ProjectConfig) -> bool:
    """
    Route to ``run_advanced_inference`` instead of baseline ``run_inference``.

    Perspective-only rectification stays on ``run_inference`` (homography in base loop).
    """
    if uses_production_advanced_features(cfg):
        return True
    return bool(cfg.use_depth_estimation or cfg.use_3d_measurement)


def clamp_experimental_flags(cfg: ProjectConfig) -> ProjectConfig:
    """
    Force depth, 3D, and perspective off unless ``FISHNET_ALLOW_EXPERIMENTAL=1``.

    Does not modify grid, HRNet, or baseline measurement settings.
    """
    if experimental_allowed() or not uses_experimental_features(cfg):
        return _sync_perspective_aliases(cfg)
    logger.warning(
        "Experimental features (depth/3D/perspective) requested but disabled by policy; "
        "set FISHNET_ALLOW_EXPERIMENTAL=1 to enable archived paths"
    )
    return _sync_perspective_aliases(
        replace(
            cfg,
            use_depth_estimation=False,
            use_depth_model=False,
            use_3d_measurement=False,
            use_perspective=False,
            apply_perspective_correction=False,
            use_depth_metric_scale=False,
        )
    )


def resolve_run_config(
    cfg: ProjectConfig,
    *,
    pipeline: str,
    perspective: bool | None = None,
    use_grid_auto_calibration: bool | None = None,
    use_depth_estimation: bool | None = None,
    use_3d_measurement: bool | None = None,
    use_hrnet_keypoints: bool | None = None,
) -> tuple[ProjectConfig, bool, bool, bool, bool, bool]:
    """
    Apply pipeline policy and return (cfg, perspective, grid, depth, 3d, hrnet).

    Baseline never enables perspective or experimental flags.
    """
    cfg = replace(cfg)

    if pipeline == "baseline":
        cfg = replace(
            cfg,
            use_perspective=False,
            apply_perspective_correction=False,
            use_depth_estimation=False,
            use_depth_model=False,
            use_3d_measurement=False,
            use_grid_auto_calibration=False,
            use_grid_calibration=False,
            use_hrnet_keypoints=False,
        )
        return cfg, False, False, False, False, False

    use_perspective = bool(perspective) if perspective is not None else cfg.use_perspective
    adv_grid = (
        use_grid_auto_calibration
        if use_grid_auto_calibration is not None
        else cfg.use_grid_auto_calibration
    )
    adv_depth = (
        use_depth_estimation if use_depth_estimation is not None else cfg.use_depth_estimation
    )
    adv_3d = use_3d_measurement if use_3d_measurement is not None else cfg.use_3d_measurement
    adv_hrnet = (
        use_hrnet_keypoints if use_hrnet_keypoints is not None else cfg.use_hrnet_keypoints
    )

    cfg = replace(
        cfg,
        use_perspective=use_perspective,
        apply_perspective_correction=use_perspective,
        use_grid_auto_calibration=adv_grid,
        use_depth_estimation=adv_depth,
        use_3d_measurement=adv_3d,
        use_hrnet_keypoints=adv_hrnet,
    )
    cfg = clamp_experimental_flags(cfg)

    if uses_production_advanced_features(cfg) and cfg.use_perspective:
        logger.info(
            "Disabling perspective: production advanced path uses raw image space (grid/HRNet)"
        )
        cfg = replace(cfg, use_perspective=False, apply_perspective_correction=False)

    return (
        cfg,
        cfg.use_perspective,
        cfg.use_grid_auto_calibration,
        cfg.use_depth_estimation,
        cfg.use_3d_measurement,
        cfg.use_hrnet_keypoints,
    )


def _sync_perspective_aliases(cfg: ProjectConfig) -> ProjectConfig:
    synced = cfg.use_perspective or cfg.apply_perspective_correction
    if synced == cfg.use_perspective and synced == cfg.apply_perspective_correction:
        return cfg
    return replace(cfg, use_perspective=synced, apply_perspective_correction=synced)
