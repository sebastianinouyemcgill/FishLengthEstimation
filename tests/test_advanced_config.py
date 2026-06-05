"""Advanced pipeline config defaults and routing."""

from __future__ import annotations

from src.config import ProjectConfig, get_config
from src.pipelines.advanced_inference import uses_advanced_features


def test_advanced_flags_default_false() -> None:
    cfg = get_config()
    assert not cfg.use_grid_auto_calibration
    assert not cfg.use_depth_estimation
    assert not cfg.use_3d_measurement
    assert not uses_advanced_features(cfg)


def test_uses_advanced_features_when_grid_on() -> None:
    cfg = ProjectConfig(use_grid_auto_calibration=True)
    assert uses_advanced_features(cfg)


def test_depth_alone_does_not_use_grid_advanced_path() -> None:
    from src.pipelines.routing import uses_advanced_inference_path

    cfg = ProjectConfig(use_depth_estimation=True)
    assert uses_advanced_inference_path(cfg)
