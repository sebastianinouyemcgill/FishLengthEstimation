"""Pipeline routing and experimental policy."""

from __future__ import annotations

from src.config import ProjectConfig
from src.pipelines.advanced_inference import uses_advanced_features
from src.pipelines.routing import (
    clamp_experimental_flags,
    resolve_run_config,
    uses_advanced_inference_path,
    uses_experimental_features,
    uses_production_advanced_features,
)


def test_baseline_clamps_experimental() -> None:
    cfg = ProjectConfig(
        use_depth_estimation=True,
        use_3d_measurement=True,
        use_perspective=True,
    )
    out, persp, grid, depth, three_d, hrnet = resolve_run_config(
        cfg, pipeline="baseline", perspective=True
    )
    assert not persp
    assert not grid
    assert not depth
    assert not three_d
    assert not hrnet
    assert not uses_experimental_features(out)


def test_advanced_grid_uses_inference_path() -> None:
    cfg = ProjectConfig(use_grid_auto_calibration=True)
    out, *_ = resolve_run_config(cfg, pipeline="advanced")
    assert uses_production_advanced_features(out)
    assert uses_advanced_inference_path(out)
    assert uses_advanced_features(out)


def test_depth_clamped_without_allow_env(monkeypatch) -> None:
    monkeypatch.delenv("FISHNET_ALLOW_EXPERIMENTAL", raising=False)
    cfg = ProjectConfig(use_grid_auto_calibration=True, use_depth_estimation=True)
    clamped = clamp_experimental_flags(cfg)
    assert not clamped.use_depth_estimation
    assert clamped.use_grid_auto_calibration


def test_perspective_only_uses_baseline_loop() -> None:
    cfg = ProjectConfig(use_perspective=True)
    assert uses_experimental_features(cfg)
    assert not uses_advanced_inference_path(cfg)
