"""Tests for automatic grid calibration."""

from __future__ import annotations

import numpy as np

from src.calibration.grid_auto import (
    estimate_grid_calibration,
    estimate_pixels_per_grid_square,
)


def test_synthetic_grid_spacing() -> None:
    """Checkerboard-like grid should yield non-zero spacing."""
    size = 400
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    step = 40
    for y in range(0, size, step):
        img[y : y + 2, :] = 30
    for x in range(0, size, step):
        img[:, x : x + 2] = 30

    spacing, n_h, n_v = estimate_pixels_per_grid_square(img)
    assert spacing > 0
    assert 20 <= spacing <= 80
    grid = estimate_grid_calibration(img)
    assert grid.success
    assert grid.grid_valid
    assert grid.grid_confidence > 0.4
    assert grid.pixels_per_mm > 0


def test_grid_calibration_failure_on_blank() -> None:
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    grid = estimate_grid_calibration(img)
    assert not grid.success
    assert not grid.grid_valid
    assert grid.pixels_per_mm == 0.0


def test_fusion_rejects_disagreeing_estimates() -> None:
    from src.calibration.grid_auto import SpacingEstimate, _fuse_spacing_estimates

    est = [
        SpacingEstimate(40.0, 0.9),
        SpacingEstimate(120.0, 0.3),
    ]
    fused, conf = _fuse_spacing_estimates(est)
    assert 35 <= fused <= 45
    assert conf > 0
