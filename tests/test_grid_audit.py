"""Tests for read-only grid audit helpers."""

from __future__ import annotations

import numpy as np

from src.calibration.grid_audit import (
    classify_failure_category,
    infer_rejection_reason,
    records_to_dataframe,
)
from src.calibration.grid_auto import GridCalibrationResult, SpacingEstimate
from src.calibration.grid_audit import GridAuditSignals
from src.config import ProjectConfig


def _dummy_signals(**kwargs) -> GridAuditSignals:
    defaults = dict(
        scale=1.0,
        fused_spacing_px=40.0,
        fusion_conf=0.5,
        n_horizontal_lines=8,
        n_vertical_lines=8,
        grid_valid=False,
        grid_confidence=0.35,
        est_h=SpacingEstimate(40.0, 0.5, n_lines=8, gap_cv=0.2),
        est_v=SpacingEstimate(0.0, 0.0, n_lines=2, gap_cv=1.0),
        est_corner=SpacingEstimate(0.0, 0.0),
        horiz_segs=[],
        vert_segs=[],
        rejected_segs=[],
        all_segments=[],
        horiz_angle=0.0,
        vert_angle=90.0,
        orthogonality_error=2.0,
        hv_ratio=float("nan"),
        spacing_variance=0.9,
        edges=np.zeros((10, 10), dtype=np.uint8),
    )
    defaults.update(kwargs)
    return GridAuditSignals(**defaults)  # type: ignore[arg-type]


def test_infer_partial_grid_reason() -> None:
    sig = _dummy_signals()
    grid = GridCalibrationResult(
        pixels_per_grid_square=0.0,
        pixels_per_mm=0.0,
        grid_square_mm=10.0,
        success=False,
        grid_valid=False,
    )
    cfg = ProjectConfig(use_grid_auto_calibration=True)
    reason = infer_rejection_reason(
        sig, grid, marker_ppm=5.0, scale_source="marker_fallback", cfg=cfg
    )
    assert "PARTIAL_GRID_DETECTED" in reason


def test_records_to_dataframe_columns() -> None:
    from src.calibration.grid_audit import GridAuditRecord

    rec = GridAuditRecord(
        image_id="test",
        grid_valid=False,
        grid_confidence=0.4,
        grid_success=False,
        rejection_reason="LOW_CONFIDENCE",
        failure_category="INSUFFICIENT_LINES",
        horizontal_spacing=0.0,
        vertical_spacing=0.0,
        hv_ratio=float("nan"),
        number_of_lines_detected=0,
        spacing_variance=1.0,
        orthogonality_error=5.0,
        fallback_triggered=True,
        marker_ppm=5.0,
        grid_ppm=0.0,
        marker_grid_ratio=float("nan"),
        final_scale_source="marker_fallback",
        absolute_error_if_grid_used=float("nan"),
        absolute_error_if_marker_used=0.0,
    )
    df = records_to_dataframe([rec])
    assert "rejection_reason" in df.columns
    assert df.iloc[0]["image_id"] == "test"
