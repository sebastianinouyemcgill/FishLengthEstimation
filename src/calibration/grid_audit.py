"""
Read-only grid calibration auditor.

Inspects accept/reject decisions and produces figures, tables, and reports.
Does not modify ``grid_auto`` calibration behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration.grid_auto import (
    GridCalibrationResult,
    LineSegment,
    SpacingEstimate,
    _angle_delta,
    _analyze_grid,
    _dominant_orthogonal_angles,
    _filter_segments_roi,
    _fuse_spacing_estimates,
    _HV_RATIO_MAX,
    _HV_RATIO_MIN,
    _MIN_FAMILY_LINES,
    _MIN_GRID_CONFIDENCE,
    _MIN_SPACING_PX,
    _MAX_SPACING_PX,
    _preprocess_gray,
    _refine_family_segments,
    _segments_for_angle,
    estimate_grid_calibration,
)
from src.calibration.marker import CalibrationResult, calibrate_sample, estimate_scale_from_markers
from src.config import ProjectConfig, get_config
from src.dataset import DatasetSample, iterate_image_ids
from src.evaluation import load_ground_truth_csv
from src.masks import mask_from_class
from src.measurement import estimate_length_mm

logger = logging.getLogger(__name__)

FAILURE_CATEGORIES = (
    "NO_GRID_DETECTED",
    "PARTIAL_GRID_DETECTED",
    "EDGE_CONFUSION",
    "PERSPECTIVE_FAILURE",
    "INSUFFICIENT_LINES",
    "HIGH_VARIANCE_SPACING",
    "ACCEPTED",
    "OTHER",
)


@dataclass
class GridAuditSignals:
    """Internal signals extracted from one image (read-only introspection)."""

    scale: float
    fused_spacing_px: float
    fusion_conf: float
    n_horizontal_lines: int
    n_vertical_lines: int
    grid_valid: bool
    grid_confidence: float
    est_h: SpacingEstimate
    est_v: SpacingEstimate
    est_corner: SpacingEstimate
    horiz_segs: list[LineSegment]
    vert_segs: list[LineSegment]
    rejected_segs: list[LineSegment]
    all_segments: list[LineSegment]
    horiz_angle: float
    vert_angle: float
    orthogonality_error: float
    hv_ratio: float
    spacing_variance: float
    edges: np.ndarray


@dataclass
class GridAuditRecord:
    """Per-image audit row with diagnostics and pipeline outcomes."""

    image_id: str
    grid_valid: bool
    grid_confidence: float
    grid_success: bool
    rejection_reason: str
    failure_category: str
    horizontal_spacing: float
    vertical_spacing: float
    hv_ratio: float
    number_of_lines_detected: int
    spacing_variance: float
    orthogonality_error: float
    fallback_triggered: bool
    marker_ppm: float
    grid_ppm: float
    marker_grid_ratio: float
    final_scale_source: str
    absolute_error_if_grid_used: float
    absolute_error_if_marker_used: float
    length_mm_gt: float | None = None
    length_mm_pred_grid: float | None = None
    length_mm_pred_marker: float | None = None
    signals: GridAuditSignals | None = field(default=None, repr=False)


def extract_grid_audit_signals(image_bgr: np.ndarray) -> GridAuditSignals:
    """Mirror ``_analyze_grid`` and expose intermediate geometry (no logic changes)."""
    from src.calibration.grid_auto import (
        _detect_line_segments_xy,
        _FUSION_DISAGREE_FRAC,
        _spacing_from_family,
        _spacing_from_harris,
    )

    gray, scale, _ = _preprocess_gray(image_bgr)
    all_segments, _edges = _detect_line_segments_xy(gray)
    all_segments = _filter_segments_roi(all_segments, gray)

    angle_a, angle_b = _dominant_orthogonal_angles(all_segments)
    if _angle_delta(angle_a, 0.0) <= _angle_delta(angle_a, 90.0):
        horiz_angle, vert_angle = angle_a, angle_b
    else:
        horiz_angle, vert_angle = angle_b, angle_a

    horiz_segs = _refine_family_segments(
        _segments_for_angle(all_segments, horiz_angle), gray, horizontal=True
    )
    vert_segs = _refine_family_segments(
        _segments_for_angle(all_segments, vert_angle), gray, horizontal=False
    )

    est_h_work = _spacing_from_family(horiz_segs, gray, horizontal=True)
    est_v_work = _spacing_from_family(vert_segs, gray, horizontal=False)
    est_corner = _spacing_from_harris(gray)

    harris_estimates: list[SpacingEstimate] = []
    if est_corner.spacing_px > 0:
        refs = [e.spacing_px for e in (est_h_work, est_v_work) if e.spacing_px > 0]
        if refs:
            med_ref = float(np.median(refs))
            if abs(est_corner.spacing_px - med_ref) / med_ref <= _FUSION_DISAGREE_FRAC:
                harris_estimates.append(est_corner)
        else:
            harris_estimates.append(est_corner)

    _fused_work, fusion_conf = _fuse_spacing_estimates([est_h_work, est_v_work, *harris_estimates])
    ortho_err = abs(_angle_delta(horiz_angle, vert_angle) - 90.0)

    (
        fused_full,
        n_h,
        n_v,
        grid_valid,
        grid_conf,
        _all,
        _kept,
        rejected,
        est_h_out,
        est_v_out,
        _est_c,
        edges_out,
        scale_out,
    ) = _analyze_grid(image_bgr)

    hv_ratio = (
        est_h_out.spacing_px / est_v_out.spacing_px
        if est_h_out.spacing_px > 0 and est_v_out.spacing_px > 0
        else float("nan")
    )
    spacing_variance = float(np.mean([est_h_out.gap_cv, est_v_out.gap_cv]))

    return GridAuditSignals(
        scale=scale_out,
        fused_spacing_px=fused_full,
        fusion_conf=fusion_conf,
        n_horizontal_lines=n_h,
        n_vertical_lines=n_v,
        grid_valid=grid_valid,
        grid_confidence=grid_conf,
        est_h=est_h_out,
        est_v=est_v_out,
        est_corner=est_corner,
        horiz_segs=horiz_segs,
        vert_segs=vert_segs,
        rejected_segs=rejected,
        all_segments=all_segments,
        horiz_angle=horiz_angle,
        vert_angle=vert_angle,
        orthogonality_error=ortho_err,
        hv_ratio=hv_ratio if np.isfinite(hv_ratio) else float("nan"),
        spacing_variance=spacing_variance,
        edges=edges_out,
    )


def infer_rejection_reason(
    sig: GridAuditSignals,
    grid: GridCalibrationResult,
    *,
    marker_ppm: float,
    scale_source: str,
    cfg: ProjectConfig,
) -> str:
    """Explain accept/reject using the same thresholds as ``grid_auto`` (read-only)."""
    tags: list[str] = []

    if len(sig.all_segments) == 0:
        tags.append("NO_GRID_DETECTED")
    if sig.n_horizontal_lines < _MIN_FAMILY_LINES:
        tags.append("INSUFFICIENT_HORIZONTAL_LINES")
    if sig.n_vertical_lines < _MIN_FAMILY_LINES:
        tags.append("INSUFFICIENT_VERTICAL_LINES")
    if sig.est_h.spacing_px <= 0 and sig.est_v.spacing_px <= 0:
        if "NO_GRID_DETECTED" not in tags:
            tags.append("NO_SPACING_ESTIMATE")
    elif sig.est_h.spacing_px <= 0 or sig.est_v.spacing_px <= 0:
        tags.append("PARTIAL_GRID_DETECTED")
    if sig.fused_spacing_px <= 0:
        tags.append("NO_FUSED_SPACING")
    if sig.orthogonality_error > 20.0:
        tags.append("PERSPECTIVE_FAILURE")
    if np.isfinite(sig.hv_ratio) and not (_HV_RATIO_MIN <= sig.hv_ratio <= _HV_RATIO_MAX):
        tags.append("HV_RATIO_OUT_OF_RANGE")
    if sig.spacing_variance > 0.75:
        tags.append("HIGH_VARIANCE_SPACING")
    if sig.grid_confidence < _MIN_GRID_CONFIDENCE:
        tags.append("LOW_CONFIDENCE")
    if len(sig.all_segments) >= 60 and not sig.grid_valid:
        tags.append("EDGE_CONFUSION")
    max_full = _MAX_SPACING_PX / sig.scale if 0 < sig.scale < 1.0 else _MAX_SPACING_PX
    if sig.fused_spacing_px > 0 and (
        sig.fused_spacing_px < _MIN_SPACING_PX or sig.fused_spacing_px > max_full
    ):
        tags.append("SPACING_OUT_OF_BOUNDS")

    if sig.grid_valid and grid.success:
        tags.append("GEOMETRY_PASSED")
    elif not tags:
        tags.append("GEOMETRY_FAILED")

    if scale_source != "grid":
        if grid.success and marker_ppm > 0:
            tags.append("MARKER_RATIO_GATE")
        elif not grid.success:
            tags.append("MARKER_FALLBACK")

    if scale_source == "grid":
        tags.append("FINAL_ACCEPTED_GRID")

    return "; ".join(dict.fromkeys(tags))


def classify_failure_category(
    sig: GridAuditSignals,
    grid: GridCalibrationResult,
    *,
    scale_source: str,
) -> str:
    """Single primary category for rejected (or accepted) images."""
    if scale_source == "grid" and grid.grid_valid:
        return "ACCEPTED"
    if len(sig.all_segments) == 0 or sig.fused_spacing_px <= 0:
        return "NO_GRID_DETECTED"
    if sig.est_h.spacing_px <= 0 or sig.est_v.spacing_px <= 0:
        if sig.est_h.spacing_px > 0 or sig.est_v.spacing_px > 0:
            return "PARTIAL_GRID_DETECTED"
        return "NO_GRID_DETECTED"
    if sig.n_horizontal_lines < _MIN_FAMILY_LINES or sig.n_vertical_lines < _MIN_FAMILY_LINES:
        return "INSUFFICIENT_LINES"
    if sig.orthogonality_error > 20.0:
        return "PERSPECTIVE_FAILURE"
    if sig.spacing_variance > 0.75:
        return "HIGH_VARIANCE_SPACING"
    if len(sig.all_segments) >= 60:
        return "EDGE_CONFUSION"
    if not grid.grid_valid:
        return "OTHER"
    return "OTHER"


def _length_predictions_mm(
    sample: DatasetSample,
    *,
    grid_ppm: float,
    marker_ppm: float,
) -> tuple[float | None, float | None]:
    if sample.image is None:
        return None, None
    mask = mask_from_class(
        sample.annotations,
        class_name="fish",
        height=sample.image.shape[0],
        width=sample.image.shape[1],
    )
    pred_grid = None
    pred_marker = None
    if grid_ppm > 0:
        pred_grid = estimate_length_mm(mask, CalibrationResult(pixels_per_mm=grid_ppm), method="bbox")
    if marker_ppm > 0:
        pred_marker = estimate_length_mm(mask, CalibrationResult(pixels_per_mm=marker_ppm), method="bbox")
    return pred_grid, pred_marker


def audit_single_image(
    sample: DatasetSample,
    cfg: ProjectConfig,
) -> GridAuditRecord:
    """Run calibration + introspection for one loaded sample."""
    from src.pipelines.advanced_inference import _choose_calibration

    if sample.image is None:
        raise ValueError(f"{sample.image_id}: image not loaded")

    img = sample.image
    sig = extract_grid_audit_signals(img)
    grid = estimate_grid_calibration(img, cfg=cfg)
    marker_calib = calibrate_sample(sample, cfg=cfg)
    marker_ppm = marker_calib.pixels_per_mm
    if marker_ppm <= 0:
        marker_ppm = estimate_scale_from_markers(
            sample.blue_annotations(),
            sample.yellow_annotations(),
            sample.width,
            sample.height,
            physical_length_mm=cfg.calibration_rect_mm,
        )

    _, scale_source, _, grid_ppm_chosen, _ = _choose_calibration(sample, cfg, img, grid=grid)
    grid_ppm = grid.pixels_per_mm if grid.success else 0.0

    abs_grid = abs(grid_ppm - marker_ppm) if grid_ppm > 0 and marker_ppm > 0 else float("nan")
    abs_marker = 0.0 if marker_ppm > 0 else float("nan")
    ratio = grid_ppm / marker_ppm if grid_ppm > 0 and marker_ppm > 0 else float("nan")

    rejection_reason = infer_rejection_reason(
        sig,
        grid,
        marker_ppm=marker_ppm,
        scale_source=scale_source,
        cfg=cfg,
    )
    failure_category = classify_failure_category(sig, grid, scale_source=scale_source)

    pred_grid, pred_marker = _length_predictions_mm(sample, grid_ppm=grid_ppm, marker_ppm=marker_ppm)

    return GridAuditRecord(
        image_id=sample.image_id,
        grid_valid=grid.grid_valid,
        grid_confidence=grid.grid_confidence,
        grid_success=grid.success,
        rejection_reason=rejection_reason,
        failure_category=failure_category,
        horizontal_spacing=sig.est_h.spacing_px,
        vertical_spacing=sig.est_v.spacing_px,
        hv_ratio=sig.hv_ratio if np.isfinite(sig.hv_ratio) else float("nan"),
        number_of_lines_detected=sig.n_horizontal_lines + sig.n_vertical_lines,
        spacing_variance=sig.spacing_variance,
        orthogonality_error=sig.orthogonality_error,
        fallback_triggered=scale_source != "grid",
        marker_ppm=marker_ppm,
        grid_ppm=grid_ppm,
        marker_grid_ratio=ratio,
        final_scale_source=scale_source,
        absolute_error_if_grid_used=abs_grid,
        absolute_error_if_marker_used=abs_marker,
        length_mm_pred_grid=pred_grid,
        length_mm_pred_marker=pred_marker,
        signals=sig,
    )


def records_to_dataframe(records: Iterable[GridAuditRecord]) -> pd.DataFrame:
    """Flatten audit records (drops non-serializable ``signals``)."""
    rows = []
    for rec in records:
        rows.append(
            {
                "image_id": rec.image_id,
                "grid_valid": rec.grid_valid,
                "grid_confidence": rec.grid_confidence,
                "grid_success": rec.grid_success,
                "rejection_reason": rec.rejection_reason,
                "failure_category": rec.failure_category,
                "horizontal_spacing": rec.horizontal_spacing,
                "vertical_spacing": rec.vertical_spacing,
                "hv_ratio": rec.hv_ratio,
                "number_of_lines_detected": rec.number_of_lines_detected,
                "spacing_variance": rec.spacing_variance,
                "orthogonality_error": rec.orthogonality_error,
                "fallback_triggered": rec.fallback_triggered,
                "marker_ppm": rec.marker_ppm,
                "grid_ppm": rec.grid_ppm,
                "marker_grid_ratio": rec.marker_grid_ratio,
                "final_scale_source": rec.final_scale_source,
                "absolute_error_if_grid_used": rec.absolute_error_if_grid_used,
                "absolute_error_if_marker_used": rec.absolute_error_if_marker_used,
                "length_mm_pred_grid": rec.length_mm_pred_grid,
                "length_mm_pred_marker": rec.length_mm_pred_marker,
                "length_mm_gt": rec.length_mm_gt,
            }
        )
    return pd.DataFrame(rows)


def attach_ground_truth(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Add ``length_mm_gt`` and length absolute errors when GT CSV exists."""
    from src.experiments import default_ground_truth_path

    out = df.copy()
    if "length_mm_gt" not in out.columns or out["length_mm_gt"].isna().all():
        gt_path = default_ground_truth_path(cfg)
        if gt_path is None or not gt_path.is_file():
            return _append_length_error_columns(out)
        gt = load_ground_truth_csv(gt_path)
        out = out.drop(columns=[c for c in out.columns if c.startswith("length_mm_gt")], errors="ignore")
        out = out.merge(
            gt[["image_id", "length_mm"]].rename(columns={"length_mm": "length_mm_gt"}),
            on="image_id",
            how="left",
        )
    return _append_length_error_columns(out)


def _append_length_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "length_mm_pred_grid" in out.columns and "length_mm_gt" in out.columns:
        out["abs_err_length_grid_mm"] = (out["length_mm_pred_grid"] - out["length_mm_gt"]).abs()
        out["abs_err_length_marker_mm"] = (out["length_mm_pred_marker"] - out["length_mm_gt"]).abs()
    return out


def analyze_grid_calibration_set(
    *,
    cfg: ProjectConfig | None = None,
    image_ids: list[str] | None = None,
    split: str = "valid",
    limit: int | None = 30,
    run_name: str = "grid_audit",
    output_dir: Path | None = None,
    generate_figures: bool = True,
) -> pd.DataFrame:
    """
    Run grid calibration audit on a dataset subset.

    Writes CSV tables and a markdown report under ``runs/<run_name>/grid_audit/``.
    """
    from src.experiments import default_ground_truth_path, load_validation_image_ids

    cfg = cfg or get_config()
    cfg_audit = ProjectConfig(
        repo_root=cfg.repo_root,
        storage_root=cfg.storage_root,
        data_root=cfg.data_root,
        data_raw=cfg.data_raw,
        data_annotations=cfg.data_annotations,
        data_processed=cfg.data_processed,
        runs_root=cfg.runs_root,
        figures_root=cfg.figures_root,
        outputs_figures=cfg.outputs_figures,
        outputs_predictions=cfg.outputs_predictions,
        outputs_metrics=cfg.outputs_metrics,
        is_colab=cfg.is_colab,
        use_grid_auto_calibration=True,
        grid_square_mm=cfg.grid_square_mm,
        grid_ppm_ratio_min=cfg.grid_ppm_ratio_min,
        grid_ppm_ratio_max=cfg.grid_ppm_ratio_max,
    )

    if image_ids is None:
        image_ids = load_validation_image_ids(cfg)
    if limit is not None:
        image_ids = image_ids[:limit]

    out_root = output_dir or (cfg.runs_root / run_name / "grid_audit")
    out_root.mkdir(parents=True, exist_ok=True)

    gt_path = default_ground_truth_path(cfg)
    gt_lookup: dict[str, float] = {}
    if gt_path and gt_path.is_file():
        gt_df = load_ground_truth_csv(gt_path)
        gt_lookup = {
            str(i): float(v) for i, v in zip(gt_df["image_id"], gt_df["length_mm"], strict=False)
        }

    records: list[GridAuditRecord] = []
    for sample in iterate_image_ids(cfg, split=split, image_ids=image_ids, load_images=True):
        if sample.image is None:
            continue
        rec = audit_single_image(sample, cfg_audit)
        if sample.image_id in gt_lookup:
            rec.length_mm_gt = gt_lookup[sample.image_id]
        records.append(rec)
        if generate_figures:
            render_grid_audit_figure(
                sample,
                rec,
                out_root / "figures" / f"{sample.image_id}.png",
                cfg=cfg,
            )

    df = attach_ground_truth(records_to_dataframe(records), cfg)
    df = df.sort_values(
        "absolute_error_if_grid_used",
        ascending=False,
        na_position="last",
    )
    df.to_csv(out_root / "grid_audit_per_image.csv", index=False)

    summary = compute_coverage_accuracy_summary(df)
    pd.DataFrame([summary]).to_csv(out_root / "grid_audit_coverage_accuracy.csv", index=False)

    rejection_table = build_rejection_analysis_table(df)
    rejection_table.to_csv(out_root / "grid_audit_rejection_table.csv", index=False)

    if (~df.grid_valid).any():
        failure_counts = (
            df.loc[~df.grid_valid, "failure_category"].value_counts().reset_index(name="count")
        )
        failure_counts.columns = ["failure_category", "count"]
    else:
        failure_counts = pd.DataFrame(columns=["failure_category", "count"])
    failure_counts.to_csv(out_root / "grid_audit_failure_categories.csv", index=False)

    write_grid_audit_report(
        df,
        summary,
        failure_counts,
        out_root / "grid_audit_report.md",
        figure_dir=out_root / "figures",
    )

    logger.info("Wrote grid audit to %s (%d images)", out_root, len(df))
    return df


def build_rejection_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table sorted by grid vs marker disagreement."""
    cols = [
        "image_id",
        "grid_valid",
        "grid_confidence",
        "rejection_reason",
        "failure_category",
        "marker_grid_ratio",
        "final_scale_source",
        "absolute_error_if_grid_used",
        "absolute_error_if_marker_used",
    ]
    if "abs_err_length_grid_mm" in df.columns:
        cols.extend(["abs_err_length_grid_mm", "abs_err_length_marker_mm", "length_mm_gt"])
    existing = [c for c in cols if c in df.columns]
    out = df[existing].copy()
    out = out.sort_values(
        "absolute_error_if_grid_used",
        ascending=False,
        na_position="last",
    )
    return out


def compute_coverage_accuracy_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Coverage and length MAE splits for grid vs fallback cohorts."""
    n = len(df)
    grid_used = df["final_scale_source"] == "grid"
    fallback = ~grid_used
    valid = df["grid_valid"]

    summary: dict[str, Any] = {
        "n_images": n,
        "pct_grid_valid": float(df["grid_valid"].mean() * 100) if n else 0.0,
        "pct_grid_success": float(df["grid_success"].mean() * 100) if n else 0.0,
        "pct_final_scale_from_grid": float(grid_used.mean() * 100) if n else 0.0,
        "pct_marker_fallback": float(fallback.mean() * 100) if n else 0.0,
        "mean_grid_confidence": float(df["grid_confidence"].mean()) if n else 0.0,
        "mean_confidence_when_valid": float(df.loc[valid, "grid_confidence"].mean())
        if valid.any()
        else float("nan"),
        "mean_confidence_when_rejected": float(df.loc[~valid, "grid_confidence"].mean())
        if (~valid).any()
        else float("nan"),
    }

    if "abs_err_length_grid_mm" in df.columns and "length_mm_gt" in df.columns:
        has_gt = df["length_mm_gt"].notna()
        grid_cohort = has_gt & grid_used
        fallback_cohort = has_gt & fallback
        geometry_valid = has_gt & valid

        def _mae(mask: pd.Series, col: str) -> float:
            if not mask.any():
                return float("nan")
            return float(df.loc[mask, col].mean())

        summary["mae_mm_if_only_grid_used"] = _mae(grid_cohort, "abs_err_length_grid_mm")
        summary["mae_mm_if_only_fallback"] = _mae(fallback_cohort, "abs_err_length_marker_mm")
        summary["mae_mm_geometry_valid_using_grid_scale"] = _mae(
            geometry_valid, "abs_err_length_grid_mm"
        )
        summary["mae_mm_geometry_valid_using_marker_scale"] = _mae(
            geometry_valid, "abs_err_length_marker_mm"
        )
        summary["mae_mm_all_images_marker_scale"] = _mae(has_gt, "abs_err_length_marker_mm")

    return summary


def _line_intersection(seg_a: LineSegment, seg_b: LineSegment) -> tuple[float, float] | None:
    x1, y1, x2, y2 = seg_a.x1, seg_a.y1, seg_a.x2, seg_a.y2
    x3, y3, x4, y4 = seg_b.x1, seg_b.y1, seg_b.x2, seg_b.y2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def _approximate_grid_intersections(
    horiz_segs: list[LineSegment],
    vert_segs: list[LineSegment],
    *,
    max_pairs: int = 40,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for h in horiz_segs[:max_pairs]:
        for v in vert_segs[:max_pairs]:
            pt = _line_intersection(h, v)
            if pt is not None:
                points.append(pt)
    return points


def _spacing_histogram_data(sig: GridAuditSignals) -> tuple[np.ndarray, np.ndarray]:
    dists: list[float] = []
    for seg in sig.horiz_segs + sig.vert_segs:
        dists.append(seg.perp_dist)
    if not dists:
        return np.array([]), np.array([])
    arr = np.asarray(dists, dtype=np.float64)
    if sig.scale > 0 and sig.scale < 1.0:
        arr = arr / sig.scale
    return arr, np.histogram(arr, bins=min(20, max(5, len(arr) // 2)))[0] if len(arr) else np.array([])


def _draw_segments(
    base: np.ndarray,
    segments: list[LineSegment],
    color: tuple[int, int, int],
    *,
    scale_to_full: float,
) -> np.ndarray:
    out = base.copy()
    inv = scale_to_full
    for seg in segments:
        x1, y1 = int(seg.x1 * inv), int(seg.y1 * inv)
        x2, y2 = int(seg.x2 * inv), int(seg.y2 * inv)
        cv2.line(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    return out


def render_grid_audit_figure(
    sample: DatasetSample,
    record: GridAuditRecord,
    output_path: Path,
    *,
    cfg: ProjectConfig | None = None,
) -> Path:
    """Save an 8-panel audit figure for one image."""
    if sample.image is None or record.signals is None:
        raise ValueError("sample image and record.signals required")

    cfg = cfg or get_config()
    sig = record.signals
    img = sample.image
    display = img
    max_side = 520
    h, w = display.shape[:2]
    disp_scale = 1.0
    if max(h, w) > max_side:
        disp_scale = max_side / max(h, w)
        display = cv2.resize(display, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_AREA)

    seg_scale = disp_scale / max(sig.scale, 1e-6) if sig.scale < 1.0 else disp_scale

    panel_h = _draw_segments(display, sig.horiz_segs, (255, 140, 0), scale_to_full=seg_scale)
    panel_v = _draw_segments(display, sig.vert_segs, (0, 140, 255), scale_to_full=seg_scale)
    panel_rej = _draw_segments(display, sig.rejected_segs, (80, 80, 220), scale_to_full=seg_scale)

    intersections = _approximate_grid_intersections(sig.horiz_segs, sig.vert_segs)
    panel_pts = display.copy()
    for x, y in intersections:
        px, py = int(x * seg_scale), int(y * seg_scale)
        if 0 <= px < panel_pts.shape[1] and 0 <= py < panel_pts.shape[0]:
            cv2.circle(panel_pts, (px, py), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    marker_overlay = display.copy()
    for ann in sample.blue_annotations():
        if ann.coords_pixels is not None and len(ann.coords_pixels) >= 2:
            pts = (ann.coords_pixels * disp_scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(marker_overlay, [pts], True, (255, 0, 0), 2)
    for ann in sample.yellow_annotations():
        if ann.coords_pixels is not None and len(ann.coords_pixels) >= 2:
            pts = (ann.coords_pixels * disp_scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(marker_overlay, [pts], True, (0, 255, 255), 2)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("A: Original")
    axes[0, 1].imshow(cv2.cvtColor(panel_h, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"B: Horizontal ({len(sig.horiz_segs)})")
    axes[0, 2].imshow(cv2.cvtColor(panel_v, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"C: Vertical ({len(sig.vert_segs)})")
    axes[0, 3].imshow(cv2.cvtColor(panel_rej, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title(f"D: Rejected ({len(sig.rejected_segs)})")

    axes[1, 0].imshow(cv2.cvtColor(panel_pts, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"E: Intersections ({len(intersections)})")

    dists, _ = _spacing_histogram_data(sig)
    if len(dists):
        axes[1, 1].hist(dists, bins=min(20, max(5, len(dists) // 2)), color="steelblue", edgecolor="white")
    axes[1, 1].set_title("F: Spacing (perp dist)")
    axes[1, 1].set_xlabel("px (full-res approx)")

    axes[1, 2].imshow(cv2.cvtColor(marker_overlay, cv2.COLOR_BGR2RGB))
    g_txt = f"grid {record.grid_ppm:.3f} px/mm" if record.grid_ppm > 0 else "grid N/A"
    m_txt = f"marker {record.marker_ppm:.3f} px/mm"
    ratio_txt = (
        f"ratio {record.marker_grid_ratio:.2f}"
        if np.isfinite(record.marker_grid_ratio)
        else "ratio N/A"
    )
    axes[1, 2].set_title(f"G: {g_txt}\n{m_txt}\n{ratio_txt}", fontsize=8)

    overlay = (
        f"grid_valid={record.grid_valid}\n"
        f"grid_confidence={record.grid_confidence:.3f}\n"
        f"rejection_reason:\n{record.rejection_reason}\n"
        f"H/V ratio={record.hv_ratio:.3f}\n"
        f"spacing H={record.horizontal_spacing:.1f} V={record.vertical_spacing:.1f}\n"
        f"fused={sig.fused_spacing_px:.1f}px\n"
        f"source={record.final_scale_source}\n"
        f"category={record.failure_category}"
    )
    axes[1, 3].text(0.02, 0.98, overlay, va="top", fontsize=8, family="monospace")
    axes[1, 3].set_title("H: Summary")
    axes[1, 3].axis("off")

    for ax in axes.ravel():
        if ax != axes[1, 3]:
            ax.axis("off")
    fig.suptitle(record.image_id, fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_grid_audit_set(
    *,
    cfg: ProjectConfig | None = None,
    image_ids: list[str] | None = None,
    split: str = "valid",
    limit: int = 30,
    run_name: str = "grid_audit",
    output_dir: Path | None = None,
    records: list[GridAuditRecord] | None = None,
) -> list[Path]:
    """Generate audit figures for many images."""
    from src.experiments import load_validation_image_ids

    cfg = cfg or get_config()
    cfg_audit = ProjectConfig(
        repo_root=cfg.repo_root,
        storage_root=cfg.storage_root,
        data_root=cfg.data_root,
        data_raw=cfg.data_raw,
        data_annotations=cfg.data_annotations,
        data_processed=cfg.data_processed,
        runs_root=cfg.runs_root,
        figures_root=cfg.figures_root,
        outputs_figures=cfg.outputs_figures,
        outputs_predictions=cfg.outputs_predictions,
        outputs_metrics=cfg.outputs_metrics,
        is_colab=cfg.is_colab,
        use_grid_auto_calibration=True,
        grid_square_mm=cfg.grid_square_mm,
        grid_ppm_ratio_min=cfg.grid_ppm_ratio_min,
        grid_ppm_ratio_max=cfg.grid_ppm_ratio_max,
    )

    out_root = output_dir or (cfg.runs_root / run_name / "grid_audit" / "figures")
    out_root.mkdir(parents=True, exist_ok=True)

    if image_ids is None:
        image_ids = load_validation_image_ids(cfg)
    if limit is not None:
        image_ids = image_ids[:limit]

    paths: list[Path] = []
    record_by_id = {r.image_id: r for r in records} if records else {}

    for sample in iterate_image_ids(cfg, split=split, image_ids=image_ids, load_images=True):
        if sample.image is None:
            continue
        rec = record_by_id.get(sample.image_id) or audit_single_image(sample, cfg_audit)
        path = out_root / f"{sample.image_id}.png"
        render_grid_audit_figure(sample, rec, path, cfg=cfg)
        paths.append(path)

    return paths


def visualize_grid_audit(
    image_id: str,
    *,
    cfg: ProjectConfig | None = None,
    split: str = "valid",
    run_name: str = "grid_audit",
    output_dir: Path | None = None,
) -> Path:
    """Audit and render one image by ``image_id`` (8-panel figure)."""
    cfg = cfg or get_config()
    out_root = output_dir or (cfg.runs_root / run_name / "grid_audit" / "figures")
    cfg_audit = ProjectConfig(
        repo_root=cfg.repo_root,
        use_grid_auto_calibration=True,
        grid_square_mm=cfg.grid_square_mm,
        grid_ppm_ratio_min=cfg.grid_ppm_ratio_min,
        grid_ppm_ratio_max=cfg.grid_ppm_ratio_max,
    )
    for sample in iterate_image_ids(cfg, split=split, image_ids=[image_id], load_images=True):
        rec = audit_single_image(sample, cfg_audit)
        path = out_root / f"{image_id}.png"
        return render_grid_audit_figure(sample, rec, path, cfg=cfg)
    raise FileNotFoundError(f"image_id {image_id} not found in split={split}")


# Backward-compatible alias
visualize_grid_audit_by_id = visualize_grid_audit


def write_grid_audit_report(
    df: pd.DataFrame,
    summary: dict[str, Any],
    failure_counts: pd.DataFrame,
    output_path: Path,
    *,
    figure_dir: Path | None = None,
) -> None:
    """Write markdown report with acceptance analysis and examples."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = int(summary.get("n_images", len(df)))
    valid = df["grid_valid"]
    grid_final = df["final_scale_source"] == "grid"

    false_rejection = df[~valid & df["marker_grid_ratio"].between(0.85, 1.15, inclusive="both")]
    false_acceptance = df[valid & ~df["marker_grid_ratio"].between(0.85, 1.15, inclusive="both")]

    example_ids = {
        "correct_acceptance": df[grid_final].head(3)["image_id"].tolist(),
        "false_rejection": false_rejection.head(3)["image_id"].tolist(),
        "false_acceptance": false_acceptance.head(3)["image_id"].tolist(),
        "high_disagreement": df.sort_values("absolute_error_if_grid_used", ascending=False)
        .head(3)["image_id"]
        .tolist(),
    }

    rec_lines: list[str] = []
    pct_valid = summary.get("pct_grid_valid", 0)
    pct_grid_used = summary.get("pct_final_scale_from_grid", 0)
    if pct_valid < 15 and pct_grid_used <= 5:
        rec_lines.append(
            "High rejection with very few grid scales used in production — likely **correctly strict** "
            "given prior audit (29/30 marker mismatch). Review false-rejection examples before loosening."
        )
    elif pct_valid > 40 and pct_grid_used > 25:
        mae_g = summary.get("mae_mm_if_only_grid_used", float("nan"))
        mae_m = summary.get("mae_mm_if_only_fallback", float("nan"))
        if np.isfinite(mae_g) and np.isfinite(mae_m) and mae_g < mae_m:
            rec_lines.append(
                "Grid cohort MAE is lower than fallback — consider **slightly loosening** confidence or H/V ratio."
            )
        else:
            rec_lines.append("Keep current strictness; grid cohort does not beat marker on length MAE.")
    else:
        rec_lines.append(
            "Moderate acceptance — use per-image figures to judge false rejections vs true failures."
        )

    lines = [
        "# Grid calibration audit report",
        "",
        f"Generated: {ts}",
        "",
        "## 1. Grid acceptance rate analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Images audited | {n} |",
        f"| `grid_valid` rate | {pct_valid:.1f}% |",
        f"| `grid_success` rate | {summary.get('pct_grid_success', 0):.1f}% |",
        f"| Final scale from grid | {pct_grid_used:.1f}% |",
        f"| Marker fallback | {summary.get('pct_marker_fallback', 0):.1f}% |",
        f"| Mean confidence (all) | {summary.get('mean_grid_confidence', 0):.3f} |",
        f"| Mean confidence (valid) | {summary.get('mean_confidence_when_valid', float('nan')):.3f} |",
        f"| Mean confidence (rejected) | {summary.get('mean_confidence_when_rejected', float('nan')):.3f} |",
        "",
        "## 2. Rejection reason breakdown",
        "",
    ]

    if failure_counts is not None and not failure_counts.empty:
        lines.append("| failure_category | count |")
        lines.append("|------------------|-------|")
        for _, row in failure_counts.iterrows():
            cat = row.get("failure_category", row.get("index", ""))
            cnt = row.get("count", 0)
            lines.append(f"| {cat} | {cnt} |")
    else:
        lines.append("_No rejected images._")

    lines.extend(
        [
            "",
            "## 3. Coverage vs accuracy",
            "",
            f"- MAE (bbox) if only images with **final grid** scale: "
            f"{summary.get('mae_mm_if_only_grid_used', 'N/A')} mm",
            f"- MAE (bbox) if only **fallback** images (marker scale): "
            f"{summary.get('mae_mm_if_only_fallback', 'N/A')} mm",
            f"- MAE using grid scale on geometry-valid images: "
            f"{summary.get('mae_mm_geometry_valid_using_grid_scale', 'N/A')} mm",
            f"- MAE using marker scale on geometry-valid images: "
            f"{summary.get('mae_mm_geometry_valid_using_marker_scale', 'N/A')} mm",
            "",
            "## 4. Visual examples",
            "",
        ]
    )

    fig_rel = "figures" if figure_dir is None else figure_dir.name
    for label, ids in example_ids.items():
        lines.append(f"### {label.replace('_', ' ').title()}")
        if not ids:
            lines.append("_None identified._")
        else:
            for iid in ids:
                lines.append(f"- `{iid}`: `{fig_rel}/{iid}.png`")
        lines.append("")

    lines.extend(
        [
            "## 5. Recommendation",
            "",
            *rec_lines,
            "",
            "### Interpretation guide",
            "",
            "- **A (filtering correct):** Most rejections show edge confusion / H-V mismatch in figures; "
            "marker ratio gate would have failed anyway.",
            "- **B (too strict):** Multiple false-rejection IDs show clean grids and marker ratio in band; "
            "confidence just below threshold.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
