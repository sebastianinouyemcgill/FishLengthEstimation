"""
Automatic tank-grid calibration without fiduciary markers.

Detects periodic parallel line families (Hough), filters non-grid structure,
fuses spacing estimates with confidence, and validates geometry before returning scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import ProjectConfig, get_config
from src.utils import get_logger

logger = get_logger(__name__)

# Tunables (grid module only)
_MIN_SPACING_PX = 15.0
_MAX_SPACING_PX = 250.0
_FUSION_DISAGREE_FRAC = 0.18
_HV_RATIO_MIN = 0.82
_HV_RATIO_MAX = 1.22
_MIN_FAMILY_LINES = 4
_MIN_GRID_CONFIDENCE = 0.40
_ANGLE_TOL_DEG = 12.0
_GAP_BIN_PX = 6.0
_MIN_LINE_LEN_FRAC = 0.07
_ROI_TOP_FRAC = 0.12  # ignore top strip (glare / air-water)


@dataclass
class LineSegment:
    """Single Hough segment in working (possibly scaled) coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float
    length: float
    perp_dist: float


@dataclass
class SpacingEstimate:
    """One spacing hypothesis with confidence in [0, 1]."""

    spacing_px: float
    confidence: float
    n_lines: int = 0
    gap_cv: float = 1.0


@dataclass
class GridCalibrationResult:
    """Outputs of grid-based automatic calibration for one image."""

    pixels_per_grid_square: float
    pixels_per_mm: float
    grid_square_mm: float
    success: bool
    n_horizontal_lines: int = 0
    n_vertical_lines: int = 0
    depth_scale_mm_per_unit: float = 1.0
    grid_valid: bool = False
    grid_confidence: float = 0.0


def _normalize_angle_deg(theta_deg: float) -> float:
    return theta_deg % 180.0


def _angle_delta(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _preprocess_gray(image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale, (h, w)


def _detect_line_segments_xy(
    gray: np.ndarray,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 80,
    max_line_gap: int = 15,
) -> tuple[list[LineSegment], np.ndarray]:
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    segments: list[LineSegment] = []
    if lines is None:
        return segments, edges
    h, w = gray.shape[:2]
    min_len = max(min_line_length, _MIN_LINE_LEN_FRAC * min(h, w))
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        angle = _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rad = np.radians(angle)
        # Signed distance along line normal (spacing between parallel lines)
        perp_dist = abs(-mx * np.sin(rad) + my * np.cos(rad))
        segments.append(
            LineSegment(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                angle_deg=angle,
                length=length,
                perp_dist=perp_dist,
            )
        )
    return segments, edges


def _filter_segments_roi(segments: list[LineSegment], gray: np.ndarray) -> list[LineSegment]:
    """Drop segments outside the tank grid region (exclude top strip and side margins)."""
    h, w = gray.shape[:2]
    y_cut = h * _ROI_TOP_FRAC
    x_lo = w * 0.05
    x_hi = w * 0.95
    kept: list[LineSegment] = []
    for seg in segments:
        my = 0.5 * (seg.y1 + seg.y2)
        mx = 0.5 * (seg.x1 + seg.x2)
        if my >= y_cut and x_lo <= mx <= x_hi:
            kept.append(seg)
    return kept


def _dominant_orthogonal_angles(segments: list[LineSegment]) -> tuple[float, float]:
    """Estimate two dominant line directions (degrees, [0, 180))."""
    if not segments:
        return 0.0, 90.0
    angles = np.array([s.angle_deg for s in segments], dtype=np.float64)
    hist, edges = np.histogram(angles, bins=36, range=(0.0, 180.0))
    peak_idx = int(np.argmax(hist))
    a0 = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))
    a1 = _normalize_angle_deg(a0 + 90.0)
    return a0, a1


def _segments_for_angle(
    segments: list[LineSegment],
    angle_target: float,
    *,
    angle_tol: float = _ANGLE_TOL_DEG,
) -> list[LineSegment]:
    return [s for s in segments if _angle_delta(s.angle_deg, angle_target) <= angle_tol]


def _coverage_score(segments: list[LineSegment], gray: np.ndarray, *, horizontal: bool) -> float:
    """Fraction of image span covered by line midpoints along relevant axis."""
    if not segments:
        return 0.0
    h, w = gray.shape[:2]
    if horizontal:
        coords = [0.5 * (s.x1 + s.x2) for s in segments]
        span = max(coords) - min(coords)
        return float(np.clip(span / max(w, 1), 0.0, 1.0))
    coords = [0.5 * (s.y1 + s.y2) for s in segments]
    span = max(coords) - min(coords)
    return float(np.clip(span / max(h, 1), 0.0, 1.0))


def _gap_histogram_peaks(distances: list[float], bin_px: float = _GAP_BIN_PX) -> tuple[float, float, float]:
    """
    Fundamental spacing from parallel-line distances.

    Returns (spacing_px, peak_dominance, gap_cv).
    peak_dominance in [0,1]: share of gaps within 15% of modal gap.
    """
    if len(distances) < 3:
        return 0.0, 0.0, 1.0
    sorted_d = np.sort(np.asarray(distances, dtype=np.float64))
    quantized = np.round(sorted_d / bin_px) * bin_px
    unique = np.unique(quantized)
    if len(unique) < 2:
        return 0.0, 0.0, 1.0
    gaps = np.diff(unique)
    gaps = gaps[gaps > bin_px * 0.5]
    if len(gaps) == 0:
        return 0.0, 0.0, 1.0

    rounded = np.round(gaps / bin_px) * bin_px
    uniq, counts = np.unique(rounded, return_counts=True)
    min_support = max(2, int(0.12 * len(gaps)))
    spacing = 0.0
    dominance = 0.0
    for g, c in sorted(zip(uniq, counts), key=lambda x: x[0]):
        if g < _MIN_SPACING_PX or g > _MAX_SPACING_PX:
            continue
        if int(c) < min_support:
            continue
        spacing = float(g)
        dominance = float(c / len(gaps))
        break
    if spacing <= 0:
        gap_hist, gap_edges = np.histogram(gaps, bins=max(8, int(len(gaps) * 2)))
        peak_i = int(np.argmax(gap_hist))
        spacing = float(0.5 * (gap_edges[peak_i] + gap_edges[peak_i + 1]))
        if spacing <= 0:
            spacing = float(np.median(gaps))
        modal = spacing
        tol = max(modal * 0.15, bin_px)
        dominance = float(np.mean(np.abs(gaps - modal) <= tol))
    else:
        modal = spacing
    gap_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-6))
    return spacing, dominance, gap_cv


def _dedupe_by_perp_distance(
    segments: list[LineSegment],
    *,
    bin_px: float = 8.0,
) -> list[LineSegment]:
    """Keep the longest segment per perpendicular-distance bin."""
    buckets: dict[int, LineSegment] = {}
    for seg in segments:
        key = int(round(seg.perp_dist / bin_px))
        if key not in buckets or seg.length > buckets[key].length:
            buckets[key] = seg
    return list(buckets.values())


def _filter_to_lattice(
    segments: list[LineSegment],
    spacing: float,
    *,
    tol_frac: float = 0.22,
) -> list[LineSegment]:
    """Drop segments that do not lie on a regular lattice with the given spacing."""
    if spacing <= 0 or not segments:
        return []
    dists = [s.perp_dist for s in segments]
    offset = float(np.median(dists)) % spacing
    tol = max(spacing * tol_frac, 4.0)
    kept: list[LineSegment] = []
    for seg in segments:
        d = seg.perp_dist
        k = round((d - offset) / spacing)
        if abs(d - (offset + k * spacing)) <= tol:
            kept.append(seg)
    return kept


def _refine_family_segments(
    segments: list[LineSegment],
    gray: np.ndarray,
    *,
    horizontal: bool,
) -> list[LineSegment]:
    """Deduplicate, estimate spacing, and keep only lattice-consistent lines."""
    segs = _dedupe_by_perp_distance(segments)
    for _ in range(2):
        est = _spacing_from_family(segs, gray, horizontal=horizontal)
        if est.spacing_px <= 0:
            break
        filtered = _filter_to_lattice(segs, est.spacing_px)
        if len(filtered) < _MIN_FAMILY_LINES:
            break
        segs = filtered
    return segs


def _spacing_from_family(
    segments: list[LineSegment],
    gray: np.ndarray,
    *,
    horizontal: bool,
) -> SpacingEstimate:
    if len(segments) < _MIN_FAMILY_LINES:
        return SpacingEstimate(0.0, 0.0, n_lines=len(segments), gap_cv=1.0)
    distances = [s.perp_dist for s in segments]
    spacing, dominance, gap_cv = _gap_histogram_peaks(distances)
    coverage = _coverage_score(segments, gray, horizontal=horizontal)
    periodic = 1.0 / (1.0 + gap_cv)
    conf = float(np.clip(0.35 * dominance + 0.35 * periodic + 0.30 * coverage, 0.0, 1.0))
    if spacing < _MIN_SPACING_PX or spacing > _MAX_SPACING_PX:
        return SpacingEstimate(0.0, 0.0, n_lines=len(segments), gap_cv=gap_cv)
    return SpacingEstimate(spacing, conf, n_lines=len(segments), gap_cv=gap_cv)


def _spacing_from_harris(gray: np.ndarray) -> SpacingEstimate:
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=400,
        qualityLevel=0.02,
        minDistance=15,
        blockSize=5,
    )
    if corners is None or len(corners) < 12:
        return SpacingEstimate(0.0, 0.0)
    pts = corners.reshape(-1, 2)
    if len(pts) > 150:
        idx = np.linspace(0, len(pts) - 1, 150, dtype=int)
        pts = pts[idx]
    dists: list[float] = []
    for i, p in enumerate(pts):
        others = np.delete(pts, i, axis=0)
        nn = float(np.min(np.linalg.norm(others - p, axis=1)))
        if _MIN_SPACING_PX < nn < _MAX_SPACING_PX:
            dists.append(nn)
    if len(dists) < 8:
        return SpacingEstimate(0.0, 0.0)
    arr = np.asarray(dists)
    spacing = float(np.median(arr))
    # Unimodality: peak tightness
    med = spacing
    tol = max(med * 0.2, 4.0)
    dominance = float(np.mean(np.abs(arr - med) <= tol))
    conf = float(np.clip(0.5 * dominance, 0.0, 0.55))  # cap — secondary cue only
    return SpacingEstimate(spacing, conf, n_lines=len(dists))


def _fuse_spacing_estimates(estimates: list[SpacingEstimate]) -> tuple[float, float]:
    """
    Confidence-weighted fusion with outlier rejection.

    Returns (spacing_px, fusion_confidence).
    """
    valid = [e for e in estimates if e.spacing_px > 0 and e.confidence > 0.05]
    if not valid:
        return 0.0, 0.0
    if len(valid) == 1:
        e = valid[0]
        return e.spacing_px, e.confidence

    # Drop lowest-confidence estimate if strong disagreement
    for _ in range(len(valid) - 1):
        spacings = np.array([e.spacing_px for e in valid])
        median_s = float(np.median(spacings))
        disagree = [
            abs(e.spacing_px - median_s) / max(median_s, 1e-6) > _FUSION_DISAGREE_FRAC
            for e in valid
        ]
        if not any(disagree):
            break
        worst = min(valid, key=lambda e: e.confidence)
        valid = [e for e in valid if e is not worst]
        if len(valid) < 1:
            break

    if not valid:
        return 0.0, 0.0

    spacings = np.array([e.spacing_px for e in valid], dtype=np.float64)
    weights = np.array([e.confidence for e in valid], dtype=np.float64)
    wsum = float(weights.sum())
    if wsum < 1e-6:
        return float(np.median(spacings)), 0.0
    fused = float(np.sum(spacings * weights) / wsum)
    spread = float(np.std(spacings) / (np.mean(spacings) + 1e-6))
    fusion_conf = float(np.clip(np.mean(weights) * (1.0 / (1.0 + spread)), 0.0, 1.0))
    return fused, fusion_conf


def _compute_grid_quality(
    *,
    est_h: SpacingEstimate,
    est_v: SpacingEstimate,
    angle_a: float,
    angle_b: float,
    fusion_conf: float,
    fused_spacing: float,
) -> tuple[bool, float]:
    """grid_valid and grid_confidence in [0, 1]."""
    if fused_spacing <= 0:
        return False, 0.0

    scores: list[float] = []

    # Orthogonality between families
    ortho_err = abs(_angle_delta(angle_a, angle_b) - 90.0)
    ortho_score = float(np.clip(1.0 - ortho_err / 25.0, 0.0, 1.0))
    scores.append(ortho_score)

    # H/V spacing consistency
    if est_h.spacing_px > 0 and est_v.spacing_px > 0:
        ratio = est_h.spacing_px / est_v.spacing_px
        if _HV_RATIO_MIN <= ratio <= _HV_RATIO_MAX:
            hv_score = 1.0 - min(abs(ratio - 1.0), 0.3) / 0.3
        else:
            hv_score = 0.0
        scores.append(float(np.clip(hv_score, 0.0, 1.0)))

    scores.append(fusion_conf)
    scores.append(min(est_h.confidence, est_v.confidence) if est_h.spacing_px and est_v.spacing_px else 0.0)

    confidence = float(np.mean(scores)) if scores else 0.0
    valid = (
        confidence >= _MIN_GRID_CONFIDENCE
        and est_h.n_lines >= _MIN_FAMILY_LINES
        and est_v.n_lines >= _MIN_FAMILY_LINES
        and est_h.spacing_px > 0
        and est_v.spacing_px > 0
        and _HV_RATIO_MIN <= (est_h.spacing_px / est_v.spacing_px) <= _HV_RATIO_MAX
    )
    return valid, confidence


def _analyze_grid(
    image_bgr: np.ndarray,
) -> tuple[
    float,
    int,
    int,
    bool,
    float,
    list[LineSegment],
    list[LineSegment],
    list[LineSegment],
    SpacingEstimate,
    SpacingEstimate,
    SpacingEstimate,
    np.ndarray,
    float,
]:
    """
    Core analysis on one image.

    Returns fused spacing (full-res px), n_h, n_v, grid_valid, grid_confidence,
    all/rejected/kept segments, estimates, edges, scale.
    """
    gray, scale, _ = _preprocess_gray(image_bgr)
    all_segments, edges = _detect_line_segments_xy(gray)
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

    est_h = _spacing_from_family(horiz_segs, gray, horizontal=True)
    est_v = _spacing_from_family(vert_segs, gray, horizontal=False)
    est_corner = _spacing_from_harris(gray)

    # Only include Harris if it agrees with H or V
    harris_estimates = []
    if est_corner.spacing_px > 0:
        refs = [e.spacing_px for e in (est_h, est_v) if e.spacing_px > 0]
        if refs:
            med_ref = float(np.median(refs))
            if abs(est_corner.spacing_px - med_ref) / med_ref <= _FUSION_DISAGREE_FRAC:
                harris_estimates.append(est_corner)
        else:
            harris_estimates.append(est_corner)

    fused, fusion_conf = _fuse_spacing_estimates([est_h, est_v, *harris_estimates])

    # Validate in working (scaled) coordinates before converting to full-res pixels
    grid_valid, grid_conf = _compute_grid_quality(
        est_h=est_h,
        est_v=est_v,
        angle_a=horiz_angle,
        angle_b=vert_angle,
        fusion_conf=fusion_conf,
        fused_spacing=fused,
    )

    if scale != 1.0 and fused > 0:
        fused /= scale
        est_h = SpacingEstimate(
            est_h.spacing_px / scale if est_h.spacing_px > 0 else 0.0,
            est_h.confidence,
            est_h.n_lines,
            est_h.gap_cv,
        )
        est_v = SpacingEstimate(
            est_v.spacing_px / scale if est_v.spacing_px > 0 else 0.0,
            est_v.confidence,
            est_v.n_lines,
            est_v.gap_cv,
        )

    max_spacing_full = _MAX_SPACING_PX / scale if 0 < scale < 1.0 else _MAX_SPACING_PX
    if fused > 0 and (fused < _MIN_SPACING_PX or fused > max_spacing_full):
        grid_valid = False
        fused = 0.0

    kept = horiz_segs + vert_segs
    rejected = [s for s in all_segments if s not in kept]

    return (
        fused,
        len(horiz_segs),
        len(vert_segs),
        grid_valid,
        grid_conf,
        all_segments,
        kept,
        rejected,
        est_h,
        est_v,
        est_corner,
        edges,
        scale,
    )


def estimate_pixels_per_grid_square(
    image_bgr: np.ndarray,
    *,
    min_spacing_px: float = _MIN_SPACING_PX,
    max_spacing_px: float = _MAX_SPACING_PX,
) -> tuple[float, int, int]:
    """
    Estimate average pixel spacing between adjacent grid lines.

    Returns (spacing_px, n_horizontal, n_vertical) for backward compatibility.
    """
    spacing, n_h, n_v, grid_valid, *_ = _analyze_grid(image_bgr)
    if not grid_valid or spacing < min_spacing_px or spacing > max_spacing_px:
        return 0.0, n_h, n_v
    return spacing, n_h, n_v


def estimate_depth_metric_scale(
    depth_map: np.ndarray,
    pixels_per_grid_square: float,
    grid_square_mm: float,
) -> float:
    """Map relative depth units to millimeters using grid spacing as reference."""
    if pixels_per_grid_square <= 0 or grid_square_mm <= 0:
        return 1.0
    d = depth_map.astype(np.float64)
    if d.size == 0 or not np.isfinite(d).any():
        return 1.0
    gy, gx = np.gradient(d)
    grad = np.sqrt(gx * gx + gy * gy)
    h, w = grad.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    roi = grad[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    median_grad = float(np.median(roi[np.isfinite(roi)])) if roi.size else 0.0
    if median_grad < 1e-8:
        return 1.0
    depth_delta_per_cell = median_grad * pixels_per_grid_square
    return grid_square_mm / max(depth_delta_per_cell, 1e-8)


def save_grid_debug_figure(
    image_bgr: np.ndarray,
    output_path: Path,
    *,
    image_id: str = "",
    est_h: SpacingEstimate | None = None,
    est_v: SpacingEstimate | None = None,
    grid_valid: bool = False,
    grid_confidence: float = 0.0,
    all_segments: list[LineSegment] | None = None,
    kept_segments: list[LineSegment] | None = None,
    rejected_segments: list[LineSegment] | None = None,
    edges: np.ndarray | None = None,
    fused_spacing: float = 0.0,
) -> None:
    """Save optional debug montage under run_dir/grid_debug/."""
    import matplotlib.pyplot as plt

    gray, scale, _ = _preprocess_gray(image_bgr)
    if all_segments is None:
        all_segments, edges = _detect_line_segments_xy(gray)
        all_segments = _filter_segments_roi(all_segments, gray)
        angle_a, angle_b = _dominant_orthogonal_angles(all_segments)
        kept_segments = _segments_for_angle(all_segments, angle_a) + _segments_for_angle(
            all_segments, angle_b
        )
        rejected_segments = [s for s in all_segments if s not in kept_segments]

    display = image_bgr
    if max(display.shape[:2]) > 640:
        s = 640 / max(display.shape[:2])
        display = cv2.resize(display, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        inv = s
    else:
        inv = 1.0

    def _draw(segs: list[LineSegment], base: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        out = base.copy()
        for seg in segs:
            x1, y1 = int(seg.x1 * inv), int(seg.y1 * inv)
            x2, y2 = int(seg.x2 * inv), int(seg.y2 * inv)
            cv2.line(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        return out

    kept = kept_segments or []
    rejected = rejected_segments or []
    panel_all = _draw(all_segments or [], display, (160, 160, 160))
    panel_kept = _draw(kept, display, (80, 200, 80))
    panel_rej = _draw(rejected, display, (80, 80, 220))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(cv2.cvtColor(panel_kept, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Kept H/V ({len(kept)})")
    axes[0, 2].imshow(cv2.cvtColor(panel_rej, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"Rejected ({len(rejected)})")
    if edges is not None:
        axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title("Canny")
    axes[1, 1].imshow(cv2.cvtColor(panel_all, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"All segments ({len(all_segments or [])})")

    eh = est_h or SpacingEstimate(0, 0)
    ev = est_v or SpacingEstimate(0, 0)
    txt = (
        f"spacing={fused_spacing:.1f}px  valid={grid_valid}  conf={grid_confidence:.2f}\n"
        f"H: {eh.spacing_px:.1f}px conf={eh.confidence:.2f} n={eh.n_lines}\n"
        f"V: {ev.spacing_px:.1f}px conf={ev.confidence:.2f} n={ev.n_lines}"
    )
    axes[1, 2].text(0.05, 0.5, txt, fontsize=10, va="center", family="monospace")
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Fusion")

    for ax in axes.ravel():
        if ax != axes[1, 2]:
            ax.axis("off")
    fig.suptitle(image_id or "grid_debug", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def estimate_grid_calibration(
    image_bgr: np.ndarray,
    cfg: ProjectConfig | None = None,
    *,
    debug_output_dir: Path | None = None,
    image_id: str = "",
) -> GridCalibrationResult:
    """
    Full grid calibration for one RGB image (no marker polygons).

    ``success`` is True only when ``grid_valid`` and a positive spacing were found.
    """
    cfg = cfg or get_config()
    grid_square_mm = cfg.grid_square_mm

    (
        spacing,
        n_h,
        n_v,
        grid_valid,
        grid_conf,
        _all,
        kept,
        rejected,
        est_h,
        est_v,
        _est_c,
        edges,
        _scale,
    ) = _analyze_grid(image_bgr)

    success = grid_valid and spacing > 0
    pixels_per_mm = spacing / grid_square_mm if success else 0.0

    debug_dir = debug_output_dir
    if debug_dir is None and getattr(cfg, "visualize_grid_debug", False):
        debug_dir = cfg.outputs_figures / "grid_debug"

    if debug_dir is not None:
        name = f"{image_id}.png" if image_id else "grid_debug.png"
        save_grid_debug_figure(
            image_bgr,
            Path(debug_dir) / name,
            image_id=image_id,
            est_h=est_h,
            est_v=est_v,
            grid_valid=grid_valid,
            grid_confidence=grid_conf,
            all_segments=_all,
            kept_segments=kept,
            rejected_segments=rejected,
            edges=edges,
            fused_spacing=spacing,
        )

    if success:
        logger.info(
            "Grid calibration: %.2f px/square (%.4f px/mm) conf=%.2f H=%d V=%d",
            spacing,
            pixels_per_mm,
            grid_conf,
            n_h,
            n_v,
        )
    else:
        logger.warning(
            "Grid calibration unreliable (conf=%.2f, spacing=%.2f); markers should be used",
            grid_conf,
            spacing,
        )

    return GridCalibrationResult(
        pixels_per_grid_square=spacing if success else 0.0,
        pixels_per_mm=pixels_per_mm,
        grid_square_mm=grid_square_mm,
        success=success,
        n_horizontal_lines=n_h,
        n_vertical_lines=n_v,
        grid_valid=grid_valid,
        grid_confidence=grid_conf,
    )


def grid_result_to_marker_calibration(grid: GridCalibrationResult):
    """Build a ``CalibrationResult``-compatible object for metric conversion."""
    from src.calibration.marker import CalibrationResult

    return CalibrationResult(pixels_per_mm=grid.pixels_per_mm)


def _detect_line_segments(gray: np.ndarray) -> list[tuple[float, float]]:
    """Legacy audit API: (angle_deg, perp_dist) per segment."""
    segs, _ = _detect_line_segments_xy(gray)
    return [(s.angle_deg, s.perp_dist) for s in segs]


def _cluster_line_distances(
    segments: list[tuple[float, float]],
    *,
    angle_target: float = 0.0,
    angle_tol: float = _ANGLE_TOL_DEG,
) -> list[float]:
    """Legacy audit API: perpendicular distances for one angle family."""
    return [
        d
        for angle, d in segments
        if _angle_delta(angle, angle_target) <= angle_tol
    ]


def _median_spacing(distances: list[float]) -> float:
    """Legacy audit API: dominant spacing from distance samples."""
    spacing, _, _ = _gap_histogram_peaks(distances)
    return spacing


def _spacing_from_intersections(gray: np.ndarray) -> float:
    """Legacy audit API: Harris-based spacing (float only)."""
    return _spacing_from_harris(gray).spacing_px
