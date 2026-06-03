#!/usr/bin/env python3
"""
Grid calibration diagnostic audit (read-only w.r.t. grid_auto algorithm).

Produces per-image debug figures and a CSV comparing grid vs marker scale.
Does not modify experiment run artifacts unless --run-dir is set (writes only grid_debug/).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.grid_auto import (  # noqa: E402
    GridCalibrationResult,
    _cluster_line_distances,
    _detect_line_segments,
    _median_spacing,
    _normalize_angle_deg,
    _spacing_from_intersections,
    estimate_grid_calibration,
    estimate_pixels_per_grid_square,
)
from src.calibration.marker import estimate_scale_from_markers  # noqa: E402
from src.config import ProjectConfig, get_config  # noqa: E402
from src.dataset import iterate_image_ids  # noqa: E402
from src.experiments import load_validation_image_ids  # noqa: E402
from src.pipelines.advanced_inference import _choose_calibration  # noqa: E402


@dataclass
class GridDiagnostics:
    """Intermediate values from the grid spacing pipeline (audit only)."""

    scale: float
    gray: np.ndarray
    edges: np.ndarray
    segments: list[tuple[float, float]]
    hough_lines: np.ndarray | None
    horiz_dists: list[float]
    vert_dists: list[float]
    sp_h: float
    sp_v: float
    sp_corner: float
    spacing_px: float
    n_h_samples: int
    n_v_samples: int
    harris_count: int


def _detect_hough_lines_xy(gray: np.ndarray) -> np.ndarray | None:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=15,
    )
    return lines


def _preprocess_gray(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def _count_harris_corners(gray: np.ndarray) -> int:
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=5,
    )
    return 0 if corners is None else len(corners)


def analyze_grid_pipeline(image_bgr: np.ndarray) -> GridDiagnostics:
    """Mirror estimate_pixels_per_grid_square with exposed intermediates."""
    gray, scale = _preprocess_gray(image_bgr)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    hough_lines = _detect_hough_lines_xy(gray)
    segments = _detect_line_segments(gray)
    horiz_dists = _cluster_line_distances(segments, angle_target=0.0)
    vert_dists = _cluster_line_distances(segments, angle_target=90.0)
    sp_h = _median_spacing(horiz_dists)
    sp_v = _median_spacing(vert_dists)
    sp_corner = _spacing_from_intersections(gray)
    harris_count = _count_harris_corners(gray)

    min_spacing_px, max_spacing_px = 15.0, 250.0
    candidates = [s for s in (sp_h, sp_v, sp_corner) if min_spacing_px <= s <= max_spacing_px]
    if not candidates:
        spacing_px = 0.0
    else:
        spacing_px = float(np.median(candidates))
        if scale != 1.0:
            spacing_px /= scale

    return GridDiagnostics(
        scale=scale,
        gray=gray,
        edges=edges,
        segments=segments,
        hough_lines=hough_lines,
        horiz_dists=horiz_dists,
        vert_dists=vert_dists,
        sp_h=sp_h,
        sp_v=sp_v,
        sp_corner=sp_corner,
        spacing_px=spacing_px,
        n_h_samples=len(horiz_dists),
        n_v_samples=len(vert_dists),
        harris_count=harris_count,
    )


def _segment_angle_ok(angle: float, target: float, tol: float = 12.0) -> bool:
    delta = min(abs(angle - target), abs(angle - target + 180))
    return delta <= tol


def _draw_hough_segments(
    base_bgr: np.ndarray,
    hough_lines: np.ndarray | None,
    segments: list[tuple[float, float]],
) -> np.ndarray:
    out = base_bgr.copy()
    if hough_lines is not None:
        for x1, y1, x2, y2 in hough_lines[:, 0]:
            dx, dy = float(x2 - x1), float(y2 - y1)
            angle = _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
            if _segment_angle_ok(angle, 0.0):
                color = (255, 120, 0)
            elif _segment_angle_ok(angle, 90.0):
                color = (0, 120, 255)
            else:
                color = (180, 180, 180)
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, cv2.LINE_AA)
    return out


def _draw_clustered_only(
    base_bgr: np.ndarray,
    hough_lines: np.ndarray | None,
) -> np.ndarray:
    out = base_bgr.copy()
    if hough_lines is None:
        return out
    for x1, y1, x2, y2 in hough_lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        angle = _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
        if _segment_angle_ok(angle, 0.0):
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 80, 0), 2, cv2.LINE_AA)
        elif _segment_angle_ok(angle, 90.0):
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 80, 255), 2, cv2.LINE_AA)
    return out


def _draw_harris(gray: np.ndarray, base_bgr: np.ndarray) -> np.ndarray:
    out = base_bgr.copy()
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=5,
    )
    if corners is not None:
        for x, y in corners.reshape(-1, 2):
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return out


def _draw_markers(image_bgr: np.ndarray, sample) -> np.ndarray:
    out = image_bgr.copy()
    for ann in sample.blue_annotations():
        if ann.coords_pixels is None or len(ann.coords_pixels) < 2:
            continue
        pts = ann.coords_pixels.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (255, 0, 0), 2)
    for ann in sample.yellow_annotations():
        if ann.coords_pixels is None or len(ann.coords_pixels) < 2:
            continue
        pts = ann.coords_pixels.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2)
    return out


def _resize_panel(img: np.ndarray, max_side: int = 480) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def save_debug_figure(
    path: Path,
    image_bgr: np.ndarray,
    diag: GridDiagnostics,
    sample,
    grid: GridCalibrationResult,
    marker_ppm: float,
    cfg: ProjectConfig,
) -> None:
    """7-panel montage: A–G per investigation plan."""
    display = _resize_panel(image_bgr)
    gray_disp = cv2.resize(diag.gray, (display.shape[1], display.shape[0]))
    edges_disp = cv2.resize(diag.edges, (display.shape[1], display.shape[0]))
    edges_bgr = cv2.cvtColor(edges_disp, cv2.COLOR_GRAY2BGR)

    hough_all = _draw_hough_segments(display, diag.hough_lines, diag.segments)
    hough_clust = _draw_clustered_only(display, diag.hough_lines)
    harris_panel = _draw_harris(gray_disp, display)
    markers_panel = _draw_markers(display, sample)

    ratio = grid.pixels_per_mm / marker_ppm if marker_ppm > 0 and grid.success else float("nan")
    pct = 100.0 * (grid.pixels_per_mm - marker_ppm) / marker_ppm if marker_ppm > 0 and grid.success else float("nan")
    expected_px = marker_ppm * cfg.grid_square_mm if marker_ppm > 0 else float("nan")
    accepted = (
        grid.success
        and marker_ppm > 0
        and cfg.grid_ppm_ratio_min <= ratio <= cfg.grid_ppm_ratio_max
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels = [
        (axes[0, 0], display, "A: Original"),
        (axes[0, 1], hough_all, "B: Hough (H=orange V=blue)"),
        (axes[0, 2], hough_clust, "C: Clustered H/V only"),
        (axes[0, 3], harris_panel, f"D: Harris corners (n={diag.harris_count})"),
        (axes[1, 0], markers_panel, f"F: Markers (ppm={marker_ppm:.4f})"),
    ]
    for ax, img, title in panels:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    axes[1, 1].imshow(edges_disp, cmap="gray")
    axes[1, 1].set_title("B2: Canny edges", fontsize=9)
    axes[1, 1].axis("off")

    e_text = (
        f"E: Grid spacing\n"
        f"sp_h={diag.sp_h:.1f} sp_v={diag.sp_v:.1f} sp_corner={diag.sp_corner:.1f}\n"
        f"fused={grid.pixels_per_grid_square:.1f} px/sq\n"
        f"grid_ppm={grid.pixels_per_mm:.4f}\n"
        f"expected_from_marker={expected_px:.1f} px/sq\n"
        f"n_h={diag.n_h_samples} n_v={diag.n_v_samples} segs={len(diag.segments)}"
    )
    axes[1, 2].text(0.05, 0.5, e_text, fontsize=9, va="center", family="monospace")
    axes[1, 2].set_title("E: Numerical", fontsize=9)
    axes[1, 2].axis("off")

    g_text = (
        f"G: vs marker\n"
        f"pct_diff={pct:+.1f}%\n"
        f"ratio={ratio:.3f}\n"
        f"gate [{cfg.grid_ppm_ratio_min}, {cfg.grid_ppm_ratio_max}]\n"
        f"{'ACCEPT grid' if accepted else 'REJECT / fallback'}\n"
        f"grid_success={grid.success}"
    )
    axes[1, 3].text(0.05, 0.5, g_text, fontsize=9, va="center", family="monospace")
    axes[1, 3].set_title("G: Comparison", fontsize=9)
    axes[1, 3].axis("off")

    fig.suptitle(f"{sample.image_id}  scale_down={diag.scale:.3f}", fontsize=11)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def simulate_scale_source(
    grid: GridCalibrationResult,
    marker_ppm: float,
    cfg: ProjectConfig,
) -> str:
    if not cfg.use_grid_auto_calibration:
        return "marker"
    if not grid.success:
        return "marker_fallback"
    if marker_ppm <= 0:
        return "grid"
    ratio = grid.pixels_per_mm / marker_ppm
    if cfg.grid_ppm_ratio_min <= ratio <= cfg.grid_ppm_ratio_max:
        return "grid"
    return "marker_fallback"


def classify_failure_modes(row: dict) -> str:
    """Heuristic tags for report (comma-separated)."""
    tags: list[str] = []
    if not row["grid_success"]:
        tags.append("detection_failed")
    if row["marker_scale"] <= 0:
        tags.append("G_marker_missing")
    if row["number_of_detected_lines"] < 20:
        tags.append("A_few_lines")
    if row["n_h_samples"] < 3 or row["n_v_samples"] < 3:
        tags.append("B_weak_cluster")
    sp_h, sp_v = row["sp_h"], row["sp_v"]
    sp_corner = row["sp_corner"]
    if sp_h > 0 and sp_v > 0:
        hv_rel = abs(sp_h - sp_v) / max((sp_h + sp_v) / 2, 1e-6)
        if hv_rel > 0.25:
            tags.append("D_hv_mismatch")
    if sp_h > 0 and sp_v > 0 and sp_corner > 0:
        vals = [sp_h, sp_v, sp_corner]
        if max(vals) / max(min(vals), 1e-6) > 1.5:
            tags.append("C_spacing_disagree")
    est = row["estimated_grid_spacing_pixels"]
    if est in (15.0, 250.0) or (est > 0 and (est <= 16 or est >= 240)):
        tags.append("C_band_edge")
    if row["marker_scale"] > 0 and row["grid_success"]:
        ratio = row["grid_scale"] / row["marker_scale"]
        for mult in (2, 3, 4, 0.5, 0.33):
            if abs(ratio - mult) < 0.15 * mult:
                tags.append("F_integer_ratio")
                break
    if row["scale_source"] == "marker_fallback" and row["grid_success"]:
        tags.append("rejected_by_gate")
    return ",".join(tags) if tags else "ok"


def run_audit(
    cfg: ProjectConfig,
    image_ids: list[str],
    split: str,
    figure_dir: Path,
    csv_path: Path,
) -> pd.DataFrame:
    figure_dir.mkdir(parents=True, exist_ok=True)

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

    rows: list[dict] = []
    for sample in iterate_image_ids(cfg, split=split, image_ids=image_ids, load_images=True):
        if sample.image is None:
            continue
        img = sample.image
        diag = analyze_grid_pipeline(img)
        grid = estimate_grid_calibration(img, cfg=cfg_audit)
        marker_ppm = estimate_scale_from_markers(
            sample.blue_annotations(),
            sample.yellow_annotations(),
            sample.width,
            sample.height,
        )
        _, scale_source, _, _, _ = _choose_calibration(
            sample, cfg_audit, img, grid=grid
        )

        expected_px = marker_ppm * cfg.grid_square_mm if marker_ppm > 0 else float("nan")
        abs_diff = abs(grid.pixels_per_mm - marker_ppm) if grid.success and marker_ppm > 0 else float("nan")
        pct_diff = (
            100.0 * (grid.pixels_per_mm - marker_ppm) / marker_ppm
            if grid.success and marker_ppm > 0
            else float("nan")
        )

        inv_scale = 1.0 / diag.scale if diag.scale > 0 else 1.0

        row = {
            "image_id": sample.image_id,
            "marker_scale": marker_ppm,
            "grid_scale": grid.pixels_per_mm if grid.success else float("nan"),
            "absolute_difference": abs_diff,
            "percent_difference": pct_diff,
            "number_of_detected_lines": len(diag.segments),
            "number_of_detected_intersections": diag.harris_count,
            "estimated_grid_spacing_pixels": grid.pixels_per_grid_square,
            "expected_grid_spacing_pixels": expected_px,
            "sp_h": diag.sp_h * inv_scale,
            "sp_v": diag.sp_v * inv_scale,
            "sp_corner": diag.sp_corner * inv_scale,
            "n_h_samples": diag.n_h_samples,
            "n_v_samples": diag.n_v_samples,
            "scale_source": scale_source,
            "grid_success": grid.success,
            "ppm_ratio": grid.pixels_per_mm / marker_ppm if marker_ppm > 0 and grid.success else float("nan"),
        }
        row["failure_modes"] = classify_failure_modes(row)
        rows.append(row)

        save_debug_figure(
            figure_dir / f"{sample.image_id}.png",
            img,
            diag,
            sample,
            grid,
            marker_ppm,
            cfg_audit,
        )

    df = pd.DataFrame(rows)
    if not df.empty and "percent_difference" in df.columns:
        df = df.sort_values(
            "percent_difference",
            key=lambda s: s.abs(),
            ascending=False,
            na_position="last",
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")
    print(f"Figures in {figure_dir}")
    return df


def print_summary(df: pd.DataFrame, cfg: ProjectConfig) -> None:
    if df.empty:
        return
    valid = df[df["grid_success"] & (df["marker_scale"] > 0)]
    print("\n--- Summary ---")
    print(f"grid_square_mm: {cfg.grid_square_mm}")
    print(f"images: {len(df)}  grid_success: {df['grid_success'].sum()}")
    if len(valid):
        ratios = valid["ppm_ratio"]
        print(f"ppm_ratio median={ratios.median():.3f} mean={ratios.mean():.3f}")
        rejected = valid[
            (valid["ppm_ratio"] < cfg.grid_ppm_ratio_min)
            | (valid["ppm_ratio"] > cfg.grid_ppm_ratio_max)
        ]
        print(
            f"would reject by gate: {len(rejected)}/{len(valid)} "
            f"({100*len(rejected)/len(valid):.1f}%)"
        )
        print(f"scale_source counts:\n{df['scale_source'].value_counts().to_string()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid calibration investigation audit")
    p.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    p.add_argument("--limit", type=int, default=30, help="Max images (default: all validation)")
    p.add_argument("--all", action="store_true", help="Process full validation set")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "grid_calibration_audit",
        help="Audit output root (CSV + grid_debug/)",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="If set, write figures to {run_dir}/grid_debug/ only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    ids = load_validation_image_ids(cfg)
    if not args.all and args.limit:
        ids = ids[: args.limit]

    csv_path = args.output_dir / "grid_calibration_audit.csv"
    if args.run_dir is not None:
        figure_dir = Path(args.run_dir) / "grid_debug"
    else:
        figure_dir = args.output_dir / "grid_debug"

    df = run_audit(cfg, ids, args.split, figure_dir, csv_path)
    print_summary(df, cfg)


if __name__ == "__main__":
    main()
