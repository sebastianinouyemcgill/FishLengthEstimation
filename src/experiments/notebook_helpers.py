"""
Shared configuration and plotting helpers for experiment notebooks.

Used by ``03_run_experiments.ipynb`` and ``04_analyze_results.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ProjectConfig, apply_storage_preferences, get_config
from src.paths import get_storage_paths


@dataclass
class RunExperimentsConfig:
    """
    Single-cell configuration for ``03_run_experiments.ipynb``.

    Set ``experiments`` to run an explicit list (legacy mode). Otherwise a grid
    is built from ``pipelines`` × ``methods`` × ``splits``.
    """

    pipelines: list[str] = field(default_factory=lambda: ["baseline", "advanced"])
    methods: list[str] = field(default_factory=lambda: ["bbox", "pca", "skeleton"])
    splits: list[str] = field(default_factory=lambda: ["valid"])

    image_ids: list[str] | None = None
    validation_set_only: bool = True
    limit: int | None = None

    overwrite: bool = True
    visualize: bool = False
    evaluate: bool = True

    runs_root: Path | None = None
    run_name_template: str = "{pipeline}_{method}_v1"

    perspective: bool = False
    use_grid_auto_calibration: bool = True
    use_depth_estimation: bool = False
    use_3d_measurement: bool = False
    use_hrnet_keypoints: bool = False
    use_pseudo_label_training: bool = False
    # Feature-based SH length calibration (trains on validation_lengths.csv)
    run_regression_calibration: bool = False
    regression_run_name: str = "regression_calibrated_v1"
    regression_model_type: str = "random_forest"
    regression_train_split: str = "valid"

    # Apply a saved regression model during baseline runs (requires regression_model_path)
    use_regression_model: bool = False
    regression_model_path: Path | None = None

    experiments: list[dict[str, Any]] | None = None
    stop_on_error: bool = False

    cache_results: bool = True
    save_figures_to_drive: bool = True
    save_metrics_to_drive: bool = True
    save_predictions_to_drive: bool = True


@dataclass
class AnalyzeResultsConfig:
    """Single-cell configuration for ``04_analyze_results.ipynb``."""

    run_names: list[str] | None = None
    inspect_run: str = "baseline_bbox_v1"
    compare_run: str | None = None

    metrics: list[str] = field(default_factory=lambda: ["mae_mm", "rmse_mm"])

    plot_mae_comparison: bool = True
    plot_rmse_comparison: bool = True
    plot_error_histogram: bool = True
    plot_pred_vs_gt: bool = True
    plot_residual: bool = True
    plot_pivot_table: bool = True
    plot_per_image_viz: bool = True

    top_k_worst: int = 5
    top_k_best: int = 5
    viz_image_ids: list[str] | None = None
    save_viz: bool = False

    figures_dir: Path | None = None
    histogram_bins: int = 20


def resolve_runs_root(cfg: RunExperimentsConfig | None = None) -> Path:
    project = get_config()
    if cfg and cfg.runs_root:
        return Path(cfg.runs_root)
    return project.runs_root


def project_config_for_experiments(run_cfg: RunExperimentsConfig | None = None) -> ProjectConfig:
    """Build ``ProjectConfig`` with storage preferences from notebook run config."""
    base = get_config()
    if run_cfg is None:
        return base
    return apply_storage_preferences(
        base,
        cache_results=run_cfg.cache_results,
        save_figures_to_drive=run_cfg.save_figures_to_drive,
        save_metrics_to_drive=run_cfg.save_metrics_to_drive,
        save_predictions_to_drive=run_cfg.save_predictions_to_drive,
    )


def build_experiment_specs(run_cfg: RunExperimentsConfig) -> list[dict[str, Any]]:
    """Expand config into kwargs dicts for ``run_experiment`` / ``run_experiments``."""
    if run_cfg.experiments:
        return [dict(spec) for spec in run_cfg.experiments]

    specs: list[dict[str, Any]] = []
    for pipeline in run_cfg.pipelines:
        for method in run_cfg.methods:
            for split in run_cfg.splits:
                run_name = run_cfg.run_name_template.format(
                    pipeline=pipeline,
                    method=method,
                    split=split,
                )
                spec: dict[str, Any] = {
                    "pipeline": pipeline,
                    "method": method,
                    "split": split,
                    "run_name": run_name,
                    "overwrite": run_cfg.overwrite,
                    "visualize": run_cfg.visualize,
                    "evaluate": run_cfg.evaluate,
                    "limit": run_cfg.limit,
                }
                if run_cfg.image_ids:
                    spec["image_ids"] = list(run_cfg.image_ids)
                    spec["validation_set_only"] = False
                else:
                    spec["validation_set_only"] = run_cfg.validation_set_only

                if pipeline == "advanced":
                    spec["perspective"] = run_cfg.perspective
                    spec["use_grid_auto_calibration"] = run_cfg.use_grid_auto_calibration
                    spec["use_depth_estimation"] = run_cfg.use_depth_estimation
                    spec["use_3d_measurement"] = run_cfg.use_3d_measurement

                if pipeline == "baseline" and run_cfg.use_regression_model:
                    spec["use_regression_model"] = True
                    if run_cfg.regression_model_path:
                        spec["regression_model_path"] = run_cfg.regression_model_path

                specs.append(spec)
    return specs


def benchmark_experiment_specs(
    run_cfg: RunExperimentsConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Official benchmark grid: baseline methods + grid-calibrated advanced (skeleton).

    Depth, 3D, and perspective experiments are excluded.
    """
    run_cfg = run_cfg or RunExperimentsConfig(
        pipelines=["baseline", "advanced"],
        methods=["bbox", "pca", "skeleton"],
        use_grid_auto_calibration=True,
        use_depth_estimation=False,
        use_3d_measurement=False,
        perspective=False,
    )
    return build_experiment_specs(run_cfg)


def preview_experiment_specs(
    specs: list[dict[str, Any]],
    run_cfg: RunExperimentsConfig | None = None,
) -> pd.DataFrame:
    """Human-readable preview before launching runs."""
    rows = []
    if run_cfg and run_cfg.run_regression_calibration:
        rows.append(
            {
                "run_name": run_cfg.regression_run_name,
                "pipeline": "regression",
                "method": "calibrated",
                "split": run_cfg.splits[0] if run_cfg.splits else "valid",
                "visualize": False,
                "perspective": False,
                "grid": False,
                "depth": False,
                "3d": False,
                "regression": True,
            }
        )
    for spec in specs:
        rows.append(
            {
                "run_name": spec.get("run_name"),
                "pipeline": spec.get("pipeline"),
                "method": spec.get("method"),
                "split": spec.get("split"),
                "visualize": spec.get("visualize", False),
                "perspective": spec.get("perspective", False),
                "grid": spec.get("use_grid_auto_calibration", False),
                "depth": spec.get("use_depth_estimation", False),
                "3d": spec.get("use_3d_measurement", False),
                "regression": spec.get("use_regression_model", False),
            }
        )
    return pd.DataFrame(rows)


def run_configured_experiments(
    run_cfg: RunExperimentsConfig,
    specs: list[dict[str, Any]],
    *,
    cfg: ProjectConfig | None = None,
) -> pd.DataFrame:
    """
    Run standard experiment specs plus optional regression calibration.

    Used by ``03_run_experiments.ipynb`` when ``RUN=True``.
    """
    from src.experiments import run_experiment, run_regression_experiment

    cfg = cfg or project_config_for_experiments(run_cfg)
    results = []

    if run_cfg.run_regression_calibration:
        results.append(
            run_regression_experiment(
                cfg,
                run_name=run_cfg.regression_run_name,
                train_split=run_cfg.regression_train_split,
                eval_split=run_cfg.splits[0] if run_cfg.splits else None,
                overwrite=run_cfg.overwrite,
                evaluate=run_cfg.evaluate,
                validation_set_only=run_cfg.validation_set_only,
                image_ids=run_cfg.image_ids,
                model_type=run_cfg.regression_model_type,
            )
        )

    for spec in specs:
        try:
            results.append(run_experiment(cfg=cfg, **spec))
        except Exception:
            if run_cfg.stop_on_error:
                raise
            import logging

            logging.getLogger(__name__).exception("Experiment failed: %s", spec)

    from src.experiments.results import results_to_dataframe

    return results_to_dataframe(results)


def filter_summary(
    summary: pd.DataFrame,
    run_names: list[str] | None,
) -> pd.DataFrame:
    """Keep latest row per run_name, optionally restricted to a name list."""
    if summary.empty:
        return summary
    df = summary.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["run_name"], keep="last")
    if run_names:
        df = df[df["run_name"].isin(run_names)]
    return df.reset_index(drop=True)


def load_comparison(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "comparison.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def resolve_figures_dir(cfg: AnalyzeResultsConfig, project: ProjectConfig | None = None) -> Path:
    project = project or get_config()
    if cfg.figures_dir:
        path = Path(cfg.figures_dir)
    else:
        sp = get_storage_paths()
        path = sp.figures_analysis
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_or_show(fig: plt.Figure, path: Path | None, show: bool = True) -> None:
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_mae_comparison(
    summary: pd.DataFrame,
    *,
    title: str = "MAE comparison",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    df = summary.dropna(subset=["mae_mm"]).sort_values("run_name")
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["run_name"], df["mae_mm"], color="steelblue")
    ax.set_ylabel("MAE (mm)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_or_show(fig, save_path, show=show)
    return fig


def plot_rmse_comparison(
    summary: pd.DataFrame,
    *,
    title: str = "RMSE comparison",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    df = summary.dropna(subset=["rmse_mm"]).sort_values("run_name")
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["run_name"], df["rmse_mm"], color="coral")
    ax.set_ylabel("RMSE (mm)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_or_show(fig, save_path, show=show)
    return fig


def plot_mae_rmse_panel(
    summary: pd.DataFrame,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    df = summary.dropna(subset=["mae_mm"]).sort_values("run_name")
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(df["run_name"], df["mae_mm"], color="steelblue")
    axes[0].set_title("MAE (mm)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(df["run_name"], df["rmse_mm"], color="coral")
    axes[1].set_title("RMSE (mm)")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_or_show(fig, save_path, show=show)
    return fig


def plot_error_histogram(
    comparison: pd.DataFrame,
    *,
    run_name: str = "",
    bins: int = 20,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    if "abs_error_mm" not in comparison.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(comparison["abs_error_mm"].dropna(), bins=bins, edgecolor="k")
    ax.set_xlabel("Absolute error (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error distribution — {run_name}" if run_name else "Error distribution")
    plt.tight_layout()
    _save_or_show(fig, save_path, show=show)
    return fig


def plot_pred_vs_gt(
    comparison: pd.DataFrame,
    *,
    run_name: str = "",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    if not {"length_mm", "predicted_length_mm"}.issubset(comparison.columns):
        return None
    gt = comparison["length_mm"].astype(float)
    pred = comparison["predicted_length_mm"].astype(float)
    mask = np.isfinite(gt) & np.isfinite(pred)
    if mask.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt[mask], pred[mask], alpha=0.7, edgecolors="k", linewidths=0.5)
    lo = min(gt[mask].min(), pred[mask].min())
    hi = max(gt[mask].max(), pred[mask].max())
    ax.plot([lo, hi], [lo, hi], "r--", label="y = x")
    ax.set_xlabel("Ground truth (mm)")
    ax.set_ylabel("Predicted (mm)")
    ax.set_title(f"Prediction vs ground truth — {run_name}" if run_name else "Prediction vs GT")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    _save_or_show(fig, save_path, show=show)
    return fig


def plot_residuals(
    comparison: pd.DataFrame,
    *,
    run_name: str = "",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    if not {"length_mm", "predicted_length_mm"}.issubset(comparison.columns):
        return None
    gt = comparison["length_mm"].astype(float)
    pred = comparison["predicted_length_mm"].astype(float)
    residual = pred - gt
    mask = np.isfinite(residual) & np.isfinite(gt)
    if mask.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(gt[mask], residual[mask], alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Ground truth (mm)")
    ax.set_ylabel("Residual (pred − GT, mm)")
    ax.set_title(f"Residual plot — {run_name}" if run_name else "Residual plot")
    plt.tight_layout()
    _save_or_show(fig, save_path, show=show)
    return fig


def top_k_errors(
    comparison: pd.DataFrame,
    *,
    k: int,
    worst: bool = True,
) -> pd.DataFrame:
    if "abs_error_mm" not in comparison.columns:
        return comparison.head(0)
    ascending = not worst
    return comparison.sort_values("abs_error_mm", ascending=ascending).head(k)


def run_analysis_plots(
    summary: pd.DataFrame,
    analysis_cfg: AnalyzeResultsConfig,
    runs_root: Path,
    *,
    show: bool = True,
) -> dict[str, Path | None]:
    """Generate all plots enabled in ``analysis_cfg``."""
    saved: dict[str, Path | None] = {}
    fig_dir = resolve_figures_dir(analysis_cfg) if analysis_cfg.save_viz else None

    if analysis_cfg.plot_mae_comparison and analysis_cfg.plot_rmse_comparison:
        path = fig_dir / "mae_rmse_comparison.png" if fig_dir else None
        plot_mae_rmse_panel(summary, save_path=path, show=show)
        saved["mae_rmse"] = path
    else:
        if analysis_cfg.plot_mae_comparison:
            path = fig_dir / "mae_comparison.png" if fig_dir else None
            plot_mae_comparison(summary, save_path=path, show=show)
            saved["mae"] = path
        if analysis_cfg.plot_rmse_comparison:
            path = fig_dir / "rmse_comparison.png" if fig_dir else None
            plot_rmse_comparison(summary, save_path=path, show=show)
            saved["rmse"] = path

    inspect_dir = runs_root / analysis_cfg.inspect_run
    comparison = load_comparison(inspect_dir)
    if comparison is not None:
        if analysis_cfg.plot_error_histogram:
            path = fig_dir / f"{analysis_cfg.inspect_run}_error_hist.png" if fig_dir else None
            plot_error_histogram(
                comparison,
                run_name=analysis_cfg.inspect_run,
                bins=analysis_cfg.histogram_bins,
                save_path=path,
                show=show,
            )
            saved["error_hist"] = path
        if analysis_cfg.plot_pred_vs_gt:
            path = fig_dir / f"{analysis_cfg.inspect_run}_pred_vs_gt.png" if fig_dir else None
            plot_pred_vs_gt(comparison, run_name=analysis_cfg.inspect_run, save_path=path, show=show)
            saved["pred_vs_gt"] = path
        if analysis_cfg.plot_residual:
            path = fig_dir / f"{analysis_cfg.inspect_run}_residual.png" if fig_dir else None
            plot_residuals(comparison, run_name=analysis_cfg.inspect_run, save_path=path, show=show)
            saved["residual"] = path

    return saved
