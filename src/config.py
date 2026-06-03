"""
Central configuration for paths, class IDs, and calibration constants.

Adjust ``ProjectConfig`` fields or override via environment variables where noted.
See ``assignment.pdf`` for dataset-specific conventions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from src.paths import PROJECT_ROOT, get_storage_paths, is_colab

# Backwards-compatible alias (repository root).
REPO_ROOT: Final[Path] = PROJECT_ROOT

# Default YOLO class IDs (see data/fishnet.yaml when present)
CLASS_BLUE: Final[int] = 0
CLASS_YELLOW: Final[int] = 1
CLASS_FISH: Final[int] = 2

CLASS_NAMES: Final[dict[int, str]] = {
    CLASS_BLUE: "blue",
    CLASS_YELLOW: "yellow",
    CLASS_FISH: "fish",
}

# Physical size of calibration rectangles (mm), per assignment spec
CALIBRATION_RECT_MM: Final[float] = 100.0

# Physical size of one tank grid cell (mm) for automatic grid calibration
GRID_SQUARE_MM: Final[float] = 10.0


def _default_paths() -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, bool]:
    """Initialize path fields from :func:`src.paths.get_storage_paths`."""
    sp = get_storage_paths()
    return (
        sp.storage_root,
        sp.data_root,
        sp.data_raw,
        sp.data_annotations,
        sp.intermediate_cache_root,
        sp.runs_root,
        sp.figures_root,
        sp.legacy_outputs_predictions,
        sp.logs_root,
        sp.colab,
    )


@dataclass
class ProjectConfig:
    """
    Runtime configuration for dataset I/O, calibration, and outputs.

    Path fields default from :mod:`src.paths` (local legacy layout or Colab Drive).
    """

    repo_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    storage_root: Path = field(default_factory=lambda: _default_paths()[0])
    data_root: Path = field(default_factory=lambda: _default_paths()[1])
    data_raw: Path = field(default_factory=lambda: _default_paths()[2])
    data_annotations: Path = field(default_factory=lambda: _default_paths()[3])
    data_processed: Path = field(default_factory=lambda: _default_paths()[4])

    runs_root: Path = field(default_factory=lambda: _default_paths()[5])
    figures_root: Path = field(default_factory=lambda: _default_paths()[6])

    outputs_figures: Path = field(default_factory=lambda: _default_paths()[6])
    outputs_predictions: Path = field(default_factory=lambda: _default_paths()[7])
    outputs_metrics: Path = field(default_factory=lambda: _default_paths()[8])

    is_colab: bool = field(default_factory=lambda: _default_paths()[9])
    cache_depth_maps: bool = True

    calibration_rect_mm: float = CALIBRATION_RECT_MM
    default_split: str = "test"
    measurement_method: str = "bbox"
    apply_perspective_correction: bool = False

    use_grid_auto_calibration: bool = False
    use_depth_estimation: bool = False
    use_3d_measurement: bool = False
    grid_square_mm: float = GRID_SQUARE_MM
    depth_model_name: str = "depth-anything/DA3-SMALL"
    grid_ppm_ratio_min: float = 0.85
    grid_ppm_ratio_max: float = 1.15
    use_depth_metric_scale: bool = False
    visualize_grid_debug: bool = False

    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def resolve_dataset_root(self) -> Path:
        """
        Return the directory containing ``images/`` and ``labels/`` subtrees.

        Checks, in order:
        1. ``data/fishnet`` (Drive layout or in-repo extract)
        2. ``data/raw/fishnet``
        3. ``data/raw``
        """
        candidates = [
            self.data_root / "fishnet",
            self.data_raw / "fishnet",
            self.repo_root / "data" / "fishnet",
            self.data_raw,
        ]
        for path in candidates:
            if (path / "images").is_dir() and (path / "labels").is_dir():
                return path
        return candidates[0]

    def images_dir(self, split: str) -> Path:
        """Path to images for a split (``train``, ``valid``, ``test``)."""
        root = self.resolve_dataset_root()
        split_aliases = {"valid": ("valid", "val"), "val": ("val", "valid")}
        names = split_aliases.get(split, (split,))
        for name in names:
            candidate = root / "images" / name
            if candidate.is_dir():
                return candidate
        return root / "images" / split

    def labels_dir(self, split: str) -> Path:
        """Path to YOLO label ``.txt`` files for a split."""
        root = self.resolve_dataset_root()
        split_aliases = {"valid": ("valid", "val"), "val": ("val", "valid")}
        names = split_aliases.get(split, (split,))
        for name in names:
            candidate = root / "labels" / name
            if candidate.is_dir():
                return candidate
        return root / "labels" / split

    def depth_cache_dir(self) -> Path:
        """Directory for cached monocular depth maps."""
        sp = get_storage_paths()
        base = sp.depth_cache_root if self.cache_depth_maps else sp.ephemeral_cache_root() / "depth_maps"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def ensure_output_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for path in (
            self.data_processed,
            self.data_annotations,
            self.runs_root,
            self.figures_root,
            self.outputs_figures,
            self.outputs_predictions,
            self.outputs_metrics,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def with_repo_root(cls, repo_root: Path) -> ProjectConfig:
        """Config with all storage paths under ``repo_root`` (for tests and temp dirs)."""
        root = Path(repo_root)
        return cls(
            repo_root=root,
            storage_root=root,
            data_root=root / "data",
            data_raw=root / "data" / "raw",
            data_annotations=root / "data" / "annotations",
            data_processed=root / "data" / "processed",
            runs_root=root / "outputs" / "runs",
            figures_root=root / "outputs" / "figures",
            outputs_figures=root / "outputs" / "figures",
            outputs_predictions=root / "outputs" / "predictions",
            outputs_metrics=root / "outputs" / "metrics",
            is_colab=False,
        )


def apply_storage_preferences(
    cfg: ProjectConfig,
    *,
    cache_results: bool = True,
    save_figures_to_drive: bool = True,
    save_metrics_to_drive: bool = True,
    save_predictions_to_drive: bool = True,
) -> ProjectConfig:
    """
    Adjust ``ProjectConfig`` paths for Colab ephemeral vs Drive persistence.

    On local machines all flags default to True and paths are unchanged.
    """
    from dataclasses import replace

    sp = get_storage_paths()
    updates: dict = {"cache_depth_maps": cache_results}

    if sp.colab:
        if not save_predictions_to_drive:
            updates["runs_root"] = sp.ephemeral_runs_root()
        if not save_metrics_to_drive:
            updates["outputs_metrics"] = sp.ephemeral_runs_root() / "metrics"
        if not save_figures_to_drive:
            updates["figures_root"] = sp.ephemeral_runs_root() / "figures"
            updates["outputs_figures"] = updates["figures_root"]
        if not cache_results:
            updates["data_processed"] = sp.ephemeral_cache_root()

    cfg_out = replace(cfg, **updates)
    return cfg_out


def get_config() -> ProjectConfig:
    """
    Build configuration, optionally overriding split/method from environment.

    Environment variables:
    - ``FISHNET_SPLIT``: train | valid | test
    - ``FISHNET_METHOD``: bbox | pca | skeleton
    - ``FISHNET_PERSPECTIVE``: 1 / true to enable homography rectification
    - ``FISHNET_GRID_AUTO``: 1 / true for automatic grid calibration
    - ``FISHNET_DEPTH``: 1 / true for depth estimation
    - ``FISHNET_3D``: 1 / true for 3D skeleton measurement
    - ``FISHNET_DRIVE_ROOT`` / ``FISHNET_STORAGE_ROOT``: override storage root
    """
    cfg = ProjectConfig()

    def _env_bool(name: str) -> bool:
        return os.environ.get(name, "").lower() in ("1", "true", "yes")

    if split := os.environ.get("FISHNET_SPLIT"):
        cfg.default_split = split
    if method := os.environ.get("FISHNET_METHOD"):
        cfg.measurement_method = method
    if _env_bool("FISHNET_PERSPECTIVE"):
        cfg.apply_perspective_correction = True
    if _env_bool("FISHNET_GRID_AUTO"):
        cfg.use_grid_auto_calibration = True
    if _env_bool("FISHNET_DEPTH"):
        cfg.use_depth_estimation = True
    if _env_bool("FISHNET_3D"):
        cfg.use_3d_measurement = True
    if grid_mm := os.environ.get("FISHNET_GRID_SQUARE_MM"):
        cfg.grid_square_mm = float(grid_mm)
    if depth_model := os.environ.get("FISHNET_DEPTH_MODEL"):
        cfg.depth_model_name = depth_model
    return cfg
