"""
Ground-truth CSV paths and validation image ID loading.

Kept separate from ``experiments.__init__`` to avoid circular imports with ``regression``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

from src.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)

# Keys for ``ProjectConfig.ground_truth_source`` / ``RunExperimentsConfig.ground_truth_source``
GROUND_TRUTH_SOURCES: Final[dict[str, str]] = {
    "validation_lengths": "validation_lengths.csv",
    "validation_lengths2": "validation_lengths2.csv",
    "test_lengths": "test_lengths.csv",
    "test_lengths2": "test_lengths2.csv",
    "lengths_mm": "lengths_mm.csv",
}


def resolve_ground_truth_path(
    cfg: ProjectConfig | None = None,
    *,
    source: str | None = None,
) -> Path | None:
    """
    Resolve a ground-truth CSV under ``cfg.data_annotations``.

    Parameters
    ----------
    source:
        One of ``validation_lengths``, ``validation_lengths2``, ``lengths_mm``.
        Defaults to ``cfg.ground_truth_source``.
    """
    cfg = cfg or get_config()
    key = source or cfg.ground_truth_source
    if key not in GROUND_TRUTH_SOURCES:
        raise ValueError(
            f"Unknown ground_truth_source={key!r}; "
            f"choose from {list(GROUND_TRUTH_SOURCES)}"
        )
    path = cfg.data_annotations / GROUND_TRUTH_SOURCES[key]
    if path.is_file():
        return path
    logger.warning("Ground truth file not found: %s", path)
    return None


def default_ground_truth_path(cfg: ProjectConfig | None = None) -> Path | None:
    """Return the configured ground-truth CSV if it exists."""
    return resolve_ground_truth_path(cfg)


def load_validation_image_ids(
    cfg: ProjectConfig | None = None,
    *,
    ground_truth_source: str | None = None,
) -> list[str]:
    """
    Load ``image_id`` list for evaluation / regression training.

    Uses the configured ground-truth CSV first, then ``validation_images.csv``.
    """
    cfg = cfg or get_config()
    gt_path = resolve_ground_truth_path(cfg, source=ground_truth_source)
    if gt_path is not None:
        df = pd.read_csv(gt_path)
        if "image_id" not in df.columns:
            raise ValueError(f"{gt_path} must contain an image_id column")
        ids = df["image_id"].astype(str).tolist()
        logger.info("Loaded %d validation image IDs from %s", len(ids), gt_path.name)
        return ids

    fallback = cfg.data_annotations / "validation_images.csv"
    if fallback.is_file():
        df = pd.read_csv(fallback)
        if "image_id" not in df.columns:
            raise ValueError(f"{fallback} must contain an image_id column")
        ids = df["image_id"].astype(str).tolist()
        logger.info("Loaded %d validation image IDs from %s", len(ids), fallback.name)
        return ids

    raise FileNotFoundError(
        f"No validation CSV found for source={cfg.ground_truth_source!r}. "
        "Create data/annotations/validation_lengths.csv (notebook 02) or set "
        "ground_truth_source to an existing file."
    )
