"""
Ground-truth CSV paths and validation image ID loading.

Kept separate from ``experiments.__init__`` to avoid circular imports with ``regression``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)


def load_validation_image_ids(cfg: ProjectConfig | None = None) -> list[str]:
    """
    Load ``image_id`` list from the manual validation set.

    Prefers ``validation_lengths.csv`` (annotated fish only), then
    ``validation_images.csv`` from notebook 02.
    """
    cfg = cfg or get_config()
    candidates = [
        cfg.data_annotations / "validation_lengths.csv",
        cfg.data_annotations / "validation_images.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "image_id" not in df.columns:
            raise ValueError(f"{path} must contain an image_id column")
        ids = df["image_id"].astype(str).tolist()
        logger.info("Loaded %d validation image IDs from %s", len(ids), path.name)
        return ids
    raise FileNotFoundError(
        "No validation CSV found. Run notebook 02 first to create "
        "data/annotations/validation_lengths.csv"
    )


def default_ground_truth_path(cfg: ProjectConfig | None = None) -> Path | None:
    """
    Prefer manual validation lengths, then generic ``lengths_mm.csv``.

    Returns ``None`` if no ground-truth file exists (run proceeds without eval).
    """
    cfg = cfg or get_config()
    candidates = [
        cfg.data_annotations / "validation_lengths.csv",
        cfg.data_annotations / "lengths_mm.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None
