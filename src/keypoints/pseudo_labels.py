"""
Skeleton-based pseudo keypoints for future HRNet training.

No model inference — reads masks and exports structured 2D points per image.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.measurement.core import get_contour_endpoints
from src.masks import skeletonize_mask


@dataclass
class PseudoKeypointRecord:
    """Two endpoint keypoints (head/tail proxy) from skeleton geometry."""

    image_id: str
    keypoints_xy: list[tuple[float, float]]
    keypoint_names: tuple[str, str] = ("endpoint_a", "endpoint_b")
    skeleton_length_px: float = 0.0
    source: str = "skeleton_pseudo"


def extract_pseudo_keypoints(
    image_id: str,
    fish_mask: np.ndarray,
) -> PseudoKeypointRecord | None:
    """
    Derive pseudo keypoints from the largest skeleton component.

    Returns None when the mask is empty or endpoints cannot be found.
    """
    skel = skeletonize_mask(fish_mask)
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return None

    endpoints = get_contour_endpoints(skel)
    if endpoints is None:
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        keypoints = [(cx, cy), (cx, cy)]
        length_px = 0.0
    else:
        (x0, y0), (x1, y1) = endpoints
        keypoints = [(float(x0), float(y0)), (float(x1), float(y1))]
        length_px = float(np.hypot(x1 - x0, y1 - y0))

    return PseudoKeypointRecord(
        image_id=image_id,
        keypoints_xy=keypoints,
        skeleton_length_px=length_px,
    )


def save_pseudo_labels_jsonl(records: list[PseudoKeypointRecord], path: Path) -> Path:
    """Write one JSON object per line (image_id, keypoints, metadata)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            row = asdict(rec)
            row["keypoints_xy"] = [list(pt) for pt in rec.keypoints_xy]
            f.write(json.dumps(row) + "\n")
    return path
