"""
Build a manifest for HRNet training from pseudo keypoints (no training loop).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig, get_config
from src.dataset import iterate_image_ids
from src.keypoints.pseudo_labels import PseudoKeypointRecord, extract_pseudo_keypoints
from src.masks import mask_from_class


@dataclass
class KeypointDatasetManifest:
    """Paths and counts for a pseudo-label export."""

    manifest_csv: Path
    labels_jsonl: Path
    n_images: int
    n_labeled: int


def build_keypoint_manifest(
    *,
    cfg: ProjectConfig | None = None,
    split: str = "train",
    image_ids: list[str] | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> KeypointDatasetManifest:
    """
    Iterate images, extract skeleton pseudo keypoints, write CSV + JSONL.

    Does not train HRNet.
    """
    cfg = cfg or get_config()
    out = output_dir or (cfg.data_processed / "keypoints" / split)
    out.mkdir(parents=True, exist_ok=True)

    labels_path = out / "pseudo_keypoints.jsonl"
    rows: list[dict] = []
    n_seen = 0

    iterator = iterate_image_ids(cfg, split=split, image_ids=image_ids, load_images=True)
    with labels_path.open("w", encoding="utf-8") as jf:
        for sample in iterator:
            if limit is not None and n_seen >= limit:
                break
            n_seen += 1
            if sample.image is None:
                continue
            h, w = sample.image.shape[:2]
            mask = mask_from_class(
                sample.annotations,
                class_name="fish",
                height=h,
                width=w,
            )
            rec = extract_pseudo_keypoints(sample.image_id, mask)
            if rec is None:
                rows.append(
                    {
                        "image_id": sample.image_id,
                        "has_keypoints": False,
                        "skeleton_length_px": 0.0,
                    }
                )
                continue
            row_dict = {
                "image_id": rec.image_id,
                "has_keypoints": True,
                "skeleton_length_px": rec.skeleton_length_px,
                "x0": rec.keypoints_xy[0][0],
                "y0": rec.keypoints_xy[0][1],
                "x1": rec.keypoints_xy[1][0],
                "y1": rec.keypoints_xy[1][1],
            }
            rows.append(row_dict)
            payload = {
                "image_id": rec.image_id,
                "keypoints_xy": [list(p) for p in rec.keypoints_xy],
                "keypoint_names": list(rec.keypoint_names),
                "skeleton_length_px": rec.skeleton_length_px,
                "source": rec.source,
            }
            jf.write(json.dumps(payload) + "\n")

    manifest_csv = out / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)
    n_labeled = sum(1 for r in rows if r.get("has_keypoints"))
    return KeypointDatasetManifest(
        manifest_csv=manifest_csv,
        labels_jsonl=labels_path,
        n_images=len(rows),
        n_labeled=n_labeled,
    )
