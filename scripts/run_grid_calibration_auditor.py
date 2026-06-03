#!/usr/bin/env python3
"""
Run the read-only grid calibration auditor on validation images.

Outputs under ``runs/<run_name>/grid_audit/``:
  - grid_audit_per_image.csv
  - grid_audit_rejection_table.csv
  - grid_audit_coverage_accuracy.csv
  - grid_audit_failure_categories.csv
  - grid_audit_report.md
  - figures/<image_id>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.grid_audit import (  # noqa: E402
    analyze_grid_calibration_set,
    visualize_grid_audit,
    visualize_grid_audit_set,
)
from src.config import get_config  # noqa: E402
from src.experiments import load_validation_image_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid calibration auditor (read-only)")
    parser.add_argument("--run-name", default="grid_audit", help="Subfolder under runs/")
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--limit", type=int, default=30, help="Max images (default 30)")
    parser.add_argument("--image-id", default=None, help="Audit a single image only")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip CSV/report; only write figures (uses --image-id or --limit)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default runs/<run_name>/grid_audit)",
    )
    args = parser.parse_args()
    cfg = get_config()
    out_dir = args.output_dir or (cfg.runs_root / args.run_name / "grid_audit")

    if args.image_id:
        path = visualize_grid_audit(
            args.image_id,
            cfg=cfg,
            split=args.split,
            run_name=args.run_name,
            output_dir=out_dir / "figures",
        )
        print(f"Wrote {path}")
        if not args.figures_only:
            analyze_grid_calibration_set(
                cfg=cfg,
                image_ids=[args.image_id],
                split=args.split,
                limit=None,
                run_name=args.run_name,
                output_dir=out_dir,
            )
        return

    if args.figures_only:
        visualize_grid_audit_set(
            cfg=cfg,
            split=args.split,
            limit=args.limit,
            run_name=args.run_name,
            output_dir=out_dir / "figures",
        )
        print(f"Figures written to {out_dir / 'figures'}")
        return

    image_ids = load_validation_image_ids(cfg)
    if args.limit:
        image_ids = image_ids[: args.limit]

    df = analyze_grid_calibration_set(
        cfg=cfg,
        image_ids=image_ids,
        split=args.split,
        limit=None,
        run_name=args.run_name,
        output_dir=out_dir,
        generate_figures=True,
    )
    print(f"Audited {len(df)} images -> {out_dir}")
    print(f"Report: {out_dir / 'grid_audit_report.md'}")


if __name__ == "__main__":
    main()
