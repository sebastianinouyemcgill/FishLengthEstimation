"""Tests for length regression features and model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ProjectConfig
from src.measurement.features import FEATURE_COLUMNS, extract_length_features
from src.models.length_regression import LengthRegressionModel


def _synthetic_mask() -> np.ndarray:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 20:100] = 255
    return mask


def test_extract_length_features_positive() -> None:
    from src.calibration import CalibrationResult

    calib = CalibrationResult(pixels_per_mm=2.0, homography=None)
    feat = extract_length_features(_synthetic_mask(), calib)
    d = feat.as_dict()
    assert set(d) == set(FEATURE_COLUMNS)
    assert d["skeleton_length"] > 0
    assert d["aspect_ratio"] > 0


def test_length_regression_model_roundtrip(tmp_path: Path) -> None:
    n = 20
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "skeleton_length": rng.uniform(80, 120, n),
            "pca_length": rng.uniform(80, 120, n),
            "bounding_box_diagonal_length": rng.uniform(80, 130, n),
            "mask_area": rng.uniform(500, 2000, n),
            "mask_perimeter": rng.uniform(100, 400, n),
            "aspect_ratio": rng.uniform(1.5, 4.0, n),
        }
    )
    y = X["skeleton_length"] * 1.05 + 2.0 + rng.normal(0, 1, n)
    model = LengthRegressionModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (n,)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = LengthRegressionModel.load(path)
    np.testing.assert_allclose(loaded.predict(X), preds, rtol=1e-5)


def test_run_inference_unchanged_when_regression_disabled(tmp_path: Path) -> None:
    """Baseline run_inference path must not depend on regression modules."""
    import cv2

    root = tmp_path / "data" / "fishnet"
    valid_img = root / "images" / "valid"
    valid_lbl = root / "labels" / "valid"
    valid_img.mkdir(parents=True)
    valid_lbl.mkdir(parents=True)
    cv2.imwrite(str(valid_img / "fish1.JPG"), np.zeros((40, 60, 3), dtype=np.uint8))
    (valid_lbl / "fish1.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
        "1 0.7 0.7 0.8 0.7 0.8 0.8 0.7 0.8\n"
        "2 0.3 0.3 0.7 0.3 0.7 0.6 0.35 0.55\n"
    )
    cfg = ProjectConfig.with_repo_root(tmp_path)
    cfg.use_regression_model = False
    from src.pipelines.base import run_inference

    out = tmp_path / "pred.csv"
    run_inference(cfg, split="valid", method="bbox", predictions_path=out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["image_id", "predicted_length_mm"]
