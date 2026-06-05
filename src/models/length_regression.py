"""
Feature-based regression calibration for fish SH length (mm).

Skeleton length is the strongest single geometric estimator but system bias varies with
pose and mask quality. A sklearn regressor learns a correction from multiple mask
features (skeleton, PCA, bbox diagonal, area, perimeter, aspect ratio) to manual
``true_fish_length_mm`` labels, improving MAE over skeleton-only predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.measurement.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

TARGET_COLUMN = "true_fish_length_mm"


def _make_estimator(model_type: str = "random_forest", **kwargs: Any):
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost not installed; use model_type='random_forest' or pip install xgboost"
            ) from exc
        return XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 4),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=kwargs.get("subsample", 0.9),
            random_state=kwargs.get("random_state", 42),
        )
    return RandomForestRegressor(
        n_estimators=kwargs.get("n_estimators", 200),
        max_depth=kwargs.get("max_depth", 8),
        min_samples_leaf=kwargs.get("min_samples_leaf", 1),
        random_state=kwargs.get("random_state", 42),
    )


class LengthRegressionModel:
    """Sklearn wrapper for mask-feature → true SH length (mm) regression."""

    def __init__(self, model_type: str = "random_forest", **estimator_kwargs: Any) -> None:
        self.model_type = model_type
        self.estimator_kwargs = estimator_kwargs
        self._model = _make_estimator(model_type, **estimator_kwargs)
        self.feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LengthRegressionModel:
        X_arr = self._to_array(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
        if mask.sum() < 2:
            raise ValueError(f"Need at least 2 finite training rows; got {mask.sum()}")
        self._model.fit(X_arr[mask], y_arr[mask])
        logger.info("Fitted length regression on %d samples (%s)", mask.sum(), self.model_type)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = self._to_array(X)
        preds = self._model.predict(X_arr)
        return np.asarray(preds, dtype=np.float64)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "model_type": self.model_type,
            "estimator_kwargs": self.estimator_kwargs,
            "feature_columns": list(self.feature_columns),
        }
        joblib.dump(payload, path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "model_type": self.model_type,
                    "feature_columns": list(self.feature_columns),
                    "target": TARGET_COLUMN,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path | str) -> LengthRegressionModel:
        path = Path(path)
        payload = joblib.load(path)
        obj = cls(
            model_type=payload.get("model_type", "random_forest"),
            **payload.get("estimator_kwargs", {}),
        )
        obj._model = payload["model"]
        obj.feature_columns = tuple(payload.get("feature_columns", FEATURE_COLUMNS))
        return obj

    def _to_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            return X[list(self.feature_columns)].to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)
