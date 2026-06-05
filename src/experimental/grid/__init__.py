"""
Archived grid auto-calibration (inactive by default).

Enable with ``cfg.use_grid_calibration`` or ``cfg.use_grid_auto_calibration``.
Implementation remains in ``src.calibration.grid_auto``.
"""

from src.calibration.grid_auto import GridCalibrationResult, estimate_grid_calibration

__all__ = ["GridCalibrationResult", "estimate_grid_calibration"]
