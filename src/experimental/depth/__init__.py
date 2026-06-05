"""
Archived monocular depth estimation (inactive by default).

Enable with ``cfg.use_depth_model`` or ``cfg.use_depth_estimation`` and
``FISHNET_ALLOW_EXPERIMENTAL=1`` for depth/3D/perspective paths.
"""

from src.depth import DepthEstimator, get_depth_estimator
from src.depth.cache import depth_cache_path, load_cached_depth

__all__ = [
    "DepthEstimator",
    "depth_cache_path",
    "get_depth_estimator",
    "load_cached_depth",
]
