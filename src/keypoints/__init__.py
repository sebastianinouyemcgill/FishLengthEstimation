"""
HRNet keypoint pipeline (scaffold only — no model training yet).

Pseudo-labels are derived from the baseline skeleton; training and inference hooks
will be added in a later phase.
"""

from src.keypoints.dataset_builder import KeypointDatasetManifest, build_keypoint_manifest
from src.keypoints.evaluation import KeypointEvalSummary, evaluate_keypoint_predictions
from src.keypoints.pseudo_labels import PseudoKeypointRecord, extract_pseudo_keypoints

__all__ = [
    "KeypointDatasetManifest",
    "KeypointEvalSummary",
    "PseudoKeypointRecord",
    "build_keypoint_manifest",
    "evaluate_keypoint_predictions",
    "extract_pseudo_keypoints",
]
