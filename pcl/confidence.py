"""Confidence computation for PCL datapoints.

Confidence = min(1.0, base_weight * log2(max(1, n)) / log2(plateau))

Method quality × evidence quantity = confidence.
"""

import math

CONFIDENCE_WEIGHTS = {
    "automated": {"base": 0.95, "plateau": 10},
    "doe": {"base": 0.90, "plateau": 8},
    "workbench": {"base": 0.85, "plateau": 15},
    "time_study": {"base": 0.70, "plateau": 30},
    "manual": {"base": 0.50, "plateau": 30},
    "estimate": {"base": 0.25, "plateau": 30},
    "calculator": {"base": 0.60, "plateau": 10},
}

DEFAULT_WEIGHT = {"base": 0.50, "plateau": 30}


def compute_confidence(source_type: str, observation_count: int) -> float:
    """Compute confidence from source type and observation count.

    Returns float in [0.0, 1.0].
    """
    w = CONFIDENCE_WEIGHTS.get(source_type, DEFAULT_WEIGHT)
    n = max(1, observation_count)
    plateau = w["plateau"]
    raw = w["base"] * math.log2(max(1, n)) / math.log2(plateau)
    return min(1.0, max(0.0, raw))
