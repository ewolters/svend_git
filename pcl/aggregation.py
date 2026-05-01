"""Bayesian weighted aggregation for PCL measures.

Every datapoint contributes. Weight = confidence x consistency_factor.
Outliers (low confidence + high deviation) get near-zero effective weight.
Shift detection via rolling window comparison.
"""

import math
from typing import Optional


def _consistency_factor(value: float, mean: float, std: float) -> float:
    """Penalize values that deviate from the running estimate.

    Returns 1.0 for values at the mean, decays toward 0 for outliers.
    Uses a Gaussian-shaped penalty: exp(-0.5 * z^2).
    """
    if std <= 0:
        return 1.0
    z = abs(value - mean) / std
    return math.exp(-0.5 * z * z)


def _decay_factor(age_days: float, halflife_days: float) -> float:
    """Exponential decay based on age. Returns 1.0 for age=0, 0.5 at halflife."""
    if halflife_days <= 0 or age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / halflife_days)


def compute_aggregate(
    values: list[float],
    confidences: list[float],
    timestamps_age_days: list[float],
    decay_enabled: bool = False,
    decay_halflife_days: Optional[float] = None,
) -> dict:
    """Compute the Bayesian weighted aggregate of all datapoints.

    Returns dict with: value, confidence, variance, n, effective_n, shift_detected.
    """
    n = len(values)
    if n == 0:
        return {
            "value": None,
            "confidence": 0.0,
            "variance": 0.0,
            "n": 0,
            "effective_n": 0.0,
            "shift_detected": False,
        }

    # First pass: raw weighted mean (confidence + optional decay, no consistency yet)
    raw_weights = []
    for i in range(n):
        w = confidences[i]
        if decay_enabled and decay_halflife_days:
            w *= _decay_factor(timestamps_age_days[i], decay_halflife_days)
        raw_weights.append(w)

    total_raw = sum(raw_weights)
    if total_raw <= 0:
        return {
            "value": values[-1] if values else None,
            "confidence": 0.0,
            "variance": 0.0,
            "n": n,
            "effective_n": 0.0,
            "shift_detected": False,
        }

    raw_mean = sum(v * w for v, w in zip(values, raw_weights)) / total_raw

    # Weighted standard deviation from raw mean
    if n >= 2:
        var_sum = sum(w * (v - raw_mean) ** 2 for v, w in zip(values, raw_weights))
        raw_std = math.sqrt(var_sum / total_raw) if total_raw > 0 else 0.0
    else:
        raw_std = 0.0

    # Second pass: apply consistency factor
    effective_weights = []
    for i in range(n):
        cf = _consistency_factor(values[i], raw_mean, raw_std) if raw_std > 0 else 1.0
        effective_weights.append(raw_weights[i] * cf)

    total_ew = sum(effective_weights)
    if total_ew <= 0:
        return {
            "value": raw_mean,
            "confidence": 0.0,
            "variance": raw_std**2,
            "n": n,
            "effective_n": 0.0,
            "shift_detected": False,
        }

    agg_value = sum(v * w for v, w in zip(values, effective_weights)) / total_ew

    # Weighted variance around aggregate
    if n >= 2:
        agg_var = sum(w * (v - agg_value) ** 2 for v, w in zip(values, effective_weights)) / total_ew
    else:
        agg_var = 0.0

    effective_n = total_ew
    agg_confidence = min(1.0, effective_n / n) if n > 0 else 0.0

    # Shift detection: compare recent window (last 30%) vs older window
    shift_detected = False
    if n >= 6:
        split = max(3, n * 7 // 10)  # 70/30 split
        old_vals = values[:split]
        new_vals = values[split:]
        if old_vals and new_vals:
            old_mean = sum(old_vals) / len(old_vals)
            new_mean = sum(new_vals) / len(new_vals)
            if raw_std > 0:
                shift_z = abs(new_mean - old_mean) / raw_std
                shift_detected = shift_z > 1.5

    return {
        "value": agg_value,
        "confidence": agg_confidence,
        "variance": agg_var,
        "n": n,
        "effective_n": effective_n,
        "shift_detected": shift_detected,
    }


def update_aggregate_incremental(
    cached_value: Optional[float],
    cached_variance: Optional[float],
    cached_confidence: Optional[float],
    cached_n: int,
    cached_effective_n: float,
    new_value: float,
    new_confidence: float,
    decay_enabled: bool = False,
    decay_halflife_days: Optional[float] = None,
    new_age_days: float = 0.0,
) -> dict:
    """Incrementally update the cached aggregate with a new datapoint.

    Fast path for pcl.write() — avoids recomputing from scratch.
    """
    w = new_confidence
    if decay_enabled and decay_halflife_days:
        w *= _decay_factor(new_age_days, decay_halflife_days)

    if cached_value is None or cached_n == 0:
        return {
            "value": new_value,
            "variance": 0.0,
            "confidence": new_confidence,
            "n": 1,
            "effective_n": w,
        }

    # Consistency factor against existing aggregate
    std = math.sqrt(cached_variance) if cached_variance and cached_variance > 0 else 0.0
    cf = _consistency_factor(new_value, cached_value, std) if std > 0 else 1.0
    ew = w * cf

    total_ew = cached_effective_n + ew
    if total_ew <= 0:
        return {
            "value": cached_value,
            "variance": cached_variance or 0.0,
            "confidence": cached_confidence or 0.0,
            "n": cached_n + 1,
            "effective_n": cached_effective_n,
        }

    # Online weighted mean update
    new_agg = (cached_value * cached_effective_n + new_value * ew) / total_ew

    # Online variance update (Welford-like)
    new_var = (
        cached_effective_n * ((cached_variance or 0.0) + (cached_value - new_agg) ** 2)
        + ew * (new_value - new_agg) ** 2
    ) / total_ew

    new_n = cached_n + 1

    return {
        "value": new_agg,
        "variance": new_var,
        "confidence": min(1.0, total_ew / new_n) if new_n > 0 else 0.0,
        "n": new_n,
        "effective_n": total_ew,
    }
