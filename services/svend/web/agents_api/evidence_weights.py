"""
Evidence weighting methodology — CANON-002 §2-3.

Implements the epistemological hierarchy, tool→source rank mapping,
tool→function mapping, and composite evidence weight computation.

All functions are pure (no side effects) except _compute_measurement_validity(),
which performs a lazy DB lookup for MeasurementSystem.current_validity.

Reference: docs/standards/CANON-002.md §2.1-2.3, §3.1-3.3, §4, §11.1
"""

from __future__ import annotations

import math
from enum import Enum


class SourceRank(float, Enum):
    """Epistemological hierarchy — CANON-002 §2.1."""

    DESIGNED_EXPERIMENT = 0.95
    CONTROLLED_OBSERVATION = 0.85
    STATISTICAL_TEST = 0.75
    STRUCTURED_ANALYSIS = 0.60
    SIMULATION = 0.50
    OBSERVATIONAL_STUDY = 0.45
    EXPERT_JUDGMENT = 0.35
    ANECDOTAL = 0.20


# Tool → source rank mapping. Used by compute_evidence_weight().
# CANON-002 §2.3
TOOL_SOURCE_RANKS: dict[str, SourceRank | None] = {
    # Layer 1
    "spc": SourceRank.CONTROLLED_OBSERVATION,
    "dsw": SourceRank.STATISTICAL_TEST,
    "doe_design": None,  # Design phase produces no evidence
    "doe_results": SourceRank.DESIGNED_EXPERIMENT,
    "ml": SourceRank.SIMULATION,
    "forecast": SourceRank.SIMULATION,
    "triage": None,  # Triage produces no evidence
    # Layer 2
    "rca": SourceRank.STRUCTURED_ANALYSIS,
    "ishikawa": SourceRank.STRUCTURED_ANALYSIS,
    "ce_matrix": SourceRank.STRUCTURED_ANALYSIS,
    "fmea": SourceRank.STRUCTURED_ANALYSIS,
    "a3": None,  # Report sink
    "vsm": None,  # Feeds Layer 3 directly
    "report": None,  # Report sink (8D)
    # Layer 3 (when generating own evidence)
    "ncr": SourceRank.STRUCTURED_ANALYSIS,
    "capa": SourceRank.STRUCTURED_ANALYSIS,
    # User-supplied
    "user": SourceRank.EXPERT_JUDGMENT,
    "observation": SourceRank.ANECDOTAL,
}


# Tool → function mapping. Used by InvestigationToolLink.
# CANON-002 §11.1, CANON-001 §1.3 (three tool functions)
TOOL_FUNCTIONS: dict[str, str | None] = {
    "spc": "inference",
    "dsw": "inference",
    "doe_design": "intent",
    "doe_results": "inference",
    "ml": "inference",
    "forecast": "inference",
    "triage": None,  # Cannot be linked — no graph interaction
    "rca": "information",
    "ishikawa": "information",
    "ce_matrix": "information",
    "fmea": "information",
    "a3": "report",
    "vsm": None,  # Feeds Layer 3 directly, not investigations
    "report": "report",
}


# Applicable study quality factors per source type — CANON-002 §3.3
_APPLICABLE_QUALITY_FACTORS: dict[str, list[str]] = {
    "doe_results": [
        "randomization",
        "replication",
        "blocking",
        "blinding",
        "pre_registration",
    ],
    "dsw": ["blinding", "pre_registration"],
    "forecast": ["replication", "pre_registration"],
    "ml": [],  # No applicable quality factors
    "spc": [],
}


def compute_evidence_weight(
    source_tool: str,
    sample_size: int | None = None,
    measurement_system_id: str | None = None,
    study_quality_factors: dict | None = None,
) -> float:
    """
    Compute evidence weight per CANON-002 §3.1.

    evidence_weight = source_rank × sample_modifier × measurement_validity × study_quality

    Returns 0.0 for tools that produce no evidence (rank is None).
    Otherwise returns clamped float in [0.05, 0.99].
    """
    rank = TOOL_SOURCE_RANKS.get(source_tool)
    if rank is None:
        return 0.0  # Tool produces no evidence

    source_rank = float(rank)
    sample_modifier = _compute_sample_modifier(sample_size)
    measurement_validity = _compute_measurement_validity(measurement_system_id)
    study_quality = _compute_study_quality(source_tool, study_quality_factors)

    weight = source_rank * sample_modifier * measurement_validity * study_quality
    return max(0.05, min(0.99, weight))


def _compute_sample_modifier(n: int | None) -> float:
    """
    CANON-002 §3.2 — sample size modifier.

    n < 5:    0.50  (insufficient for meaningful inference)
    5-14:     0.70  (marginal — wide CIs)
    15-29:    0.85  (adequate for most parametric tests)
    30-99:    0.95  (strong — CLT reliable)
    n >= 100: 1.00  (large sample — no discount)
    None:     1.00  (non-sample tools: information, reports)
    """
    if n is None:
        return 1.0
    if n < 5:
        return 0.50
    if n < 15:
        return 0.70
    if n < 30:
        return 0.85
    if n < 100:
        return 0.95
    return 1.0


def _compute_measurement_validity(measurement_system_id: str | None) -> float:
    """
    CANON-002 §4 — measurement system validity gate.

    Lazy DB lookup: imports MeasurementSystem only when a measurement_system_id
    is provided. Returns current_validity from the most recent GageStudy,
    or 0.55 (unvalidated default per §4.3) if no study exists or ID not found.
    """
    if measurement_system_id is None:
        return 0.55  # No MSA linked — assumed unvalidated (§4.3)

    try:
        from core.models import MeasurementSystem

        ms = MeasurementSystem.objects.get(id=measurement_system_id)
        return ms.current_validity
    except (ImportError, Exception):
        # ImportError: MeasurementSystem model not yet created (FEAT-100)
        # DoesNotExist: measurement system ID not found
        return 0.55


def _compute_study_quality(source_tool: str, factors: dict | None) -> float:
    """
    CANON-002 §3.3 — study quality modifier (geometric mean of applicable factors).

    Each factor is expected to be 1.0 (full credit), 0.7 (partial), or 0.5 (penalty).
    Only factors applicable to the source type are considered.
    Returns 1.0 if no applicable factors or no factors provided.
    """
    if factors is None:
        return 1.0

    applicable_keys = _APPLICABLE_QUALITY_FACTORS.get(source_tool, [])
    if not applicable_keys:
        return 1.0

    values = [factors[key] for key in applicable_keys if key in factors]
    if not values:
        return 1.0

    product = math.prod(values)
    return product ** (1.0 / len(values))
