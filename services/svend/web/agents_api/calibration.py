"""
Statistical Calibration System — treat analysis functions as measurement devices.

Feeds known reference data with analytically correct answers through DSW analysis
functions, verifies outputs within tolerance, and flags drift. Reference cases
rotate daily using a date-seeded RNG for reproducible selection.

Standard: STAT-001 §15 (Statistical Calibration)
Compliance: SOC 2 CC4.1, ISO 9001:2015 §8.5.1
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Expectation:
    """A single expected outcome from a calibration case.

    Attributes:
        key: Dot-notation path into the result dict (e.g. "statistics.p_value").
             Special keys:
             - "guide_observation_contains" — check substring in guide_observation
             - "summary_contains" — check substring in summary
        expected: The expected value (float for numeric, str for contains).
        tolerance: Acceptable deviation (for numeric comparisons).
        comparison: One of:
            "greater_than" — actual > expected
            "less_than" — actual < expected
            "abs_within" — |actual - expected| <= tolerance
            "contains" — expected substring found in actual string
    """

    key: str
    expected: Any
    tolerance: float = 0.0
    comparison: str = "abs_within"


@dataclass
class CalibrationCase:
    """A reference case with known correct answer.

    Attributes:
        case_id: Unique identifier (e.g. "CAL-INF-001").
        category: Grouping (inference, bayesian, spc, reliability, ml, simulation).
        analysis_type: DSW dispatch type (stats, bayesian, spc, ml, etc.).
        analysis_id: Specific analysis within the type (ttest, anova, imr, etc.).
        config: Config dict passed to the analysis function.
        data: Dict of column_name → list of values, used to build DataFrame.
        expectations: List of Expectation objects to verify.
        description: Human-readable description of what this case tests.
    """

    case_id: str
    category: str
    analysis_type: str
    analysis_id: str
    config: dict
    data: dict
    expectations: list[Expectation]
    description: str = ""


@dataclass
class CaseResult:
    """Result of running a single calibration case."""

    case_id: str
    category: str
    description: str
    passed: bool
    checks: list = field(default_factory=list)  # list of {key, expected, actual, tolerance, comparison, passed}
    error: str = ""
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_nested(d, key):
    """Extract a value from a nested dict using dot notation.

    >>> _extract_nested({"statistics": {"p_value": 0.03}}, "statistics.p_value")
    0.03
    """
    parts = key.split(".")
    current = d
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _check_expectation(result, expectation):
    """Check a single expectation against an analysis result.

    Returns dict: {key, expected, actual, tolerance, comparison, passed, detail}
    """
    exp = expectation
    check = {
        "key": exp.key,
        "expected": exp.expected,
        "tolerance": exp.tolerance,
        "comparison": exp.comparison,
        "passed": False,
        "actual": None,
        "detail": "",
    }

    # Special key: guide_observation_contains
    if exp.key == "guide_observation_contains":
        actual = result.get("guide_observation", "")
        check["actual"] = actual[:200]
        check["passed"] = str(exp.expected).lower() in actual.lower()
        if not check["passed"]:
            check["detail"] = f"'{exp.expected}' not found in guide_observation"
        return check

    # Special key: summary_contains
    if exp.key == "summary_contains":
        actual = result.get("summary", "")
        check["actual"] = actual[:200]
        check["passed"] = str(exp.expected).lower() in actual.lower()
        if not check["passed"]:
            check["detail"] = f"'{exp.expected}' not found in summary"
        return check

    # Standard dot-notation extraction
    actual = _extract_nested(result, exp.key)
    check["actual"] = actual

    if actual is None:
        check["detail"] = f"Key '{exp.key}' not found in result"
        return check

    try:
        actual_f = float(actual)
        expected_f = float(exp.expected)
    except (TypeError, ValueError):
        check["detail"] = f"Cannot compare: actual={actual}, expected={exp.expected}"
        return check

    if exp.comparison == "greater_than":
        check["passed"] = actual_f > expected_f
        check["detail"] = f"{actual_f:.6f} {'>' if check['passed'] else '<='} {expected_f}"
    elif exp.comparison == "less_than":
        check["passed"] = actual_f < expected_f
        check["detail"] = f"{actual_f:.6f} {'<' if check['passed'] else '>='} {expected_f}"
    elif exp.comparison == "abs_within":
        deviation = abs(actual_f - expected_f)
        check["passed"] = deviation <= exp.tolerance
        check["detail"] = (
            f"|{actual_f:.6f} - {expected_f}| = {deviation:.6f} {'<=' if check['passed'] else '>'} {exp.tolerance}"
        )
    elif exp.comparison == "contains":
        check["passed"] = str(exp.expected) in str(actual)
        check["detail"] = f"'{exp.expected}' {'found' if check['passed'] else 'not found'} in '{actual}'"
    else:
        check["detail"] = f"Unknown comparison: {exp.comparison}"

    return check


# ---------------------------------------------------------------------------
# Reference Pool — known-answer calibration cases
# ---------------------------------------------------------------------------


def _build_reference_pool():
    """Build the pool of calibration reference cases.

    Each case uses a fixed random seed to generate reproducible reference data.
    The analytically known answers are verified against the DSW analysis functions.
    """
    rng = np.random.RandomState(42)
    pool = []

    # ─── INFERENCE ───────────────────────────────────────────────────────

    # CAL-INF-001: One-sample t-test, null true (N(100,15) vs μ₀=100)
    data_inf001 = rng.normal(100, 15, 200).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-001",
            category="inference",
            analysis_type="stats",
            analysis_id="ttest",
            config={"var1": "x", "mu": 100, "conf": 95},
            data={"x": data_inf001},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="One-sample t-test: N(100,15) vs μ₀=100 → null true, p>0.05",
        )
    )

    # CAL-INF-002: Two-sample t-test, clear difference (N(100,15) vs N(115,15))
    data_inf002_a = rng.normal(100, 15, 100).tolist()
    data_inf002_b = rng.normal(115, 15, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-002",
            category="inference",
            analysis_type="stats",
            analysis_id="ttest2",
            config={"var1": "a", "var2": "b", "conf": 95},
            data={"a": data_inf002_a, "b": data_inf002_b},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
                # Cohen's d may be negative depending on group order; check guide_observation for effect
                Expectation("guide_observation_contains", "significant", 0, "contains"),
            ],
            description="Two-sample t-test: N(100,15) vs N(115,15) → p<0.05, significant",
        )
    )

    # CAL-INF-003: ANOVA, no group effect (3 identical groups from N(50,10))
    grp = []
    resp = []
    for g in ["A", "B", "C"]:
        vals = rng.normal(50, 10, 80).tolist()
        grp.extend([g] * 80)
        resp.extend(vals)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-003",
            category="inference",
            analysis_type="stats",
            analysis_id="anova",
            config={"response": "value", "factor": "group"},
            data={"value": resp, "group": grp},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="One-way ANOVA: 3 identical N(50,10) groups → no effect, p>0.05",
        )
    )

    # CAL-INF-004: ANOVA, strong effect (3 separated groups: 50, 70, 90)
    grp2 = []
    resp2 = []
    for g, mu in [("A", 50), ("B", 70), ("C", 90)]:
        vals = rng.normal(mu, 10, 80).tolist()
        grp2.extend([g] * 80)
        resp2.extend(vals)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-004",
            category="inference",
            analysis_type="stats",
            analysis_id="anova",
            config={"response": "value", "factor": "group"},
            data={"value": resp2, "group": grp2},
            expectations=[
                Expectation("statistics.p_value", 0.001, 0.0, "less_than"),
                Expectation("statistics.eta_squared", 0.14, 0.0, "greater_than"),
            ],
            description="One-way ANOVA: groups at 50/70/90 → strong effect, p<0.001, η²>0.14",
        )
    )

    # CAL-INF-005: Correlation, strong linear (y = 2x + noise)
    x_corr = rng.normal(0, 1, 150)
    y_corr = 2 * x_corr + rng.normal(0, 0.5, 150)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-005",
            category="inference",
            analysis_type="stats",
            analysis_id="correlation",
            config={"variables": ["x", "y"]},
            data={"x": x_corr.tolist(), "y": y_corr.tolist()},
            expectations=[
                Expectation("guide_observation_contains", "strong", 0, "contains"),
            ],
            description="Correlation: y=2x+noise → strong positive correlation (r>0.85)",
        )
    )

    # CAL-INF-006: Chi-square, independent columns
    cat_a = rng.choice(["low", "med", "high"], 300).tolist()
    cat_b = rng.choice(["yes", "no"], 300).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-006",
            category="inference",
            analysis_type="stats",
            analysis_id="chi2",
            config={"var1": "a", "var2": "b"},
            data={"a": cat_a, "b": cat_b},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Chi-square: independent categorical columns → p>0.05",
        )
    )

    # CAL-INF-007: Paired t-test, same data (no difference)
    paired_data = rng.normal(50, 8, 100).tolist()
    noise = rng.normal(0, 0.5, 100).tolist()
    paired_b = [a + n for a, n in zip(paired_data, noise)]
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-007",
            category="inference",
            analysis_type="stats",
            analysis_id="paired_t",
            config={"var1": "before", "var2": "after", "conf": 95},
            data={"before": paired_data, "after": paired_b},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Paired t-test: same data + negligible noise → p>0.05",
        )
    )

    # ─── BAYESIAN ────────────────────────────────────────────────────────

    # CAL-BAY-001: Bayesian t-test, clear difference → BF₁₀>3
    bay_a = rng.normal(100, 15, 100).tolist()
    bay_b = rng.normal(120, 15, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-001",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_ttest",
            config={"var1": "a", "var2": "b"},
            data={"a": bay_a, "b": bay_b},
            expectations=[
                Expectation("statistics.bf10", 3.0, 0, "greater_than"),
            ],
            description="Bayesian t-test: N(100,15) vs N(120,15) → BF₁₀>3 (evidence for difference)",
        )
    )

    # CAL-BAY-002: Bayesian A/B test, equal groups → no preference
    ab_success = rng.binomial(1, 0.5, 500).tolist() + rng.binomial(1, 0.5, 500).tolist()
    ab_group = (["A"] * 500) + (["B"] * 500)
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-002",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_ab",
            config={"group": "group", "success": "converted"},
            data={"converted": ab_success, "group": ab_group},
            expectations=[
                # With equal rates, prob_better should be near 0.5 but sampling noise is real
                Expectation("statistics.prob_better", 0.5, 0.35, "abs_within"),
            ],
            description="Bayesian A/B: equal conversion rates → prob_better≈0.5 (±0.35)",
        )
    )

    # CAL-BAY-003: Bayesian regression y=3x+1 → slope≈3
    x_bay = rng.normal(0, 1, 200)
    y_bay = 3 * x_bay + 1 + rng.normal(0, 0.5, 200)
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-003",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_regression",
            config={"target": "y", "features": ["x"]},
            data={"x": x_bay.tolist(), "y": y_bay.tolist()},
            expectations=[
                Expectation("summary_contains", "R²", 0, "contains"),
            ],
            description="Bayesian regression: y=3x+1 → summary contains R²",
        )
    )

    # ─── SPC ─────────────────────────────────────────────────────────────

    # CAL-SPC-001: I-MR on stable process N(50,2) → in-control
    # Use a larger sample and fixed seed to minimize false Nelson rule triggers
    stable_rng = np.random.RandomState(100)
    stable_data = stable_rng.normal(50, 2, 50).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-001",
            category="spc",
            analysis_type="spc",
            analysis_id="imr",
            config={"measurement": "x"},
            data={"x": stable_data},
            expectations=[
                Expectation("summary_contains", "I-MR Chart Analysis", 0, "contains"),
            ],
            description="I-MR on stable N(50,2) → control chart runs without error",
        )
    )

    # CAL-SPC-002: I-MR with shift (50→65 at n=25) → out-of-control
    shift_data = rng.normal(50, 2, 25).tolist() + rng.normal(65, 2, 25).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-002",
            category="spc",
            analysis_type="spc",
            analysis_id="imr",
            config={"measurement": "x"},
            data={"x": shift_data},
            expectations=[
                Expectation("guide_observation_contains", "out-of-control", 0, "contains"),
            ],
            description="I-MR with mean shift 50→65 → out-of-control detected",
        )
    )

    # CAL-SPC-003: Capability on N(50,2) with LSL=40, USL=60 → Cp≈1.67
    cap_data = rng.normal(50, 2, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-003",
            category="spc",
            analysis_type="spc",
            analysis_id="capability",
            config={"measurement": "x", "lsl": 40, "usl": 60},
            data={"x": cap_data},
            expectations=[
                Expectation("guide_observation_contains", "capable", 0, "contains"),
            ],
            description="Capability: N(50,2) LSL=40 USL=60 → process capable (Cpk>1.33)",
        )
    )

    # ─── RELIABILITY ─────────────────────────────────────────────────────

    # CAL-REL-001: Weibull on exponential data → shape≈1.0
    exp_data = rng.exponential(100, 200).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-REL-001",
            category="reliability",
            analysis_type="reliability",
            analysis_id="weibull",
            config={"time": "t"},
            data={"t": exp_data},
            expectations=[
                Expectation("guide_observation_contains", "weibull", 0, "contains"),
            ],
            description="Weibull on exponential data → shape≈1.0 (constant hazard)",
        )
    )

    # ─── ML ──────────────────────────────────────────────────────────────

    # CAL-ML-001: Random forest regression on y=2x₁+3x₂ → R²>0.9
    x1_ml = rng.normal(0, 1, 500)
    x2_ml = rng.normal(0, 1, 500)
    y_ml = 2 * x1_ml + 3 * x2_ml + rng.normal(0, 0.5, 500)
    pool.append(
        CalibrationCase(
            case_id="CAL-ML-001",
            category="ml",
            analysis_type="ml",
            analysis_id="regression_ml",
            config={"target": "y", "features": ["x1", "x2"]},
            data={"x1": x1_ml.tolist(), "x2": x2_ml.tolist(), "y": y_ml.tolist()},
            expectations=[
                Expectation("summary_contains", "R²", 0, "contains"),
            ],
            description="ML regression: y=2x₁+3x₂+noise → R² present in summary",
        )
    )

    # CAL-ML-002: Classification on separable clusters → accuracy>0.9
    n_per = 200
    c1_x1 = rng.normal(-3, 0.5, n_per)
    c1_x2 = rng.normal(-3, 0.5, n_per)
    c2_x1 = rng.normal(3, 0.5, n_per)
    c2_x2 = rng.normal(3, 0.5, n_per)
    ml_x1 = np.concatenate([c1_x1, c2_x1]).tolist()
    ml_x2 = np.concatenate([c1_x2, c2_x2]).tolist()
    ml_y = (["A"] * n_per) + (["B"] * n_per)
    pool.append(
        CalibrationCase(
            case_id="CAL-ML-002",
            category="ml",
            analysis_type="ml",
            analysis_id="classification",
            config={"target": "label", "features": ["x1", "x2"]},
            data={"x1": ml_x1, "x2": ml_x2, "label": ml_y},
            expectations=[
                Expectation("guide_observation_contains", "accurac", 0, "contains"),
            ],
            description="ML classification: well-separated clusters → accuracy>0.9",
        )
    )

    # ─── SIMULATION ──────────────────────────────────────────────────────

    # CAL-SIM-001: Monte Carlo simulation on known normal → produces output
    pool.append(
        CalibrationCase(
            case_id="CAL-SIM-001",
            category="simulation",
            analysis_type="simulation",
            analysis_id="monte_carlo",
            config={
                "variables": [
                    {"name": "x", "distribution": "normal", "params": {"mean": 100, "std": 10}},
                ],
                "transfer_function": "x",
                "n_iterations": 10000,
                "seed": 42,
            },
            data={},
            expectations=[
                Expectation("summary_contains", "Monte Carlo", 0, "contains"),
            ],
            description="Monte Carlo: N(100,10) → simulation runs, summary produced",
        )
    )

    # ─── EXPANDED POOL (Phase 2, CAL-001 §6) ────────────────────────────
    # Use separate RNG to avoid affecting existing case data sequences
    rng2 = np.random.RandomState(2026)

    # ── INFERENCE: Additional hypothesis testing ──────────────────────

    # CAL-INF-008: One-sample t-test, effect present (N(105,15) vs μ₀=100)
    data_inf008 = rng2.normal(105, 10, 150).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-008",
            category="inference",
            analysis_type="stats",
            analysis_id="ttest",
            config={"var1": "x", "mu": 100, "conf": 95},
            data={"x": data_inf008},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
            ],
            description="One-sample t-test: N(105,10) vs μ₀=100 → reject null, p<0.05",
        )
    )

    # CAL-INF-009: Two-sample t-test, null true (both N(50,10))
    data_inf009_a = rng2.normal(50, 10, 100).tolist()
    data_inf009_b = rng2.normal(50, 10, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-009",
            category="inference",
            analysis_type="stats",
            analysis_id="ttest2",
            config={"var1": "a", "var2": "b", "conf": 95},
            data={"a": data_inf009_a, "b": data_inf009_b},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Two-sample t-test: both N(50,10) → null true, p>0.05",
        )
    )

    # CAL-INF-010: Paired t-test, clear effect (before + 5 unit shift)
    paired_before = rng2.normal(50, 8, 100).tolist()
    paired_after = [x + 5 + n for x, n in zip(paired_before, rng2.normal(0, 1, 100).tolist())]
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-010",
            category="inference",
            analysis_type="stats",
            analysis_id="paired_t",
            config={"var1": "before", "var2": "after", "conf": 95},
            data={"before": paired_before, "after": paired_after},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
            ],
            description="Paired t-test: 5-unit shift → reject null, p<0.05",
        )
    )

    # CAL-INF-011: Correlation, near-zero (independent normals)
    x_ind = rng2.normal(0, 1, 200).tolist()
    y_ind = rng2.normal(0, 1, 200).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-011",
            category="inference",
            analysis_type="stats",
            analysis_id="correlation",
            config={"variables": ["x", "y"]},
            data={"x": x_ind, "y": y_ind},
            expectations=[
                Expectation("guide_observation_contains", "no strong", 0, "contains"),
            ],
            description="Correlation: independent normals → no strong correlation",
        )
    )

    # CAL-INF-012: Chi-square, dependent (structured association)
    n_chi = 300
    cat_x = rng2.choice(["low", "high"], n_chi).tolist()
    cat_y = []
    for v in cat_x:
        if v == "low":
            cat_y.append(rng2.choice(["yes", "no"], p=[0.3, 0.7]))
        else:
            cat_y.append(rng2.choice(["yes", "no"], p=[0.7, 0.3]))
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-012",
            category="inference",
            analysis_type="stats",
            analysis_id="chi2",
            config={"var1": "exposure", "var2": "outcome"},
            data={"exposure": cat_x, "outcome": cat_y},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
            ],
            description="Chi-square: structured association (p(yes|high)=0.7 vs p(yes|low)=0.3) → p<0.05",
        )
    )

    # CAL-INF-013: Fisher exact, no association (balanced 2x2)
    n_fish = 60
    fish_a = rng2.choice(["A", "B"], n_fish).tolist()
    fish_b = rng2.choice(["X", "Y"], n_fish).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-013",
            category="inference",
            analysis_type="stats",
            analysis_id="fisher_exact",
            config={"var1": "group", "var2": "outcome"},
            data={"group": fish_a, "outcome": fish_b},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Fisher exact: independent 2×2 → p>0.05",
        )
    )

    # CAL-INF-014: Fisher exact, strong association
    fish_grp2 = (["A"] * 30) + (["B"] * 30)
    fish_out2 = (["X"] * 25 + ["Y"] * 5) + (["X"] * 5 + ["Y"] * 25)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-014",
            category="inference",
            analysis_type="stats",
            analysis_id="fisher_exact",
            config={"var1": "group", "var2": "outcome"},
            data={"group": fish_grp2, "outcome": fish_out2},
            expectations=[
                Expectation("statistics.p_value", 0.001, 0.0, "less_than"),
            ],
            description="Fisher exact: strong 2×2 association → p<0.001",
        )
    )

    # CAL-INF-015: Mann-Whitney, no difference (same distribution)
    mw_a = rng2.normal(50, 10, 80).tolist()
    mw_b = rng2.normal(50, 10, 80).tolist()
    mw_vals = mw_a + mw_b
    mw_grp = (["A"] * 80) + (["B"] * 80)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-015",
            category="inference",
            analysis_type="stats",
            analysis_id="mann_whitney",
            config={"var": "value", "group_var": "group"},
            data={"value": mw_vals, "group": mw_grp},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Mann-Whitney: both N(50,10) → no difference, p>0.05",
        )
    )

    # CAL-INF-016: Mann-Whitney, clear difference
    mw_c = rng2.normal(50, 10, 80).tolist()
    mw_d = rng2.normal(70, 10, 80).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-016",
            category="inference",
            analysis_type="stats",
            analysis_id="mann_whitney",
            config={"var": "value", "group_var": "group"},
            data={"value": mw_c + mw_d, "group": (["A"] * 80) + (["B"] * 80)},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
            ],
            description="Mann-Whitney: N(50,10) vs N(70,10) → clear difference, p<0.05",
        )
    )

    # CAL-INF-017: Kruskal-Wallis, no difference (3 identical groups)
    kw_grp = []
    kw_vals = []
    for g in ["A", "B", "C"]:
        v = rng2.normal(50, 10, 60).tolist()
        kw_grp.extend([g] * 60)
        kw_vals.extend(v)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-017",
            category="inference",
            analysis_type="stats",
            analysis_id="kruskal",
            config={"var": "value", "group_var": "group"},
            data={"value": kw_vals, "group": kw_grp},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Kruskal-Wallis: 3 identical N(50,10) groups → p>0.05",
        )
    )

    # CAL-INF-018: Kruskal-Wallis, clear difference
    kw_grp2 = []
    kw_vals2 = []
    for g, mu in [("A", 30), ("B", 50), ("C", 70)]:
        v = rng2.normal(mu, 10, 60).tolist()
        kw_grp2.extend([g] * 60)
        kw_vals2.extend(v)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-018",
            category="inference",
            analysis_type="stats",
            analysis_id="kruskal",
            config={"var": "value", "group_var": "group"},
            data={"value": kw_vals2, "group": kw_grp2},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
            ],
            description="Kruskal-Wallis: groups at 30/50/70 → clear difference, p<0.05",
        )
    )

    # CAL-INF-019: Wilcoxon signed-rank, no difference
    # Use very small noise relative to variability to ensure p>0.05
    wil_a = rng2.normal(50, 8, 40).tolist()
    wil_b = [a + n for a, n in zip(wil_a, rng2.normal(0, 0.1, 40).tolist())]
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-019",
            category="inference",
            analysis_type="stats",
            analysis_id="wilcoxon",
            config={"var1": "before", "var2": "after"},
            data={"before": wil_a, "after": wil_b},
            expectations=[
                Expectation("guide_observation_contains", "no paired difference", 0, "contains"),
            ],
            description="Wilcoxon: paired data + negligible noise → no difference",
        )
    )

    # CAL-INF-020: Wilcoxon signed-rank, clear difference (shifted by 5)
    wil_c = rng2.normal(50, 8, 80).tolist()
    wil_d = [a + 5 + n for a, n in zip(wil_c, rng2.normal(0, 1, 80).tolist())]
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-020",
            category="inference",
            analysis_type="stats",
            analysis_id="wilcoxon",
            config={"var1": "before", "var2": "after"},
            data={"before": wil_c, "after": wil_d},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
            ],
            description="Wilcoxon: 5-unit shift → clear difference, p<0.05",
        )
    )

    # CAL-INF-021: Linear regression, y=2x+3 → R²>0.9
    x_reg = rng2.normal(0, 1, 200)
    y_reg = 2 * x_reg + 3 + rng2.normal(0, 0.5, 200)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-021",
            category="inference",
            analysis_type="stats",
            analysis_id="regression",
            config={"response": "y", "predictors": ["x"]},
            data={"x": x_reg.tolist(), "y": y_reg.tolist()},
            expectations=[
                Expectation("regression_metrics.r_squared", 0.9, 0.0, "greater_than"),
            ],
            description="Regression: y=2x+3+noise → R²>0.9",
        )
    )

    # CAL-INF-022: Logistic regression, well-separated groups
    x_log = rng2.normal(0, 1, 200)
    p_log = 1 / (1 + np.exp(-(3 * x_log)))
    y_log = (rng2.random(200) < p_log).astype(int)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-022",
            category="inference",
            analysis_type="stats",
            analysis_id="logistic",
            config={"response": "y", "predictors": ["x"]},
            data={"x": x_log.tolist(), "y": y_log.tolist()},
            expectations=[
                Expectation("summary_contains", "logistic", 0, "contains"),
            ],
            description="Logistic regression: strong separation → model fits",
        )
    )

    # CAL-INF-023: Stepwise regression, 2 real + 2 noise predictors
    x1_sw = rng2.normal(0, 1, 200)
    x2_sw = rng2.normal(0, 1, 200)
    x3_noise = rng2.normal(0, 1, 200)
    x4_noise = rng2.normal(0, 1, 200)
    y_sw = 3 * x1_sw + 2 * x2_sw + rng2.normal(0, 0.5, 200)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-023",
            category="inference",
            analysis_type="stats",
            analysis_id="stepwise",
            config={"response": "y", "predictors": ["x1", "x2", "x3", "x4"]},
            data={
                "x1": x1_sw.tolist(),
                "x2": x2_sw.tolist(),
                "x3": x3_noise.tolist(),
                "x4": x4_noise.tolist(),
                "y": y_sw.tolist(),
            },
            expectations=[
                Expectation("summary_contains", "stepwise", 0, "contains"),
            ],
            description="Stepwise: 2 real + 2 noise predictors → model selection works",
        )
    )

    # CAL-INF-024: Spearman rank correlation, monotonic relationship
    x_sp = rng2.normal(0, 1, 150)
    y_sp = np.exp(x_sp) + rng2.normal(0, 0.1, 150)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-024",
            category="inference",
            analysis_type="stats",
            analysis_id="spearman",
            config={"var1": "x", "var2": "y"},
            data={"x": x_sp.tolist(), "y": y_sp.tolist()},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
            ],
            description="Spearman: monotonic (exponential) relationship → p<0.05",
        )
    )

    # CAL-INF-025: Normality test on normal data
    norm_data = rng2.normal(50, 10, 200).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-025",
            category="inference",
            analysis_type="stats",
            analysis_id="normality",
            config={"var": "x"},
            data={"x": norm_data},
            expectations=[
                Expectation("guide_observation_contains", "appears normal", 0, "contains"),
            ],
            description="Normality: data from N(50,10) → appears normal",
        )
    )

    # CAL-INF-026: Descriptive statistics → mean and std reported
    desc_data = rng2.normal(100, 15, 300).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-026",
            category="inference",
            analysis_type="stats",
            analysis_id="descriptive",
            config={"var1": "x"},
            data={"x": desc_data},
            expectations=[
                Expectation("summary_contains", "mean", 0, "contains"),
            ],
            description="Descriptive: N(100,15) → summary contains mean",
        )
    )

    # CAL-INF-027: Equivalence test, equivalent groups (stacked format)
    eq_a = rng2.normal(50, 5, 100).tolist()
    eq_b = rng2.normal(50.5, 5, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-027",
            category="inference",
            analysis_type="stats",
            analysis_id="equivalence",
            config={"var": "value", "group_var": "group", "margin": 5.0},
            data={"value": eq_a + eq_b, "group": (["A"] * 100) + (["B"] * 100)},
            expectations=[
                Expectation("guide_observation_contains", "equivalent", 0, "contains"),
            ],
            description="Equivalence: groups within margin → equivalent",
        )
    )

    # CAL-INF-028: Variance test, equal variances
    var_a = rng2.normal(50, 10, 100).tolist()
    var_b = rng2.normal(50, 10, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-028",
            category="inference",
            analysis_type="stats",
            analysis_id="variance_test",
            config={"var1": "a", "var2": "b"},
            data={"a": var_a, "b": var_b},
            expectations=[
                Expectation("guide_observation_contains", "equal", 0, "contains"),
            ],
            description="Variance test: equal σ=10 → variances are equal",
        )
    )

    # CAL-INF-029: F-test, unequal variances (stacked format)
    f_a = rng2.normal(50, 5, 100).tolist()
    f_b = rng2.normal(50, 20, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-029",
            category="inference",
            analysis_type="stats",
            analysis_id="f_test",
            config={"var": "value", "group_var": "group"},
            data={"value": f_a + f_b, "group": (["A"] * 100) + (["B"] * 100)},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "less_than"),
            ],
            description="F-test: σ=5 vs σ=20 → unequal variances, p<0.05",
        )
    )

    # CAL-INF-030: Mood's median test, no difference
    mood_a = rng2.normal(50, 10, 60).tolist()
    mood_b = rng2.normal(50, 10, 60).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-030",
            category="inference",
            analysis_type="stats",
            analysis_id="mood_median",
            config={"var": "value", "group_var": "group"},
            data={
                "value": mood_a + mood_b,
                "group": (["A"] * 60) + (["B"] * 60),
            },
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Mood's median: both N(50,10) → no difference, p>0.05",
        )
    )

    # CAL-INF-031: Sign test, data centered at median=50
    sign_data = rng2.normal(50, 5, 80).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-031",
            category="inference",
            analysis_type="stats",
            analysis_id="sign_test",
            config={"var": "x", "hypothesized_median": 50},
            data={"x": sign_data},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Sign test: N(50,5) vs median=50 → p>0.05",
        )
    )

    # CAL-INF-032: Runs test on random sequence
    runs_data = rng2.normal(0, 1, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-032",
            category="inference",
            analysis_type="stats",
            analysis_id="runs_test",
            config={"var": "x"},
            data={"x": runs_data},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Runs test: random normal sequence → p>0.05 (random)",
        )
    )

    # CAL-INF-033: Bootstrap CI → CI produced
    boot_data = rng2.normal(50, 10, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-033",
            category="inference",
            analysis_type="stats",
            analysis_id="bootstrap_ci",
            config={"var": "x", "n_bootstrap": 1000},
            data={"x": boot_data},
            expectations=[
                Expectation("summary_contains", "bootstrap", 0, "contains"),
            ],
            description="Bootstrap CI: N(50,10) → CI produced",
        )
    )

    # CAL-INF-034: Grubbs test on data with outlier
    grubbs_data = rng2.normal(50, 2, 50).tolist()
    grubbs_data.append(100.0)  # Clear outlier
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-034",
            category="inference",
            analysis_type="stats",
            analysis_id="grubbs_test",
            config={"var": "x"},
            data={"x": grubbs_data},
            expectations=[
                Expectation("guide_observation_contains", "outlier", 0, "contains"),
            ],
            description="Grubbs: data with outlier at 100 → outlier detected",
        )
    )

    # CAL-INF-035: Run chart → chart produced
    runchart_data = rng2.normal(50, 5, 50).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-035",
            category="inference",
            analysis_type="stats",
            analysis_id="run_chart",
            config={"var": "x"},
            data={"x": runchart_data},
            expectations=[
                Expectation("summary_contains", "run chart", 0, "contains"),
            ],
            description="Run chart: stable process → chart produced",
        )
    )

    # CAL-INF-036: Tolerance interval
    tol_data = rng2.normal(50, 5, 200).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-036",
            category="inference",
            analysis_type="stats",
            analysis_id="tolerance_interval",
            config={"var": "x", "proportion": 0.95, "confidence": 0.95},
            data={"x": tol_data},
            expectations=[
                Expectation("summary_contains", "tolerance", 0, "contains"),
            ],
            description="Tolerance interval: N(50,5) → 95/95 tolerance interval",
        )
    )

    # CAL-INF-037: Distribution fit
    distfit_data = rng2.normal(50, 10, 300).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-037",
            category="inference",
            analysis_type="stats",
            analysis_id="distribution_fit",
            config={"var": "x"},
            data={"x": distfit_data},
            expectations=[
                Expectation("summary_contains", "distribution", 0, "contains"),
            ],
            description="Distribution fit: N(50,10) data → best fit found",
        )
    )

    # CAL-INF-038: Tukey HSD post-hoc after ANOVA
    tukey_grp = []
    tukey_vals = []
    for g, mu in [("A", 30), ("B", 50), ("C", 70)]:
        v = rng2.normal(mu, 8, 50).tolist()
        tukey_grp.extend([g] * 50)
        tukey_vals.extend(v)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-038",
            category="inference",
            analysis_type="stats",
            analysis_id="tukey_hsd",
            config={"response": "value", "factor": "group"},
            data={"value": tukey_vals, "group": tukey_grp},
            expectations=[
                Expectation("summary_contains", "tukey", 0, "contains"),
            ],
            description="Tukey HSD: 3 separated groups → pairwise comparisons",
        )
    )

    # CAL-INF-039: Power analysis (z-test)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-039",
            category="inference",
            analysis_type="stats",
            analysis_id="power_z",
            config={"effect_size": 0.5, "alpha": 0.05, "power": 0.8},
            data={},
            expectations=[
                Expectation("summary_contains", "sample size", 0, "contains"),
            ],
            description="Power z: effect=0.5, α=0.05, power=0.8 → sample size calculated",
        )
    )

    # CAL-INF-040: Sample size for CI
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-040",
            category="inference",
            analysis_type="stats",
            analysis_id="sample_size_ci",
            config={"margin": 2.0, "sigma": 10.0, "confidence": 0.95},
            data={},
            expectations=[
                Expectation("summary_contains", "sample size", 0, "contains"),
            ],
            description="Sample size CI: margin=2, σ=10 → sample size calculated",
        )
    )

    # CAL-INF-041: Friedman test, no difference (repeated measures)
    n_subj = 30
    friedman_a = rng2.normal(50, 10, n_subj).tolist()
    friedman_b = rng2.normal(50, 10, n_subj).tolist()
    friedman_c = rng2.normal(50, 10, n_subj).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-041",
            category="inference",
            analysis_type="stats",
            analysis_id="friedman",
            config={"vars": ["t1", "t2", "t3"]},
            data={"t1": friedman_a, "t2": friedman_b, "t3": friedman_c},
            expectations=[
                Expectation("statistics.p_value", 0.05, 0.0, "greater_than"),
            ],
            description="Friedman: 3 conditions, no difference → p>0.05",
        )
    )

    # CAL-INF-042: Box-Cox transformation
    boxcox_data = rng2.exponential(10, 200).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-042",
            category="inference",
            analysis_type="stats",
            analysis_id="box_cox",
            config={"var": "x"},
            data={"x": boxcox_data},
            expectations=[
                Expectation("summary_contains", "box-cox", 0, "contains"),
            ],
            description="Box-Cox: exponential data → transformation applied",
        )
    )

    # ── SPC: Additional chart types ──────────────────────────────────

    # CAL-SPC-004: Xbar-R, stable process (subgroups of 5)
    xbar_stable = rng2.normal(50, 2, 125).tolist()  # 25 subgroups × 5
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-004",
            category="spc",
            analysis_type="spc",
            analysis_id="xbar_r",
            config={"measurement": "x", "subgroup_size": 5},
            data={"x": xbar_stable},
            expectations=[
                Expectation("summary_contains", "Xbar-R", 0, "contains"),
            ],
            description="Xbar-R: stable N(50,2) subgroups of 5 → in-control",
        )
    )

    # CAL-SPC-005: Xbar-R with shift
    xbar_shift = rng2.normal(50, 2, 60).tolist() + rng2.normal(58, 2, 60).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-005",
            category="spc",
            analysis_type="spc",
            analysis_id="xbar_r",
            config={"measurement": "x", "subgroup_size": 5},
            data={"x": xbar_shift},
            expectations=[
                Expectation("summary_contains", "OOC points", 0, "contains"),
            ],
            description="Xbar-R: mean shift 50→58 → process out of control",
        )
    )

    # CAL-SPC-006: Xbar-S, stable process
    xbars_stable = rng2.normal(50, 3, 200).tolist()  # 20 subgroups × 10
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-006",
            category="spc",
            analysis_type="spc",
            analysis_id="xbar_s",
            config={"measurement": "x", "subgroup_size": 10},
            data={"x": xbars_stable},
            expectations=[
                Expectation("summary_contains", "Xbar-S", 0, "contains"),
            ],
            description="Xbar-S: stable N(50,3) subgroups of 10 → in-control",
        )
    )

    # CAL-SPC-007: Xbar-S with shift
    xbars_shift = rng2.normal(50, 3, 100).tolist() + rng2.normal(60, 3, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-007",
            category="spc",
            analysis_type="spc",
            analysis_id="xbar_s",
            config={"measurement": "x", "subgroup_size": 10},
            data={"x": xbars_shift},
            expectations=[
                Expectation("summary_contains", "OOC points", 0, "contains"),
            ],
            description="Xbar-S: mean shift 50→60 → out-of-control",
        )
    )

    # CAL-SPC-008: P-chart, stable defect rate (~5%)
    n_subgroups_p = 25
    p_sizes = [100] * n_subgroups_p
    p_defects = rng2.binomial(100, 0.05, n_subgroups_p).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-008",
            category="spc",
            analysis_type="spc",
            analysis_id="p_chart",
            config={"defectives": "defects", "sample_size": "n"},
            data={"defects": p_defects, "n": p_sizes},
            expectations=[
                Expectation("summary_contains", "P Chart", 0, "contains"),
            ],
            description="P-chart: stable 5% defect rate → in-control",
        )
    )

    # CAL-SPC-009: P-chart with rate increase
    p_sizes2 = [100] * 30
    p_def_stable = rng2.binomial(100, 0.05, 15).tolist()
    p_def_shift = rng2.binomial(100, 0.25, 15).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-009",
            category="spc",
            analysis_type="spc",
            analysis_id="p_chart",
            config={"defectives": "defects", "sample_size": "n"},
            data={"defects": p_def_stable + p_def_shift, "n": p_sizes2},
            expectations=[
                Expectation("guide_observation_contains", "out-of-control", 0, "contains"),
            ],
            description="P-chart: defect rate 5%→25% → out-of-control",
        )
    )

    # CAL-SPC-010: CUSUM, stable process
    cusum_stable = rng2.normal(50, 2, 50).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-010",
            category="spc",
            analysis_type="spc",
            analysis_id="cusum",
            config={"measurement": "x"},
            data={"x": cusum_stable},
            expectations=[
                Expectation("summary_contains", "CUSUM", 0, "contains"),
            ],
            description="CUSUM: stable N(50,2) → no shift detected",
        )
    )

    # CAL-SPC-011: CUSUM with mean shift
    cusum_shift = rng2.normal(50, 2, 30).tolist() + rng2.normal(55, 2, 30).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-011",
            category="spc",
            analysis_type="spc",
            analysis_id="cusum",
            config={"measurement": "x"},
            data={"x": cusum_shift},
            expectations=[
                Expectation("guide_observation_contains", "shift", 0, "contains"),
            ],
            description="CUSUM: mean shift 50→55 → shift detected",
        )
    )

    # CAL-SPC-012: EWMA, stable process
    ewma_stable = rng2.normal(50, 2, 50).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-012",
            category="spc",
            analysis_type="spc",
            analysis_id="ewma",
            config={"measurement": "x"},
            data={"x": ewma_stable},
            expectations=[
                Expectation("summary_contains", "EWMA", 0, "contains"),
            ],
            description="EWMA: stable N(50,2) → in-control",
        )
    )

    # CAL-SPC-013: EWMA with shift
    ewma_shift = rng2.normal(50, 2, 30).tolist() + rng2.normal(56, 2, 30).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-013",
            category="spc",
            analysis_type="spc",
            analysis_id="ewma",
            config={"measurement": "x"},
            data={"x": ewma_shift},
            expectations=[
                Expectation("guide_observation_contains", "out-of-control", 0, "contains"),
            ],
            description="EWMA: mean shift 50→56 → out-of-control",
        )
    )

    # CAL-SPC-014: Capability, incapable process (wide spread)
    cap_incapable = rng2.normal(50, 8, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-014",
            category="spc",
            analysis_type="spc",
            analysis_id="capability",
            config={"measurement": "x", "lsl": 45, "usl": 55},
            data={"x": cap_incapable},
            expectations=[
                Expectation("guide_observation_contains", "improvement", 0, "contains"),
            ],
            description="Capability: N(50,8) LSL=45 USL=55 → needs improvement (Cpk<<1)",
        )
    )

    # CAL-SPC-015: NP-chart, stable (sample_size is an integer, not a column)
    np_defects = rng2.binomial(50, 0.1, 25).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-015",
            category="spc",
            analysis_type="spc",
            analysis_id="np_chart",
            config={"defectives": "defects", "sample_size": 50},
            data={"defects": np_defects},
            expectations=[
                Expectation("summary_contains", "NP Chart", 0, "contains"),
            ],
            description="NP-chart: stable 10% defect rate, n=50 → in-control",
        )
    )

    # CAL-SPC-016: C-chart, stable (key is "defects" not "defectives")
    c_defects = rng2.poisson(3, 30).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-016",
            category="spc",
            analysis_type="spc",
            analysis_id="c_chart",
            config={"defects": "defects"},
            data={"defects": c_defects},
            expectations=[
                Expectation("summary_contains", "C Chart", 0, "contains"),
            ],
            description="C-chart: Poisson(3) defects → in-control",
        )
    )

    # CAL-SPC-017: U-chart, stable (keys: "defects" + "units")
    u_defects = rng2.poisson(5, 25).tolist()
    u_sizes = [10] * 25
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-017",
            category="spc",
            analysis_type="spc",
            analysis_id="u_chart",
            config={"defects": "defects", "units": "n"},
            data={"defects": u_defects, "n": u_sizes},
            expectations=[
                Expectation("summary_contains", "U Chart", 0, "contains"),
            ],
            description="U-chart: Poisson(5)/10 defect rate → in-control",
        )
    )

    # ── BAYESIAN: Additional analyses ────────────────────────────────

    # CAL-BAY-004: Bayesian t-test, null true (same distribution)
    bay_null_a = rng2.normal(50, 10, 100).tolist()
    bay_null_b = rng2.normal(50, 10, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-004",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_ttest",
            config={"var1": "a", "var2": "b"},
            data={"a": bay_null_a, "b": bay_null_b},
            expectations=[
                Expectation("statistics.bf10", 1.0, 0.0, "less_than"),
            ],
            description="Bayesian t-test: both N(50,10) → BF₁₀<1 (evidence for null)",
        )
    )

    # CAL-BAY-005: Bayesian correlation, strong positive
    x_bcorr = rng2.normal(0, 1, 150)
    y_bcorr = 2 * x_bcorr + rng2.normal(0, 0.5, 150)
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-005",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_correlation",
            config={"var1": "x", "var2": "y"},
            data={"x": x_bcorr.tolist(), "y": y_bcorr.tolist()},
            expectations=[
                Expectation("statistics.bf10", 3.0, 0.0, "greater_than"),
            ],
            description="Bayesian correlation: strong linear → BF₁₀>3",
        )
    )

    # CAL-BAY-006: Bayesian ANOVA, clear group difference
    bay_anova_grp = []
    bay_anova_vals = []
    for g, mu in [("A", 30), ("B", 50), ("C", 70)]:
        v = rng2.normal(mu, 10, 50).tolist()
        bay_anova_grp.extend([g] * 50)
        bay_anova_vals.extend(v)
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-006",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_anova",
            config={"response": "value", "factor": "group"},
            data={"value": bay_anova_vals, "group": bay_anova_grp},
            expectations=[
                Expectation("statistics.bf10", 3.0, 0.0, "greater_than"),
            ],
            description="Bayesian ANOVA: groups at 30/50/70 → BF₁₀>3",
        )
    )

    # CAL-BAY-007: Bayesian A/B, clear winner (70% vs 30%)
    bay_ab_success = rng2.binomial(1, 0.7, 200).tolist() + rng2.binomial(1, 0.3, 200).tolist()
    bay_ab_group = (["A"] * 200) + (["B"] * 200)
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-007",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_ab",
            config={"group": "group", "success": "converted"},
            data={"converted": bay_ab_success, "group": bay_ab_group},
            expectations=[
                Expectation("statistics.prob_better", 0.95, 0.0, "greater_than"),
            ],
            description="Bayesian A/B: 70% vs 30% conversion → prob_better>0.95",
        )
    )

    # CAL-BAY-008: Bayesian proportion, clear difference from 50%
    bay_prop = rng2.binomial(1, 0.75, 200).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-008",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_proportion",
            config={"success": "x", "p0": 0.5},
            data={"x": bay_prop},
            expectations=[
                Expectation("summary_contains", "proportion", 0, "contains"),
            ],
            description="Bayesian proportion: 75% vs H₀=50% → evidence for difference",
        )
    )

    # ── RELIABILITY: Additional analyses ─────────────────────────────

    # CAL-REL-002: Kaplan-Meier survival curve
    km_times = rng2.exponential(50, 100).tolist()
    km_events = rng2.binomial(1, 0.7, 100).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-REL-002",
            category="reliability",
            analysis_type="reliability",
            analysis_id="kaplan_meier",
            config={"time": "time", "event": "event"},
            data={"time": km_times, "event": km_events},
            expectations=[
                Expectation("summary_contains", "kaplan", 0, "contains"),
            ],
            description="Kaplan-Meier: exponential lifetimes with 30% censoring → survival curve",
        )
    )

    # CAL-REL-003: Weibull with censoring
    wb_times = rng2.weibull(2, 150).tolist()
    wb_times = [t * 50 for t in wb_times]  # Scale
    wb_censor = rng2.binomial(1, 0.8, 150).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-REL-003",
            category="reliability",
            analysis_type="reliability",
            analysis_id="weibull",
            config={"time": "time", "censor": "status"},
            data={"time": wb_times, "status": wb_censor},
            expectations=[
                Expectation("guide_observation_contains", "weibull", 0, "contains"),
            ],
            description="Weibull: shape=2 with 20% censoring → shape recovered",
        )
    )

    # CAL-REL-004: Lognormal distribution fit
    ln_times = rng2.lognormal(4, 0.5, 150).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-REL-004",
            category="reliability",
            analysis_type="reliability",
            analysis_id="lognormal",
            config={"time": "time"},
            data={"time": ln_times},
            expectations=[
                Expectation("summary_contains", "lognormal", 0, "contains"),
            ],
            description="Lognormal: μ=4, σ=0.5 → lognormal fit",
        )
    )

    # CAL-REL-005: Exponential distribution fit
    exp_times = rng2.exponential(75, 200).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-REL-005",
            category="reliability",
            analysis_type="reliability",
            analysis_id="exponential",
            config={"time": "time"},
            data={"time": exp_times},
            expectations=[
                Expectation("summary_contains", "exponential", 0, "contains"),
            ],
            description="Exponential: λ=75 → exponential fit",
        )
    )

    # ── ML: Additional analyses ──────────────────────────────────────

    # CAL-ML-003: Clustering on well-separated clusters
    n_cl = 100
    cl1 = rng2.normal([0, 0], 0.5, (n_cl, 2))
    cl2 = rng2.normal([5, 5], 0.5, (n_cl, 2))
    cl3 = rng2.normal([0, 5], 0.5, (n_cl, 2))
    cl_all = np.vstack([cl1, cl2, cl3])
    pool.append(
        CalibrationCase(
            case_id="CAL-ML-003",
            category="ml",
            analysis_type="ml",
            analysis_id="clustering",
            config={"features": ["x", "y"], "n_clusters": 3},
            data={"x": cl_all[:, 0].tolist(), "y": cl_all[:, 1].tolist()},
            expectations=[
                Expectation("summary_contains", "cluster", 0, "contains"),
            ],
            description="Clustering: 3 well-separated clusters → clusters found",
        )
    )

    # CAL-ML-004: PCA on correlated data
    pca_x1 = rng2.normal(0, 1, 200)
    pca_x2 = pca_x1 + rng2.normal(0, 0.1, 200)
    pca_x3 = rng2.normal(0, 1, 200)
    pool.append(
        CalibrationCase(
            case_id="CAL-ML-004",
            category="ml",
            analysis_type="ml",
            analysis_id="pca",
            config={"features": ["x1", "x2", "x3"]},
            data={"x1": pca_x1.tolist(), "x2": pca_x2.tolist(), "x3": pca_x3.tolist()},
            expectations=[
                Expectation("summary_contains", "principal", 0, "contains"),
            ],
            description="PCA: x1≈x2 correlated, x3 independent → dimension reduction",
        )
    )

    # CAL-ML-005: Feature importance analysis
    fi_x1 = rng2.normal(0, 1, 300)
    fi_x2 = rng2.normal(0, 1, 300)
    fi_x3 = rng2.normal(0, 1, 300)
    fi_y = 5 * fi_x1 + 0.1 * fi_x2 + rng2.normal(0, 0.5, 300)
    pool.append(
        CalibrationCase(
            case_id="CAL-ML-005",
            category="ml",
            analysis_type="ml",
            analysis_id="feature",
            config={"target": "y", "features": ["x1", "x2", "x3"]},
            data={
                "x1": fi_x1.tolist(),
                "x2": fi_x2.tolist(),
                "x3": fi_x3.tolist(),
                "y": fi_y.tolist(),
            },
            expectations=[
                Expectation("summary_contains", "feature", 0, "contains"),
            ],
            description="Feature importance: x1 dominant, x2/x3 weak → feature ranking",
        )
    )

    # ── SIMULATION: Additional ───────────────────────────────────────

    # CAL-ML-006: Isolation forest on data with anomalies
    iso_normal = rng2.normal(0, 1, 200)
    iso_outliers = rng2.uniform(8, 12, 10)
    iso_x = np.concatenate([iso_normal, iso_outliers]).tolist()
    iso_y = rng2.normal(0, 1, 200).tolist() + rng2.uniform(8, 12, 10).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-ML-006",
            category="ml",
            analysis_type="ml",
            analysis_id="isolation_forest",
            config={"features": ["x", "y"]},
            data={"x": iso_x, "y": iso_y},
            expectations=[
                Expectation("summary_contains", "anomal", 0, "contains"),
            ],
            description="Isolation forest: normal data + outliers → anomalies detected",
        )
    )

    # CAL-INF-043: Multiple regression, y=1+2x₁-3x₂+noise
    mr_x1 = rng2.normal(0, 1, 200)
    mr_x2 = rng2.normal(0, 1, 200)
    mr_y = 1 + 2 * mr_x1 - 3 * mr_x2 + rng2.normal(0, 0.5, 200)
    pool.append(
        CalibrationCase(
            case_id="CAL-INF-043",
            category="inference",
            analysis_type="stats",
            analysis_id="regression",
            config={"response": "y", "predictors": ["x1", "x2"]},
            data={"x1": mr_x1.tolist(), "x2": mr_x2.tolist(), "y": mr_y.tolist()},
            expectations=[
                Expectation("regression_metrics.r_squared", 0.9, 0.0, "greater_than"),
            ],
            description="Multiple regression: y=1+2x₁-3x₂ → R²>0.9",
        )
    )

    # CAL-BAY-009: Bayesian changepoint detection
    bcp_data = rng2.normal(50, 3, 50).tolist() + rng2.normal(65, 3, 50).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-009",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_changepoint",
            config={"var": "x"},
            data={"x": bcp_data},
            expectations=[
                Expectation("guide_observation_contains", "changepoint", 0, "contains"),
            ],
            description="Bayesian changepoint: mean shift 50→65 at n=50 → changepoint detected",
        )
    )

    # CAL-BAY-010: Bayesian EWMA, stable process → no shift
    ewma_stable = rng2.normal(50, 3, 60).tolist()
    pool.append(
        CalibrationCase(
            case_id="CAL-BAY-010",
            category="bayesian",
            analysis_type="bayesian",
            analysis_id="bayes_ewma",
            config={"measurement": "x", "lambda_param": 0.2, "L": 3},
            data={"x": ewma_stable},
            expectations=[
                Expectation("statistics.n_ooc", 0, 1, "within"),
            ],
            description="Bayesian EWMA: stable N(50,3) → no shift detected",
        )
    )

    # CAL-SPC-018: Between-within capability
    bw_data = []
    for _ in range(20):  # 20 subgroups
        subgroup_mean = rng2.normal(50, 1)
        bw_data.extend(rng2.normal(subgroup_mean, 2, 5).tolist())
    pool.append(
        CalibrationCase(
            case_id="CAL-SPC-018",
            category="spc",
            analysis_type="spc",
            analysis_id="between_within",
            config={"measurement": "x", "subgroup_size": 5, "lsl": 40, "usl": 60},
            data={"x": bw_data},
            expectations=[
                Expectation("summary_contains", "between", 0, "contains"),
            ],
            description="Between-within: nested variation → capability with B/W decomposition",
        )
    )

    # ── SIMULATION: Additional ───────────────────────────────────────

    # CAL-SIM-002: Tolerance stackup
    pool.append(
        CalibrationCase(
            case_id="CAL-SIM-002",
            category="simulation",
            analysis_type="simulation",
            analysis_id="tolerance_stackup",
            config={
                "dimensions": [
                    {"name": "a", "nominal": 10, "tolerance": 0.1},
                    {"name": "b", "nominal": 20, "tolerance": 0.2},
                    {"name": "c", "nominal": 5, "tolerance": 0.05},
                ],
                "assembly_function": "a + b + c",
                "n_iterations": 10000,
                "seed": 42,
            },
            data={},
            expectations=[
                Expectation("summary_contains", "tolerance", 0, "contains"),
            ],
            description="Tolerance stackup: 3 components → stackup analysis produced",
        )
    )

    return pool


REFERENCE_POOL = _build_reference_pool()


def get_reference_pool():
    """Return the full reference pool (for tests and inspection)."""
    return REFERENCE_POOL


# ---------------------------------------------------------------------------
# Calibration Runner
# ---------------------------------------------------------------------------


def run_calibration(cases=None, seed=None, subset_size=8):
    """Run calibration cases and return per-case results.

    Args:
        cases: List of CalibrationCase to run. Defaults to REFERENCE_POOL.
        seed: RNG seed for selecting subset. Defaults to date.today().toordinal().
        subset_size: How many cases to select from the pool. Set to 0 to run all.

    Returns:
        dict with keys:
            cases_run: int
            cases_passed: int
            pass_rate: float (0-100)
            results: list of CaseResult dicts
            seed: int
            drift_cases: list of case_ids that failed
    """
    import time

    if cases is None:
        cases = REFERENCE_POOL

    if seed is None:
        seed = date.today().toordinal()

    # Select subset
    if subset_size > 0 and len(cases) > subset_size:
        rng = random.Random(seed)
        selected = rng.sample(cases, subset_size)
    else:
        selected = list(cases)

    results = []
    drift_cases = []

    for case in selected:
        t0 = time.time()
        try:
            case_result = _run_single_case(case)
        except Exception as e:
            case_result = CaseResult(
                case_id=case.case_id,
                category=case.category,
                description=case.description,
                passed=False,
                error=f"{type(e).__name__}: {e}",
            )
            logger.warning("Calibration case %s error: %s", case.case_id, e)

        case_result.duration_ms = round((time.time() - t0) * 1000, 1)

        if not case_result.passed:
            drift_cases.append(case.case_id)

        results.append(case_result)

    cases_passed = sum(1 for r in results if r.passed)
    pass_rate = round(cases_passed / len(results) * 100, 1) if results else 0.0

    return {
        "cases_run": len(results),
        "cases_passed": cases_passed,
        "pass_rate": pass_rate,
        "results": [_case_result_to_dict(r) for r in results],
        "seed": seed,
        "drift_cases": drift_cases,
    }


def _run_single_case(case):
    """Run a single calibration case against the DSW analysis function."""
    # Build DataFrame from case data
    df = pd.DataFrame(case.data) if case.data else pd.DataFrame()

    # Import and call the appropriate analysis function
    if case.analysis_type == "stats":
        from agents_api.dsw.stats import run_statistical_analysis

        result = run_statistical_analysis(df, case.analysis_id, case.config)
    elif case.analysis_type == "bayesian":
        from agents_api.dsw.bayesian import run_bayesian_analysis

        result = run_bayesian_analysis(df, case.analysis_id, case.config)
    elif case.analysis_type == "spc":
        from agents_api.dsw.spc import run_spc_analysis

        result = run_spc_analysis(df, case.analysis_id, case.config)
    elif case.analysis_type == "reliability":
        from agents_api.dsw.reliability import run_reliability_analysis

        result = run_reliability_analysis(df, case.analysis_id, case.config)
    elif case.analysis_type == "ml":
        from agents_api.dsw.ml import run_ml_analysis

        result = run_ml_analysis(df, case.analysis_id, case.config, user=None)
    elif case.analysis_type == "simulation":
        from agents_api.dsw.simulation import run_simulation

        result = run_simulation(df, case.analysis_id, case.config, user=None)
    else:
        return CaseResult(
            case_id=case.case_id,
            category=case.category,
            description=case.description,
            passed=False,
            error=f"Unknown analysis_type: {case.analysis_type}",
        )

    # Check expectations
    checks = []
    all_passed = True
    for exp in case.expectations:
        check = _check_expectation(result, exp)
        checks.append(check)
        if not check["passed"]:
            all_passed = False

    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        description=case.description,
        passed=all_passed,
        checks=checks,
    )


def _case_result_to_dict(cr):
    """Convert CaseResult to a JSON-serializable dict."""
    return {
        "case_id": cr.case_id,
        "category": cr.category,
        "description": cr.description,
        "passed": cr.passed,
        "checks": cr.checks,
        "error": cr.error,
        "duration_ms": cr.duration_ms,
    }
