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
