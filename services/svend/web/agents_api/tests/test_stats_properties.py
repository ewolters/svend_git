"""Property-based tests — verify statistical invariants hold for all inputs.

Uses the hypothesis library to generate random valid inputs and check
that universal properties hold: p in [0,1], CI_lower < CI_upper, etc.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
"""

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase


def _run_stats(analysis_id, config, data_dict):
    """Run stats analysis, catching exceptions as the HTTP dispatch layer would."""
    from agents_api.dsw.stats import run_statistical_analysis

    try:
        df = pd.DataFrame(data_dict)
        return run_statistical_analysis(df, analysis_id, config)
    except Exception as e:
        return {"error": str(e)}


def _run_spc(analysis_id, config, data_dict):
    from agents_api.dsw.spc import run_spc_analysis

    try:
        df = pd.DataFrame(data_dict)
        return run_spc_analysis(df, analysis_id, config)
    except Exception as e:
        return {"error": str(e)}


# --- Strategies ---

# Finite floats suitable for statistical data (no nan/inf, reasonable range)
stat_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

# Lists of floats for one-sample data
sample_data = st.lists(stat_float, min_size=10, max_size=200)

# Two independent samples
two_samples = st.tuples(
    st.lists(stat_float, min_size=10, max_size=100),
    st.lists(stat_float, min_size=10, max_size=100),
)

PROP_SETTINGS = settings(
    max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.too_slow]
)


# ---------------------------------------------------------------------------
# Property: p-values always in [0, 1]
# ---------------------------------------------------------------------------
class PValueBoundsTest(TestCase):
    """p-values must be in [0, 1] for any valid input."""

    @given(data=sample_data)
    @PROP_SETTINGS
    def test_ttest_p_value_bounded(self, data):
        assume(len(set(data)) > 1)  # need nonzero variance
        result = _run_stats("ttest", {"var1": "x", "mu": 0}, {"x": data})
        p = result.get("statistics", {}).get("p_value")
        if p is not None:
            self.assertGreaterEqual(p, 0.0, f"p-value {p} < 0")
            self.assertLessEqual(p, 1.0, f"p-value {p} > 1")

    @given(samples=two_samples)
    @PROP_SETTINGS
    def test_ttest2_p_value_bounded(self, samples):
        a, b = samples
        assume(len(set(a)) > 1 and len(set(b)) > 1)  # need nonzero variance
        result = _run_stats("ttest2", {"var1": "a", "var2": "b"}, {"a": a, "b": b})
        p = result.get("statistics", {}).get("p_value")
        if p is not None:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    @given(data=sample_data)
    @PROP_SETTINGS
    def test_normality_p_value_bounded(self, data):
        result = _run_stats("normality", {"var": "x"}, {"x": data})
        p = result.get("statistics", {}).get("p_value")
        if p is not None:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


# ---------------------------------------------------------------------------
# Property: correlation r in [-1, 1]
# ---------------------------------------------------------------------------
class CorrelationBoundsTest(TestCase):
    """Pearson r must be in [-1, 1]."""

    @given(
        x=st.lists(stat_float, min_size=10, max_size=100),
        y=st.lists(stat_float, min_size=10, max_size=100),
    )
    @PROP_SETTINGS
    def test_correlation_r_bounded(self, x, y):
        # Make same length
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        assume(n >= 10)
        # Check variance is nonzero
        assume(len(set(x)) > 1 and len(set(y)) > 1)

        result = _run_stats("correlation", {"variables": ["x", "y"]}, {"x": x, "y": y})
        r = result.get("statistics", {}).get("r(x,y)")
        if r is not None:
            self.assertGreaterEqual(r, -1.0 - 1e-10, f"r={r} < -1")
            self.assertLessEqual(r, 1.0 + 1e-10, f"r={r} > 1")


# ---------------------------------------------------------------------------
# Property: R² in [0, 1]
# ---------------------------------------------------------------------------
class RSquaredBoundsTest(TestCase):
    """R-squared must be in [0, 1] for standard regression."""

    @given(
        x=st.lists(stat_float, min_size=20, max_size=100),
        y=st.lists(stat_float, min_size=20, max_size=100),
    )
    @PROP_SETTINGS
    def test_regression_r2_bounded(self, x, y):
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        assume(n >= 20)
        assume(len(set(x)) > 5)  # need real variance in predictor
        assume(len(set(y)) > 5)

        result = _run_stats(
            "regression", {"response": "y", "predictors": ["x"]}, {"x": x, "y": y}
        )
        r2 = (result.get("regression_metrics") or {}).get("r_squared")
        if r2 is not None:
            self.assertGreaterEqual(r2, -0.01, f"R²={r2} < 0")
            self.assertLessEqual(r2, 1.01, f"R²={r2} > 1")


# ---------------------------------------------------------------------------
# Property: confidence intervals are ordered
# ---------------------------------------------------------------------------
class CIOrderTest(TestCase):
    """CI lower bound must be <= upper bound."""

    @given(data=sample_data)
    @PROP_SETTINGS
    def test_ttest_ci_ordered(self, data):
        assume(len(set(data)) > 1)  # need nonzero variance
        result = _run_stats("ttest", {"var1": "x", "mu": 0, "conf": 95}, {"x": data})
        stats = result.get("statistics", {})
        lo = stats.get("ci_lower")
        hi = stats.get("ci_upper")
        if lo is not None and hi is not None:
            self.assertLessEqual(lo, hi, f"CI: {lo} > {hi}")


# ---------------------------------------------------------------------------
# Property: effect sizes are finite
# ---------------------------------------------------------------------------
class EffectSizeFiniteTest(TestCase):
    """Cohen's d and other effect sizes must be finite numbers."""

    @given(samples=two_samples)
    @PROP_SETTINGS
    def test_ttest2_cohens_d_finite(self, samples):
        a, b = samples
        # Need non-constant samples for Cohen's d (pooled std > 0)
        assume(len(set(a)) > 2 and len(set(b)) > 2)

        result = _run_stats("ttest2", {"var1": "a", "var2": "b"}, {"a": a, "b": b})
        d = result.get("statistics", {}).get("cohens_d")
        if d is not None:
            self.assertTrue(np.isfinite(d), f"Cohen's d is not finite: {d}")


# ---------------------------------------------------------------------------
# Property: symmetric tests give consistent results
# ---------------------------------------------------------------------------
class SymmetryTest(TestCase):
    """Swapping group labels should give the same p-value."""

    @given(samples=two_samples)
    @PROP_SETTINGS
    def test_ttest2_symmetric_p(self, samples):
        a, b = samples
        assume(len(set(a)) > 2 and len(set(b)) > 2)

        r1 = _run_stats("ttest2", {"var1": "a", "var2": "b"}, {"a": a, "b": b})
        r2 = _run_stats("ttest2", {"var1": "a", "var2": "b"}, {"a": b, "b": a})

        p1 = r1.get("statistics", {}).get("p_value")
        p2 = r2.get("statistics", {}).get("p_value")
        if p1 is not None and p2 is not None:
            self.assertAlmostEqual(
                p1, p2, delta=1e-10, msg="p-value not symmetric on group swap"
            )


# ---------------------------------------------------------------------------
# Property: SPC control limits bracket center line
# ---------------------------------------------------------------------------
class SPCControlLimitTest(TestCase):
    """UCL > CL > LCL for any valid process data."""

    @given(data=st.lists(stat_float, min_size=25, max_size=200))
    @PROP_SETTINGS
    def test_imr_limits_ordered(self, data):
        assume(len(set(data)) > 1)  # need some variation

        result = _run_spc("imr", {"measurement": "x"}, {"x": data})
        charts = result.get("charts", [])
        for chart in charts:
            params = chart.get("params", {})
            ucl = params.get("ucl")
            cl = params.get("cl")
            lcl = params.get("lcl")
            if ucl is not None and cl is not None and lcl is not None:
                self.assertGreaterEqual(ucl, cl, f"UCL {ucl} < CL {cl}")
                self.assertGreaterEqual(cl, lcl, f"CL {cl} < LCL {lcl}")


# ---------------------------------------------------------------------------
# Property: descriptive stats are consistent
# ---------------------------------------------------------------------------
class DescriptiveConsistencyTest(TestCase):
    """Basic sanity: min <= mean <= max, std >= 0."""

    @given(data=st.lists(stat_float, min_size=5, max_size=200))
    @PROP_SETTINGS
    def test_descriptive_min_mean_max(self, data):
        result = _run_stats("descriptive", {"var1": "x"}, {"x": data})
        stats = result.get("statistics", {})
        mn = stats.get("min")
        mx = stats.get("max")
        mean = stats.get("mean")
        std = stats.get("std")
        if mn is not None and mx is not None and mean is not None:
            self.assertLessEqual(mn, mean + 1e-10, f"min {mn} > mean {mean}")
            self.assertGreaterEqual(mx, mean - 1e-10, f"max {mx} < mean {mean}")
        if std is not None:
            self.assertGreaterEqual(std, -1e-10, f"std {std} < 0")
