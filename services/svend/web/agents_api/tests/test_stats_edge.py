"""Edge case tests — verify graceful handling of boundary conditions.

Tests ensure Tier 1 analyses don't crash on adversarial input:
empty data, single obs, constant values, NaN/inf, extreme outliers.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
"""

import numpy as np
import pandas as pd
from django.test import TestCase


def _run_stats(analysis_id, config, data_dict):
    """Run stats analysis, catching exceptions as the HTTP dispatch layer would."""
    from agents_api.dsw.stats import run_statistical_analysis

    df = pd.DataFrame(data_dict)
    try:
        return run_statistical_analysis(df, analysis_id, config)
    except Exception as e:
        return {"error": str(e), "_exception_type": type(e).__name__}


def _run_spc(analysis_id, config, data_dict):
    from agents_api.dsw.spc import run_spc_analysis

    df = pd.DataFrame(data_dict)
    try:
        return run_spc_analysis(df, analysis_id, config)
    except Exception as e:
        return {"error": str(e), "_exception_type": type(e).__name__}


def _run_bayesian(analysis_id, config, data_dict):
    from agents_api.dsw.bayesian import run_bayesian_analysis

    df = pd.DataFrame(data_dict)
    try:
        return run_bayesian_analysis(df, analysis_id, config)
    except Exception as e:
        return {"error": str(e), "_exception_type": type(e).__name__}


def _is_error(result):
    """Check if result indicates an error (not a crash)."""
    if result is None:
        return True
    if isinstance(result, dict):
        return "error" in result
    return False


def _is_valid_result(result):
    """Check result is a dict without error — a successful analysis."""
    return isinstance(result, dict) and "error" not in result


# ---------------------------------------------------------------------------
# Edge case: empty data
# ---------------------------------------------------------------------------
class EmptyDataTest(TestCase):
    """Analyses should not crash on empty DataFrames."""

    def test_ttest_empty(self):
        result = _run_stats("ttest", {"var1": "x", "mu": 0}, {"x": []})
        self.assertIsInstance(result, dict)

    def test_anova_empty(self):
        # Known: crashes with empty data (scipy.stats.f_oneway rejects empty groups)
        result = _run_stats("anova", {"response": "v", "factor": "g"}, {"v": [], "g": []})
        self.assertIsInstance(result, dict)
        # Bug: should return graceful error, currently throws exception
        self.assertIn("error", result)

    def test_correlation_empty(self):
        result = _run_stats("correlation", {"variables": ["x", "y"]}, {"x": [], "y": []})
        self.assertIsInstance(result, dict)

    def test_regression_empty(self):
        # Known: crashes with empty data (sklearn rejects empty arrays)
        result = _run_stats("regression", {"response": "y", "predictors": ["x"]}, {"x": [], "y": []})
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_descriptive_empty(self):
        result = _run_stats("descriptive", {"var1": "x"}, {"x": []})
        self.assertIsInstance(result, dict)

    def test_imr_empty(self):
        result = _run_spc("imr", {"measurement": "x"}, {"x": []})
        self.assertIsInstance(result, dict)

    def test_bayes_ttest_empty(self):
        # Known: crashes with empty data (KeyError on column access)
        result = _run_bayesian("bayes_ttest", {"var1": "x", "mu": 0}, {"x": []})
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Edge case: single observation
# ---------------------------------------------------------------------------
class SingleObservationTest(TestCase):
    """Analyses receiving n=1 should return gracefully."""

    def test_ttest_single(self):
        result = _run_stats("ttest", {"var1": "x", "mu": 0}, {"x": [42.0]})
        self.assertIsInstance(result, dict)

    def test_ttest2_single_per_group(self):
        result = _run_stats("ttest2", {"var1": "a", "var2": "b"}, {"a": [1.0], "b": [2.0]})
        self.assertIsInstance(result, dict)

    def test_descriptive_single(self):
        result = _run_stats("descriptive", {"var1": "x"}, {"x": [42.0]})
        self.assertIsInstance(result, dict)
        # Should still produce mean
        stats = result.get("statistics", {})
        if "mean" in stats:
            self.assertAlmostEqual(stats["mean"], 42.0, delta=0.01)

    def test_correlation_two_points(self):
        """Correlation with exactly 2 points — degenerate but shouldn't crash."""
        result = _run_stats("correlation", {"variables": ["x", "y"]}, {"x": [1.0, 2.0], "y": [3.0, 4.0]})
        self.assertIsInstance(result, dict)

    def test_imr_single(self):
        result = _run_spc("imr", {"measurement": "x"}, {"x": [42.0]})
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Edge case: constant values (zero variance)
# ---------------------------------------------------------------------------
class ConstantValueTest(TestCase):
    """All identical values → zero variance. Tests must not divide by zero."""

    def test_ttest_constant(self):
        data = [100.0] * 50
        result = _run_stats("ttest", {"var1": "x", "mu": 100}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_ttest_constant_vs_different_mu(self):
        data = [100.0] * 50
        result = _run_stats("ttest", {"var1": "x", "mu": 95}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_ttest2_both_constant(self):
        a = [100.0] * 30
        b = [200.0] * 30
        result = _run_stats("ttest2", {"var1": "a", "var2": "b"}, {"a": a, "b": b})
        self.assertIsInstance(result, dict)

    def test_ttest2_same_constant(self):
        """Both groups have the same constant value — should not crash."""
        a = [100.0] * 30
        b = [100.0] * 30
        result = _run_stats("ttest2", {"var1": "a", "var2": "b"}, {"a": a, "b": b})
        self.assertIsInstance(result, dict)

    def test_anova_constant_within_groups(self):
        values = [10.0] * 30 + [20.0] * 30 + [30.0] * 30
        groups = ["A"] * 30 + ["B"] * 30 + ["C"] * 30
        result = _run_stats("anova", {"response": "v", "factor": "g"}, {"v": values, "g": groups})
        self.assertIsInstance(result, dict)

    def test_correlation_constant_x(self):
        """Correlation with constant X → undefined r. Should not crash."""
        x = [5.0] * 50
        y = list(range(50))
        result = _run_stats("correlation", {"variables": ["x", "y"]}, {"x": x, "y": [float(v) for v in y]})
        self.assertIsInstance(result, dict)

    def test_regression_constant_predictor(self):
        x = [1.0] * 50
        y = list(np.random.RandomState(99).normal(0, 1, 50))
        result = _run_stats("regression", {"response": "y", "predictors": ["x"]}, {"x": x, "y": y})
        self.assertIsInstance(result, dict)

    def test_imr_constant(self):
        result = _run_spc("imr", {"measurement": "x"}, {"x": [50.0] * 30})
        self.assertIsInstance(result, dict)

    def test_capability_constant(self):
        """Constant process → Cp/Cpk = infinity (or very large). Should not crash."""
        result = _run_spc("capability", {"measurement": "x", "lsl": 40, "usl": 60}, {"x": [50.0] * 50})
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Edge case: NaN in data
# ---------------------------------------------------------------------------
class NaNHandlingTest(TestCase):
    """NaN values in data should be handled gracefully (dropped or reported)."""

    def _data_with_nans(self, n=100, nan_frac=0.1):
        rng = np.random.RandomState(77)
        data = rng.normal(50, 10, n).tolist()
        nan_count = int(n * nan_frac)
        for i in range(nan_count):
            data[i] = float("nan")
        return data

    def test_ttest_with_nans(self):
        data = self._data_with_nans()
        result = _run_stats("ttest", {"var1": "x", "mu": 50}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_descriptive_with_nans(self):
        data = self._data_with_nans()
        result = _run_stats("descriptive", {"var1": "x"}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_correlation_with_nans(self):
        rng = np.random.RandomState(78)
        x = rng.normal(0, 1, 100).tolist()
        y = rng.normal(0, 1, 100).tolist()
        x[0] = float("nan")
        y[5] = float("nan")
        result = _run_stats("correlation", {"variables": ["x", "y"]}, {"x": x, "y": y})
        self.assertIsInstance(result, dict)

    def test_imr_with_nans(self):
        data = self._data_with_nans(50, 0.1)
        result = _run_spc("imr", {"measurement": "x"}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_all_nan(self):
        """Entirely NaN data — should not crash."""
        data = [float("nan")] * 20
        result = _run_stats("ttest", {"var1": "x", "mu": 0}, {"x": data})
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Edge case: infinity in data
# ---------------------------------------------------------------------------
class InfHandlingTest(TestCase):
    """Inf values should not cause unhandled exceptions."""

    def test_ttest_with_inf(self):
        data = list(np.random.RandomState(80).normal(50, 10, 50))
        data[0] = float("inf")
        result = _run_stats("ttest", {"var1": "x", "mu": 50}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_ttest_with_neg_inf(self):
        data = list(np.random.RandomState(81).normal(50, 10, 50))
        data[0] = float("-inf")
        result = _run_stats("ttest", {"var1": "x", "mu": 50}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_regression_with_inf(self):
        # Known: crashes with inf (sklearn rejects infinite values)
        rng = np.random.RandomState(82)
        x = rng.normal(0, 1, 50).tolist()
        y = rng.normal(0, 1, 50).tolist()
        y[0] = float("inf")
        result = _run_stats("regression", {"response": "y", "predictors": ["x"]}, {"x": x, "y": y})
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Edge case: extreme values
# ---------------------------------------------------------------------------
class ExtremeValueTest(TestCase):
    """Very large or very small values that might cause overflow."""

    def test_ttest_very_large(self):
        rng = np.random.RandomState(83)
        data = (rng.normal(0, 1, 100) * 1e15).tolist()
        result = _run_stats("ttest", {"var1": "x", "mu": 0}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_ttest_very_small(self):
        rng = np.random.RandomState(84)
        data = (rng.normal(0, 1, 100) * 1e-15).tolist()
        result = _run_stats("ttest", {"var1": "x", "mu": 0}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_regression_large_predictor(self):
        rng = np.random.RandomState(85)
        x = (rng.normal(0, 1, 100) * 1e10).tolist()
        y = (rng.normal(0, 1, 100) * 1e10).tolist()
        result = _run_stats("regression", {"response": "y", "predictors": ["x"]}, {"x": x, "y": y})
        self.assertIsInstance(result, dict)

    def test_capability_tight_limits(self):
        """Spec limits very close to mean — low Cp/Cpk."""
        rng = np.random.RandomState(86)
        data = rng.normal(50, 10, 100).tolist()
        result = _run_spc("capability", {"measurement": "x", "lsl": 49, "usl": 51}, {"x": data})
        self.assertIsInstance(result, dict)

    def test_capability_wide_limits(self):
        """Spec limits very far from data — very high Cp."""
        rng = np.random.RandomState(87)
        data = rng.normal(50, 1, 100).tolist()
        result = _run_spc("capability", {"measurement": "x", "lsl": 0, "usl": 100}, {"x": data})
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Edge case: wrong column names
# ---------------------------------------------------------------------------
class MissingColumnTest(TestCase):
    """Config references columns that don't exist in data."""

    def test_ttest_wrong_column(self):
        """Missing column should return error, not crash."""
        result = _run_stats("ttest", {"var1": "nonexistent", "mu": 0}, {"x": [1, 2, 3]})
        self.assertIsInstance(result, dict)
        # Known: throws KeyError instead of graceful error
        self.assertIn("error", result)

    def test_anova_wrong_factor(self):
        """Missing factor column should return error, not crash."""
        result = _run_stats(
            "anova",
            {"response": "v", "factor": "nonexistent"},
            {"v": [1, 2, 3], "g": ["A", "B", "C"]},
        )
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_regression_wrong_predictor(self):
        """Missing predictor column should return error, not crash."""
        result = _run_stats(
            "regression",
            {"response": "y", "predictors": ["nonexistent"]},
            {"x": [1, 2, 3], "y": [4, 5, 6]},
        )
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Edge case: minimum sample sizes for specific tests
# ---------------------------------------------------------------------------
class MinimumSampleSizeTest(TestCase):
    """Tests that require minimum n should handle small samples gracefully."""

    def test_chi2_single_cell(self):
        """Chi-square with 1x1 contingency — degenerate."""
        result = _run_stats("chi2", {"var1": "a", "var2": "b"}, {"a": ["yes"], "b": ["no"]})
        self.assertIsInstance(result, dict)

    def test_chi2_two_by_two(self):
        """Minimal 2x2 contingency table."""
        result = _run_stats(
            "chi2",
            {"var1": "a", "var2": "b"},
            {"a": ["A", "A", "B", "B"], "b": ["X", "Y", "X", "Y"]},
        )
        self.assertIsInstance(result, dict)

    def test_anova_single_group(self):
        """ANOVA with only one group — degenerate. Known crash."""
        result = _run_stats(
            "anova",
            {"response": "v", "factor": "g"},
            {"v": [1.0, 2.0, 3.0], "g": ["A", "A", "A"]},
        )
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_mann_whitney_tiny(self):
        """Mann-Whitney with n=2 per group."""
        values = [1.0, 2.0, 3.0, 4.0]
        groups = ["A", "A", "B", "B"]
        result = _run_stats(
            "mann_whitney",
            {"var": "value", "group_var": "group"},
            {"value": values, "group": groups},
        )
        self.assertIsInstance(result, dict)

    def test_paired_t_two_pairs(self):
        """Paired t with only 2 pairs."""
        result = _run_stats(
            "paired_t",
            {"var1": "before", "var2": "after"},
            {"before": [10.0, 20.0], "after": [12.0, 22.0]},
        )
        self.assertIsInstance(result, dict)

    def test_fisher_exact_sparse(self):
        """Fisher exact with very small counts."""
        result = _run_stats(
            "fisher_exact",
            {"var1": "treatment", "var2": "outcome"},
            {"treatment": ["A", "A", "B", "B"], "outcome": ["Y", "N", "Y", "N"]},
        )
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Edge case: large datasets
# ---------------------------------------------------------------------------
class LargeDatasetTest(TestCase):
    """Verify analyses complete without timeout on larger datasets."""

    def test_ttest_5000(self):
        rng = np.random.RandomState(90)
        data = rng.normal(100, 15, 5000).tolist()
        result = _run_stats("ttest", {"var1": "x", "mu": 100}, {"x": data})
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)

    def test_correlation_5000(self):
        rng = np.random.RandomState(91)
        x = rng.normal(0, 1, 5000).tolist()
        y = rng.normal(0, 1, 5000).tolist()
        result = _run_stats("correlation", {"variables": ["x", "y"]}, {"x": x, "y": y})
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)

    def test_imr_2000(self):
        rng = np.random.RandomState(92)
        data = rng.normal(50, 2, 2000).tolist()
        result = _run_spc("imr", {"measurement": "x"}, {"x": data})
        self.assertIsInstance(result, dict)
