"""Unit tests for calibration.py — reference pool and calibration runner.

Verifies the calibration infrastructure that underpins CAL-001 coverage.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1, ISO/IEC 17025:2017
<!-- test: agents_api.tests.test_calibration -->
"""

from django.test import TestCase


class ReferencePoolTest(TestCase):
    """Tests for get_reference_pool()."""

    def test_pool_not_empty(self):
        from agents_api.calibration import get_reference_pool

        pool = get_reference_pool()
        self.assertGreater(len(pool), 0)

    def test_pool_has_required_fields(self):
        from agents_api.calibration import get_reference_pool

        pool = get_reference_pool()
        for case in pool[:5]:
            self.assertTrue(hasattr(case, "case_id"))
            self.assertTrue(hasattr(case, "analysis_type"))
            self.assertTrue(hasattr(case, "analysis_id"))
            self.assertTrue(hasattr(case, "config"))
            self.assertTrue(hasattr(case, "expectations"))

    def test_pool_has_multiple_categories(self):
        from agents_api.calibration import get_reference_pool

        pool = get_reference_pool()
        categories = {c.category for c in pool}
        self.assertIn("inference", categories)
        self.assertGreater(len(categories), 1)

    def test_case_ids_unique(self):
        from agents_api.calibration import get_reference_pool

        pool = get_reference_pool()
        ids = [c.case_id for c in pool]
        self.assertEqual(len(ids), len(set(ids)))

    def test_expectations_are_lists(self):
        from agents_api.calibration import get_reference_pool

        pool = get_reference_pool()
        for case in pool:
            self.assertIsInstance(case.expectations, list)

    def test_pool_covers_stats(self):
        from agents_api.calibration import get_reference_pool

        pool = get_reference_pool()
        stats_cases = [c for c in pool if c.analysis_type == "stats"]
        self.assertGreater(len(stats_cases), 10)


class CalibrationRunnerTest(TestCase):
    """Tests for run_calibration()."""

    def test_run_subset(self):
        from agents_api.calibration import run_calibration

        result = run_calibration(subset_size=3, seed=42)
        self.assertIn("cases_run", result)
        self.assertEqual(result["cases_run"], 3)
        self.assertIn("pass_rate", result)
        self.assertIn("results", result)

    def test_run_all(self):
        from agents_api.calibration import run_calibration

        result = run_calibration(subset_size=0)
        self.assertGreater(result["cases_run"], 0)
        self.assertIn("drift_cases", result)

    def test_pass_rate_is_percentage(self):
        from agents_api.calibration import run_calibration

        result = run_calibration(subset_size=5, seed=42)
        self.assertGreaterEqual(result["pass_rate"], 0)
        self.assertLessEqual(result["pass_rate"], 100)

    def test_seed_reproducibility(self):
        from agents_api.calibration import run_calibration

        r1 = run_calibration(subset_size=5, seed=123)
        r2 = run_calibration(subset_size=5, seed=123)
        ids1 = [r["case_id"] for r in r1["results"]]
        ids2 = [r["case_id"] for r in r2["results"]]
        self.assertEqual(ids1, ids2)


class CheckExpectationTest(TestCase):
    """Tests for _check_expectation()."""

    def test_abs_within(self):
        from agents_api.calibration import Expectation, _check_expectation

        exp = Expectation(key="statistics.p_value", expected=0.05, tolerance=0.01)
        result = {"statistics": {"p_value": 0.052}}
        check = _check_expectation(result, exp)
        self.assertTrue(check["passed"])

    def test_abs_within_fail(self):
        from agents_api.calibration import Expectation, _check_expectation

        exp = Expectation(key="statistics.p_value", expected=0.05, tolerance=0.001)
        result = {"statistics": {"p_value": 0.1}}
        check = _check_expectation(result, exp)
        self.assertFalse(check["passed"])

    def test_greater_than(self):
        from agents_api.calibration import Expectation, _check_expectation

        exp = Expectation(key="statistics.n", expected=10, comparison="greater_than")
        result = {"statistics": {"n": 50}}
        check = _check_expectation(result, exp)
        self.assertTrue(check["passed"])

    def test_contains(self):
        from agents_api.calibration import Expectation, _check_expectation

        exp = Expectation(key="summary_contains", expected="significant", comparison="contains")
        result = {"summary": "The result is statistically significant at p < 0.05"}
        check = _check_expectation(result, exp)
        self.assertTrue(check["passed"])


class ExtractNestedTest(TestCase):
    """Tests for _extract_nested() dot-notation accessor."""

    def test_simple_key(self):
        from agents_api.calibration import _extract_nested

        self.assertEqual(_extract_nested({"a": 1}, "a"), 1)

    def test_nested_key(self):
        from agents_api.calibration import _extract_nested

        self.assertEqual(_extract_nested({"a": {"b": 2}}, "a.b"), 2)

    def test_missing_key_returns_none(self):
        from agents_api.calibration import _extract_nested

        self.assertIsNone(_extract_nested({"a": 1}, "b"))

    def test_deep_nested_key(self):
        from agents_api.calibration import _extract_nested

        data = {"a": {"b": {"c": 42}}}
        self.assertEqual(_extract_nested(data, "a.b.c"), 42)
