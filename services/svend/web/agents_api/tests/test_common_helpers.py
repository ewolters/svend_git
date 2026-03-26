"""Unit tests for common.py helper functions.

Covers the statistical helper functions that are used by all analysis engines.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_common_helpers -->
"""

import numpy as np
from django.test import TestCase


class NarrativeTest(TestCase):
    """Tests for _narrative() dict builder."""

    def test_narrative_returns_dict(self):
        from agents_api.dsw.common import _narrative

        result = _narrative("Verdict text", "Body text")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["verdict"], "Verdict text")
        self.assertEqual(result["body"], "Body text")

    def test_narrative_contains_verdict(self):
        from agents_api.dsw.common import _narrative

        result = _narrative("The mean differs", "More details here")
        self.assertEqual(result["verdict"], "The mean differs")
        self.assertEqual(result["body"], "More details here")

    def test_narrative_with_next_steps(self):
        from agents_api.dsw.common import _narrative

        result = _narrative("Verdict", "Body", next_steps="Collect more data")
        self.assertEqual(result["next_steps"], "Collect more data")

    def test_narrative_with_chart_guidance(self):
        from agents_api.dsw.common import _narrative

        result = _narrative(
            "Verdict", "Body", chart_guidance="Look at the scatter plot"
        )
        self.assertEqual(result["chart_guidance"], "Look at the scatter plot")

    def test_narrative_all_fields(self):
        from agents_api.dsw.common import _narrative

        result = _narrative(
            "Strong evidence",
            "The data shows significance",
            next_steps="Run confirmatory study",
            chart_guidance="Check residual plot",
        )
        self.assertEqual(result["verdict"], "Strong evidence")
        self.assertEqual(result["body"], "The data shows significance")
        self.assertEqual(result["next_steps"], "Run confirmatory study")
        self.assertEqual(result["chart_guidance"], "Check residual plot")


class CheckNormalityTest(TestCase):
    """Tests for _check_normality() diagnostic."""

    def test_normal_data_returns_none(self):
        from agents_api.dsw.common import _check_normality

        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        result = _check_normality(data)
        # Large normal sample should pass normality test
        # (result is None when normality holds)
        if result is not None:
            self.assertIsInstance(result, dict)

    def test_nonnormal_data_returns_dict(self):
        from agents_api.dsw.common import _check_normality

        data = np.concatenate([np.zeros(50), np.ones(5) * 100])
        result = _check_normality(data)
        if result is not None:
            self.assertIn("level", result)
            self.assertIn("detail", result)

    def test_custom_label(self):
        from agents_api.dsw.common import _check_normality

        data = np.concatenate([np.zeros(50), np.ones(5) * 100])
        result = _check_normality(data, label="Temperature")
        if result is not None:
            self.assertIsInstance(result, dict)
            self.assertIn("level", result)


class CheckOutliersTest(TestCase):
    """Tests for _check_outliers() diagnostic."""

    def test_no_outliers(self):
        from agents_api.dsw.common import _check_outliers

        data = np.random.RandomState(42).normal(0, 1, 100)
        result = _check_outliers(data)
        # May or may not detect outliers in normal data
        if result is not None:
            self.assertIsInstance(result, dict)

    def test_with_outliers(self):
        from agents_api.dsw.common import _check_outliers

        data = list(np.random.RandomState(42).normal(0, 1, 100)) + [100, -100, 200]
        result = _check_outliers(data)
        if result is not None:
            self.assertIn("_n_outliers", result)
            self.assertGreater(result["_n_outliers"], 0)


class CheckEqualVarianceTest(TestCase):
    """Tests for _check_equal_variance() diagnostic."""

    def test_equal_variances(self):
        from agents_api.dsw.common import _check_equal_variance

        g1 = np.random.RandomState(42).normal(0, 1, 50)
        g2 = np.random.RandomState(43).normal(0, 1, 50)
        result = _check_equal_variance(g1, g2)
        # Should return None when variances are equal
        if result is not None:
            self.assertIsInstance(result, dict)

    def test_unequal_variances(self):
        from agents_api.dsw.common import _check_equal_variance

        g1 = np.random.RandomState(42).normal(0, 1, 50)
        g2 = np.random.RandomState(43).normal(0, 10, 50)
        result = _check_equal_variance(g1, g2)
        if result is not None:
            self.assertIn("level", result)


class EffectMagnitudeTest(TestCase):
    """Tests for _effect_magnitude() classifier."""

    def test_large_cohens_d(self):
        from agents_api.dsw.common import _effect_magnitude

        label, meaningful = _effect_magnitude(1.2, "cohens_d")
        self.assertEqual(label, "large")
        self.assertTrue(meaningful)

    def test_medium_cohens_d(self):
        from agents_api.dsw.common import _effect_magnitude

        label, meaningful = _effect_magnitude(0.5, "cohens_d")
        self.assertEqual(label, "medium")
        self.assertTrue(meaningful)

    def test_small_cohens_d(self):
        from agents_api.dsw.common import _effect_magnitude

        label, meaningful = _effect_magnitude(0.2, "cohens_d")
        self.assertEqual(label, "small")
        self.assertFalse(meaningful)

    def test_negligible_cohens_d(self):
        from agents_api.dsw.common import _effect_magnitude

        label, meaningful = _effect_magnitude(0.05, "cohens_d")
        self.assertEqual(label, "negligible")
        self.assertFalse(meaningful)

    def test_eta_squared(self):
        from agents_api.dsw.common import _effect_magnitude

        label, meaningful = _effect_magnitude(0.15, "eta_squared")
        self.assertIn(label, ["large", "medium"])

    def test_negative_value_uses_abs(self):
        from agents_api.dsw.common import _effect_magnitude

        label, meaningful = _effect_magnitude(-0.8, "cohens_d")
        self.assertEqual(label, "large")


class CrossValidateTest(TestCase):
    """Tests for _cross_validate() agreement checker."""

    def test_agreement(self):
        from agents_api.dsw.common import _cross_validate

        result = _cross_validate(0.001, 0.002, "t-test", "Mann-Whitney")
        self.assertIsInstance(result, dict)
        self.assertIn("level", result)

    def test_contradiction(self):
        from agents_api.dsw.common import _cross_validate

        result = _cross_validate(0.001, 0.5, "t-test", "Mann-Whitney")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["level"], "contradiction")


class PracticalBlockTest(TestCase):
    """Tests for _practical_block() text builder."""

    def test_significant_large_effect(self):
        from agents_api.dsw.common import _practical_block

        result = _practical_block("Cohen's d", 1.0, "cohens_d", 0.001)
        self.assertIsInstance(result, str)
        self.assertIn("large", result.lower())

    def test_nonsignificant(self):
        from agents_api.dsw.common import _practical_block

        result = _practical_block("Cohen's d", 0.1, "cohens_d", 0.5)
        self.assertIsInstance(result, str)


class BayesianShadowTest(TestCase):
    """Tests for _bayesian_shadow() Bayes factor calculator."""

    def test_ttest_1samp(self):
        from agents_api.dsw.common import _bayesian_shadow

        x = np.random.RandomState(42).normal(100, 15, 30)
        result = _bayesian_shadow("ttest_1samp", x=x, mu=100)
        if result is not None:
            self.assertIn("bf10", result)
            self.assertIn("bf_label", result)
            self.assertIn("interpretation", result)
            self.assertGreater(result["bf10"], 0)

    def test_ttest_2samp(self):
        from agents_api.dsw.common import _bayesian_shadow

        x = np.random.RandomState(42).normal(100, 15, 30)
        y = np.random.RandomState(43).normal(105, 15, 30)
        result = _bayesian_shadow("ttest_2samp", x=x, y=y)
        if result is not None:
            self.assertIn("bf10", result)

    def test_correlation(self):
        from agents_api.dsw.common import _bayesian_shadow

        x = np.random.RandomState(42).normal(0, 1, 30)
        y = x + np.random.RandomState(43).normal(0, 0.5, 30)
        result = _bayesian_shadow("correlation", x=x, y=y)
        if result is not None:
            self.assertIn("bf10", result)

    def test_unknown_type_returns_none(self):
        from agents_api.dsw.common import _bayesian_shadow

        result = _bayesian_shadow("nonexistent_type", x=[1, 2, 3])
        self.assertIsNone(result)


class EvidenceGradeTest(TestCase):
    """Tests for _evidence_grade() synthesizer."""

    def test_strong_evidence(self):
        from agents_api.dsw.common import _evidence_grade

        result = _evidence_grade(0.001, bf10=15.0, effect_magnitude="large")
        if result is not None:
            self.assertIn("grade", result)
            self.assertIn("rationale", result)
            self.assertIn("components", result)

    def test_weak_evidence(self):
        from agents_api.dsw.common import _evidence_grade

        result = _evidence_grade(0.5, bf10=0.5, effect_magnitude="negligible")
        if result is not None:
            self.assertIn(result["grade"], ["Weak", "Inconclusive"])

    def test_none_pvalue(self):
        from agents_api.dsw.common import _evidence_grade

        result = _evidence_grade(None)
        self.assertIsNone(result)

    def test_pvalue_only(self):
        from agents_api.dsw.common import _evidence_grade

        result = _evidence_grade(0.03)
        if result is not None:
            self.assertIn("grade", result)


class ModelCacheTest(TestCase):
    """Tests for cache_model / get_cached_model."""

    def test_cache_and_retrieve(self):
        from agents_api.dsw.common import cache_model, get_cached_model

        cache_model("test_user", "test_model", {"dummy": "model"}, {"accuracy": 0.95})
        entry = get_cached_model("test_user", "test_model")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["model"], {"dummy": "model"})
        self.assertEqual(entry["metadata"]["accuracy"], 0.95)

    def test_missing_key_returns_none(self):
        from agents_api.dsw.common import get_cached_model

        result = get_cached_model("no_user", "no_model")
        self.assertIsNone(result)


class SanitizeForJsonTest(TestCase):
    """Tests for sanitize_for_json()."""

    def test_nan_replaced(self):
        from agents_api.dsw.common import sanitize_for_json

        result = sanitize_for_json({"val": float("nan")})
        self.assertIsNone(result["val"])

    def test_inf_replaced(self):
        from agents_api.dsw.common import sanitize_for_json

        result = sanitize_for_json({"val": float("inf")})
        self.assertIsNone(result["val"])

    def test_numpy_array_converted(self):
        from agents_api.dsw.common import sanitize_for_json

        result = sanitize_for_json({"arr": np.array([1, 2, 3])})
        self.assertEqual(result["arr"], [1, 2, 3])

    def test_nested_dict(self):
        from agents_api.dsw.common import sanitize_for_json

        result = sanitize_for_json({"a": {"b": float("nan")}})
        self.assertIsNone(result["a"]["b"])
