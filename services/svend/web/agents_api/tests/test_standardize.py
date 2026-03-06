"""Unit tests for standardize.py — output normalization.

Verifies standardize_output() enforces canonical schema (QUAL-001 §6),
bounds validation, and narrative conversion.

Standard: CAL-001 §6, QUAL-001 §6
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_standardize -->
"""

from django.test import TestCase


class StandardizeOutputTest(TestCase):
    """Tests for standardize_output()."""

    def test_fills_missing_fields(self):
        from agents_api.dsw.standardize import standardize_output

        result = standardize_output({"summary": "Hello"}, "stats", "ttest")
        self.assertIn("plots", result)
        self.assertIn("narrative", result)
        self.assertIn("diagnostics", result)
        self.assertIn("education", result)

    def test_string_narrative_converted_to_dict(self):
        from agents_api.dsw.standardize import standardize_output

        result = standardize_output(
            {"summary": "Test summary", "narrative": "Plain string narrative"},
            "stats",
            "ttest",
        )
        self.assertIsInstance(result["narrative"], dict)
        self.assertIn("verdict", result["narrative"])

    def test_dict_narrative_preserved(self):
        from agents_api.dsw.standardize import standardize_output

        narr = {"verdict": "V", "body": "B", "next_steps": "N", "chart_guidance": "C"}
        result = standardize_output(
            {"summary": "S", "narrative": narr},
            "stats",
            "ttest",
        )
        self.assertEqual(result["narrative"]["verdict"], "V")

    def test_non_dict_input_returned_unchanged(self):
        from agents_api.dsw.standardize import standardize_output

        self.assertEqual(standardize_output("not a dict", "stats", "ttest"), "not a dict")
        self.assertIsNone(standardize_output(None, "stats", "ttest"))

    def test_metadata_tags_added(self):
        from agents_api.dsw.standardize import standardize_output

        result = standardize_output({"summary": "X"}, "bayesian", "bayes_ttest")
        self.assertEqual(result["_analysis_type"], "bayesian")
        self.assertEqual(result["_analysis_id"], "bayes_ttest")

    def test_guide_observation_generated(self):
        from agents_api.dsw.standardize import standardize_output

        result = standardize_output(
            {"summary": "<<COLOR:title>>ANOVA<</COLOR>>\nF = 5.2, p = 0.03"},
            "stats",
            "anova",
        )
        obs = result.get("guide_observation", "")
        self.assertNotIn("<<COLOR:", obs)  # Color tags stripped


class ValidateStatisticsBoundsTest(TestCase):
    """Tests for _validate_statistics_bounds()."""

    def test_p_value_clamped(self):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        result = {"statistics": {"p_value": 1.5}}
        _validate_statistics_bounds(result)
        self.assertLessEqual(result["statistics"]["p_value"], 1.0)

    def test_negative_p_value_clamped(self):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        result = {"statistics": {"p_value": -0.1}}
        _validate_statistics_bounds(result)
        self.assertGreaterEqual(result["statistics"]["p_value"], 0.0)

    def test_correlation_clamped(self):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        result = {"statistics": {"correlation": 1.5}}
        _validate_statistics_bounds(result)
        self.assertLessEqual(result["statistics"]["correlation"], 1.0)

    def test_r_squared_clamped(self):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        result = {"statistics": {"r_squared": -0.1}}
        _validate_statistics_bounds(result)
        self.assertGreaterEqual(result["statistics"]["r_squared"], 0.0)

    def test_nan_set_to_none(self):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        result = {"statistics": {"p_value": float("nan")}}
        _validate_statistics_bounds(result)
        self.assertIsNone(result["statistics"]["p_value"])

    def test_inf_set_to_none(self):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        result = {"statistics": {"bf10": float("inf")}}
        _validate_statistics_bounds(result)
        self.assertIsNone(result["statistics"]["bf10"])

    def test_valid_values_unchanged(self):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        result = {"statistics": {"p_value": 0.05, "correlation": -0.7, "r_squared": 0.85}}
        _validate_statistics_bounds(result)
        self.assertAlmostEqual(result["statistics"]["p_value"], 0.05)
        self.assertAlmostEqual(result["statistics"]["correlation"], -0.7)
        self.assertAlmostEqual(result["statistics"]["r_squared"], 0.85)

    def test_none_values_skipped(self):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        result = {"statistics": {"p_value": None}}
        _validate_statistics_bounds(result)
        self.assertIsNone(result["statistics"]["p_value"])


class NarrativeFromSummaryTest(TestCase):
    """Tests for _narrative_from_summary()."""

    def test_strips_color_tags(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        result = _narrative_from_summary("<<COLOR:title>>ANOVA<</COLOR>>\nF = 5.2")
        self.assertIsNotNone(result)
        self.assertNotIn("<<COLOR:", result["verdict"])

    def test_first_line_is_verdict(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        result = _narrative_from_summary("Main conclusion\nDetails here\nMore details")
        self.assertEqual(result["verdict"], "Main conclusion")
        self.assertIn("Details here", result["body"])

    def test_empty_string_returns_empty_narrative(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        result = _narrative_from_summary("")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["verdict"], "")
        self.assertEqual(result["body"], "")

    def test_whitespace_only_returns_empty_narrative(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        result = _narrative_from_summary("   \n  \n  ")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["verdict"], "")
        self.assertEqual(result["body"], "")

    def test_single_line(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        result = _narrative_from_summary("Just a verdict")
        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "Just a verdict")
        self.assertEqual(result["body"], "")

    def test_has_required_keys(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        result = _narrative_from_summary("V\nB")
        for key in ("verdict", "body", "next_steps", "chart_guidance"):
            self.assertIn(key, result)
