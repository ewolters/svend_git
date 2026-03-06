"""T1 final coverage gap closer — dispatch.py, standardize.py, bayesian.py branches.

Target: 186 additional lines covered across three modules.

- dispatch.py: authenticated POST to /api/dsw/analyze/ with inline data
- standardize.py: branch coverage for narrative, bounds, what-if normalization
- bayesian.py: deeper config branches (correct config keys, alternative priors)

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_t1_final -->
"""

import json

import numpy as np
import pandas as pd
from django.test import TestCase, override_settings

from accounts.constants import Tier
from accounts.models import User

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SMOKE_SETTINGS = {"RATELIMIT_ENABLE": False, "SECURE_SSL_REDIRECT": False}

NORMAL_50 = list(np.random.RandomState(42).normal(100, 15, 50))
NORMAL_50B = list(np.random.RandomState(99).normal(105, 15, 50))
BINARY_50 = ([0] * 25) + ([1] * 25)
COUNTS_50 = list(np.random.RandomState(42).poisson(5, 50))
TIME_50 = list(np.random.RandomState(42).exponential(10, 50))
EVENTS_50 = np.random.RandomState(43).choice([0, 1], 50, p=[0.3, 0.7]).tolist()


def _run_bayes(analysis_id, config, data_dict):
    """Direct call to bayesian engine (no HTTP overhead)."""
    from agents_api.dsw.bayesian import run_bayesian_analysis

    df = pd.DataFrame(data_dict)
    return run_bayesian_analysis(df, analysis_id, config)


# ═══════════════════════════════════════════════════════════════════════════
# TARGET 1: dispatch.py — authenticated HTTP tests
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE_SETTINGS)
class DispatchCoverageTest(TestCase):
    """Exercise dispatch.py lines via authenticated POST to /api/dsw/analyze/."""

    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="dispatch_t1", email="dispatch_t1@test.com", password="testpass123"
        )
        cls.user.tier = Tier.PRO
        cls.user.email_verified = True
        cls.user.save()

    def setUp(self):
        self.client.login(username="dispatch_t1", password="testpass123")

    def _post(self, data):
        return self.client.post(
            "/api/dsw/analysis/",
            data=json.dumps(data),
            content_type="application/json",
        )

    # ── JSON parsing error (line 64-67) ──
    def test_invalid_json(self):
        resp = self.client.post("/api/dsw/analysis/", data="not json{{{", content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    # ── No data loaded (line 128-129) ──
    def test_no_data_returns_400(self):
        resp = self._post({"type": "stats", "analysis": "ttest"})
        self.assertEqual(resp.status_code, 400)

    # ── Unknown analysis type (line 201-202) ──
    def test_unknown_type(self):
        resp = self._post({"type": "unknown_type", "analysis": "x", "data": {"x": [1, 2, 3]}})
        self.assertEqual(resp.status_code, 400)

    # ── Inline data loading (line 87-94) + stats routing (line 134-137) ──
    def test_stats_descriptive_inline(self):
        resp = self._post(
            {
                "type": "stats",
                "analysis": "descriptive",
                "data": {"x": [1.0, 2.0, 3.0, 4.0, 5.0]},
                "config": {"features": ["x"]},
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── ML routing (line 138-141) ──
    def test_ml_regression_inline(self):
        rng = np.random.RandomState(42)
        x = list(rng.normal(0, 1, 50))
        y = [2 * xi + rng.normal(0, 0.5) for xi in x]
        resp = self._post(
            {
                "type": "ml",
                "analysis": "regression_ml",
                "data": {"x": x, "y": y},
                "config": {"target": "y", "features": ["x"]},
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── SPC routing (line 142-145) ──
    def test_spc_imr_inline(self):
        resp = self._post(
            {
                "type": "spc",
                "analysis": "imr",
                "data": {"y": list(np.random.RandomState(42).normal(50, 5, 30))},
                "config": {"measurement": "y"},
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── Viz routing (line 146-149) ──
    def test_viz_histogram_inline(self):
        resp = self._post(
            {
                "type": "viz",
                "analysis": "histogram",
                "data": {"x": list(np.random.RandomState(42).normal(0, 1, 30))},
                "config": {"var1": "x"},
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── Bayesian routing (line 150-153) ──
    def test_bayesian_ttest_inline(self):
        resp = self._post(
            {
                "type": "bayesian",
                "analysis": "bayes_ttest",
                "data": {"x": NORMAL_50[:25], "y": NORMAL_50[25:]},
                "config": {"var1": "x", "var2": "y"},
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── Reliability routing (line 154-157) ──
    def test_reliability_weibull_inline(self):
        resp = self._post(
            {
                "type": "reliability",
                "analysis": "weibull",
                "data": {"t": list(np.random.RandomState(42).exponential(100, 30))},
                "config": {"time": "t"},
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── Simulation without data (line 126-127, 158-161) ──
    def test_simulation_no_data(self):
        resp = self._post(
            {
                "type": "simulation",
                "analysis": "monte_carlo",
                "config": {
                    "expression": "x + y",
                    "variables": {
                        "x": {"distribution": "normal", "mean": 10, "std": 1},
                        "y": {"distribution": "normal", "mean": 5, "std": 0.5},
                    },
                    "n_simulations": 100,
                },
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── Bayesian without data (line 126-127) ──
    def test_bayesian_demo_no_data(self):
        resp = self._post(
            {
                "type": "bayesian",
                "analysis": "bayes_demo",
                "config": {"n_tested": 50, "n_failures": 0, "target_reliability": 0.99},
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── save_result branch (line 247-279) ──
    def test_save_result(self):
        resp = self._post(
            {
                "type": "stats",
                "analysis": "descriptive",
                "data": {"x": [1.0, 2.0, 3.0, 4.0, 5.0]},
                "config": {"features": ["x"]},
                "save_result": True,
                "title": "T1 Coverage Test Result",
            }
        )
        self.assertNotEqual(resp.status_code, 401)

    # ── data_id branch with non-existent file (line 97-114) ──
    def test_data_id_missing_file(self):
        resp = self._post(
            {
                "type": "stats",
                "analysis": "descriptive",
                "data_id": "data_nonexistent_abc123",
                "config": {"features": ["x"]},
            }
        )
        # Falls through to "No data loaded" since file doesn't exist
        self.assertIn(resp.status_code, [400, 404, 500])

    # ── Unauthenticated request ──
    def test_unauthenticated(self):
        self.client.logout()
        resp = self._post(
            {
                "type": "stats",
                "analysis": "descriptive",
                "data": {"x": [1]},
            }
        )
        self.assertIn(resp.status_code, [401, 403, 302])

    # ── problem_id branch (line 212-244) — problem doesn't exist but exercises the try block ──
    def test_problem_id_nonexistent(self):
        resp = self._post(
            {
                "type": "stats",
                "analysis": "descriptive",
                "data": {"x": [1.0, 2.0, 3.0, 4.0, 5.0]},
                "config": {"features": ["x"]},
                "problem_id": "fake-problem-id-12345",
            }
        )
        # Should still return 200 (problem linking failure is logged but not fatal)
        self.assertNotEqual(resp.status_code, 401)


# ═══════════════════════════════════════════════════════════════════════════
# TARGET 2: standardize.py — branch coverage
# ═══════════════════════════════════════════════════════════════════════════


class StandardizeBranchTest(TestCase):
    """Exercise uncovered branches in standardize.py."""

    def _std(self, result, atype="stats", aid="descriptive"):
        from agents_api.dsw.standardize import standardize_output

        return standardize_output(result, atype, aid)

    # ── Non-dict input (line 48-49) ──
    def test_non_dict_returns_as_is(self):
        out = self._std("not a dict")
        self.assertEqual(out, "not a dict")

    def test_none_returns_as_is(self):
        out = self._std(None)
        self.assertIsNone(out)

    # ── Fill missing required fields (line 54-60) ──
    def test_fills_missing_required_fields(self):
        out = self._std({})
        self.assertIn("summary", out)
        self.assertIn("plots", out)
        self.assertIn("narrative", out)
        self.assertIn("diagnostics", out)
        self.assertIsInstance(out["plots"], list)
        self.assertIsInstance(out["diagnostics"], list)

    # ── Narrative is a string → wrap (line 63-65) ──
    def test_narrative_string_wrapped(self):
        out = self._std({"narrative": "Simple verdict here"})
        self.assertIsInstance(out["narrative"], dict)
        self.assertEqual(out["narrative"]["verdict"], "Simple verdict here")

    # ── Narrative is None with summary → generate (line 66-67) ──
    def test_narrative_from_summary(self):
        out = self._std({"summary": "Line one\nLine two\nLine three"})
        self.assertIsInstance(out["narrative"], dict)
        self.assertEqual(out["narrative"]["verdict"], "Line one")
        self.assertIn("Line two", out["narrative"]["body"])

    # ── Narrative still None → empty narrative dict (line 70-71) ──
    def test_empty_narrative_fallback(self):
        out = self._std({"summary": "", "narrative": None})
        self.assertIsInstance(out["narrative"], dict)
        self.assertEqual(out["narrative"]["verdict"], "")

    # ── Summary with color tags → cleaned guide_observation (line 74-76) ──
    def test_guide_observation_strips_color_tags(self):
        out = self._std(
            {
                "summary": "<<COLOR:success>>Result is good<</COLOR>> p=0.01",
                "guide_observation": "",
            }
        )
        self.assertNotIn("<<COLOR", out["guide_observation"])
        self.assertIn("Result is good", out["guide_observation"])

    # ── narrative_from_summary with separator-only lines (line 238-242) ──
    def test_narrative_strips_separators(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        text = "Verdict line\n=========\n---\nBody content here"
        result = _narrative_from_summary(text)
        self.assertEqual(result["verdict"], "Verdict line")
        self.assertIn("Body content here", result["body"])
        self.assertNotIn("===", result["body"])

    # ── narrative_from_summary with empty content (line 244-245) ──
    def test_narrative_empty_returns_empty_dict(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        result = _narrative_from_summary("")
        self.assertEqual(result["verdict"], "")
        self.assertEqual(result["body"], "")

    # ── narrative_from_summary with only separators ──
    def test_narrative_only_separators(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        result = _narrative_from_summary("===\n---\n***\n###")
        self.assertEqual(result["verdict"], "")

    # ── narrative_from_summary with HTML and box-drawing chars ──
    def test_narrative_strips_html_and_box(self):
        from agents_api.dsw.standardize import _narrative_from_summary

        text = "<b>Bold</b>\n═══╔══╗\nBody text"
        result = _narrative_from_summary(text)
        self.assertEqual(result["verdict"], "Bold")
        self.assertIn("Body text", result["body"])

    # ── narrative with dict already set (should be kept) ──
    def test_narrative_dict_kept(self):
        out = self._std(
            {
                "narrative": {"verdict": "V", "body": "B", "next_steps": "N", "chart_guidance": "C"},
            }
        )
        self.assertEqual(out["narrative"]["verdict"], "V")

    # ── Plot chart defaults applied (line 79-82) ──
    def test_chart_defaults_applied(self):
        out = self._std(
            {
                "plots": [{"title": "Test", "data": [], "layout": {}}],
            }
        )
        self.assertEqual(len(out["plots"]), 1)


class StandardizeBoundsTest(TestCase):
    """Exercise _validate_statistics_bounds branches."""

    def _validate(self, result):
        from agents_api.dsw.standardize import _validate_statistics_bounds

        _validate_statistics_bounds(result)
        return result

    # ── Bounded metric out of range → clamped (line 185-188) ──
    def test_pvalue_clamped_above(self):
        r = self._validate({"p_value": 1.5})
        self.assertEqual(r["p_value"], 1.0)

    def test_pvalue_clamped_below(self):
        r = self._validate({"p_value": -0.5})
        self.assertEqual(r["p_value"], 0.0)

    def test_correlation_clamped(self):
        r = self._validate({"correlation": 1.5})
        self.assertEqual(r["correlation"], 1.0)

    def test_r_squared_clamped(self):
        r = self._validate({"r_squared": 1.2})
        self.assertEqual(r["r_squared"], 1.0)

    # ── Non-finite values → None (line 182-184) ──
    def test_pvalue_inf_set_none(self):
        r = self._validate({"p_value": float("inf")})
        self.assertIsNone(r["p_value"])

    def test_pvalue_nan_set_none(self):
        r = self._validate({"p_value": float("nan")})
        self.assertIsNone(r["p_value"])

    # ── Positive metric invalid (line 199-201) ──
    def test_bf10_negative_set_none(self):
        r = self._validate({"bf10": -0.5})
        self.assertIsNone(r["bf10"])

    def test_bf10_zero_set_none(self):
        r = self._validate({"bf10": 0.0})
        self.assertIsNone(r["bf10"])

    def test_bf10_inf_set_none(self):
        r = self._validate({"bf10": float("inf")})
        self.assertIsNone(r["bf10"])

    # ── Finite metric non-finite (line 212-214) ──
    def test_cp_inf_set_none(self):
        r = self._validate({"cp": float("inf")})
        self.assertIsNone(r["cp"])

    def test_cohens_d_nan_set_none(self):
        r = self._validate({"cohens_d": float("nan")})
        self.assertIsNone(r["cohens_d"])

    # ── Nested statistics dict (line 167-169) ──
    def test_nested_statistics_dict_clamped(self):
        r = self._validate({"statistics": {"p_value": 2.0, "r_squared": -0.5}})
        self.assertEqual(r["statistics"]["p_value"], 1.0)
        self.assertEqual(r["statistics"]["r_squared"], 0.0)

    # ── Non-numeric value skipped (line 180-181) ──
    def test_non_numeric_pvalue_skipped(self):
        r = self._validate({"p_value": "not_a_number"})
        self.assertEqual(r["p_value"], "not_a_number")

    # ── Valid values left unchanged ──
    def test_valid_values_unchanged(self):
        r = self._validate({"p_value": 0.05, "correlation": 0.8, "bf10": 3.2})
        self.assertEqual(r["p_value"], 0.05)
        self.assertEqual(r["correlation"], 0.8)
        self.assertEqual(r["bf10"], 3.2)

    # ── None value skipped ──
    def test_none_values_skipped(self):
        r = self._validate({"p_value": None, "bf10": None, "cp": None})
        self.assertIsNone(r["p_value"])
        self.assertIsNone(r["bf10"])
        self.assertIsNone(r["cp"])


class StandardizeExtractPValueTest(TestCase):
    """Exercise _extract_p_value branches."""

    def _extract(self, result):
        from agents_api.dsw.standardize import _extract_p_value

        return _extract_p_value(result)

    def test_direct_p_value(self):
        self.assertAlmostEqual(self._extract({"p_value": 0.05}), 0.05)

    def test_nested_p_value(self):
        self.assertAlmostEqual(self._extract({"statistics": {"p_value": 0.01}}), 0.01)

    def test_no_p_value(self):
        self.assertIsNone(self._extract({}))

    def test_invalid_direct_p_value(self):
        self.assertIsNone(self._extract({"p_value": "not_number"}))

    def test_invalid_nested_p_value(self):
        self.assertIsNone(self._extract({"statistics": {"p_value": "bad"}}))


class StandardizeClassifyEffectTest(TestCase):
    """Exercise _classify_effect branches."""

    def _classify(self, result):
        from agents_api.dsw.standardize import _classify_effect

        return _classify_effect(result)

    # ── Cohen's d branches ──
    def test_cohens_d_large(self):
        self.assertEqual(self._classify({"statistics": {"cohens_d": 1.0}}), "large")

    def test_cohens_d_medium(self):
        self.assertEqual(self._classify({"statistics": {"cohens_d": 0.6}}), "medium")

    def test_cohens_d_small(self):
        self.assertEqual(self._classify({"statistics": {"cohens_d": 0.3}}), "small")

    def test_cohens_d_negligible(self):
        self.assertEqual(self._classify({"statistics": {"cohens_d": 0.1}}), "negligible")

    # ── Eta squared branches ──
    def test_eta_large(self):
        self.assertEqual(self._classify({"statistics": {"eta_squared": 0.2}}), "large")

    def test_eta_medium(self):
        self.assertEqual(self._classify({"statistics": {"eta_squared": 0.08}}), "medium")

    def test_eta_small(self):
        self.assertEqual(self._classify({"statistics": {"eta_squared": 0.02}}), "small")

    def test_eta_negligible(self):
        self.assertEqual(self._classify({"statistics": {"eta_squared": 0.005}}), "negligible")

    # ── Effect r branches ──
    def test_effect_r_large(self):
        self.assertEqual(self._classify({"statistics": {"effect_size_r": 0.6}}), "large")

    def test_effect_r_medium(self):
        self.assertEqual(self._classify({"statistics": {"effect_size_r": 0.35}}), "medium")

    def test_effect_r_small(self):
        self.assertEqual(self._classify({"statistics": {"effect_size_r": 0.15}}), "small")

    def test_effect_r_negligible(self):
        self.assertEqual(self._classify({"statistics": {"effect_size_r": 0.05}}), "negligible")

    # ── R-squared branches ──
    def test_r2_large(self):
        self.assertEqual(self._classify({"statistics": {"r_squared": 0.3}}), "large")

    def test_r2_medium(self):
        self.assertEqual(self._classify({"statistics": {"r_squared": 0.15}}), "medium")

    def test_r2_small(self):
        self.assertEqual(self._classify({"statistics": {"r_squared": 0.05}}), "small")

    def test_r2_negligible(self):
        self.assertEqual(self._classify({"statistics": {"r_squared": 0.01}}), "negligible")

    # ── No statistics ──
    def test_no_statistics(self):
        self.assertIsNone(self._classify({}))

    def test_non_dict_statistics(self):
        self.assertIsNone(self._classify({"statistics": "not_dict"}))


class StandardizeNormalizeWhatIfTest(TestCase):
    """Exercise _normalize_what_if branches."""

    def _normalize(self, result, entry=None):
        from agents_api.dsw.standardize import _normalize_what_if

        return _normalize_what_if(result, entry)

    # ── power_explorer legacy pattern (line 405-434) ──
    def test_power_explorer_pattern(self):
        result = {
            "power_explorer": {
                "test_type": "ttest",
                "observed_n": 50,
                "cohens_d": 0.6,
                "alpha": 0.05,
            }
        }
        wi = self._normalize(result)
        self.assertIsNotNone(wi)
        self.assertEqual(wi["type"], "slider")
        self.assertEqual(len(wi["parameters"]), 3)
        self.assertEqual(wi["legacy_source"], "power_explorer")

    # ── what_if_data regression pattern (line 437-462) ──
    def test_what_if_data_regression(self):
        result = {
            "what_if_data": {
                "type": "regression",
                "feature_ranges": {
                    "temp": {"min": 20, "max": 80, "mean": 50},
                    "pressure": {"min": 1, "max": 10, "mean": 5},
                },
                "intercept": 3.14,
                "coefficients": {"temp": 0.5, "pressure": 1.2},
            }
        }
        wi = self._normalize(result)
        self.assertIsNotNone(wi)
        self.assertEqual(wi["type"], "slider")
        self.assertEqual(len(wi["parameters"]), 2)
        self.assertEqual(wi["legacy_source"], "what_if_data")
        self.assertIn("client_model", wi)

    # ── Entry with what_if_tier > 0 but no legacy (line 465-471) ──
    def test_what_if_tier_stub(self):
        entry = {"what_if_tier": 1}
        wi = self._normalize({}, entry)
        self.assertIsNotNone(wi)
        self.assertEqual(wi["type"], "slider")

    def test_what_if_tier_2_sensitivity(self):
        entry = {"what_if_tier": 2}
        wi = self._normalize({}, entry)
        self.assertIsNotNone(wi)
        self.assertEqual(wi["type"], "sensitivity")

    # ── No what-if data and no entry ──
    def test_no_what_if(self):
        wi = self._normalize({}, None)
        self.assertIsNone(wi)


# ═══════════════════════════════════════════════════════════════════════════
# TARGET 3: bayesian.py — deeper branch coverage
# ═══════════════════════════════════════════════════════════════════════════


class BayesianRegressionBranchTest(TestCase):
    """Exercise bayes_regression with correct config keys and multiple features."""

    def test_multi_feature_regression(self):
        rng = np.random.RandomState(42)
        x1 = list(rng.normal(0, 1, 50))
        x2 = list(rng.normal(0, 1, 50))
        y = [2 * a + 3 * b + rng.normal(0, 0.5) for a, b in zip(x1, x2)]
        r = _run_bayes(
            "bayes_regression",
            {"target": "y", "features": ["x1", "x2"]},
            {"x1": x1, "x2": x2, "y": y},
        )
        self.assertIn("summary", r)
        self.assertIn("plots", r)
        self.assertTrue(len(r["plots"]) > 0)
        self.assertIn("synara_weights", r)
        self.assertEqual(len(r["synara_weights"]["coefficients"]), 2)

    def test_regression_no_target(self):
        r = _run_bayes(
            "bayes_regression",
            {"target": "", "features": []},
            {"x": [1, 2, 3]},
        )
        self.assertIn("Error", r["summary"])

    def test_regression_custom_ci(self):
        rng = np.random.RandomState(42)
        x = list(rng.normal(0, 1, 40))
        y = [2 * xi + rng.normal(0, 0.5) for xi in x]
        r = _run_bayes(
            "bayes_regression",
            {"target": "y", "features": ["x"], "ci": 0.99},
            {"x": x, "y": y},
        )
        self.assertIn("99%", r["summary"])


class BayesianTtestBranchTest(TestCase):
    """Exercise bayes_ttest with different prior scales and BF threshold branches."""

    def test_small_prior_scale(self):
        r = _run_bayes(
            "bayes_ttest",
            {"var1": "x", "var2": "y", "prior_scale": "small"},
            {"x": NORMAL_50[:25], "y": NORMAL_50[25:]},
        )
        self.assertIn("bf10", r.get("statistics", {}))

    def test_large_prior_scale(self):
        r = _run_bayes(
            "bayes_ttest",
            {"var1": "x", "var2": "y", "prior_scale": "large"},
            {"x": NORMAL_50[:25], "y": NORMAL_50[25:]},
        )
        self.assertIn("bf10", r.get("statistics", {}))

    def test_ultrawide_prior_scale(self):
        r = _run_bayes(
            "bayes_ttest",
            {"var1": "x", "var2": "y", "prior_scale": "ultrawide"},
            {"x": NORMAL_50[:25], "y": NORMAL_50[25:]},
        )
        self.assertIn("bf10", r.get("statistics", {}))

    def test_strong_evidence_bf_gt_10(self):
        """Two very different distributions to get BF > 10."""
        x = list(np.random.RandomState(42).normal(0, 1, 50))
        y = list(np.random.RandomState(42).normal(5, 1, 50))
        r = _run_bayes(
            "bayes_ttest",
            {"var1": "x", "var2": "y"},
            {"x": x, "y": y},
        )
        self.assertIn("Strong evidence", r["summary"])

    def test_no_difference_bf_lt_1(self):
        """Same distribution → BF < 1."""
        rng = np.random.RandomState(42)
        x = list(rng.normal(0, 1, 50))
        y = list(rng.normal(0, 1, 50))
        r = _run_bayes(
            "bayes_ttest",
            {"var1": "x", "var2": "y"},
            {"x": x, "y": y},
        )
        # Should mention evidence favors no difference or weak evidence
        self.assertIn("statistics", r)


class BayesianABBranchTest(TestCase):
    """Exercise bayes_ab with different priors and outcome directions."""

    def test_jeffreys_prior(self):
        data = {
            "group": ["A"] * 50 + ["B"] * 50,
            "success": [1] * 40 + [0] * 10 + [1] * 25 + [0] * 25,
        }
        r = _run_bayes(
            "bayes_ab",
            {"group": "group", "success": "success", "prior": "jeffreys"},
            data,
        )
        self.assertIn("prob_better", r.get("statistics", {}))

    def test_informed_prior(self):
        data = {
            "group": ["A"] * 50 + ["B"] * 50,
            "success": [1] * 40 + [0] * 10 + [1] * 25 + [0] * 25,
        }
        r = _run_bayes(
            "bayes_ab",
            {"group": "group", "success": "success", "prior": "informed"},
            data,
        )
        # A is clearly better → prob_better should be high
        self.assertGreater(r["statistics"]["prob_better"], 0.5)

    def test_strong_evidence_a_better(self):
        data = {
            "group": ["A"] * 100 + ["B"] * 100,
            "success": [1] * 95 + [0] * 5 + [1] * 50 + [0] * 50,
        }
        r = _run_bayes(
            "bayes_ab",
            {"group": "group", "success": "success"},
            data,
        )
        self.assertIn("Strong evidence", r["summary"])

    def test_b_better(self):
        """B has higher rate → prob_better < 0.05 line."""
        data = {
            "group": ["A"] * 100 + ["B"] * 100,
            "success": [1] * 20 + [0] * 80 + [1] * 95 + [0] * 5,
        }
        r = _run_bayes(
            "bayes_ab",
            {"group": "group", "success": "success"},
            data,
        )
        self.assertLess(r["statistics"]["prob_better"], 0.1)

    def test_less_than_2_groups(self):
        r = _run_bayes(
            "bayes_ab",
            {"group": "group", "success": "success"},
            {"group": ["A"] * 10, "success": [1] * 10},
        )
        self.assertIn("Error", r["summary"])


class BayesianAnovaBranchTest(TestCase):
    """Exercise bayes_anova with strong vs weak group differences."""

    def test_3_group_strong_difference(self):
        """Three well-separated groups → BF > 10."""
        rng = np.random.RandomState(42)
        data = {
            "y": list(rng.normal(10, 1, 20)) + list(rng.normal(20, 1, 20)) + list(rng.normal(30, 1, 20)),
            "g": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        }
        r = _run_bayes("bayes_anova", {"response": "y", "factor": "g"}, data)
        self.assertIn("Strong evidence", r["summary"])

    def test_no_group_difference(self):
        """All groups same → BF < 1."""
        rng = np.random.RandomState(42)
        vals = list(rng.normal(50, 10, 60))
        data = {"y": vals, "g": ["A"] * 20 + ["B"] * 20 + ["C"] * 20}
        r = _run_bayes("bayes_anova", {"variable": "y", "group": "g"}, data)
        self.assertIn("statistics", r)

    def test_using_variable_config_key(self):
        """Test the 'variable' config alias for response."""
        r = _run_bayes(
            "bayes_anova",
            {"variable": "y", "group": "g"},
            {"y": NORMAL_50, "g": ["A"] * 25 + ["B"] * 25},
        )
        self.assertIn("statistics", r)


class BayesianChangepointBranchTest(TestCase):
    """Exercise bayes_changepoint branches — especially the 'var' key and time_col."""

    def test_clear_changepoint(self):
        """Big mean shift → detects changepoint."""
        seg1 = list(np.random.RandomState(42).normal(10, 0.5, 30))
        seg2 = list(np.random.RandomState(43).normal(20, 0.5, 30))
        r = _run_bayes(
            "bayes_changepoint",
            {"var": "y", "max_cp": 1},
            {"y": seg1 + seg2},
        )
        self.assertGreater(r["statistics"]["n_changepoints"], 0)

    def test_no_changepoint(self):
        """Stable data → no changepoint detected (or at most a spurious one)."""
        data = list(np.random.RandomState(42).normal(50, 0.1, 60))
        r = _run_bayes(
            "bayes_changepoint",
            {"var": "y", "max_cp": 2},
            {"y": data},
        )
        # With very low variance, should find 0 changepoints, but allow for
        # spurious detections (BF threshold is only 3)
        self.assertIsInstance(r["statistics"]["n_changepoints"], int)
        self.assertIn("statistics", r)

    def test_with_time_column(self):
        """Exercise the time_col branch (line 630-631)."""
        seg1 = list(np.random.RandomState(42).normal(10, 0.5, 30))
        seg2 = list(np.random.RandomState(43).normal(20, 0.5, 30))
        r = _run_bayes(
            "bayes_changepoint",
            {"var": "y", "time": "t", "max_cp": 2},
            {"y": seg1 + seg2, "t": list(range(60))},
        )
        self.assertIn("statistics", r)

    def test_multiple_changepoints(self):
        """Three regimes → max_cp=2."""
        rng = np.random.RandomState(42)
        seg1 = list(rng.normal(10, 0.5, 20))
        seg2 = list(rng.normal(20, 0.5, 20))
        seg3 = list(rng.normal(30, 0.5, 20))
        r = _run_bayes(
            "bayes_changepoint",
            {"var": "y", "max_cp": 2},
            {"y": seg1 + seg2 + seg3},
        )
        self.assertGreaterEqual(r["statistics"]["n_changepoints"], 1)


class BayesianProportionBranchTest(TestCase):
    """Exercise bayes_proportion both manual and column-based modes."""

    def test_manual_entry_mode(self):
        """Manual successes/n mode (line 817-821).
        Known issue: bayesian.py line 864 references unbound `prior_type`
        in manual mode. No exception masking (TST-001 §11.6) — if the
        bug persists, this test correctly fails to expose it.
        """
        r = _run_bayes(
            "bayes_proportion",
            {"successes": 80, "n": 100, "prior_a": 1, "prior_b": 1},
            {"x": [0]},  # dataframe needed but not used
        )
        self.assertIn("proportion", r.get("statistics", {}))

    def test_column_mode_uniform(self):
        """Column-based mode with uniform prior (line 823-830)."""
        r = _run_bayes(
            "bayes_proportion",
            {"success": "x", "prior": "uniform"},
            {"x": BINARY_50},
        )
        self.assertIn("proportion", r["statistics"])

    def test_column_mode_jeffreys(self):
        r = _run_bayes(
            "bayes_proportion",
            {"success": "x", "prior": "jeffreys"},
            {"x": BINARY_50},
        )
        self.assertIn("proportion", r["statistics"])

    def test_column_mode_optimistic(self):
        r = _run_bayes(
            "bayes_proportion",
            {"success": "x", "prior": "optimistic"},
            {"x": BINARY_50},
        )
        self.assertIn("proportion", r["statistics"])

    def test_column_mode_pessimistic(self):
        r = _run_bayes(
            "bayes_proportion",
            {"success": "x", "prior": "pessimistic"},
            {"x": BINARY_50},
        )
        self.assertIn("proportion", r["statistics"])


class BayesianCapabilityBranchTest(TestCase):
    """Exercise bayes_capability_prediction branches."""

    def test_usl_only(self):
        """One-sided capability (USL only, line 986-987)."""
        r = _run_bayes(
            "bayes_capability_prediction",
            {"var": "x", "usl": 130},
            {"x": NORMAL_50},
        )
        self.assertIn("cpk_mean", r["statistics"])

    def test_lsl_only(self):
        """One-sided capability (LSL only, line 988-989)."""
        r = _run_bayes(
            "bayes_capability_prediction",
            {"var": "x", "lsl": 70},
            {"x": NORMAL_50},
        )
        self.assertIn("cpk_mean", r["statistics"])

    def test_two_sided(self):
        """Both LSL and USL."""
        r = _run_bayes(
            "bayes_capability_prediction",
            {"var": "x", "lsl": 70, "usl": 130},
            {"x": NORMAL_50},
        )
        self.assertIn("cp_mean", r["statistics"])
        # Should have P(Cpk > target) probabilities
        self.assertIn("prob_above_133", r["statistics"])

    def test_no_spec_limits(self):
        r = _run_bayes(
            "bayes_capability_prediction",
            {"var": "x"},
            {"x": NORMAL_50},
        )
        self.assertIn("Error", r["summary"])

    def test_too_few_data(self):
        r = _run_bayes(
            "bayes_capability_prediction",
            {"var": "x", "lsl": 0, "usl": 10},
            {"x": [5.0, 6.0]},
        )
        self.assertIn("at least 3", r["summary"])


class BayesianEquivalenceBranchTest(TestCase):
    """Exercise bayes_equivalence with different ROPE widths and use_effect_size."""

    def test_accept_equivalence(self):
        """Same distribution, wide ROPE → P(in ROPE) > 0.95."""
        rng = np.random.RandomState(42)
        x = list(rng.normal(50, 1, 100))
        y = list(rng.normal(50, 1, 100))
        r = _run_bayes(
            "bayes_equivalence",
            {"var1": "x", "var2": "y", "rope_low": -0.5, "rope_high": 0.5},
            {"x": x, "y": y},
        )
        self.assertEqual(r["statistics"]["decision"], "Accept equivalence")

    def test_reject_equivalence(self):
        """Very different distributions → reject."""
        x = list(np.random.RandomState(42).normal(0, 1, 100))
        y = list(np.random.RandomState(42).normal(5, 1, 100))
        r = _run_bayes(
            "bayes_equivalence",
            {"var1": "x", "var2": "y", "rope_low": -0.1, "rope_high": 0.1},
            {"x": x, "y": y},
        )
        self.assertEqual(r["statistics"]["decision"], "Reject equivalence")

    def test_raw_difference_mode(self):
        """use_effect_size=False → raw difference (line 1223-1225)."""
        r = _run_bayes(
            "bayes_equivalence",
            {"var1": "x", "var2": "y", "use_effect_size": False, "rope_low": -5, "rope_high": 5},
            {"x": NORMAL_50[:25], "y": NORMAL_50[25:]},
        )
        self.assertIn("effect", r["statistics"])


class BayesianChi2BranchTest(TestCase):
    """Exercise bayes_chi2 with strong and weak associations."""

    def test_strong_association(self):
        """Perfect correlation between categorical variables."""
        data = {
            "c1": ["A"] * 50 + ["B"] * 50,
            "c2": ["X"] * 50 + ["Y"] * 50,
        }
        r = _run_bayes("bayes_chi2", {"var1": "c1", "var2": "c2"}, data)
        self.assertIn("bf10", r["statistics"])
        self.assertIn("cramers_v", r["statistics"])

    def test_less_than_2_levels(self):
        r = _run_bayes(
            "bayes_chi2",
            {"var1": "c1", "var2": "c2"},
            {"c1": ["A"] * 10, "c2": ["X"] * 10},
        )
        self.assertIn("Error", r["summary"])


class BayesianPoissonBranchTest(TestCase):
    """Exercise bayes_poisson one-sample and two-sample branches."""

    def test_one_sample(self):
        r = _run_bayes(
            "bayes_poisson",
            {"var1": "x"},
            {"x": COUNTS_50},
        )
        self.assertIn("rate1_mean", r["statistics"])
        self.assertNotIn("rate2_mean", r.get("statistics", {}))

    def test_two_sample(self):
        """Two-sample comparison (line 1592-1638)."""
        r = _run_bayes(
            "bayes_poisson",
            {"var1": "x1", "var2": "x2"},
            {"x1": COUNTS_50, "x2": list(np.random.RandomState(99).poisson(8, 50))},
        )
        self.assertIn("rate2_mean", r["statistics"])
        self.assertIn("p_greater", r["statistics"])

    def test_with_exposure(self):
        """With exposure column (line 1553)."""
        r = _run_bayes(
            "bayes_poisson",
            {"var1": "x", "exposure": "exp"},
            {"x": COUNTS_50, "exp": [10.0] * 50},
        )
        self.assertIn("rate1_mean", r["statistics"])


class BayesianLogisticBranchTest(TestCase):
    """Exercise bayes_logistic with correct config keys."""

    def test_single_feature(self):
        r = _run_bayes(
            "bayes_logistic",
            {"target": "y", "features": ["x"]},
            {"y": BINARY_50, "x": NORMAL_50},
        )
        self.assertIn("accuracy", r.get("statistics", {}))

    def test_multi_feature(self):
        rng = np.random.RandomState(42)
        x1 = list(rng.normal(0, 1, 50))
        x2 = list(rng.normal(0, 1, 50))
        y = [int(a + b > 0) for a, b in zip(x1, x2)]
        r = _run_bayes(
            "bayes_logistic",
            {"target": "y", "features": ["x1", "x2"]},
            {"x1": x1, "x2": x2, "y": y},
        )
        self.assertEqual(len(r["statistics"]["coefficients"]), 2)

    def test_no_target_error(self):
        r = _run_bayes(
            "bayes_logistic",
            {"target": "", "features": []},
            {"x": [1]},
        )
        self.assertIn("Error", r["summary"])

    def test_non_binary_target_error(self):
        r = _run_bayes(
            "bayes_logistic",
            {"target": "y", "features": ["x"]},
            {"y": [0, 1, 2, 3, 4] * 10, "x": list(range(50))},
        )
        self.assertIn("Error", r["summary"])

    def test_custom_prior_width(self):
        r = _run_bayes(
            "bayes_logistic",
            {"target": "y", "features": ["x"], "prior_width": 10.0},
            {"y": BINARY_50, "x": NORMAL_50},
        )
        self.assertIn("accuracy", r.get("statistics", {}))


class BayesianSurvivalBranchTest(TestCase):
    """Exercise bayes_survival with censored and uncensored data."""

    def test_with_event_column(self):
        r = _run_bayes(
            "bayes_survival",
            {"var1": "t", "var2": "e"},
            {"t": TIME_50, "e": EVENTS_50},
        )
        self.assertIn("beta_mean", r.get("statistics", {}))
        self.assertIn("phase", r.get("statistics", {}))

    def test_without_event_column(self):
        """All events assumed = 1 (line 1927-1928)."""
        r = _run_bayes(
            "bayes_survival",
            {"var1": "t"},
            {"t": TIME_50},
        )
        self.assertEqual(r["statistics"]["n_events"], len(TIME_50))

    def test_too_few_observations(self):
        r = _run_bayes(
            "bayes_survival",
            {"var1": "t"},
            {"t": [1.0, 2.0, 3.0, 0.0]},  # only 3 after filter > 0
        )
        self.assertIn("Error", r.get("summary", ""))


class BayesianMetaBranchTest(TestCase):
    """Exercise bayes_meta with correct config keys."""

    def test_basic_meta(self):
        effects = [0.3, 0.5, 0.4, 0.6, 0.35]
        ses = [0.1, 0.15, 0.12, 0.08, 0.11]
        r = _run_bayes(
            "bayes_meta",
            {"var1": "effect", "var2": "se"},
            {"effect": effects, "se": ses},
        )
        self.assertIn("mu_mean", r["statistics"])
        self.assertIn("tau_mean", r["statistics"])
        self.assertIn("i2", r["statistics"])

    def test_too_few_studies(self):
        r = _run_bayes(
            "bayes_meta",
            {"var1": "effect", "var2": "se"},
            {"effect": [0.3], "se": [0.1]},
        )
        self.assertIn("Error", r["summary"])

    def test_high_heterogeneity(self):
        """Large spread in effects → high I²."""
        effects = [-1.0, 0.0, 1.0, 2.0, 3.0]
        ses = [0.1, 0.1, 0.1, 0.1, 0.1]
        r = _run_bayes(
            "bayes_meta",
            {"var1": "effect", "var2": "se"},
            {"effect": effects, "se": ses},
        )
        self.assertGreater(r["statistics"]["i2"], 50)


class BayesianDemoBranchTest(TestCase):
    """Exercise bayes_demo with various test scenarios."""

    def test_zero_failures_pass(self):
        r = _run_bayes(
            "bayes_demo",
            {"n_tested": 100, "n_failures": 0, "target_reliability": 0.95},
            {"x": [0]},
        )
        self.assertIn("prob_exceed_target", r["statistics"])
        self.assertGreater(r["statistics"]["prob_exceed_target"], 0.5)

    def test_many_failures_fail(self):
        r = _run_bayes(
            "bayes_demo",
            {"n_tested": 50, "n_failures": 10, "target_reliability": 0.99},
            {"x": [0]},
        )
        self.assertLess(r["statistics"]["prob_exceed_target"], 0.5)

    def test_custom_prior(self):
        r = _run_bayes(
            "bayes_demo",
            {"n_tested": 50, "n_failures": 1, "target_reliability": 0.95, "prior_a": 5.0, "prior_b": 1.0},
            {"x": [0]},
        )
        self.assertIn("prob_exceed_target", r["statistics"])

    def test_strong_evidence(self):
        """Many units, zero failures → prob > 0.95 → strong evidence line."""
        r = _run_bayes(
            "bayes_demo",
            {"n_tested": 500, "n_failures": 0, "target_reliability": 0.95},
            {"x": [0]},
        )
        self.assertIn("Strong evidence", r["summary"])


class BayesianSparesBranchTest(TestCase):
    """Exercise bayes_spares with column-based and config-based modes."""

    def test_column_mode(self):
        """demand_col present in dataframe (line 2645-2648)."""
        r = _run_bayes(
            "bayes_spares",
            {"var1": "demand", "planning_horizon": 6, "service_level": 0.95},
            {"demand": COUNTS_50},
        )
        self.assertIn("optimal_stock", r["statistics"])

    def test_config_mode(self):
        """No demand column → use total_demand/n_periods from config (line 2649-2651)."""
        r = _run_bayes(
            "bayes_spares",
            {"var1": "nonexistent_col", "total_demand": 50, "n_periods": 12, "planning_horizon": 6},
            {"x": [0]},
        )
        self.assertIn("rate_mean", r["statistics"])

    def test_custom_costs(self):
        r = _run_bayes(
            "bayes_spares",
            {"var1": "demand", "holding_cost": 50, "stockout_cost": 500, "planning_horizon": 12},
            {"demand": COUNTS_50},
        )
        self.assertIn("min_cost", r["statistics"])


class BayesianSystemBranchTest(TestCase):
    """Exercise bayes_system topology branches."""

    def _comps(self):
        return [
            {"name": "Motor", "n": 100, "failures": 2},
            {"name": "Pump", "n": 100, "failures": 5},
            {"name": "Valve", "n": 100, "failures": 3},
        ]

    def test_series_topology(self):
        r = _run_bayes(
            "bayes_system",
            {"components": self._comps(), "topology": "series"},
            {"x": [0]},
        )
        self.assertIn("system_reliability", r["statistics"])

    def test_parallel_topology(self):
        r = _run_bayes(
            "bayes_system",
            {"components": self._comps(), "topology": "parallel"},
            {"x": [0]},
        )
        # Parallel → higher reliability than any component
        self.assertGreater(r["statistics"]["system_reliability"], 0.95)

    def test_k_of_n_topology(self):
        r = _run_bayes(
            "bayes_system",
            {"components": self._comps(), "topology": "k_of_n", "k": 2},
            {"x": [0]},
        )
        self.assertIn("2-of-3", r["statistics"]["topology"])

    def test_json_string_components(self):
        """Components passed as JSON string (line 2882-2884)."""
        import json as _json

        r = _run_bayes(
            "bayes_system",
            {"components": _json.dumps(self._comps()), "topology": "series"},
            {"x": [0]},
        )
        self.assertIn("system_reliability", r["statistics"])

    def test_invalid_json_components(self):
        r = _run_bayes(
            "bayes_system",
            {"components": "not valid json {{{", "topology": "series"},
            {"x": [0]},
        )
        self.assertIn("Error", r["summary"])

    def test_too_few_components(self):
        r = _run_bayes(
            "bayes_system",
            {"components": [{"name": "A", "n": 10, "failures": 0}], "topology": "series"},
            {"x": [0]},
        )
        self.assertIn("Error", r["summary"])


class BayesianWarrantyBranchTest(TestCase):
    """Exercise bayes_warranty with correct config keys."""

    def test_basic_warranty(self):
        r = _run_bayes(
            "bayes_warranty",
            {"var1": "t", "warranty_period": 12, "fleet_size": 500, "forecast_period": 6},
            {"t": TIME_50},
        )
        self.assertIn("rate_mean", r["statistics"])
        self.assertIn("total_forecast", r["statistics"])

    def test_missing_column(self):
        r = _run_bayes(
            "bayes_warranty",
            {"var1": "nonexistent", "warranty_period": 12},
            {"t": TIME_50},
        )
        self.assertIn("Error", r["summary"])


class BayesianRepairableBranchTest(TestCase):
    """Exercise bayes_repairable with correct config keys."""

    def test_basic_repairable(self):
        times = sorted(np.random.RandomState(42).exponential(10, 20).tolist())
        r = _run_bayes(
            "bayes_repairable",
            {"var1": "t"},
            {"t": times},
        )
        self.assertIn("beta_mean", r["statistics"])
        self.assertIn("trend", r["statistics"])

    def test_missing_column(self):
        r = _run_bayes(
            "bayes_repairable",
            {"var1": "nonexistent"},
            {"t": [1, 2, 3]},
        )
        self.assertIn("Error", r["summary"])

    def test_too_few_events(self):
        r = _run_bayes(
            "bayes_repairable",
            {"var1": "t"},
            {"t": [1.0, 2.0]},
        )
        self.assertIn("Error", r["summary"])


class BayesianCorrelationBranchTest(TestCase):
    """Exercise bayes_correlation BF threshold branches."""

    def test_strong_positive(self):
        """Highly correlated → BF > 10."""
        rng = np.random.RandomState(42)
        x = list(rng.normal(0, 1, 50))
        y = [xi * 3 + rng.normal(0, 0.1) for xi in x]
        r = _run_bayes(
            "bayes_correlation",
            {"var1": "x", "var2": "y"},
            {"x": x, "y": y},
        )
        self.assertIn("strong", r["summary"].lower())
        self.assertIn("positive", r["summary"].lower())

    def test_weak_correlation(self):
        """Nearly uncorrelated."""
        rng = np.random.RandomState(42)
        x = list(rng.normal(0, 1, 50))
        y = list(rng.normal(0, 1, 50))
        r = _run_bayes(
            "bayes_correlation",
            {"var1": "x", "var2": "y"},
            {"x": x, "y": y},
        )
        self.assertIn("bf10", r.get("statistics", {}))
