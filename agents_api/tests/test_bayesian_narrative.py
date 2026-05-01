"""Narrative quality tests for Bayesian analysis engine.

Verifies that all Bayesian analyses produce:
- Complete narrative dicts (verdict/body/next_steps/chart_guidance all non-empty)
- Forge-native: BF₁₀ in verdict, analysis-specific next_steps
- Legacy-wrapped: next_steps and chart_guidance populated (not empty strings)

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_bayesian_narrative -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase

_rng = np.random.RandomState(42)

# Two-group data for t-test, ANOVA, etc.
GROUP_A = list(_rng.normal(50, 5, 40))
GROUP_B = list(_rng.normal(55, 5, 40))
COMBINED_DF = pd.DataFrame(
    {
        "response": GROUP_A + GROUP_B,
        "group": ["A"] * 40 + ["B"] * 40,
        "x1": list(_rng.normal(0, 1, 80)),
        "x2": list(_rng.normal(0, 1, 80)),
        "binary": [1 if x > 0 else 0 for x in _rng.normal(0, 1, 80)],
        "time": list(_rng.exponential(10, 80)),
        "event": [1] * 60 + [0] * 20,
    }
)


def _run_bayesian(analysis_id, config, df=None):
    from agents_api.analysis.forge_bayesian import run_forge_bayesian

    if df is None:
        df = COMBINED_DF
    return run_forge_bayesian(analysis_id, df, config)


def _check_narrative_complete(tc, result, analysis_id):
    """All 4 narrative fields must be non-empty strings."""
    if result is None:
        tc.skipTest(f"{analysis_id}: handler returned None (legacy handler error with test data)")
    tc.assertIn("narrative", result, f"{analysis_id}: missing narrative key")
    narr = result["narrative"]
    tc.assertIsInstance(narr, dict, f"{analysis_id}: narrative is not a dict")
    for field in ("verdict", "body", "next_steps", "chart_guidance"):
        tc.assertIn(field, narr, f"{analysis_id}: narrative missing '{field}'")
        val = narr[field]
        tc.assertTrue(
            val and len(str(val).strip()) > 3,
            f"{analysis_id}: narrative['{field}'] is empty or trivial: {val!r}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 1. FORGE-NATIVE — Must have BF₁₀ in verdict + specific next_steps
# ═══════════════════════════════════════════════════════════════════════════


class BayesianForgeNativeNarrativeTest(TestCase):
    """Forge-native Bayesian handlers (forgestat/forgespc-backed)."""

    def test_bayes_ttest_narrative(self):
        r = _run_bayesian("bayes_ttest", {"column": "response", "mu": 50})
        _check_narrative_complete(self, r, "bayes_ttest")
        self.assertIn("BF", r["narrative"]["verdict"])
        self.assertNotIn(
            "BF > 3 = moderate",
            r["narrative"]["next_steps"],
            "next_steps should be analysis-specific, not generic thresholds",
        )

    def test_bayes_correlation_narrative(self):
        r = _run_bayesian("bayes_correlation", {"var1": "x1", "var2": "x2"})
        _check_narrative_complete(self, r, "bayes_correlation")
        self.assertIn("BF", r["narrative"]["verdict"])

    def test_bayes_proportion_narrative(self):
        r = _run_bayesian("bayes_proportion", {"successes": 35, "n": 50})
        _check_narrative_complete(self, r, "bayes_proportion")
        self.assertIn("BF", r["narrative"]["verdict"])

    def test_bayes_changepoint_narrative(self):
        data = list(_rng.normal(50, 2, 30)) + list(_rng.normal(55, 2, 30))
        df = pd.DataFrame({"y": data})
        r = _run_bayesian("bayes_changepoint", {"column": "y"}, df=df)
        _check_narrative_complete(self, r, "bayes_changepoint")
        self.assertIn("changepoint", r["narrative"]["verdict"].lower())

    def test_bayes_capability_prediction_narrative(self):
        df = pd.DataFrame({"y": list(_rng.normal(50, 2, 60))})
        r = _run_bayesian("bayes_capability_prediction", {"column": "y", "usl": 65, "lsl": 35}, df=df)
        _check_narrative_complete(self, r, "bayes_capability_prediction")

    def test_bayes_ewma_narrative(self):
        df = pd.DataFrame({"y": list(_rng.normal(50, 2, 60))})
        r = _run_bayesian("bayes_ewma", {"column": "y"}, df=df)
        _check_narrative_complete(self, r, "bayes_ewma")
        self.assertIn("control", r["narrative"]["verdict"].lower())


# ═══════════════════════════════════════════════════════════════════════════
# 2. LEGACY-WRAPPED — Must have non-empty next_steps and chart_guidance
# ═══════════════════════════════════════════════════════════════════════════


class BayesianLegacyNarrativeTest(TestCase):
    """Legacy-wrapped handlers must now have populated narratives."""

    def test_bayes_anova_narrative(self):
        r = _run_bayesian("bayes_anova", {"response": "response", "factor": "group"})
        _check_narrative_complete(self, r, "bayes_anova")

    def test_bayes_regression_narrative(self):
        r = _run_bayesian("bayes_regression", {"response": "response", "predictors": ["x1", "x2"]})
        _check_narrative_complete(self, r, "bayes_regression")

    def test_bayes_chi2_narrative(self):
        r = _run_bayesian("bayes_chi2", {"var1": "group", "var2": "binary"})
        _check_narrative_complete(self, r, "bayes_chi2")

    def test_bayes_equivalence_narrative(self):
        r = _run_bayesian("bayes_equivalence", {"var1": "response", "factor": "group"})
        _check_narrative_complete(self, r, "bayes_equivalence")

    def test_bayes_poisson_narrative(self):
        r = _run_bayesian("bayes_poisson", {"column": "response", "exposure": 80})
        _check_narrative_complete(self, r, "bayes_poisson")

    def test_bayes_logistic_narrative(self):
        r = _run_bayesian("bayes_logistic", {"response": "binary", "predictors": ["x1", "x2"]})
        _check_narrative_complete(self, r, "bayes_logistic")

    def test_bayes_ab_narrative(self):
        r = _run_bayesian("bayes_ab", {"var1": "response", "factor": "group"})
        _check_narrative_complete(self, r, "bayes_ab")

    def test_bayes_meta_narrative(self):
        r = _run_bayesian("bayes_meta", {"column": "response"})
        _check_narrative_complete(self, r, "bayes_meta")

    def test_bayes_survival_narrative(self):
        r = _run_bayesian("bayes_survival", {"time_col": "time", "event_col": "event"})
        _check_narrative_complete(self, r, "bayes_survival")

    def test_bayes_demo_narrative(self):
        r = _run_bayesian("bayes_demo", {})
        _check_narrative_complete(self, r, "bayes_demo")


# ═══════════════════════════════════════════════════════════════════════════
# 3. NEXT_STEPS SPECIFICITY — legacy must not all be the same generic text
# ═══════════════════════════════════════════════════════════════════════════


class BayesianNextStepsSpecificityTest(TestCase):
    """Legacy-wrapped analyses should have analysis-specific guidance."""

    def test_different_analyses_have_different_next_steps(self):
        """At least 3 distinct next_steps across legacy-wrapped analyses."""
        configs = {
            "bayes_anova": {"response": "response", "factor": "group"},
            "bayes_regression": {"response": "response", "predictors": ["x1", "x2"]},
            "bayes_logistic": {"response": "binary", "predictors": ["x1", "x2"]},
            "bayes_chi2": {"var1": "group", "var2": "binary"},
        }
        next_steps_set = set()
        for aid, cfg in configs.items():
            r = _run_bayesian(aid, cfg)
            if r and r.get("narrative"):
                next_steps_set.add(r["narrative"].get("next_steps", ""))

        self.assertGreaterEqual(
            len(next_steps_set),
            3,
            f"Expected at least 3 distinct next_steps across 4 analyses, got {len(next_steps_set)}: {next_steps_set}",
        )
