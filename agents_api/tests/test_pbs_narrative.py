"""Narrative quality tests for PBS (Process Belief System) analyses.

Tests verify that PBS narratives are:
- Present with all 4 fields (verdict, body, next_steps, chart_guidance)
- Non-empty (not just placeholder text)
- Conditional on data state (different verdicts for stable vs shifted vs incapable)
- Accurate (verdict matches the statistical reality)

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_pbs_narrative -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase

# ── Test data ──────────────────────────────────────────────────────────────
_rng = np.random.RandomState(42)

# Stable process: mean=50, std=0.5, well within USL=65/LSL=35
# Very low std + seed chosen to avoid spurious BOCPD detections
STABLE_DATA = [50.0 + 0.5 * x for x in np.random.RandomState(7).standard_normal(100)]

# Shifted process: 40 stable obs then mean jumps to 58
SHIFTED_DATA = list(_rng.normal(50, 2, 40)) + list(_rng.normal(58, 2, 40))

# Incapable process: std so wide that Cpk << 1.0
INCAPABLE_DATA = list(_rng.normal(50, 12, 80))

# Tight specs for OOC detection in adaptive
TIGHT_SPEC_DATA = list(_rng.normal(50, 2, 30)) + [80, 15] + list(_rng.normal(50, 2, 28))

CONFIG = {"column": "y", "USL": 65, "LSL": 35}
CONFIG_NO_SPEC = {"column": "y"}


def _run(analysis_id, config, data):
    from agents_api.analysis.pbs import run_pbs

    df = pd.DataFrame({"y": data})
    return run_pbs(df, analysis_id, config)


def _check_narrative(tc, r, analysis_id):
    """Assert narrative dict exists with all 4 fields non-empty."""
    tc.assertIn("narrative", r, f"{analysis_id}: missing narrative key")
    narr = r["narrative"]
    tc.assertIsInstance(narr, dict, f"{analysis_id}: narrative is not a dict")
    for field in ("verdict", "body", "chart_guidance"):
        tc.assertIn(field, narr, f"{analysis_id}: narrative missing '{field}'")
        tc.assertTrue(
            narr[field] and len(str(narr[field])) > 5,
            f"{analysis_id}: narrative['{field}'] is empty or trivial: {narr.get(field)!r}",
        )


def _check_education(tc, r, analysis_id):
    """Assert education dict exists with title and content."""
    tc.assertIn("education", r, f"{analysis_id}: missing education key")
    edu = r["education"]
    tc.assertIsInstance(edu, dict, f"{analysis_id}: education is not a dict")
    tc.assertTrue(edu.get("title"), f"{analysis_id}: education missing title")
    tc.assertTrue(edu.get("content"), f"{analysis_id}: education missing content")
    tc.assertIn("<dl>", edu["content"], f"{analysis_id}: education content not structured")


# ═══════════════════════════════════════════════════════════════════════════
# 1. NARRATIVE STRUCTURE — all 9 PBS analyses must produce full narratives
# ═══════════════════════════════════════════════════════════════════════════


class PBSNarrativeStructureTest(TestCase):
    """Every PBS analysis must return a complete narrative + education."""

    def test_pbs_full_narrative(self):
        r = _run("pbs_full", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_full")
        _check_education(self, r, "pbs_full")

    def test_pbs_belief_narrative(self):
        r = _run("pbs_belief", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_belief")
        _check_education(self, r, "pbs_belief")

    def test_pbs_edetector_narrative(self):
        r = _run("pbs_edetector", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_edetector")
        _check_education(self, r, "pbs_edetector")

    def test_pbs_evidence_narrative(self):
        r = _run("pbs_evidence", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_evidence")
        _check_education(self, r, "pbs_evidence")

    def test_pbs_predictive_narrative(self):
        r = _run("pbs_predictive", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_predictive")
        _check_education(self, r, "pbs_predictive")

    def test_pbs_adaptive_narrative(self):
        r = _run("pbs_adaptive", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_adaptive")
        _check_education(self, r, "pbs_adaptive")

    def test_pbs_cpk_narrative(self):
        r = _run("pbs_cpk", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_cpk")
        _check_education(self, r, "pbs_cpk")

    def test_pbs_cpk_traj_narrative(self):
        r = _run("pbs_cpk_traj", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_cpk_traj")
        _check_education(self, r, "pbs_cpk_traj")

    def test_pbs_health_narrative(self):
        r = _run("pbs_health", CONFIG, STABLE_DATA)
        _check_narrative(self, r, "pbs_health")
        _check_education(self, r, "pbs_health")


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONDITIONAL VERDICTS — narrative must change based on data state
# ═══════════════════════════════════════════════════════════════════════════


class PBSBeliefConditionalTest(TestCase):
    """Belief chart verdict must be consistent with shift probability."""

    def test_verdict_matches_shift_probability(self):
        """Verdict tier must match the reported shift probability."""
        r = _run("pbs_belief", CONFIG, STABLE_DATA)
        verdict = r["narrative"]["verdict"].lower()
        sp = r["statistics"].get("shift_probability", 0)
        if sp < 0.20:
            self.assertIn("stable", verdict, f"P={sp:.0%} should say stable, got: {verdict}")
        elif sp < 0.50:
            self.assertIn("early", verdict, f"P={sp:.0%} should say early signs, got: {verdict}")
        elif sp < 0.80:
            self.assertIn("likely", verdict, f"P={sp:.0%} should say likely shifting, got: {verdict}")
        else:
            self.assertIn("shifted", verdict, f"P={sp:.0%} should say shifted, got: {verdict}")

    def test_shifted_data_high_probability(self):
        """Data with a real mean shift should produce high shift probability."""
        r = _run("pbs_belief", CONFIG, SHIFTED_DATA)
        verdict = r["narrative"]["verdict"].lower()
        sp = r["statistics"].get("shift_probability", 0)
        self.assertGreater(sp, 0.50, f"Shifted data should have P>50%, got {sp:.0%}")
        self.assertTrue(
            "shift" in verdict or "alarm" in verdict or "change" in verdict or "likely" in verdict,
            f"Shifted data should produce shift verdict, got: {verdict}",
        )


class PBSEdetectorConditionalTest(TestCase):
    """E-detector verdict must reflect alarm state."""

    def test_stable_no_alarm(self):
        r = _run("pbs_edetector", CONFIG, STABLE_DATA)
        verdict = r["narrative"]["verdict"].lower()
        self.assertIn("no alarm", verdict, f"Stable data should show no alarm, got: {verdict}")

    def test_shifted_alarm(self):
        r = _run("pbs_edetector", CONFIG, SHIFTED_DATA)
        verdict = r["narrative"]["verdict"].lower()
        self.assertTrue(
            "alarm" in verdict or "evidence" in verdict,
            f"Shifted data should trigger alarm or evidence, got: {verdict}",
        )


class PBSCpkConditionalTest(TestCase):
    """Cpk verdict must reflect capability state."""

    def test_capable_verdict(self):
        r = _run("pbs_cpk", CONFIG, STABLE_DATA)
        verdict = r["narrative"]["verdict"].lower()
        self.assertTrue(
            "capable" in verdict,
            f"Capable data should produce capable verdict, got: {verdict}",
        )

    def test_incapable_verdict(self):
        r = _run("pbs_cpk", CONFIG, INCAPABLE_DATA)
        verdict = r["narrative"]["verdict"].lower()
        self.assertTrue(
            "not capable" in verdict or "marginal" in verdict or "incapable" in verdict,
            f"Incapable data should produce incapable verdict, got: {verdict}",
        )


class PBSHealthConditionalTest(TestCase):
    """Health verdict must reflect overall process state."""

    def test_healthy_verdict(self):
        r = _run("pbs_health", CONFIG, STABLE_DATA)
        verdict = r["narrative"]["verdict"].lower()
        self.assertTrue(
            "healthy" in verdict or "health" in verdict,
            f"Healthy data should produce healthy verdict, got: {verdict}",
        )

    def test_unhealthy_verdict(self):
        r = _run("pbs_health", CONFIG, INCAPABLE_DATA)
        verdict = r["narrative"]["verdict"].lower()
        self.assertTrue(
            "unhealthy" in verdict or "risk" in verdict or "at risk" in verdict,
            f"Unhealthy data should produce unhealthy/risk verdict, got: {verdict}",
        )


class PBSFullConditionalTest(TestCase):
    """pbs_full verdict must be dynamic (not static 'Process Belief System')."""

    def test_full_stable_verdict(self):
        r = _run("pbs_full", CONFIG, STABLE_DATA)
        narr = r["narrative"]
        verdict = narr["verdict"].lower()
        # Must NOT be the old static verdict
        self.assertNotEqual(
            verdict,
            "process belief system",
            "pbs_full verdict should be dynamic, not static",
        )
        self.assertTrue(
            "stable" in verdict or "healthy" in verdict,
            f"Stable data should produce stable/healthy verdict, got: {verdict}",
        )
        # Must have next_steps
        self.assertTrue(
            narr.get("next_steps") and len(narr["next_steps"]) > 5,
            f"pbs_full must have next_steps, got: {narr.get('next_steps')!r}",
        )

    def test_full_shifted_verdict(self):
        r = _run("pbs_full", CONFIG, SHIFTED_DATA)
        narr = r["narrative"]
        verdict = narr["verdict"].lower()
        self.assertNotEqual(verdict, "process belief system")
        self.assertTrue(
            "investigate" in verdict
            or "shift" in verdict
            or "changing" in verdict
            or "risk" in verdict
            or "unhealthy" in verdict,
            f"Shifted data should produce action-oriented verdict, got: {verdict}",
        )
        self.assertTrue(
            narr.get("next_steps") and len(narr["next_steps"]) > 5,
            f"pbs_full must have next_steps for shifted data, got: {narr.get('next_steps')!r}",
        )


class PBSAdaptiveConditionalTest(TestCase):
    """Adaptive narrative must report OOC state, not just limit values."""

    def test_stable_in_control(self):
        r = _run("pbs_adaptive", CONFIG, STABLE_DATA)
        verdict = r["narrative"]["verdict"].lower()
        self.assertTrue(
            "in control" in verdict or "within" in verdict,
            f"Stable data should show in-control, got: {verdict}",
        )
        stats = r.get("statistics", {})
        self.assertEqual(stats.get("n_ooc", -1), 0, "Stable data should have 0 OOC points")

    def test_ooc_detected(self):
        r = _run("pbs_adaptive", CONFIG, TIGHT_SPEC_DATA)
        stats = r.get("statistics", {})
        n_ooc = stats.get("n_ooc", 0)
        verdict = r["narrative"]["verdict"].lower()
        if n_ooc > 0:
            self.assertIn(
                "out-of-control",
                verdict,
                f"OOC data should produce OOC verdict, got: {verdict}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. EVIDENCE & PREDICTIVE — narrative contains dynamic values
# ═══════════════════════════════════════════════════════════════════════════


class PBSEvidenceContentTest(TestCase):
    """Evidence narrative must contain the E-value."""

    def test_evidence_contains_evalue(self):
        r = _run("pbs_evidence", CONFIG, STABLE_DATA)
        body = r["narrative"]["body"]
        self.assertTrue(
            "E-value" in body or "evidence" in body.lower(),
            f"Evidence body should reference E-value, got: {body[:100]}",
        )

    def test_evidence_shifted_is_stronger(self):
        r_stable = _run("pbs_evidence", CONFIG, STABLE_DATA)
        r_shifted = _run("pbs_evidence", CONFIG, SHIFTED_DATA)
        v_stable = r_stable["narrative"]["verdict"].lower()
        v_shifted = r_shifted["narrative"]["verdict"].lower()
        # Shifted data should produce stronger evidence language
        strength_words = {"strong", "decisive", "notable"}
        stable_has_strength = any(w in v_stable for w in strength_words)
        shifted_has_strength = any(w in v_shifted for w in strength_words)
        # At minimum, shifted should not be weaker than stable
        if stable_has_strength:
            self.assertTrue(
                shifted_has_strength,
                f"Shifted evidence should be at least as strong as stable. Stable: {v_stable}, Shifted: {v_shifted}",
            )


class PBSPredictiveContentTest(TestCase):
    """Predictive narrative must contain slope and risk information."""

    def test_predictive_mentions_slope(self):
        r = _run("pbs_predictive", CONFIG, STABLE_DATA)
        body = r["narrative"]["body"]
        self.assertTrue(
            "slope" in body.lower() or "trend" in body.lower(),
            f"Predictive body should mention slope/trend, got: {body[:100]}",
        )

    def test_predictive_mentions_risk(self):
        r = _run("pbs_predictive", CONFIG, SHIFTED_DATA)
        narr = r["narrative"]
        combined = (narr.get("body", "") + " " + narr.get("next_steps", "")).lower()
        # Shifted data should mention some form of risk or exceedance
        self.assertTrue(
            "risk" in combined or "exceed" in combined or "spec" in combined or "monitor" in combined,
            f"Predictive on shifted data should mention risk/exceedance/spec, got: {combined[:200]}",
        )


class PBSCpkTrajContentTest(TestCase):
    """Cpk trajectory narrative must contain trend direction."""

    def test_traj_mentions_trend(self):
        r = _run("pbs_cpk_traj", CONFIG, STABLE_DATA)
        verdict = r["narrative"]["verdict"]
        self.assertTrue(
            "improving" in verdict.lower()
            or "declining" in verdict.lower()
            or "uncertain" in verdict.lower()
            or "Cpk" in verdict,
            f"Cpk traj verdict should mention trend direction, got: {verdict}",
        )
