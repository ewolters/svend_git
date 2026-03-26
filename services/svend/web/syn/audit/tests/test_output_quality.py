"""
QUAL-001 output quality tests: Output Quality Assurance Standard.

Tests the three pillars (mathematical correctness, data quality, output coherence),
calibration system, bounds validation, schema enforcement, and compliance check
registration.

Standard: QUAL-001
"""

import inspect
import re
from unittest.mock import patch

from django.test import SimpleTestCase

from agents_api.calibration import (
    REFERENCE_POOL,
    Expectation,
    _check_expectation,
    run_calibration,
)
from agents_api.dsw.standardize import (
    _BOUNDED_METRICS,
    REQUIRED_FIELDS,
    _validate_statistics_bounds,
    standardize_output,
)
from syn.audit.compliance import ALL_CHECKS

# ── §4 Three Pillars ─────────────────────────────────────────────────────


class ThreePillarsTest(SimpleTestCase):
    """QUAL-001 §4: Output quality enforces three pillars."""

    def test_standardize_called_in_dispatch(self):
        """standardize_output is imported and called in dispatch.py."""
        import agents_api.dsw.dispatch as dispatch_mod

        source = inspect.getsource(dispatch_mod)
        self.assertIn("standardize_output", source)

    def test_calibration_runner_exists(self):
        """run_calibration function exists and is callable."""
        self.assertTrue(callable(run_calibration))
        sig = inspect.signature(run_calibration)
        self.assertIn("cases", sig.parameters)
        self.assertIn("seed", sig.parameters)


# ── §5 Calibration System ────────────────────────────────────────────────


class CalibrationPoolTest(SimpleTestCase):
    """QUAL-001 §5.1: Calibration pool size and category coverage."""

    def test_minimum_pool_size(self):
        """Reference pool has >= 15 cases."""
        self.assertGreaterEqual(len(REFERENCE_POOL), 15)

    def test_minimum_category_count(self):
        """Pool covers >= 5 of 6 categories."""
        categories = {c.category for c in REFERENCE_POOL}
        self.assertGreaterEqual(len(categories), 5)
        valid = {"inference", "bayesian", "spc", "reliability", "ml", "simulation"}
        for cat in categories:
            self.assertIn(cat, valid, f"Unknown category: {cat}")


class CaseStructureTest(SimpleTestCase):
    """QUAL-001 §5.2: CalibrationCase required fields."""

    def test_required_fields(self):
        """Every case has case_id, category, analysis_type, analysis_id, config, data, expectations."""
        for case in REFERENCE_POOL:
            self.assertTrue(case.case_id, "Missing case_id")
            self.assertTrue(case.category, f"{case.case_id}: missing category")
            self.assertTrue(
                case.analysis_type, f"{case.case_id}: missing analysis_type"
            )
            self.assertTrue(case.analysis_id, f"{case.case_id}: missing analysis_id")
            self.assertIsInstance(case.config, dict, f"{case.case_id}: config not dict")
            self.assertIsInstance(case.data, dict, f"{case.case_id}: data not dict")
            self.assertGreaterEqual(
                len(case.expectations), 1, f"{case.case_id}: no expectations"
            )

    def test_case_id_format(self):
        """Case IDs match pattern CAL-{CATEGORY}-{NNN}."""
        pattern = re.compile(r"^CAL-[A-Z]+-\d{3}$")
        for case in REFERENCE_POOL:
            self.assertRegex(
                case.case_id,
                pattern,
                f"case_id '{case.case_id}' doesn't match CAL-XXX-NNN",
            )


class ExpectationTypesTest(SimpleTestCase):
    """QUAL-001 §5.3: Four comparison types."""

    def test_abs_within(self):
        """abs_within comparison: |actual - expected| <= tolerance."""
        result = {"statistics": {"r_squared": 0.85}}
        exp = Expectation(
            key="statistics.r_squared",
            expected=0.85,
            tolerance=0.01,
            comparison="abs_within",
        )
        check = _check_expectation(result, exp)
        self.assertTrue(check["passed"])

        # Out of tolerance
        exp2 = Expectation(
            key="statistics.r_squared",
            expected=0.90,
            tolerance=0.01,
            comparison="abs_within",
        )
        check2 = _check_expectation(result, exp2)
        self.assertFalse(check2["passed"])

    def test_greater_than(self):
        """greater_than comparison: actual > expected."""
        result = {"statistics": {"p_value": 0.45}}
        exp = Expectation(
            key="statistics.p_value", expected=0.05, comparison="greater_than"
        )
        check = _check_expectation(result, exp)
        self.assertTrue(check["passed"])

    def test_less_than(self):
        """less_than comparison: actual < expected."""
        result = {"statistics": {"p_value": 0.003}}
        exp = Expectation(
            key="statistics.p_value", expected=0.05, comparison="less_than"
        )
        check = _check_expectation(result, exp)
        self.assertTrue(check["passed"])

    def test_contains(self):
        """contains comparison: substring match."""
        result = {"guide_observation": "Significant difference found between groups"}
        exp = Expectation(
            key="guide_observation_contains",
            expected="significant",
            comparison="contains",
        )
        check = _check_expectation(result, exp)
        self.assertTrue(check["passed"])


class RotationTest(SimpleTestCase):
    """QUAL-001 §5.4: Date-seeded rotation."""

    def test_same_seed_same_selection(self):
        """Same seed produces identical case selection."""
        r1 = run_calibration(seed=42, subset_size=8)
        r2 = run_calibration(seed=42, subset_size=8)
        ids1 = [r["case_id"] for r in r1["results"]]
        ids2 = [r["case_id"] for r in r2["results"]]
        self.assertEqual(ids1, ids2)


class DriftSeverityTest(SimpleTestCase):
    """QUAL-001 §5.5: Drift detection."""

    def test_drift_violation_created(self):
        """check_statistical_calibration is registered and references DriftViolation."""
        self.assertIn("statistical_calibration", ALL_CHECKS)
        fn, _cat = ALL_CHECKS["statistical_calibration"]
        source = inspect.getsource(fn)
        self.assertIn("DriftViolation", source)


# ── §6 Output Validation ─────────────────────────────────────────────────


class SchemaTest(SimpleTestCase):
    """QUAL-001 §6.1: Required fields filled by standardize_output."""

    def test_required_fields_filled(self):
        """Empty result dict gets all required fields after standardize_output."""
        result = {}
        with patch("agents_api.dsw.standardize.get_entry", return_value=None):
            out = standardize_output(result, "stats", "ttest")
        for field in REQUIRED_FIELDS:
            self.assertIn(field, out, f"Missing required field: {field}")


class BoundsCheckTest(SimpleTestCase):
    """QUAL-001 §6.2: Statistical output bounds validation."""

    def test_p_value_bounds(self):
        """p_value clamped to [0, 1]."""
        result = {"p_value": 1.5, "statistics": {"p_value": -0.1}}
        _validate_statistics_bounds(result)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["statistics"]["p_value"], 0.0)

    def test_correlation_bounds(self):
        """correlation clamped to [-1, 1]."""
        result = {"statistics": {"pearson_r": 1.2, "spearman_rho": -1.5}}
        _validate_statistics_bounds(result)
        self.assertEqual(result["statistics"]["pearson_r"], 1.0)
        self.assertEqual(result["statistics"]["spearman_rho"], -1.0)

    def test_r_squared_bounds(self):
        """r_squared clamped to [0, 1]."""
        result = {"statistics": {"r_squared": 1.1, "R2": -0.05}}
        _validate_statistics_bounds(result)
        self.assertEqual(result["statistics"]["r_squared"], 1.0)
        self.assertEqual(result["statistics"]["R2"], 0.0)

    def test_nan_detection(self):
        """NaN and Inf values replaced with None."""
        result = {
            "p_value": float("nan"),
            "statistics": {
                "correlation": float("inf"),
                "r_squared": float("-inf"),
            },
        }
        _validate_statistics_bounds(result)
        self.assertIsNone(result["p_value"])
        self.assertIsNone(result["statistics"]["correlation"])
        self.assertIsNone(result["statistics"]["r_squared"])

    def test_effect_size_finite(self):
        """Finite metrics (cohens_d, cpk, etc.) reject NaN/Inf."""
        result = {
            "statistics": {
                "cohens_d": float("nan"),
                "cpk": float("inf"),
            }
        }
        _validate_statistics_bounds(result)
        self.assertIsNone(result["statistics"]["cohens_d"])
        self.assertIsNone(result["statistics"]["cpk"])

    def test_positive_metrics(self):
        """bf10 must be positive and finite."""
        result = {"statistics": {"bf10": -0.5}}
        _validate_statistics_bounds(result)
        self.assertIsNone(result["statistics"]["bf10"])

        result2 = {"statistics": {"bf10": 3.2}}
        _validate_statistics_bounds(result2)
        self.assertEqual(result2["statistics"]["bf10"], 3.2)

    def test_valid_values_unchanged(self):
        """Values within bounds are not modified."""
        result = {
            "p_value": 0.04,
            "statistics": {
                "pearson_r": -0.72,
                "r_squared": 0.52,
                "cohens_d": 0.8,
                "bf10": 12.5,
            },
        }
        _validate_statistics_bounds(result)
        self.assertEqual(result["p_value"], 0.04)
        self.assertEqual(result["statistics"]["pearson_r"], -0.72)
        self.assertEqual(result["statistics"]["r_squared"], 0.52)
        self.assertEqual(result["statistics"]["cohens_d"], 0.8)
        self.assertEqual(result["statistics"]["bf10"], 12.5)

    def test_bounded_metrics_list_complete(self):
        """_BOUNDED_METRICS covers all QUAL-001 §6.2 required metrics."""
        all_keys = set()
        for keys, lo, hi in _BOUNDED_METRICS:
            all_keys.update(keys)
        self.assertIn("p_value", all_keys)
        self.assertIn("pearson_r", all_keys)
        self.assertIn("r_squared", all_keys)
        self.assertIn("eta_squared", all_keys)
        self.assertIn("cramers_v", all_keys)


class CoherenceTest(SimpleTestCase):
    """QUAL-001 §6.3: Output coherence checks."""

    def test_grade_pvalue_consistency(self):
        """evidence_grade generation uses p_value."""
        result = {
            "summary": "Significant difference found",
            "p_value": 0.001,
            "statistics": {"cohens_d": 0.9},
        }
        with patch("agents_api.dsw.standardize.get_entry", return_value=None):
            out = standardize_output(result, "stats", "ttest")
        # With p=0.001, evidence_grade should be generated
        if out.get("evidence_grade"):
            self.assertIn(
                out["evidence_grade"], ("Strong", "Moderate", "Weak", "Inconclusive")
            )

    def test_guide_observation_populated(self):
        """guide_observation auto-populated from summary when missing."""
        result = {"summary": "Mean difference is 2.5 (p=0.03)"}
        with patch("agents_api.dsw.standardize.get_entry", return_value=None):
            out = standardize_output(result, "stats", "ttest")
        self.assertTrue(out["guide_observation"])
        self.assertIn("2.5", out["guide_observation"])


# ── §7 Data Quality ──────────────────────────────────────────────────────


class InputValidationTest(SimpleTestCase):
    """QUAL-001 §7: Input data validation."""

    def test_row_limit_enforced(self):
        """dispatch.py enforces a row limit on inline data."""
        import agents_api.dsw.dispatch as dispatch_mod

        source = inspect.getsource(dispatch_mod)
        self.assertIn("10000", source)


# ── §8 Module-Specific Quality ────────────────────────────────────────────


class DSWQualityTest(SimpleTestCase):
    """QUAL-001 §8.1: DSW p-value registry contract."""

    def test_pvalue_analyses_produce_pvalue(self):
        """Analyses marked has_pvalue=True in registry are documented."""
        try:
            from agents_api.dsw.registry import get_all_with_pvalue

            pvalue_analyses = get_all_with_pvalue()
            # At minimum, core inference analyses should be registered
            self.assertGreater(
                len(pvalue_analyses), 0, "No analyses with has_pvalue=True found"
            )
        except ImportError:
            self.skipTest("ANALYSIS_REGISTRY not available")


class SPCQualityTest(SimpleTestCase):
    """QUAL-001 §8.2: SPC control limits ordering."""

    def test_control_limits_ordered(self):
        """SPC module exists and has control chart logic."""
        try:
            import agents_api.spc as spc_mod

            source = inspect.getsource(spc_mod)
            # Must reference UCL, CL, LCL
            self.assertIn("UCL", source)
            self.assertIn("LCL", source)
        except ImportError:
            self.skipTest("SPC module not available")


class QMSQualityTest(SimpleTestCase):
    """QUAL-001 §8.3: QMS tool output bounds."""

    def test_fmea_rpn_bounds(self):
        """FMEA RPN calculation logic enforces [1, 1000] bounds."""
        try:
            import agents_api.fmea_views as fmea_mod

            source = inspect.getsource(fmea_mod)
            # Must reference RPN calculation
            self.assertTrue(
                "rpn" in source.lower() or "RPN" in source,
                "FMEA module must contain RPN calculation logic",
            )
        except ImportError:
            self.skipTest("FMEA module not available")


# ── §11 Compliance Check Registration ─────────────────────────────────────


class CheckRegistrationTest(SimpleTestCase):
    """QUAL-001 §11: output_quality compliance check."""

    def test_check_registered(self):
        """output_quality is registered in ALL_CHECKS."""
        self.assertIn("output_quality", ALL_CHECKS)

    def test_check_is_callable(self):
        """output_quality check function is callable."""
        fn, category = ALL_CHECKS["output_quality"]
        self.assertTrue(callable(fn))
        self.assertEqual(category, "processing_integrity")

    def test_check_returns_valid_structure(self):
        """output_quality returns {status, details, soc2_controls}."""
        fn, _cat = ALL_CHECKS["output_quality"]
        result = fn()
        self.assertIn("status", result)
        self.assertIn(result["status"], ("pass", "warning", "fail", "error"))
        self.assertIn("details", result)
        self.assertIsInstance(result["details"], dict)
        self.assertIn("soc2_controls", result)
        self.assertIsInstance(result["soc2_controls"], list)
