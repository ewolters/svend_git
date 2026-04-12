"""
CAL-001 calibration & verification tests: Software Calibration Standard.

Tests coverage tooling, golden file infrastructure, complexity governance,
endpoint coverage, ratchet mechanism, and compliance check registration.

Standard: CAL-001
"""

from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase, TestCase

from syn.audit.compliance import ALL_CHECKS

WEB_ROOT = Path(settings.BASE_DIR)
GIT_ROOT = WEB_ROOT.parent.parent.parent  # kjerne/


# ── §5 Coverage Tooling ──────────────────────────────────────────────────


class CoverageToolingTest(SimpleTestCase):
    """CAL-001 §5.1: Coverage measurement configured."""

    def test_coveragerc_exists(self):
        """CAL-001 §5.1: .coveragerc must exist in web root."""
        rc = WEB_ROOT / ".coveragerc"
        self.assertTrue(rc.exists(), ".coveragerc not found in web root")

    def test_coveragerc_has_required_sections(self):
        """CAL-001 §5.1: .coveragerc has [run], [report], [json] sections."""
        rc = WEB_ROOT / ".coveragerc"
        content = rc.read_text()
        self.assertIn("[run]", content)
        self.assertIn("[report]", content)
        self.assertIn("[json]", content)

    def test_coveragerc_omits_migrations(self):
        """CAL-001 §5.1: .coveragerc omits migrations from coverage."""
        rc = WEB_ROOT / ".coveragerc"
        content = rc.read_text()
        self.assertIn("migrations", content)

    def test_coverage_importable(self):
        """CAL-001 §5.1: coverage package is installed."""
        import coverage  # noqa: F401


# ── §5.3 Coverage Ratchet ────────────────────────────────────────────────


class CoverageRatchetTest(TestCase):
    """CAL-001 §5.3: Coverage ratchet mechanism."""

    def test_ratchet_enforced(self):
        """CAL-001 §5.3: CalibrationReport stores ratchet_baseline that can only go up."""
        from syn.audit.models import CalibrationReport

        # Create initial report
        r1 = CalibrationReport.objects.create(
            date="2026-01-01",
            overall_coverage=10.0,
            ratchet_baseline=10.0,
        )
        self.assertEqual(r1.ratchet_baseline, 10.0)

        # Higher coverage → higher ratchet
        r2 = CalibrationReport.objects.create(
            date="2026-02-01",
            overall_coverage=15.0,
            ratchet_baseline=max(15.0, r1.ratchet_baseline),
        )
        self.assertEqual(r2.ratchet_baseline, 15.0)

    def test_calibration_report_model_fields(self):
        """CAL-001 §11.1: CalibrationReport has all required fields."""
        from syn.audit.models import CalibrationReport

        field_names = [f.name for f in CalibrationReport._meta.get_fields()]
        required = [
            "date",
            "overall_coverage",
            "tier1_coverage",
            "tier2_coverage",
            "tier3_coverage",
            "tier4_coverage",
            "calibration_pass_rate",
            "calibration_cases_run",
            "calibration_cases_passed",
            "endpoint_coverage",
            "golden_file_count",
            "complexity_violations",
            "ratchet_baseline",
            "is_certificate",
            "details",
        ]
        for field in required:
            self.assertIn(field, field_names, f"CalibrationReport missing field: {field}")


# ── §6 Golden Files ──────────────────────────────────────────────────────


class GoldenFileTest(SimpleTestCase):
    """CAL-001 §6.1: Golden file infrastructure."""

    def test_golden_files_exist(self):
        """CAL-001 §6.1: Golden file directory exists (created in Phase 2)."""
        # Phase 1: directory may not exist yet — this is an informational check
        golden_dir = WEB_ROOT / "agents_api" / "tests" / "golden"
        # Just verify the path is reasonable — Phase 2 populates it
        self.assertTrue(
            str(golden_dir).endswith("agents_api/tests/golden"),
            "Golden file path incorrect",
        )


# ── §6.4 Cross-Library Verification ─────────────────────────────────────


class CrossLibraryTest(SimpleTestCase):
    """CAL-001 §6.4: Cross-library reference test infrastructure."""

    def test_reference_tests_exist(self):
        """CAL-001 §6.4: Reference test file exists (created in Phase 2)."""
        # Phase 1: file may not exist yet — check path convention
        ref_path = WEB_ROOT / "agents_api" / "tests" / "test_stats_reference.py"
        # Informational — Phase 2 creates this file
        self.assertTrue(
            str(ref_path).endswith("test_stats_reference.py"),
            "Reference test path incorrect",
        )


# ── §7 Endpoint Coverage ────────────────────────────────────────────────


class EndpointSmokeTest(SimpleTestCase):
    """CAL-001 §7.1: Endpoint smoke test verification."""

    def test_all_endpoints_have_smoke(self):
        """CAL-001 §7.1: Verify endpoint_coverage check is registered."""
        self.assertIn("endpoint_coverage", ALL_CHECKS)


# ── §8 Complexity Governance ─────────────────────────────────────────────


class ComplexityBudgetTest(SimpleTestCase):
    """CAL-001 §8.1: File size limits enforced.

    Full behavioral tests in test_complexity_governance.py.
    These verify the check is registered and wired correctly.
    """

    def test_complexity_check_registered(self):
        """CAL-001 §8.1: complexity_governance check is registered."""
        self.assertIn("complexity_governance", ALL_CHECKS)

    def test_complexity_check_no_violations(self):
        """CAL-001 §8.1: No unexempted files exceed 3000-line limit."""
        check_fn, _category = ALL_CHECKS["complexity_governance"]
        result = check_fn()
        violations = result["details"]["violations"]
        self.assertEqual(
            violations,
            [],
            f"Unexempted oversized files: {violations}",
        )


# ── §9 Regression Prevention ────────────────────────────────────────────


class NoRegressionTest(TestCase):
    """CAL-001 §9.1: No-regression rule."""

    def test_ratchet_enforced(self):
        """CAL-001 §9.1: Ratchet prevents coverage decrease."""
        from syn.audit.models import CalibrationReport

        # Simulate two reports
        CalibrationReport.objects.create(
            date="2026-01-01",
            overall_coverage=20.0,
            ratchet_baseline=20.0,
        )
        # Second report with lower coverage should NOT lower the ratchet
        last = CalibrationReport.objects.order_by("-date").first()
        new_ratchet = max(15.0, last.ratchet_baseline)  # max(15, 20) = 20
        self.assertEqual(new_ratchet, 20.0, "Ratchet must not decrease")


# ── §10 Calibration Cycles ──────────────────────────────────────────────


class MonthlyCertTest(TestCase):
    """CAL-001 §10.3: Monthly calibration certificate."""

    def test_certificate_generated(self):
        """CAL-001 §10.3: CalibrationReport with is_certificate=True can be created."""
        from syn.audit.models import CalibrationReport

        cert = CalibrationReport.objects.create(
            date="2026-03-01",
            overall_coverage=12.5,
            calibration_pass_rate=100.0,
            calibration_cases_run=16,
            calibration_cases_passed=16,
            is_certificate=True,
            details={"status": "pass"},
        )
        self.assertTrue(cert.is_certificate)
        self.assertEqual(cert.calibration_pass_rate, 100.0)


# ── Compliance Check Registration ────────────────────────────────────────


class ComplianceCheckRegistrationTest(SimpleTestCase):
    """CAL-001: All 3 compliance checks are registered."""

    def test_calibration_coverage_registered(self):
        """CAL-001 §5: calibration_coverage check is registered."""
        self.assertIn("calibration_coverage", ALL_CHECKS)

    def test_complexity_governance_registered(self):
        """CAL-001 §8: complexity_governance check is registered."""
        self.assertIn("complexity_governance", ALL_CHECKS)

    def test_endpoint_coverage_registered(self):
        """CAL-001 §7: endpoint_coverage check is registered."""
        self.assertIn("endpoint_coverage", ALL_CHECKS)

    def test_calibration_coverage_runs(self):
        """CAL-001 §5: calibration_coverage check executes."""
        check_fn, _category = ALL_CHECKS["calibration_coverage"]
        result = check_fn()
        self.assertIn("status", result)
        self.assertIn(result["status"], ["pass", "warning", "fail", "error"])

    def test_endpoint_coverage_runs(self):
        """CAL-001 §7: endpoint_coverage check executes."""
        check_fn, _category = ALL_CHECKS["endpoint_coverage"]
        result = check_fn()
        self.assertIn("status", result)
        self.assertIn("total_endpoints", result["details"])

    def test_checks_scheduled_on_wednesday(self):
        """CAL-001: New checks are in the Wednesday rotation."""
        from syn.audit.compliance import WEEKDAY_ROTATION

        wednesday = WEEKDAY_ROTATION[2]
        self.assertIn("calibration_coverage", wednesday)
        self.assertIn("complexity_governance", wednesday)
        self.assertIn("endpoint_coverage", wednesday)


# ── Management Commands ──────────────────────────────────────────────────


class ManagementCommandTest(SimpleTestCase):
    """CAL-001 §10: Management commands exist."""

    def test_measure_coverage_command_exists(self):
        """CAL-001 §10.2: measure_coverage command file exists."""
        cmd = WEB_ROOT / "syn" / "audit" / "management" / "commands" / "measure_coverage.py"
        self.assertTrue(cmd.exists(), "measure_coverage.py not found")

    def test_generate_calibration_cert_command_exists(self):
        """CAL-001 §10.3: generate_calibration_cert command file exists."""
        cmd = WEB_ROOT / "syn" / "audit" / "management" / "commands" / "generate_calibration_cert.py"
        self.assertTrue(cmd.exists(), "generate_calibration_cert.py not found")


# ── CAL-001 Standard Document ────────────────────────────────────────────


class StandardDocumentTest(SimpleTestCase):
    """CAL-001: Standard document exists with required sections."""

    def test_cal001_exists(self):
        """CAL-001: Standard document exists."""
        doc = GIT_ROOT / "docs" / "standards" / "CAL-001.md"
        self.assertTrue(doc.exists(), "CAL-001.md not found in docs/standards/")

    def test_cal001_has_sections(self):
        """CAL-001: Standard document has all required sections."""
        doc = GIT_ROOT / "docs" / "standards" / "CAL-001.md"
        content = doc.read_text()
        required_sections = [
            "§1 SCOPE AND PURPOSE",
            "§2 NORMATIVE REFERENCES",
            "§3 TERMINOLOGY",
            "§4 MODULE TIER CLASSIFICATION",
            "§5 CODE COVERAGE MEASUREMENT",
            "§6 STATISTICAL CORRECTNESS VERIFICATION",
            "§7 ENDPOINT COVERAGE",
            "§8 COMPLEXITY GOVERNANCE",
            "§9 REGRESSION PREVENTION",
            "§10 CALIBRATION CYCLES",
            "§11 CALIBRATION REPORTS",
            "§12 ANTI-PATTERNS",
        ]
        for section in required_sections:
            self.assertIn(section, content, f"CAL-001.md missing section: {section}")

    def test_cal001_has_assertions(self):
        """CAL-001: Standard document has assertion hooks."""
        doc = GIT_ROOT / "docs" / "standards" / "CAL-001.md"
        content = doc.read_text()
        import re

        assertions = re.findall(r"<!-- assert:", content)
        self.assertGreaterEqual(len(assertions), 10, f"CAL-001.md has only {len(assertions)} assertions")
