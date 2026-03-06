"""
FEAT-092: Severity-weighted compliance scoring tests.

Verifies weight assignment, weighted pass rate computation, and integration
with monthly report and dashboard API.

Standard: CMP-001 §7.5
"""

from django.test import SimpleTestCase, TestCase

from syn.audit.compliance import (
    _DEFAULT_SEVERITY,
    ALL_CHECKS,
    CHECK_SEVERITY,
    compute_weighted_pass_rate,
    get_check_weight,
)

# ── Weight assignment ──────────────────────────────────────────────────


class SeverityWeightTest(SimpleTestCase):
    """CMP-001 §7.5: Every registered check resolves to a valid weight."""

    VALID_TIERS = {3.0, 2.0, 1.0, 0.5}

    def test_all_checks_have_weight(self):
        """Every registered check resolves to a non-zero weight."""
        for name in ALL_CHECKS:
            weight = get_check_weight(name)
            self.assertGreater(weight, 0, f"{name} has zero weight")

    def test_critical_checks_weight_3(self):
        """Critical security checks have weight 3.0."""
        critical = [
            "audit_integrity",
            "security_config",
            "change_management",
            "access_logging",
            "secret_management",
            "session_security",
            "permission_coverage",
        ]
        for name in critical:
            self.assertEqual(
                get_check_weight(name),
                3.0,
                f"{name} should be critical (3.0)",
            )

    def test_low_checks_weight_half(self):
        """Low-priority checks have weight 0.5."""
        low = ["code_style", "roadmap", "policy_review"]
        for name in low:
            self.assertEqual(
                get_check_weight(name),
                0.5,
                f"{name} should be low (0.5)",
            )

    def test_unlisted_check_defaults_to_medium(self):
        """An unlisted check name defaults to _DEFAULT_SEVERITY (1.0)."""
        self.assertEqual(get_check_weight("nonexistent_check_xyz"), _DEFAULT_SEVERITY)

    def test_weights_are_valid_tiers(self):
        """All weights in CHECK_SEVERITY are valid tier values."""
        for name, weight in CHECK_SEVERITY.items():
            self.assertIn(
                weight,
                self.VALID_TIERS,
                f"{name} has invalid weight {weight}",
            )


# ── Weighted pass rate computation ─────────────────────────────────────


class WeightedPassRateTest(SimpleTestCase):
    """CMP-001 §7.5: compute_weighted_pass_rate correctness."""

    def test_empty_input_returns_zero(self):
        """Empty dict → 0.0."""
        self.assertEqual(compute_weighted_pass_rate({}), 0.0)

    def test_all_pass_returns_100(self):
        """All checks passing → 100.0."""
        statuses = {name: "pass" for name in ALL_CHECKS}
        self.assertEqual(compute_weighted_pass_rate(statuses), 100.0)

    def test_all_fail_returns_0(self):
        """All checks failing → 0.0."""
        statuses = {name: "fail" for name in ALL_CHECKS}
        self.assertEqual(compute_weighted_pass_rate(statuses), 0.0)

    def test_warning_counts_as_fail(self):
        """Warning status does not count as pass."""
        statuses = {"audit_integrity": "warning"}
        self.assertEqual(compute_weighted_pass_rate(statuses), 0.0)

    def test_critical_fail_depresses_more(self):
        """Failing a critical check depresses score more than failing a low check."""
        # Fail one critical (3.0), pass one low (0.5)
        critical_fail = {"audit_integrity": "fail", "code_style": "pass"}
        # Fail one low (0.5), pass one critical (3.0)
        low_fail = {"audit_integrity": "pass", "code_style": "fail"}

        critical_rate = compute_weighted_pass_rate(critical_fail)
        low_rate = compute_weighted_pass_rate(low_fail)
        self.assertLess(
            critical_rate,
            low_rate,
            f"Critical fail ({critical_rate}) should depress more than low fail ({low_rate})",
        )

    def test_known_arithmetic(self):
        """audit_integrity(3.0) pass + code_style(0.5) fail = 85.7%."""
        statuses = {"audit_integrity": "pass", "code_style": "fail"}
        result = compute_weighted_pass_rate(statuses)
        # 3.0 / (3.0 + 0.5) * 100 = 85.714... → rounded to 85.7
        self.assertAlmostEqual(result, 85.7, places=1)

    def test_full_system_all_pass(self):
        """All registered checks passing → 100.0."""
        statuses = {name: "pass" for name in ALL_CHECKS}
        self.assertEqual(compute_weighted_pass_rate(statuses), 100.0)

    def test_mixed_realistic_scenario(self):
        """Mixed scenario with exact expected value."""
        # 2 critical pass (6.0), 1 high fail (2.0), 1 medium pass (1.0), 1 low fail (0.5)
        statuses = {
            "audit_integrity": "pass",  # 3.0 pass
            "security_config": "pass",  # 3.0 pass
            "encryption_status": "fail",  # 2.0 fail
            "standards_compliance": "pass",  # 1.0 pass
            "code_style": "fail",  # 0.5 fail
        }
        # passed weight = 3.0 + 3.0 + 1.0 = 7.0
        # total weight = 3.0 + 3.0 + 2.0 + 1.0 + 0.5 = 9.5
        # rate = 7.0 / 9.5 * 100 = 73.684... → 73.7
        result = compute_weighted_pass_rate(statuses)
        self.assertAlmostEqual(result, 73.7, places=1)


# ── Integration with monthly report ───────────────────────────────────


class WeightedPassRateMonthlyReportTest(TestCase):
    """CMP-001 §7.5: weighted_pass_rate appears in generated reports."""

    def test_full_report_has_weighted(self):
        """generate_monthly_report() includes weighted_pass_rate in full_report."""
        from syn.audit.compliance import generate_monthly_report

        report = generate_monthly_report()
        full = report.full_report
        self.assertIn("weighted_pass_rate", full)
        self.assertIsInstance(full["weighted_pass_rate"], float)

    def test_public_report_has_weighted(self):
        """generate_monthly_report() includes weighted_pass_rate in public_report."""
        from syn.audit.compliance import generate_monthly_report

        report = generate_monthly_report()
        public = report.public_report
        self.assertIn("weighted_pass_rate", public)
        self.assertIsInstance(public["weighted_pass_rate"], float)
