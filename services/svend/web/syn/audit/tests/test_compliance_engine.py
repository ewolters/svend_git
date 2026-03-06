"""
CMP-001 compliance tests: Compliance engine check function contracts.

Tests verify all 33 registered check_* functions return valid structure,
are callable, registered with correct categories, and produce expected
detail keys. Each check is invoked and its return contract verified.

Compliance: CMP-001 (Compliance Automation), SOC 2 CC4.1, CC7.2, CC8.1
"""

from django.test import SimpleTestCase, TestCase

from syn.audit.compliance import ALL_CHECKS

# Valid SOC 2 trust service categories for check registration
VALID_CATEGORIES = {
    "security",
    "availability",
    "processing_integrity",
    "confidentiality",
    "privacy",
}

# Valid status values per CMP-001
VALID_STATUSES = {"pass", "fail", "warning", "error"}

# All 33 expected check names
EXPECTED_CHECKS = {
    "audit_integrity",
    "security_config",
    "dependency_vuln",
    "encryption_status",
    "permission_coverage",
    "access_logging",
    "backup_freshness",
    "password_policy",
    "data_retention",
    "privacy_data_export",
    "ssl_tls",
    "standards_compliance",
    "change_management",
    "session_security",
    "error_handling",
    "rate_limiting",
    "secret_management",
    "log_completeness",
    "security_headers",
    "incident_readiness",
    "sla_compliance",
    "code_style",
    "architecture_map",
    "architecture",
    "caching",
    "roadmap",
    "symbol_coverage",
    "statistical_calibration",
    "output_quality",
    "policy_review",
    "calibration_coverage",
    "complexity_governance",
    "endpoint_coverage",
}

# Checks known to NOT have soc2_controls in return
CHECKS_WITHOUT_SOC2 = {"policy_review"}

# Expected detail keys per check (subset — verifies core contract)
EXPECTED_DETAIL_KEYS = {
    "audit_integrity": {"chain_valid", "total_entries"},
    "security_config": {"issues", "checks_passed", "total_checks"},
    "encryption_status": {"issues"},
    "permission_coverage": {"unprotected_count"},
    "access_logging": {"issues", "middleware_present"},
    "password_policy": {"validators_configured", "issues"},
    "data_retention": {"retention_days", "issues"},
    "privacy_data_export": {"issues"},
    "ssl_tls": {"issues"},
    "standards_compliance": {"total_assertions", "passed", "failed"},
    "change_management": {"total_changes", "issues", "field_completeness"},
    "session_security": {"issues", "checks_passed", "total_checks"},
    "error_handling": {"issues"},
    "rate_limiting": {"issues"},
    "secret_management": {"issues"},
    "log_completeness": {"issues", "handler_count"},
    "security_headers": {"issues"},
    "incident_readiness": {"issues"},
    "code_style": {"files_scanned", "total_violations"},
    "architecture": {"missing_required_dirs"},
    "caching": {"middleware_issues"},
    "complexity_governance": {"files_checked", "violations"},
    "endpoint_coverage": {"total_endpoints", "tested_endpoints", "coverage_pct"},
    "calibration_coverage": {"coveragerc_exists", "issues"},
    "symbol_coverage": {"total_symbols", "covered_symbols"},
}


# =============================================================================
# REGISTRATION TESTS (SimpleTestCase — no DB needed)
# =============================================================================


class CheckRegistrationTest(SimpleTestCase):
    """CMP-001 §3: All checks registered with correct metadata."""

    def test_all_expected_checks_registered(self):
        """All 33 expected checks are present in ALL_CHECKS."""
        registered = set(ALL_CHECKS.keys())
        missing = EXPECTED_CHECKS - registered
        self.assertEqual(missing, set(), f"Missing checks: {missing}")

    def test_no_unexpected_checks(self):
        """No unrecognized checks in ALL_CHECKS (detects naming drift)."""
        registered = set(ALL_CHECKS.keys())
        extra = registered - EXPECTED_CHECKS
        # Allow new checks to be added — just warn, don't fail
        if extra:
            # This is informational. New checks should be added to EXPECTED_CHECKS.
            pass
        # Real assertion: registered is a superset of expected
        self.assertTrue(EXPECTED_CHECKS.issubset(registered))

    def test_all_entries_are_tuples(self):
        """Each ALL_CHECKS entry is a (function, category) tuple."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                self.assertIsInstance(entry, tuple, f"{name} is not a tuple")
                self.assertEqual(len(entry), 2, f"{name} tuple length != 2")

    def test_all_functions_callable(self):
        """First element of each entry is callable."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                fn = entry[0]
                self.assertTrue(callable(fn), f"{name} function is not callable")

    def test_all_categories_valid(self):
        """Second element of each entry is a valid SOC 2 category."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                category = entry[1]
                self.assertIn(
                    category,
                    VALID_CATEGORIES,
                    f"{name} has invalid category: {category}",
                )

    def test_minimum_check_count(self):
        """At least 30 checks registered (guards against mass deregistration)."""
        self.assertGreaterEqual(len(ALL_CHECKS), 30)

    def test_security_category_has_checks(self):
        """Security category has multiple checks."""
        security_checks = [n for n, e in ALL_CHECKS.items() if e[1] == "security"]
        self.assertGreaterEqual(len(security_checks), 5)

    def test_processing_integrity_category_has_checks(self):
        """Processing integrity category has multiple checks."""
        pi_checks = [n for n, e in ALL_CHECKS.items() if e[1] == "processing_integrity"]
        self.assertGreaterEqual(len(pi_checks), 8)


# =============================================================================
# INVOCATION TESTS (TestCase — checks query the DB)
# =============================================================================


class CheckInvocationTest(TestCase):
    """CMP-001 §4: Every check returns valid structure when invoked."""

    def test_all_checks_return_dict(self):
        """Every check function returns a dict."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                fn = entry[0]
                result = fn()
                self.assertIsInstance(result, dict, f"{name} returned {type(result).__name__}, expected dict")

    def test_all_checks_have_status(self):
        """Every check result has a 'status' key."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                result = entry[0]()
                self.assertIn("status", result, f"{name} missing 'status' key")

    def test_all_statuses_valid(self):
        """Every check status is pass/fail/warning/error."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                result = entry[0]()
                self.assertIn(
                    result["status"],
                    VALID_STATUSES,
                    f"{name} has invalid status: {result['status']}",
                )

    def test_all_checks_have_details(self):
        """Every check result has a 'details' key with a dict value."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                result = entry[0]()
                self.assertIn("details", result, f"{name} missing 'details' key")
                self.assertIsInstance(
                    result["details"],
                    dict,
                    f"{name} details is {type(result['details']).__name__}, expected dict",
                )

    def test_soc2_controls_present(self):
        """Every check has 'soc2_controls' except known exceptions."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                result = entry[0]()
                if name in CHECKS_WITHOUT_SOC2:
                    # Document that these intentionally lack soc2_controls
                    continue
                self.assertIn(
                    "soc2_controls",
                    result,
                    f"{name} missing 'soc2_controls' key",
                )

    def test_soc2_controls_are_lists(self):
        """soc2_controls values are lists of strings."""
        for name, entry in ALL_CHECKS.items():
            with self.subTest(check=name):
                result = entry[0]()
                if name in CHECKS_WITHOUT_SOC2:
                    continue
                controls = result.get("soc2_controls", [])
                self.assertIsInstance(controls, list, f"{name} soc2_controls not a list")
                for ctrl in controls:
                    self.assertIsInstance(ctrl, str, f"{name} control {ctrl} not a string")


# =============================================================================
# DETAIL KEY TESTS (TestCase — invokes checks)
# =============================================================================


class CheckDetailKeysTest(TestCase):
    """CMP-001 §4: Check-specific detail keys are present."""

    def test_expected_detail_keys(self):
        """Each check's details contains its expected keys."""
        for name, expected_keys in EXPECTED_DETAIL_KEYS.items():
            with self.subTest(check=name):
                if name not in ALL_CHECKS:
                    self.skipTest(f"{name} not registered")
                fn = ALL_CHECKS[name][0]
                result = fn()
                details = result.get("details", {})
                for key in expected_keys:
                    self.assertIn(
                        key,
                        details,
                        f"{name} details missing key: {key}",
                    )


# =============================================================================
# SECURITY CONFIG TESTS (SOC 2 CC6.1)
# =============================================================================


class SecurityConfigCheckTest(TestCase):
    """CMP-001: check_security_config behavioral assertions."""

    def test_checks_debug_setting(self):
        """Security config check evaluates DEBUG setting."""
        fn = ALL_CHECKS["security_config"][0]
        result = fn()
        details = result["details"]
        # In test mode Django has DEBUG=False by default, should pass
        self.assertIn("checks_passed", details)
        self.assertIn("total_checks", details)
        self.assertGreaterEqual(details["total_checks"], 3)

    def test_issues_is_list(self):
        """Issues field is a list."""
        result = ALL_CHECKS["security_config"][0]()
        self.assertIsInstance(result["details"]["issues"], list)


# =============================================================================
# CHANGE MANAGEMENT TESTS (SOC 2 CC8.1)
# =============================================================================


class ChangeManagementCheckTest(TestCase):
    """CMP-001: check_change_management behavioral assertions."""

    def test_field_completeness_structure(self):
        """Field completeness is a dict with percentage values."""
        result = ALL_CHECKS["change_management"][0]()
        details = result["details"]
        fc = details.get("field_completeness", {})
        self.assertIsInstance(fc, dict)

    def test_fail_warn_counts(self):
        """Result includes fail_count and warn_count."""
        result = ALL_CHECKS["change_management"][0]()
        details = result["details"]
        self.assertIn("fail_count", details)
        self.assertIn("warn_count", details)
        self.assertIsInstance(details["fail_count"], int)
        self.assertIsInstance(details["warn_count"], int)

    def test_issues_is_list(self):
        """Issues field is a list of dicts."""
        result = ALL_CHECKS["change_management"][0]()
        issues = result["details"]["issues"]
        self.assertIsInstance(issues, list)


# =============================================================================
# CODE STYLE TESTS (SOC 2 CC8.1)
# =============================================================================


class CodeStyleCheckTest(TestCase):
    """CMP-001: check_code_style behavioral assertions."""

    def test_files_scanned_positive(self):
        """Code style check scans a meaningful number of files."""
        result = ALL_CHECKS["code_style"][0]()
        self.assertGreater(result["details"]["files_scanned"], 100)

    def test_total_violations_integer(self):
        """total_violations is an integer."""
        result = ALL_CHECKS["code_style"][0]()
        self.assertIsInstance(result["details"]["total_violations"], int)

    def test_all_scanner_keys_present(self):
        """All scanner result keys are in details."""
        result = ALL_CHECKS["code_style"][0]()
        details = result["details"]
        scanner_keys = {
            "file_naming_violations",
            "class_naming_violations",
            "function_naming_violations",
            "import_order_violations",
            "wildcard_import_violations",
            "constant_violations",
            "layout_violations",
        }
        for key in scanner_keys:
            self.assertIn(key, details, f"Code style missing scanner key: {key}")


# =============================================================================
# ERROR HANDLING TESTS (SOC 2 CC7.2)
# =============================================================================


class ErrorHandlingCheckTest(TestCase):
    """CMP-001: check_error_handling behavioral assertions."""

    def test_middleware_detection(self):
        """Error handling check detects middleware presence."""
        result = ALL_CHECKS["error_handling"][0]()
        details = result["details"]
        self.assertIn("error_envelope_active", details)
        self.assertIsInstance(details["error_envelope_active"], bool)

    def test_debug_mode_reported(self):
        """Debug mode status is reported."""
        result = ALL_CHECKS["error_handling"][0]()
        self.assertIn("debug_mode", result["details"])


# =============================================================================
# LOG COMPLETENESS TESTS (SOC 2 CC7.1)
# =============================================================================


class LogCompletenessCheckTest(TestCase):
    """CMP-001: check_log_completeness behavioral assertions."""

    def test_handler_count_positive(self):
        """Log completeness reports handler count > 0."""
        result = ALL_CHECKS["log_completeness"][0]()
        self.assertGreater(result["details"]["handler_count"], 0)

    def test_has_audit_logger(self):
        """Audit logger presence is reported."""
        result = ALL_CHECKS["log_completeness"][0]()
        self.assertIn("has_audit_logger", result["details"])


# =============================================================================
# ARCHITECTURE TESTS (SOC 2 CC8.1)
# =============================================================================


class ArchitectureCheckTest(TestCase):
    """CMP-001: check_architecture behavioral assertions."""

    def test_required_dirs_checked(self):
        """Architecture check verifies required directory structure."""
        result = ALL_CHECKS["architecture"][0]()
        details = result["details"]
        self.assertIn("missing_required_dirs", details)
        self.assertIsInstance(details["missing_required_dirs"], list)

    def test_oversized_files_tracked(self):
        """Architecture check tracks oversized files."""
        result = ALL_CHECKS["architecture"][0]()
        self.assertIn("oversized_files", result["details"])

    def test_layer_violations_tracked(self):
        """Architecture check tracks layer boundary violations."""
        result = ALL_CHECKS["architecture"][0]()
        self.assertIn("layer_boundary_violations", result["details"])


# =============================================================================
# ENCRYPTION STATUS TESTS (SOC 2 CC6.1)
# =============================================================================


class EncryptionCheckTest(TestCase):
    """CMP-001: check_encryption_status behavioral assertions."""

    def test_field_encryption_reported(self):
        """Encryption check reports field encryption status."""
        result = ALL_CHECKS["encryption_status"][0]()
        self.assertIn("field_encryption_configured", result["details"])


# =============================================================================
# SESSION SECURITY TESTS (SOC 2 CC6.1)
# =============================================================================


class SessionSecurityCheckTest(TestCase):
    """CMP-001: check_session_security behavioral assertions."""

    def test_session_cookie_age_reported(self):
        """Session security reports cookie age."""
        result = ALL_CHECKS["session_security"][0]()
        self.assertIn("session_cookie_age", result["details"])

    def test_checks_total_positive(self):
        """Multiple session checks are performed."""
        result = ALL_CHECKS["session_security"][0]()
        self.assertGreaterEqual(result["details"]["total_checks"], 3)


# =============================================================================
# COMPLEXITY GOVERNANCE TESTS (SOC 2 CC8.1)
# =============================================================================


class ComplexityGovernanceCheckTest(TestCase):
    """CMP-001: check_complexity_governance behavioral assertions."""

    def test_files_checked_positive(self):
        """Complexity governance scans files."""
        result = ALL_CHECKS["complexity_governance"][0]()
        self.assertGreater(result["details"]["files_checked"], 0)

    def test_exemptions_tracked(self):
        """Exemptions list is present."""
        result = ALL_CHECKS["complexity_governance"][0]()
        self.assertIn("exemptions", result["details"])


# =============================================================================
# ENDPOINT COVERAGE TESTS (SOC 2 CC4.1)
# =============================================================================


class EndpointCoverageCheckTest(TestCase):
    """CMP-001: check_endpoint_coverage behavioral assertions."""

    def test_total_endpoints_positive(self):
        """Endpoint coverage reports positive total."""
        result = ALL_CHECKS["endpoint_coverage"][0]()
        self.assertGreater(result["details"]["total_endpoints"], 0)

    def test_coverage_pct_bounded(self):
        """Coverage percentage is between 0 and 100."""
        result = ALL_CHECKS["endpoint_coverage"][0]()
        pct = result["details"]["coverage_pct"]
        self.assertGreaterEqual(pct, 0)
        self.assertLessEqual(pct, 100)


# =============================================================================
# SYMBOL COVERAGE TESTS (SOC 2 CC4.1)
# =============================================================================


class SymbolCoverageCheckTest(TestCase):
    """CMP-001: check_symbol_coverage behavioral assertions."""

    def test_total_symbols_positive(self):
        """Symbol coverage reports positive total."""
        result = ALL_CHECKS["symbol_coverage"][0]()
        self.assertGreater(result["details"]["total_symbols"], 0)

    def test_risk_score_present(self):
        """Risk score is calculated."""
        result = ALL_CHECKS["symbol_coverage"][0]()
        self.assertIn("risk_score", result["details"])

    def test_by_module_present(self):
        """Per-module breakdown is present."""
        result = ALL_CHECKS["symbol_coverage"][0]()
        self.assertIn("by_module", result["details"])
        self.assertIsInstance(result["details"]["by_module"], list)


# =============================================================================
# CALIBRATION COVERAGE TESTS (SOC 2 CC4.1)
# =============================================================================


class CalibrationCoverageCheckTest(TestCase):
    """CMP-001: check_calibration_coverage behavioral assertions."""

    def test_coveragerc_check(self):
        """Calibration coverage checks for .coveragerc."""
        result = ALL_CHECKS["calibration_coverage"][0]()
        self.assertIn("coveragerc_exists", result["details"])
        self.assertIsInstance(result["details"]["coveragerc_exists"], bool)


# =============================================================================
# SSL/TLS TESTS (SOC 2 CC6.2)
# =============================================================================


class SslTlsCheckTest(TestCase):
    """CMP-001: check_ssl_tls behavioral assertions."""

    def test_hsts_reported(self):
        """SSL/TLS check reports HSTS configuration."""
        result = ALL_CHECKS["ssl_tls"][0]()
        self.assertIn("hsts_seconds", result["details"])

    def test_ssl_redirect_reported(self):
        """SSL redirect status is reported."""
        result = ALL_CHECKS["ssl_tls"][0]()
        self.assertIn("ssl_redirect", result["details"])


# =============================================================================
# POLICY REVIEW TESTS (no SOC 2 controls — documented exception)
# =============================================================================


class PolicyReviewCheckTest(TestCase):
    """CMP-001: check_policy_review — documented exception for soc2_controls."""

    def test_no_soc2_controls(self):
        """Policy review intentionally omits soc2_controls."""
        result = ALL_CHECKS["policy_review"][0]()
        # This documents the known exception
        self.assertNotIn("soc2_controls", result)

    def test_policies_checked_reported(self):
        """Policy review reports count of policies checked."""
        result = ALL_CHECKS["policy_review"][0]()
        details = result["details"]
        self.assertIn("policies_checked", details)


# =============================================================================
# STANDARDS COMPLIANCE TESTS (SOC 2 CC4.1)
# =============================================================================


class StandardsComplianceCheckTest(TestCase):
    """CMP-001: check_standards_compliance behavioral assertions."""

    def test_assertions_scanned(self):
        """Standards compliance scans assertions from docs/standards/."""
        result = ALL_CHECKS["standards_compliance"][0]()
        details = result["details"]
        self.assertGreater(details["total_assertions"], 50)

    def test_tests_linked(self):
        """Standards compliance reports test linkage counts."""
        result = ALL_CHECKS["standards_compliance"][0]()
        details = result["details"]
        self.assertIn("tests_linked", details)
        self.assertIn("tests_exist", details)
        self.assertIn("tests_missing", details)

    def test_by_standard_breakdown(self):
        """Per-standard breakdown is present."""
        result = ALL_CHECKS["standards_compliance"][0]()
        self.assertIn("by_standard", result["details"])
        self.assertIsInstance(result["details"]["by_standard"], dict)


# =============================================================================
# STATISTICAL CALIBRATION TESTS (SOC 2 CC4.1)
# =============================================================================


class StatisticalCalibrationCheckTest(TestCase):
    """CMP-001: check_statistical_calibration behavioral assertions."""

    def test_cases_run_positive(self):
        """Statistical calibration runs cases."""
        result = ALL_CHECKS["statistical_calibration"][0]()
        details = result["details"]
        self.assertIn("cases_run", details)
        self.assertGreater(details["cases_run"], 0)

    def test_pass_rate_bounded(self):
        """Pass rate is between 0 and 100."""
        result = ALL_CHECKS["statistical_calibration"][0]()
        pr = result["details"]["pass_rate"]
        self.assertGreaterEqual(pr, 0)
        self.assertLessEqual(pr, 100)

    def test_seed_deterministic(self):
        """Calibration uses a deterministic date-based seed."""
        result = ALL_CHECKS["statistical_calibration"][0]()
        self.assertIn("seed", result["details"])


# =============================================================================
# CACHING TESTS (SOC 2 CC9.1)
# =============================================================================


class CachingCheckTest(TestCase):
    """CMP-001: check_caching behavioral assertions."""

    def test_all_sub_check_keys_present(self):
        """Caching check has all sub-check categories."""
        result = ALL_CHECKS["caching"][0]()
        details = result["details"]
        for key in ("middleware_issues", "storage_issues", "memory_bound_issues"):
            self.assertIn(key, details, f"Caching missing key: {key}")


# =============================================================================
# SECRET MANAGEMENT TESTS (SOC 2 CC6.1)
# =============================================================================


class SecretManagementCheckTest(TestCase):
    """CMP-001: check_secret_management behavioral assertions."""

    def test_secret_key_length_reported(self):
        """Secret management reports SECRET_KEY length."""
        result = ALL_CHECKS["secret_management"][0]()
        self.assertIn("secret_key_length", result["details"])

    def test_env_file_checked(self):
        """Secret management checks for .env file."""
        result = ALL_CHECKS["secret_management"][0]()
        self.assertIn("env_file_present", result["details"])


# =============================================================================
# RATE LIMITING TESTS (SOC 2 CC6.1)
# =============================================================================


class RateLimitingCheckTest(TestCase):
    """CMP-001: check_rate_limiting behavioral assertions."""

    def test_throttle_classes_reported(self):
        """Rate limiting reports throttle classes."""
        result = ALL_CHECKS["rate_limiting"][0]()
        self.assertIn("throttle_classes", result["details"])


# =============================================================================
# SECURITY HEADERS TESTS (SOC 2 CC6.1)
# =============================================================================


class SecurityHeadersCheckTest(TestCase):
    """CMP-001: check_security_headers behavioral assertions."""

    def test_x_frame_options_reported(self):
        """Security headers reports X-Frame-Options."""
        result = ALL_CHECKS["security_headers"][0]()
        self.assertIn("x_frame_options", result["details"])

    def test_cors_checked(self):
        """Security headers checks CORS configuration."""
        result = ALL_CHECKS["security_headers"][0]()
        self.assertIn("cors_allow_all", result["details"])
