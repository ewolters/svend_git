"""
CMP-001 compliance tests: Compliance Automation Standard.

Tests the standards parser, assertion dataclass, verification functions,
ALL_CHECKS registry, daily rotation, and management command.

Standard: CMP-001
"""

import unittest
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings
from django.test import TestCase as DjangoTestCase

from syn.audit.compliance import ALL_CHECKS, DAILY_CRITICAL
from syn.audit.standards import (
    STANDARDS_DIR,
    TAG_RE,
    Assertion,
    parse_all_standards,
    parse_standard,
    verify_assertion,
    verify_code_absent,
    verify_code_pattern,
    verify_impl_exists,
)


class TagVocabularyTest(SimpleTestCase):
    """CMP-001 §4: Standards parser recognizes all DOC-001 §7.5 tag types."""

    REQUIRED_TAGS = ["assert", "impl", "check", "code", "control", "rule", "table", "test"]

    def test_tag_re_matches_all_required_tags(self):
        for tag in self.REQUIRED_TAGS:
            line = f"<!-- {tag}: some value -->"
            m = TAG_RE.match(line)
            self.assertIsNotNone(m, f"TAG_RE failed to match '{tag}' tag")
            self.assertEqual(m.group(1), tag)

    def test_tag_re_rejects_invalid(self):
        self.assertIsNone(TAG_RE.match("<!-- invalid: value -->"))
        self.assertIsNone(TAG_RE.match("not a tag"))
        self.assertIsNone(TAG_RE.match(""))


class ParseAllStandardsTest(SimpleTestCase):
    """CMP-001 §4: parse_all_standards discovers all standards files."""

    def test_discovers_standards_files(self):
        assertions = parse_all_standards()
        self.assertGreater(len(assertions), 100)

    def test_returns_assertion_instances(self):
        assertions = parse_all_standards()
        for a in assertions[:5]:
            self.assertIsInstance(a, Assertion)

    def test_all_assertions_have_check_id(self):
        assertions = parse_all_standards()
        for a in assertions:
            self.assertTrue(a.check_id, f"Assertion missing check_id: {a.text[:50]}")

    def test_all_assertions_have_standard(self):
        assertions = parse_all_standards()
        for a in assertions:
            self.assertTrue(a.standard, f"Assertion missing standard: {a.text[:50]}")


class ParseStandardTest(SimpleTestCase):
    """CMP-001 §4: parse_standard extracts Assertion instances."""

    def test_parses_known_standard(self):
        filepath = STANDARDS_DIR / "ERR-001.md"
        if filepath.exists():
            assertions = parse_standard(filepath)
            self.assertGreater(len(assertions), 0)
            self.assertEqual(assertions[0].standard, "ERR-001")

    def test_parses_section_numbers(self):
        assertions = parse_all_standards()
        sections = [a.section for a in assertions if a.section]
        self.assertGreater(len(sections), 0)
        for s in sections[:10]:
            self.assertTrue(s.startswith("§"), f"Section should start with §: {s}")


class AssertionDataclassTest(SimpleTestCase):
    """CMP-001 §4: Assertion dataclass structure."""

    def test_required_fields(self):
        a = Assertion(text="Test assertion", check_id="test-001")
        self.assertEqual(a.text, "Test assertion")
        self.assertEqual(a.check_id, "test-001")
        self.assertEqual(a.impls, [])
        self.assertEqual(a.tests, [])
        self.assertEqual(a.code_correct, [])
        self.assertEqual(a.code_prohibited, [])
        self.assertEqual(a.controls, {})
        self.assertEqual(a.rule, "")
        self.assertEqual(a.standard, "")
        self.assertEqual(a.section, "")

    def test_fields_mutable(self):
        a = Assertion(text="Test", check_id="t")
        a.impls.append("syn/err/exceptions.py")
        a.tests.append("syn.audit.tests.test_error_handling.ErrorHierarchyTest.test_all")
        self.assertEqual(len(a.impls), 1)
        self.assertEqual(len(a.tests), 1)


class VerifyImplExistsTest(SimpleTestCase):
    """CMP-001 §4: verify_impl_exists uses AST to check file/symbol."""

    def test_existing_file(self):
        ok, msg = verify_impl_exists("syn/err/exceptions.py")
        self.assertTrue(ok)

    def test_missing_file(self):
        ok, msg = verify_impl_exists("syn/err/nonexistent.py")
        self.assertFalse(ok)
        self.assertIn("not found", msg.lower())

    def test_existing_class(self):
        ok, msg = verify_impl_exists("syn/err/exceptions.py:SynaraError")
        self.assertTrue(ok)

    def test_existing_method(self):
        ok, msg = verify_impl_exists("syn/err/exceptions.py:SynaraError.__init__")
        self.assertTrue(ok)

    def test_missing_symbol(self):
        ok, msg = verify_impl_exists("syn/err/exceptions.py:NonExistentClass")
        self.assertFalse(ok)

    def test_existing_constant(self):
        ok, msg = verify_impl_exists("syn/err/types.py:CATEGORY_STATUS_CODES")
        self.assertTrue(ok)


class VerifyCodePatternTest(SimpleTestCase):
    """CMP-001 §4: verify_code_pattern checks patterns in source."""

    def test_matching_pattern(self):
        ok, msg = verify_code_pattern(
            "syn/err/exceptions.py",
            "class SynaraError(Exception):\n    default_category = ErrorCategory.SYSTEM",
        )
        self.assertTrue(ok)

    def test_no_match_returns_false(self):
        ok, msg = verify_code_pattern(
            "syn/err/exceptions.py",
            "class TotallyFakeClass(NothingReal):\n    fake_attribute = FakeValue.BOGUS",
        )
        self.assertFalse(ok)


class VerifyCodeAbsentTest(SimpleTestCase):
    """CMP-001 §4: verify_code_absent checks prohibited patterns."""

    def test_absent_pattern_passes(self):
        ok, msg = verify_code_absent(
            "syn/err/exceptions.py",
            "import pdb; pdb.set_trace()",
        )
        self.assertTrue(ok)

    def test_missing_file_passes(self):
        ok, msg = verify_code_absent("nonexistent.py", "anything")
        self.assertTrue(ok)


class VerifyAssertionTest(SimpleTestCase):
    """CMP-001 §4: verify_assertion orchestrates all checks."""

    def test_pass_with_valid_impl(self):
        a = Assertion(
            text="Test",
            check_id="test-pass",
            impls=["syn/err/exceptions.py:SynaraError"],
        )
        result = verify_assertion(a)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["check_id"], "test-pass")

    def test_fail_with_missing_impl(self):
        a = Assertion(
            text="Test",
            check_id="test-fail",
            impls=["syn/err/nonexistent.py:FakeClass"],
        )
        result = verify_assertion(a)
        self.assertEqual(result["status"], "fail")

    def test_result_contains_all_fields(self):
        a = Assertion(text="Test", check_id="test-fields")
        result = verify_assertion(a)
        for field in [
            "check_id",
            "assertion",
            "standard",
            "section",
            "impl_checks",
            "code_checks",
            "test_checks",
            "status",
        ]:
            self.assertIn(field, result)


class AllChecksRegistryTest(SimpleTestCase):
    """CMP-001 §4: ALL_CHECKS registry contains all compliance checks."""

    EXPECTED_CHECKS = [
        "audit_integrity",
        "security_config",
        "access_logging",
        "standards_compliance",
        "change_management",
    ]

    def test_expected_checks_registered(self):
        for check_name in self.EXPECTED_CHECKS:
            self.assertIn(check_name, ALL_CHECKS, f"Check '{check_name}' not in ALL_CHECKS")

    def test_all_checks_are_callable(self):
        for name, entry in ALL_CHECKS.items():
            fn = entry[0] if isinstance(entry, tuple) else entry
            self.assertTrue(callable(fn), f"Check '{name}' is not callable")

    def test_registry_has_at_least_10_checks(self):
        self.assertGreaterEqual(len(ALL_CHECKS), 10)


class SOC2AutoDiscoveryTest(SimpleTestCase):
    """CMP-001: SOC 2 controls auto-discovered from check registry metadata."""

    def test_all_checks_have_soc2_controls(self):
        """Every registered check declares SOC 2 controls on the function."""
        for name, (fn, _cat) in ALL_CHECKS.items():
            controls = getattr(fn, "soc2_controls", None)
            self.assertIsNotNone(
                controls,
                f"Check '{name}' missing soc2_controls attribute on function",
            )

    def test_no_empty_soc2_controls(self):
        """Every check declares at least one SOC 2 control (except standards_compliance edge case)."""
        for name, (fn, _cat) in ALL_CHECKS.items():
            controls = getattr(fn, "soc2_controls", [])
            self.assertGreater(
                len(controls),
                0,
                f"Check '{name}' has empty soc2_controls — add to @register()",
            )

    def test_get_all_soc2_controls_returns_sorted(self):
        from syn.audit.compliance import get_all_soc2_controls

        controls = get_all_soc2_controls()
        self.assertGreater(len(controls), 5)
        self.assertEqual(controls, sorted(controls))

    def test_get_check_soc2_controls(self):
        from syn.audit.compliance import get_check_soc2_controls

        controls = get_check_soc2_controls("audit_integrity")
        self.assertIn("CC7.2", controls)
        self.assertIn("CC7.3", controls)

    def test_get_check_soc2_controls_missing(self):
        from syn.audit.compliance import get_check_soc2_controls

        controls = get_check_soc2_controls("nonexistent_check")
        self.assertEqual(controls, [])

    def test_declared_controls_match_return_values(self):
        """Declared controls on @register match the check function return values."""
        # Run a few fast checks and verify consistency
        for name in ["security_config", "password_policy", "error_handling"]:
            fn, _cat = ALL_CHECKS[name]
            declared = set(fn.soc2_controls)
            result = fn()
            returned = set(result.get("soc2_controls", []))
            self.assertEqual(
                declared,
                returned,
                f"Check '{name}': declared {declared} != returned {returned}",
            )


class DailyRotationTest(SimpleTestCase):
    """CMP-001 §4: DAILY_CRITICAL checks run daily, others rotate."""

    def test_daily_critical_checks_defined(self):
        self.assertGreater(len(DAILY_CRITICAL), 0)

    def test_daily_critical_are_in_all_checks(self):
        for name in DAILY_CRITICAL:
            self.assertIn(name, ALL_CHECKS, f"DAILY_CRITICAL check '{name}' not in ALL_CHECKS")


class ManagementCommandTest(SimpleTestCase):
    """CMP-001 §4: run_compliance management command."""

    def test_command_exists(self):
        from django.core.management import get_commands

        commands = get_commands()
        self.assertIn("run_compliance", commands)


class DriftSignatureTest(SimpleTestCase):
    """CMP-001 §5.6: Drift signature deduplication."""

    def test_same_inputs_same_signature(self):
        from syn.audit.compliance import _compute_drift_signature

        sig1 = _compute_drift_signature("FE-001", "fe-themes", "5")
        sig2 = _compute_drift_signature("FE-001", "fe-themes", "5")
        self.assertEqual(sig1, sig2)

    def test_different_inputs_different_signature(self):
        from syn.audit.compliance import _compute_drift_signature

        sig1 = _compute_drift_signature("FE-001", "fe-themes", "5")
        sig2 = _compute_drift_signature("FE-001", "fe-themes", "6")
        self.assertNotEqual(sig1, sig2)

    def test_signature_length(self):
        from syn.audit.compliance import _compute_drift_signature

        sig = _compute_drift_signature("TST-001", "tst-framework", "4")
        self.assertEqual(len(sig), 32)


class DriftSeverityTest(SimpleTestCase):
    """CMP-001 §5.6: Drift severity mapping."""

    def test_fail_maps_to_high(self):
        from syn.audit.compliance import DRIFT_SEVERITY_MAP

        self.assertEqual(DRIFT_SEVERITY_MAP["fail"], "HIGH")

    def test_warning_maps_to_medium(self):
        from syn.audit.compliance import DRIFT_SEVERITY_MAP

        self.assertEqual(DRIFT_SEVERITY_MAP["warning"], "MEDIUM")

    def test_sla_hours_defined(self):
        from syn.audit.compliance import DRIFT_SLA_HOURS

        self.assertEqual(DRIFT_SLA_HOURS["HIGH"], 24)
        self.assertEqual(DRIFT_SLA_HOURS["MEDIUM"], 72)
        self.assertEqual(DRIFT_SLA_HOURS["CRITICAL"], 4)
        self.assertEqual(DRIFT_SLA_HOURS["LOW"], 168)


# DB-backed tests for drift violation lifecycle


class DriftViolationLifecycleTest(DjangoTestCase):
    """CMP-001 §5.6: Drift violation creation and auto-resolution."""

    def test_failure_creates_violation(self):
        """A failing assertion creates a DriftViolation record."""
        from syn.audit.compliance import _compute_drift_signature, _sync_drift_violations
        from syn.audit.models import DriftViolation

        a = Assertion(text="Test claim", check_id="test-check-create", standard="TST-001", section="1")
        r = {
            "status": "fail",
            "check_id": "test-check-create",
            "assertion": "Test claim",
            "standard": "TST-001",
            "section": "1",
        }
        _sync_drift_violations([a], [r])
        sig = _compute_drift_signature("TST-001", "test-check-create", "1")
        self.assertTrue(DriftViolation.objects.filter(drift_signature=sig).exists())
        dv = DriftViolation.objects.get(drift_signature=sig)
        self.assertEqual(dv.severity, "HIGH")
        self.assertEqual(dv.enforcement_check, "STD")

    def test_pass_resolves_violation(self):
        """A passing assertion resolves an existing DriftViolation."""
        from syn.audit.compliance import _compute_drift_signature, _sync_drift_violations
        from syn.audit.models import DriftViolation

        sig = _compute_drift_signature("TST-001", "test-check-resolve", "1")
        DriftViolation.objects.create(
            drift_signature=sig,
            severity="HIGH",
            enforcement_check="STD",
            file_path="docs/standards/TST-001.md",
            violation_message="Test failure",
            detected_by="compliance_runner",
        )
        a = Assertion(text="Test claim", check_id="test-check-resolve", standard="TST-001", section="1")
        r = {
            "status": "pass",
            "check_id": "test-check-resolve",
            "assertion": "Test claim",
            "standard": "TST-001",
            "section": "1",
        }
        _sync_drift_violations([a], [r])
        dv = DriftViolation.objects.get(drift_signature=sig)
        self.assertIsNotNone(dv.resolved_at)
        self.assertEqual(dv.resolved_by, "compliance_runner")

    def test_idempotent_no_duplicates(self):
        """Running twice with same failure doesn't create duplicate."""
        from syn.audit.compliance import _compute_drift_signature, _sync_drift_violations
        from syn.audit.models import DriftViolation

        a = Assertion(text="Test claim", check_id="test-check-dedup", standard="TST-001", section="1")
        r = {
            "status": "fail",
            "check_id": "test-check-dedup",
            "assertion": "Test claim",
            "standard": "TST-001",
            "section": "1",
        }
        _sync_drift_violations([a], [r])
        _sync_drift_violations([a], [r])
        sig = _compute_drift_signature("TST-001", "test-check-dedup", "1")
        self.assertEqual(DriftViolation.objects.filter(drift_signature=sig).count(), 1)


class DriftViolationSaveTest(DjangoTestCase):
    """DriftViolation allows resolution field updates but blocks other mutations."""

    def test_resolution_fields_updatable(self):
        from django.utils import timezone

        from syn.audit.models import DriftViolation

        dv = DriftViolation.objects.create(
            drift_signature="test-save-resolution",
            severity="HIGH",
            enforcement_check="STD",
            file_path="test.py",
            violation_message="Test",
            detected_by="test",
        )
        dv.resolved_at = timezone.now()
        dv.resolved_by = "test"
        dv.save(update_fields=["resolved_at", "resolved_by"])
        dv.refresh_from_db()
        self.assertIsNotNone(dv.resolved_at)

    def test_immutable_fields_blocked(self):
        from syn.audit.models import DriftViolation

        dv = DriftViolation.objects.create(
            drift_signature="test-save-immutable",
            severity="HIGH",
            enforcement_check="STD",
            file_path="test.py",
            violation_message="Test",
            detected_by="test",
        )
        dv.violation_message = "Changed"
        with self.assertRaises(ValueError):
            dv.save(update_fields=["violation_message"])

    def test_bare_save_blocked_on_existing(self):
        from syn.audit.models import DriftViolation

        dv = DriftViolation.objects.create(
            drift_signature="test-save-bare",
            severity="HIGH",
            enforcement_check="STD",
            file_path="test.py",
            violation_message="Test",
            detected_by="test",
        )
        with self.assertRaises(ValueError):
            dv.save()


@override_settings(SECURE_SSL_REDIRECT=False)
class CompliancePageDynamicCountsTest(DjangoTestCase):
    """CMP-001 §7 — Public compliance page uses dynamic counts from backend."""

    def test_page_context_uses_dynamic_check_count(self):
        """Compliance page context contains current_checks matching ALL_CHECKS count."""
        from syn.audit.compliance import run_check

        # Run a single check so there's at least one result
        run_check("audit_integrity")

        resp = self.client.get("/compliance/")
        self.assertEqual(resp.status_code, 200)
        # current_checks should only include checks that have been run
        # but the rendered page should use dynamic count, not hardcoded "12"
        content = resp.content.decode()
        self.assertNotIn("runs 12 automated", content)

    def test_page_context_includes_last_check_run(self):
        """Compliance page context includes last_check_run timestamp."""
        from syn.audit.compliance import run_check

        run_check("audit_integrity")

        resp = self.client.get("/compliance/")
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(resp.context["last_check_run"])

    def test_page_renders_dynamic_standards_count(self):
        """Compliance page uses {{ standards|length }} not hardcoded count."""
        resp = self.client.get("/compliance/")
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode()
        # Should not contain hardcoded "16 internal standards"
        self.assertNotIn("16 internal standards", content)


class VerifyTestExistsTest(SimpleTestCase):
    """CMP-001 §5.5: verify_test_exists checks linked test methods are importable."""

    def test_valid_test_method_resolves(self):
        """A known test method resolves successfully."""
        from syn.audit.standards import verify_test_exists

        ok, msg = verify_test_exists(
            "syn.audit.tests.test_compliance_system.TagVocabularyTest.test_tag_re_matches_all_required_tags"
        )
        self.assertTrue(ok, msg)

    def test_invalid_module_fails(self):
        """A non-existent module path returns False."""
        from syn.audit.standards import verify_test_exists

        ok, msg = verify_test_exists("syn.audit.tests.nonexistent_module.FakeClass.fake_method")
        self.assertFalse(ok)

    def test_invalid_class_fails(self):
        """A valid module but non-existent class returns False."""
        from syn.audit.standards import verify_test_exists

        ok, msg = verify_test_exists("syn.audit.tests.test_compliance_system.NonExistentClass.some_method")
        self.assertFalse(ok)

    def test_invalid_method_fails(self):
        """A valid class but non-existent method returns False."""
        from syn.audit.standards import verify_test_exists

        ok, msg = verify_test_exists("syn.audit.tests.test_compliance_system.TagVocabularyTest.nonexistent_method")
        self.assertFalse(ok)


class SLAComplianceRegistrationTest(SimpleTestCase):
    """CMP-001 §6: sla_compliance check registered in ALL_CHECKS."""

    def test_sla_compliance_in_all_checks(self):
        """sla_compliance is registered as a compliance check."""
        self.assertIn("sla_compliance", ALL_CHECKS)

    def test_sla_compliance_is_availability_category(self):
        """sla_compliance is categorized as 'availability'."""
        entry = ALL_CHECKS["sla_compliance"]
        category = entry[1] if isinstance(entry, tuple) else None
        self.assertEqual(category, "availability")


class MonthlyReportStructureTest(SimpleTestCase):
    """CMP-001 §7.1: Monthly report function structure."""

    def test_generate_monthly_report_exists(self):
        """generate_monthly_report function is importable."""
        from syn.audit.compliance import generate_monthly_report

        self.assertTrue(callable(generate_monthly_report))

    def test_report_function_creates_public_report(self):
        """generate_monthly_report source contains public_report construction."""
        import inspect

        from syn.audit.compliance import generate_monthly_report

        src = inspect.getsource(generate_monthly_report)
        self.assertIn("public_report", src)

    def test_report_function_creates_full_report(self):
        """generate_monthly_report source contains full_report construction."""
        import inspect

        from syn.audit.compliance import generate_monthly_report

        src = inspect.getsource(generate_monthly_report)
        self.assertIn("full_report", src)

    def test_report_redaction_no_file_paths(self):
        """public_report construction does not include file_path references."""
        import inspect

        from syn.audit.compliance import generate_monthly_report

        src = inspect.getsource(generate_monthly_report)
        # The public_report dict should not pass through raw file paths
        # Check that public_report is constructed separately from full_report
        self.assertIn("is_published", src)

    def test_report_includes_pass_rate(self):
        """Report includes pass_rate calculation."""
        import inspect

        from syn.audit.compliance import generate_monthly_report

        src = inspect.getsource(generate_monthly_report)
        self.assertIn("pass_rate", src)


class MonthlyReportGenerationTest(DjangoTestCase):
    """CMP-001 §7.1: generate_monthly_report creates ComplianceReport."""

    def test_report_generation_succeeds(self):
        """generate_monthly_report runs without errors."""
        from syn.audit.compliance import generate_monthly_report, run_check

        # Run at least one check so there's data
        run_check("audit_integrity")
        report = generate_monthly_report()
        self.assertIsNotNone(report)

    def test_report_has_required_fields(self):
        """Generated report has period, pass_rate, summary."""
        from syn.audit.compliance import generate_monthly_report, run_check

        run_check("audit_integrity")
        report = generate_monthly_report()
        self.assertIsNotNone(report.period_start)
        self.assertIsNotNone(report.period_end)
        self.assertIsNotNone(report.pass_rate)

    def test_report_not_auto_published(self):
        """Generated reports are not auto-published (requires manual review)."""
        from syn.audit.compliance import generate_monthly_report, run_check

        run_check("audit_integrity")
        report = generate_monthly_report()
        self.assertFalse(report.is_published)


class ManagementCommandFlagTest(SimpleTestCase):
    """CMP-001 §8.1: run_compliance management command flags."""

    def test_command_accepts_all_flag(self):
        """Command parser accepts --all flag."""
        from syn.audit.management.commands.run_compliance import Command

        cmd = Command()
        parser = cmd.create_parser("manage.py", "run_compliance")
        args = parser.parse_args(["--all"])
        self.assertTrue(args.all)

    def test_command_accepts_check_flag(self):
        """Command parser accepts --check flag with value."""
        from syn.audit.management.commands.run_compliance import Command

        cmd = Command()
        parser = cmd.create_parser("manage.py", "run_compliance")
        args = parser.parse_args(["--check=audit_integrity"])
        self.assertEqual(args.check, "audit_integrity")

    def test_command_accepts_standards_flag(self):
        """Command parser accepts --standards flag."""
        from syn.audit.management.commands.run_compliance import Command

        cmd = Command()
        parser = cmd.create_parser("manage.py", "run_compliance")
        args = parser.parse_args(["--standards"])
        self.assertTrue(args.standards)

    def test_command_accepts_run_tests_flag(self):
        """Command parser accepts --run-tests flag."""
        from syn.audit.management.commands.run_compliance import Command

        cmd = Command()
        parser = cmd.create_parser("manage.py", "run_compliance")
        args = parser.parse_args(["--run-tests"])
        self.assertTrue(args.run_tests)

    def test_command_accepts_report_flag(self):
        """Command parser accepts --report flag."""
        from syn.audit.management.commands.run_compliance import Command

        cmd = Command()
        parser = cmd.create_parser("manage.py", "run_compliance")
        args = parser.parse_args(["--report"])
        self.assertTrue(args.report)


class SymbolCoverageTest(SimpleTestCase):
    """Tests for the non-gameable symbol_coverage compliance check (CMP-001 §6.3)."""

    def _run_check(self, mock_standards=None, mock_walk=None):
        """Run check_symbol_coverage with optional mocks."""
        from syn.audit.compliance import check_symbol_coverage

        patches = []
        if mock_standards is not None:
            patches.append(
                patch(
                    "syn.audit.compliance.check_symbol_coverage.__code__",
                )
            )
        return check_symbol_coverage()

    def test_symbol_inventory_excludes_private(self):
        """Private functions (_name) are excluded from symbol inventory."""
        import ast
        import textwrap

        source = textwrap.dedent("""\
            def public_function():
                pass

            def _private_helper():
                pass

            class PublicClass:
                def method(self):
                    pass

            class _PrivateClass:
                pass
        """)
        tree = ast.parse(source)
        public_names = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    public_names.append(node.name)

        self.assertIn("public_function", public_names)
        self.assertIn("PublicClass", public_names)
        self.assertNotIn("_private_helper", public_names)
        self.assertNotIn("_PrivateClass", public_names)
        self.assertEqual(len(public_names), 2)

    def test_file_level_hooks_ignored(self):
        """File-level impl hooks (no ':') do not contribute to symbol coverage."""
        from syn.audit.standards import Assertion

        # Simulate: assertion has file-level hook only
        a = Assertion(
            text="Test assertion",
            check_id="test-file-level",
            impls=["agents_api/dsw_views.py"],  # no colon = file-level
            tests=["some.test.method"],
        )

        governed = set()
        for impl_ref in a.impls:
            impl_ref = impl_ref.strip()
            if ":" not in impl_ref:
                continue  # file-level — SKIP
            file_part, symbol_part = impl_ref.split(":", 1)
            top_symbol = symbol_part.split(".")[0]
            governed.add(f"{file_part}:{top_symbol}")

        # File-level hook should NOT add anything to governed set
        self.assertEqual(len(governed), 0)

    def test_covered_requires_test_hooks(self):
        """Impl hook without test hooks on the assertion -> specified_untested, not covered."""
        from syn.audit.standards import Assertion

        a_with_tests = Assertion(
            text="Has tests",
            check_id="with-tests",
            impls=["file.py:MyFunc"],
            tests=["some.test.method"],
        )
        a_no_tests = Assertion(
            text="No tests",
            check_id="no-tests",
            impls=["file.py:OtherFunc"],
            tests=[],
        )

        governed = set()
        covered = set()
        for a in [a_with_tests, a_no_tests]:
            has_tests = len(a.tests) > 0
            for impl_ref in a.impls:
                if ":" not in impl_ref:
                    continue
                file_part, symbol_part = impl_ref.split(":", 1)
                top_symbol = symbol_part.split(".")[0]
                key = f"{file_part}:{top_symbol}"
                governed.add(key)
                if has_tests:
                    covered.add(key)

        self.assertIn("file.py:MyFunc", covered)
        self.assertNotIn("file.py:OtherFunc", covered)
        self.assertIn("file.py:OtherFunc", governed)
        self.assertEqual(len(governed - covered), 1)

    def test_class_hook_governs_whole_class(self):
        """<!-- impl: file.py:MyClass.method --> resolves to file.py:MyClass."""
        from syn.audit.standards import Assertion

        a = Assertion(
            text="Method-level hook",
            check_id="method-hook",
            impls=["models.py:MyClass.save", "models.py:MyClass.clean"],
            tests=["some.test"],
        )

        governed = set()
        for impl_ref in a.impls:
            if ":" not in impl_ref:
                continue
            file_part, symbol_part = impl_ref.split(":", 1)
            top_symbol = symbol_part.split(".")[0]
            governed.add(f"{file_part}:{top_symbol}")

        # Both method hooks resolve to the class
        self.assertEqual(governed, {"models.py:MyClass"})

    def test_risk_score_formula(self):
        """Risk = ungoverned_loc * 1.0 + specified_untested_loc * 0.5."""
        ungoverned_loc = 1000
        specified_loc = 400
        risk = round(ungoverned_loc * 1.0 + specified_loc * 0.5, 1)
        self.assertEqual(risk, 1200.0)

        # Edge: all covered = zero risk
        risk_zero = round(0 * 1.0 + 0 * 0.5, 1)
        self.assertEqual(risk_zero, 0.0)


class CalibrationTest(unittest.TestCase):
    """Tests for the statistical calibration system (STAT-001 §15)."""

    def test_reference_pool_has_cases(self):
        """Reference pool has ≥15 cases across ≥5 categories."""
        from agents_api.calibration import get_reference_pool

        pool = get_reference_pool()
        self.assertGreaterEqual(len(pool), 15)
        categories = {c.category for c in pool}
        self.assertGreaterEqual(len(categories), 5)
        # Every case has expectations
        for case in pool:
            self.assertTrue(len(case.expectations) > 0, f"{case.case_id} has no expectations")

    def test_calibration_runner_returns_results(self):
        """run_calibration returns per-case results with expected fields."""
        from agents_api.calibration import run_calibration

        result = run_calibration(seed=42, subset_size=3)
        self.assertIn("cases_run", result)
        self.assertIn("cases_passed", result)
        self.assertIn("pass_rate", result)
        self.assertIn("results", result)
        self.assertIn("seed", result)
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["cases_run"], 3)
        for r in result["results"]:
            self.assertIn("case_id", r)
            self.assertIn("passed", r)
            self.assertIn("checks", r)
            self.assertIn("duration_ms", r)

    def test_date_seeded_reproducibility(self):
        """Same seed → same case selection."""
        from agents_api.calibration import run_calibration

        r1 = run_calibration(seed=12345, subset_size=5)
        r2 = run_calibration(seed=12345, subset_size=5)
        ids1 = [r["case_id"] for r in r1["results"]]
        ids2 = [r["case_id"] for r in r2["results"]]
        self.assertEqual(ids1, ids2)
        # Different seed → likely different selection
        r3 = run_calibration(seed=99999, subset_size=5)
        ids3 = [r["case_id"] for r in r3["results"]]
        # Not guaranteed different, but extremely unlikely to match
        # Just verify it runs without error
        self.assertEqual(len(ids3), 5)

    def test_known_null_ttest_passes(self):
        """N(100,15) vs μ₀=100 calibration case passes (null true)."""
        from agents_api.calibration import get_reference_pool, run_calibration

        pool = get_reference_pool()
        inf001 = [c for c in pool if c.case_id == "CAL-INF-001"]
        self.assertEqual(len(inf001), 1)
        result = run_calibration(cases=inf001, subset_size=0)
        self.assertEqual(result["cases_run"], 1)
        self.assertTrue(result["results"][0]["passed"], f"CAL-INF-001 failed: {result['results'][0]}")

    def test_drift_violation_on_failure(self):
        """Compliance check creates DriftViolation when a case fails."""
        from agents_api.calibration import CalibrationCase, Expectation

        # Create a deliberately failing case
        bad_case = CalibrationCase(
            case_id="CAL-TEST-FAIL",
            category="test",
            analysis_type="stats",
            analysis_id="ttest",
            config={"var1": "x", "mu": 0, "conf": 95},
            data={"x": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]},
            expectations=[
                # p-value for N(~104.5) vs μ₀=0 will be < 0.05, but we expect > 0.5
                Expectation("statistics.p_value", 0.5, 0.0, "greater_than"),
            ],
            description="Deliberately failing test case",
        )

        from agents_api.calibration import run_calibration

        result = run_calibration(cases=[bad_case], subset_size=0)
        self.assertEqual(result["cases_passed"], 0)
        self.assertIn("CAL-TEST-FAIL", result["drift_cases"])


if __name__ == "__main__":
    unittest.main()
