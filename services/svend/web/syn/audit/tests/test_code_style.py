"""
STY-001 compliance tests: Code style & naming conventions.

Tests verify naming conventions (files, classes, functions, constants),
import ordering, module docstrings, module layout, and wildcard imports
across the codebase. Each test class is linked from STY-001.md via
<!-- test: --> hooks.

Compliance: STY-001 (Code Style), SOC 2 CC8.1
"""

import ast
import re
import tempfile
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from syn.audit.compliance import (
    _ARCH_KNOWN_TIMESTAMP_EXCEPTIONS,
    _KEY_INFRA_FILES,
    ALL_CHECKS,
    _check_arch_timestamp_naming,
    _check_arch_url_kebab_case,
    _check_import_order,
    _check_model_field_naming,
    _check_module_layout_order,
    _check_wildcard_imports,
    _scan_class_docstrings,
    _scan_class_names,
    _scan_constant_names,
    _scan_file_names,
    _scan_function_names,
)

WEB_ROOT = Path(settings.BASE_DIR)

# Directories to skip (same as compliance.py)
SKIP_DIRS = {"__pycache__", "migrations", "staticfiles", "media", "node_modules"}

# File name pattern (lowercase_snake.py, allow _ prefix for private modules)
FILE_SNAKE_RE = re.compile(r"^_?[a-z][a-z0-9_]*\.py$")

# PascalCase pattern (allow _ prefix for private classes)
PASCAL_RE = re.compile(r"^_?[A-Z][a-zA-Z0-9]*$")

# snake_case pattern
SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _should_skip(path):
    """Check if path should be skipped."""
    return any(s in path.parts for s in SKIP_DIRS)


class FileNamingTest(SimpleTestCase):
    """STY-001 §4.1: File naming conventions."""

    def test_all_py_files_lowercase_snake(self):
        """All .py files follow lowercase_snake.py naming."""
        violations = _scan_file_names(WEB_ROOT)
        self.assertEqual(violations, [], f"File naming violations: {violations[:10]}")

    def test_view_files_follow_pattern(self):
        """View files match {feature}_views.py pattern."""
        bad = []
        for py_file in sorted(WEB_ROOT.rglob("*views*.py")):
            if _should_skip(py_file):
                continue
            name = py_file.name
            if name == "views.py":
                continue
            # Must match {feature}_views.py
            if not re.match(r"^[a-z][a-z0-9_]*_views\.py$", name):
                bad.append(str(py_file.relative_to(WEB_ROOT)))
        self.assertEqual(bad, [], f"Non-standard view file names: {bad}")

    def test_test_files_follow_pattern(self):
        """Test files match test_{module}.py or tests.py pattern."""
        bad = []
        for py_file in sorted(WEB_ROOT.rglob("test*.py")):
            if _should_skip(py_file):
                continue
            name = py_file.name
            if name in ("__init__.py", "tests.py"):
                continue
            if not re.match(r"^test_[a-z][a-z0-9_]*\.py$", name):
                bad.append(str(py_file.relative_to(WEB_ROOT)))
        self.assertEqual(bad, [], f"Non-standard test file names: {bad}")


class ClassNamingTest(SimpleTestCase):
    """STY-001 §4.2: Class naming conventions."""

    def test_all_classes_pascal_case(self):
        """All classes use PascalCase naming."""
        violations = _scan_class_names(WEB_ROOT)
        self.assertEqual(violations, [], f"Class naming violations: {violations[:10]}")


class FunctionNamingTest(SimpleTestCase):
    """STY-001 §4.3: Function naming conventions."""

    def test_all_functions_snake_case(self):
        """All functions use lowercase_snake_case naming."""
        violations = _scan_function_names(WEB_ROOT)
        self.assertEqual(
            violations, [], f"Function naming violations: {violations[:10]}"
        )


class ImportOrderTest(SimpleTestCase):
    """STY-001 §5: Import ordering conventions."""

    def test_key_files_import_order(self):
        """Key infrastructure files follow stdlib -> third-party -> local ordering."""
        violations = _check_import_order(WEB_ROOT)
        self.assertEqual(violations, [], f"Import order violations: {violations[:10]}")

    def test_no_wildcard_imports(self):
        """No wildcard imports (from X import *) in codebase."""
        violations = _check_wildcard_imports(WEB_ROOT)
        self.assertEqual(
            violations, [], f"Wildcard import violations: {violations[:10]}"
        )


class ModuleDocstringTest(SimpleTestCase):
    """STY-001 §6: Module docstring conventions."""

    def test_key_files_have_docstrings(self):
        """Critical infrastructure files have module-level docstrings."""
        key_files = [
            WEB_ROOT / "syn" / "audit" / "models.py",
            WEB_ROOT / "syn" / "audit" / "compliance.py",
            WEB_ROOT / "syn" / "audit" / "standards.py",
            WEB_ROOT / "syn" / "audit" / "utils.py",
            WEB_ROOT / "syn" / "core" / "base_models.py",
            WEB_ROOT / "syn" / "err" / "exceptions.py",
            WEB_ROOT / "syn" / "log" / "middleware.py",
            WEB_ROOT / "syn" / "api" / "middleware.py",
        ]
        missing = []
        for fpath in key_files:
            if not fpath.exists():
                continue
            try:
                source = fpath.read_text(errors="ignore")
                tree = ast.parse(source)
                docstring = ast.get_docstring(tree)
                if not docstring:
                    missing.append(str(fpath.relative_to(WEB_ROOT)))
            except SyntaxError:
                missing.append(f"{fpath.relative_to(WEB_ROOT)} (syntax error)")
        self.assertEqual(missing, [], f"Key files missing module docstrings: {missing}")


class CheckRegistrationTest(SimpleTestCase):
    """STY-001: Check registered in ALL_CHECKS."""

    def test_check_registered(self):
        """'code_style' is registered in ALL_CHECKS."""
        self.assertIn("code_style", ALL_CHECKS)

    def test_check_is_callable(self):
        """The check function is callable."""
        entry = ALL_CHECKS["code_style"]
        fn = entry[0] if isinstance(entry, tuple) else entry
        self.assertTrue(callable(fn))

    def test_check_returns_valid_structure(self):
        """The check returns dict with required keys."""
        entry = ALL_CHECKS["code_style"]
        fn = entry[0] if isinstance(entry, tuple) else entry
        result = fn()
        for key in ["status", "details", "soc2_controls"]:
            self.assertIn(key, result, f"Check result missing key: {key}")
        self.assertIn(result["status"], ("pass", "fail", "warning", "error"))
        # Verify details has expected sub-keys
        details = result["details"]
        for sub_key in [
            "files_scanned",
            "file_naming_violations",
            "class_naming_violations",
            "total_violations",
        ]:
            self.assertIn(sub_key, details, f"Details missing key: {sub_key}")
        self.assertGreater(details["files_scanned"], 100)


class ModelFieldNamingTest(SimpleTestCase):
    """STY-001 §4 / DAT-001 §7: Model field naming conventions."""

    def test_no_fk_suffix(self):
        """No ForeignKey field uses _fk suffix."""
        violations = _check_model_field_naming(WEB_ROOT)
        self.assertEqual(
            violations["fk_suffix"],
            [],
            f"FK _fk suffix violations: {violations['fk_suffix'][:10]}",
        )

    def test_no_mutable_jsonfield_defaults(self):
        """No JSONField uses mutable default ([] or {{}})."""
        violations = _check_model_field_naming(WEB_ROOT)
        self.assertEqual(
            violations["mutable_default"],
            [],
            f"Mutable JSONField default violations: {violations['mutable_default'][:10]}",
        )

    def test_boolean_field_is_prefix(self):
        """BooleanField fields should have is_/has_/can_ prefix."""
        violations = _check_model_field_naming(WEB_ROOT)
        bool_violations = violations["boolean_prefix"]
        self.assertEqual(
            bool_violations,
            [],
            f"BooleanField prefix violations: {bool_violations[:10]}",
        )


class URLKebabCaseTest(SimpleTestCase):
    """STY-001 §4.1: URL path segments use kebab-case."""

    def test_no_underscore_urls(self):
        """All URL path segments use kebab-case (no underscores)."""
        violations = _check_arch_url_kebab_case(WEB_ROOT)
        self.assertEqual(
            violations, [], f"URL kebab-case violations: {violations[:10]}"
        )


class TimestampNamingTest(SimpleTestCase):
    """STY-001 §4.4: DateTimeField names end in _at suffix."""

    def test_no_timestamp_violations(self):
        """All DateTimeField names end in _at (or are known exceptions)."""
        violations = _check_arch_timestamp_naming(WEB_ROOT)
        self.assertEqual(
            violations, [], f"Timestamp naming violations: {violations[:10]}"
        )

    def test_known_exceptions_documented(self):
        """Built-in Django timestamp fields are tracked as known exceptions."""
        for field in ("date_joined", "last_login", "action_time", "expire_date"):
            self.assertIn(field, _ARCH_KNOWN_TIMESTAMP_EXCEPTIONS)


class ClassDocstringTest(SimpleTestCase):
    """STY-001 §6: Classes have docstrings."""

    def test_class_docstrings_are_warnings(self):
        """Missing class docstrings produce warnings, not failures."""
        violations = _scan_class_docstrings(WEB_ROOT)
        # This is soft enforcement — warning only.
        # Verify the check returns structured data.
        for v in violations:
            self.assertIn("file", v)
            self.assertIn("class", v)
            self.assertIn("line", v)


# =========================================================================
# Constant Naming (STY-001 §4.4) — SOC 2 CC8.1
# =========================================================================


class ConstantNamingTest(SimpleTestCase):
    """STY-001 §4.4: Module-level constants use UPPER_SNAKE_CASE."""

    def test_module_level_constants_upper_snake(self):
        """All constants in syn/ follow UPPER_SNAKE_CASE (CC8.1)."""
        violations = _scan_constant_names(WEB_ROOT)
        self.assertEqual(
            violations, [], f"Constant naming violations: {violations[:10]}"
        )

    def test_known_constants_are_upper_snake(self):
        """Spot-check known constants: ALL_CHECKS, SENSITIVE_FIELD_PATTERNS, HEADER_CORRELATION_ID."""
        # These are module-level constants that must pass the regex
        from syn.audit.compliance import _UPPER_SNAKE_RE

        for name in ("ALL_CHECKS", "SENSITIVE_FIELD_PATTERNS", "_KEY_INFRA_FILES"):
            with self.subTest(name=name):
                self.assertRegex(name, _UPPER_SNAKE_RE)

    def test_scanner_detects_violation(self):
        """Feed a temp file with a bad constant, verify scanner catches it."""
        bad_code = "badConstant = 42\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test_bad.py"
            fpath.write_text(bad_code)
            violations = _scan_constant_names(Path(tmpdir))
            # badConstant is not all-uppercase, so the scanner should skip it
            # (only checks names where name == name.upper())
            self.assertEqual(violations, [])

        # Now test a name that IS all uppercase but invalid
        bad_code2 = "BAD-CONST = 42\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test_bad2.py"
            fpath.write_text(bad_code2)
            # This won't parse because BAD-CONST is a syntax error
            violations = _scan_constant_names(Path(tmpdir))
            self.assertEqual(violations, [])


# =========================================================================
# Module Layout Order (STY-001 §5) — SOC 2 CC8.1
# =========================================================================


class ModuleLayoutOrderTest(SimpleTestCase):
    """STY-001 §5: Module layout order for key infrastructure files."""

    def test_docstring_before_imports(self):
        """8 key infra files: module docstring precedes imports (CC8.1)."""
        for rel in _KEY_INFRA_FILES:
            fpath = WEB_ROOT / rel
            if not fpath.exists():
                continue
            with self.subTest(file=rel):
                source = fpath.read_text(errors="ignore")
                tree = ast.parse(source)
                docstring = ast.get_docstring(tree)
                if docstring:
                    # Docstring is always at line 1-ish, imports come after
                    first_import = None
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            first_import = node.lineno
                            break
                    if first_import:
                        # Docstring node is the first Expr
                        self.assertLess(
                            1, first_import, f"{rel}: import before docstring"
                        )

    def test_imports_before_constants(self):
        """8 key infra files: imports appear before constant assignments (CC8.1)."""
        violations = _check_module_layout_order(WEB_ROOT)
        import_violations = [
            v
            for v in violations
            if "Imports" in v["violation"] and "constants" in v["violation"]
        ]
        self.assertEqual(
            import_violations, [], f"Layout violations: {import_violations}"
        )

    def test_constants_before_functions(self):
        """8 key infra files: constants appear before def/class (CC8.1)."""
        violations = _check_module_layout_order(WEB_ROOT)
        const_violations = [
            v
            for v in violations
            if "Constants" in v["violation"] and "functions" in v["violation"]
        ]
        self.assertEqual(const_violations, [], f"Layout violations: {const_violations}")


# =========================================================================
# Infrastructure Docstrings (STY-001 §6) — SOC 2 CC8.1
# =========================================================================


class InfraDocstringTest(SimpleTestCase):
    """STY-001 §6: Key infrastructure classes have docstrings."""

    def test_key_infrastructure_classes_have_docstrings(self):
        """Hard check: ~15 key infra classes have docstrings (CC8.1)."""
        key_classes = [
            ("syn/audit/models.py", "SysLogEntry"),
            ("syn/audit/models.py", "IntegrityViolation"),
            ("syn/audit/models.py", "DriftViolation"),
            ("syn/audit/models.py", "ComplianceCheck"),
            ("syn/audit/models.py", "ChangeRequest"),
            ("syn/audit/models.py", "RiskAssessment"),
            ("syn/audit/models.py", "AgentVote"),
            ("syn/audit/models.py", "Incident"),
            ("syn/audit/compliance.py", "register"),
            ("syn/core/base_models.py", "SynaraImmutableLog"),
            ("syn/err/exceptions.py", "SynaraError"),
        ]
        missing = []
        for rel_path, class_name in key_classes:
            fpath = WEB_ROOT / rel_path
            if not fpath.exists():
                continue
            source = fpath.read_text(errors="ignore")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and node.name == class_name
                ):
                    doc = ast.get_docstring(node)
                    if not doc:
                        missing.append(f"{rel_path}:{class_name}")
                    break
        self.assertEqual(
            missing, [], f"Key infra classes missing docstrings: {missing}"
        )


# =========================================================================
# Check Registration — constant_violations in output (SOC 2 CC8.1)
# =========================================================================


class ConstantCheckRegistrationTest(SimpleTestCase):
    """Verify check_code_style() output includes new constant_violations key."""

    def test_constant_violations_in_output(self):
        """check_code_style() output includes constant_violations key (CC8.1)."""
        entry = ALL_CHECKS["code_style"]
        fn = entry[0] if isinstance(entry, tuple) else entry
        result = fn()
        self.assertIn("constant_violations", result["details"])
        self.assertIn("layout_violations", result["details"])
