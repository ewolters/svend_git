"""
STY-001 compliance tests: Code style & naming conventions.

Tests verify naming conventions (files, classes, functions), import ordering,
module docstrings, and wildcard imports across the codebase. Each test class
is linked from STY-001.md via <!-- test: --> hooks.

Compliance: STY-001 (Code Style), SOC 2 CC8.1
"""

import ast
import re
import sys
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from syn.audit.compliance import (
    ALL_CHECKS,
    _ARCH_KNOWN_TIMESTAMP_EXCEPTIONS,
    _scan_class_docstrings,
    _scan_class_names,
    _scan_file_names,
    _scan_function_names,
    _check_arch_timestamp_naming,
    _check_arch_url_kebab_case,
    _check_import_order,
    _check_model_field_naming,
    _check_module_docstrings,
    _check_wildcard_imports,
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
        self.assertEqual(
            violations, [],
            f"File naming violations: {violations[:10]}"
        )

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
        self.assertEqual(
            violations, [],
            f"Class naming violations: {violations[:10]}"
        )


class FunctionNamingTest(SimpleTestCase):
    """STY-001 §4.3: Function naming conventions."""

    def test_all_functions_snake_case(self):
        """All functions use lowercase_snake_case naming."""
        violations = _scan_function_names(WEB_ROOT)
        self.assertEqual(
            violations, [],
            f"Function naming violations: {violations[:10]}"
        )


class ImportOrderTest(SimpleTestCase):
    """STY-001 §5: Import ordering conventions."""

    def test_key_files_import_order(self):
        """Key infrastructure files follow stdlib -> third-party -> local ordering."""
        violations = _check_import_order(WEB_ROOT)
        self.assertEqual(
            violations, [],
            f"Import order violations: {violations[:10]}"
        )

    def test_no_wildcard_imports(self):
        """No wildcard imports (from X import *) in codebase."""
        violations = _check_wildcard_imports(WEB_ROOT)
        self.assertEqual(
            violations, [],
            f"Wildcard import violations: {violations[:10]}"
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
        self.assertEqual(
            missing, [],
            f"Key files missing module docstrings: {missing}"
        )


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
        for sub_key in ["files_scanned", "file_naming_violations",
                        "class_naming_violations", "total_violations"]:
            self.assertIn(sub_key, details, f"Details missing key: {sub_key}")
        self.assertGreater(details["files_scanned"], 100)


class ModelFieldNamingTest(SimpleTestCase):
    """STY-001 §4 / DAT-001 §7: Model field naming conventions."""

    def test_no_fk_suffix(self):
        """No ForeignKey field uses _fk suffix."""
        violations = _check_model_field_naming(WEB_ROOT)
        self.assertEqual(
            violations["fk_suffix"], [],
            f"FK _fk suffix violations: {violations['fk_suffix'][:10]}"
        )

    def test_no_mutable_jsonfield_defaults(self):
        """No JSONField uses mutable default ([] or {{}})."""
        violations = _check_model_field_naming(WEB_ROOT)
        self.assertEqual(
            violations["mutable_default"], [],
            f"Mutable JSONField default violations: {violations['mutable_default'][:10]}"
        )

    def test_boolean_field_is_prefix(self):
        """BooleanField fields should have is_/has_/can_ prefix."""
        violations = _check_model_field_naming(WEB_ROOT)
        bool_violations = violations["boolean_prefix"]
        self.assertEqual(
            bool_violations, [],
            f"BooleanField prefix violations: {bool_violations[:10]}"
        )


class URLKebabCaseTest(SimpleTestCase):
    """STY-001 §4.1: URL path segments use kebab-case."""

    def test_no_underscore_urls(self):
        """All URL path segments use kebab-case (no underscores)."""
        violations = _check_arch_url_kebab_case(WEB_ROOT)
        self.assertEqual(
            violations, [],
            f"URL kebab-case violations: {violations[:10]}"
        )


class TimestampNamingTest(SimpleTestCase):
    """STY-001 §4.4: DateTimeField names end in _at suffix."""

    def test_no_timestamp_violations(self):
        """All DateTimeField names end in _at (or are known exceptions)."""
        violations = _check_arch_timestamp_naming(WEB_ROOT)
        self.assertEqual(
            violations, [],
            f"Timestamp naming violations: {violations[:10]}"
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
