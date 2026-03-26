"""
ARCH-001 compliance tests: Architecture & Structure Standard.

Tests verify canonical directory structure, layer boundaries,
file size growth limits, prohibited directories, and root file hygiene.

Standard: ARCH-001
Compliance: SOC 2 CC7.2, CC8.1
"""

from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from syn.audit.compliance import (
    _ARCH_CORE_LAYER_EXEMPT,
    _ARCH_CROSS_IMPORT_EXEMPT_FILES,
    _ARCH_FEATURE_APPS,
    _ARCH_FILE_SIZE_FAIL,
    _ARCH_FILE_SIZE_WARN,
    _ARCH_FILES_PER_APP_FAIL,
    _ARCH_FILES_PER_APP_WARN,
    _ARCH_KNOWN_LARGE_APPS,
    _ARCH_REQUIRED_DIRS,
    ALL_CHECKS,
    _check_arch_core_layer,
    _check_arch_cross_imports,
    _check_arch_dir_naming,
    _check_arch_empty_dirs,
    _check_arch_file_sizes,
    _check_arch_files_per_app,
    _check_arch_layer_boundaries,
    _check_arch_nested_duplicates,
    _check_arch_prohibited_dirs,
    _check_arch_required_dirs,
    _check_arch_root_files,
    _check_arch_test_init,
    _check_arch_test_placement,
    _check_arch_testcase_in_prod,
)

WEB_ROOT = Path(settings.BASE_DIR)


class DirectoryStructureTest(SimpleTestCase):
    """ARCH-001 §4: Canonical directory structure."""

    def test_required_dirs_exist(self):
        missing = _check_arch_required_dirs(WEB_ROOT)
        self.assertEqual(missing, [], f"Missing required dirs: {missing}")

    def test_syn_subdirectories_exist(self):
        syn_subdirs = [d for d in _ARCH_REQUIRED_DIRS if d.startswith("syn/")]
        for subdir in syn_subdirs:
            self.assertTrue(
                (WEB_ROOT / subdir).is_dir(),
                f"syn subdirectory missing: {subdir}",
            )

    def test_settings_py_exists(self):
        self.assertTrue((WEB_ROOT / "svend" / "settings.py").is_file())


class ProhibitedDirTest(SimpleTestCase):
    """ARCH-001 §8: No prohibited directories."""

    def test_no_prohibited_dirs(self):
        found = _check_arch_prohibited_dirs(WEB_ROOT)
        self.assertEqual(found, [], f"Prohibited dirs found: {found}")

    def test_no_tempora_dir(self):
        self.assertFalse(
            (WEB_ROOT / "tempora").is_dir(),
            "tempora/ still exists — should have been removed (replaced by syn/sched/)",
        )

    def test_no_forge_results_dir(self):
        self.assertFalse(
            (WEB_ROOT / "forge_results").is_dir(),
            "forge_results/ still exists — empty directory, remove it",
        )


class EmptyDirTest(SimpleTestCase):
    """ARCH-001 §8: No empty directories."""

    def test_no_empty_dirs(self):
        empty = _check_arch_empty_dirs(WEB_ROOT)
        self.assertEqual(empty, [], f"Empty directories found: {empty}")


class LayerBoundaryTest(SimpleTestCase):
    """ARCH-001 §5: Layer boundary enforcement."""

    def test_syn_no_feature_imports(self):
        violations = _check_arch_layer_boundaries(WEB_ROOT)
        if violations:
            detail = "\n".join(
                f"  {v['file']}:{v['line']} imports {v['import']}"
                for v in violations[:10]
            )
            self.fail(f"syn/ imports from feature apps:\n{detail}")

    def test_feature_apps_list_complete(self):
        expected = {"agents_api", "chat", "workbench", "forge", "files", "inference"}
        self.assertEqual(_ARCH_FEATURE_APPS, expected)


class GrowthBoundaryTest(SimpleTestCase):
    """ARCH-001 §7: File size growth boundaries."""

    def test_no_oversized_files(self):
        oversized = _check_arch_file_sizes(WEB_ROOT)
        failures = [f for f in oversized if f["severity"] == "fail"]
        if failures:
            detail = "\n".join(
                f"  {f['file']}: {f['lines']} lines" for f in failures[:10]
            )
            self.fail(f"Files exceed {_ARCH_FILE_SIZE_FAIL} lines:\n{detail}")

    def test_threshold_constants(self):
        self.assertEqual(_ARCH_FILE_SIZE_WARN, 2000)
        self.assertEqual(_ARCH_FILE_SIZE_FAIL, 3000)


class RootFileTest(SimpleTestCase):
    """ARCH-001 §10: Root file hygiene."""

    def test_no_unexpected_root_files(self):
        unexpected = _check_arch_root_files(WEB_ROOT)
        self.assertEqual(
            unexpected,
            [],
            f"Unexpected files at web root: {unexpected}",
        )

    def test_manage_py_exists(self):
        self.assertTrue((WEB_ROOT / "manage.py").is_file())


class DirectoryNamingTest(SimpleTestCase):
    """ARCH-001 §6 / STY-001 §4.1: Directory naming conventions."""

    def test_no_naming_violations(self):
        violations = _check_arch_dir_naming(WEB_ROOT)
        self.assertEqual(violations, [], f"Directory naming violations: {violations}")

    def test_all_app_dirs_snake_case(self):
        import re

        pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        for app in _ARCH_FEATURE_APPS:
            self.assertTrue(
                pattern.match(app),
                f"Feature app name not snake_case: {app}",
            )


class NestedDuplicateTest(SimpleTestCase):
    """ARCH-001 §10: No nested duplicate directories."""

    def test_no_nested_duplicates(self):
        violations = _check_arch_nested_duplicates(WEB_ROOT)
        self.assertEqual(violations, [], f"Nested duplicate dirs: {violations}")


class FilesPerAppTest(SimpleTestCase):
    """ARCH-001 §7: Files-per-app growth boundaries."""

    def test_no_oversized_apps(self):
        oversized = _check_arch_files_per_app(WEB_ROOT)
        failures = [f for f in oversized if f["severity"] == "fail"]
        if failures:
            detail = "\n".join(f"  {f['app']}: {f['files']} files" for f in failures)
            self.fail(f"Apps exceed {_ARCH_FILES_PER_APP_FAIL} files:\n{detail}")

    def test_threshold_constants(self):
        self.assertEqual(_ARCH_FILES_PER_APP_WARN, 20)
        self.assertEqual(_ARCH_FILES_PER_APP_FAIL, 30)

    def test_known_large_apps_tracked(self):
        self.assertIn("agents_api", _ARCH_KNOWN_LARGE_APPS)


class CoreLayerTest(SimpleTestCase):
    """ARCH-001 §5: core/ must not import from feature apps."""

    def test_no_core_feature_imports(self):
        violations = _check_arch_core_layer(WEB_ROOT)
        if violations:
            detail = "\n".join(
                f"  {v['file']}:{v['line']} imports {v['import']}"
                for v in violations[:10]
            )
            self.fail(f"core/ imports from feature apps:\n{detail}")

    def test_exempt_list_documented(self):
        self.assertIn("seed_nlp_demo.py", _ARCH_CORE_LAYER_EXEMPT)


class CrossImportTest(SimpleTestCase):
    """ARCH-001 §5: Feature apps should not cross-import."""

    def test_cross_imports_are_warnings(self):
        """Cross-imports produce warnings, not failures (tracked debt)."""
        violations = _check_arch_cross_imports(WEB_ROOT)
        # Cross-imports are soft warnings, not hard failures.
        # This test documents the current state.
        for v in violations:
            self.assertIn("from_app", v)
            self.assertIn("to_app", v)
            self.assertNotEqual(v["from_app"], v["to_app"])

    def test_management_commands_exempt(self):
        self.assertIn("management", _ARCH_CROSS_IMPORT_EXEMPT_FILES)


class TestPlacementTest(SimpleTestCase):
    """ARCH-001 §6: Test files live in tests/ directories."""

    def test_misplaced_tests_are_warnings(self):
        """Misplaced test files are tracked as debt (soft warning)."""
        violations = _check_arch_test_placement(WEB_ROOT)
        # These are soft warnings — existing debt to be migrated.
        for v in violations:
            self.assertIn(".py", v)

    def test_syn_tests_properly_placed(self):
        """syn/ tests are all in tests/ directories."""
        violations = _check_arch_test_placement(WEB_ROOT)
        syn_violations = [v for v in violations if v.startswith("syn/")]
        self.assertEqual(
            syn_violations, [], f"syn/ tests outside tests/ dirs: {syn_violations}"
        )


class TestInitTest(SimpleTestCase):
    """ARCH-001 §6: tests/ directories have __init__.py."""

    def test_all_test_dirs_have_init(self):
        missing = _check_arch_test_init(WEB_ROOT)
        self.assertEqual(missing, [], f"tests/ dirs missing __init__.py: {missing}")


class TestCaseInProdTest(SimpleTestCase):
    """ARCH-001 §6: No TestCase subclasses in production code."""

    def test_no_testcase_in_prod(self):
        violations = _check_arch_testcase_in_prod(WEB_ROOT)
        if violations:
            detail = "\n".join(
                f"  {v['file']}:{v['line']} class {v['class']}({v['base']})"
                for v in violations[:10]
            )
            self.fail(f"TestCase subclasses in non-test files:\n{detail}")


class CheckRegistrationTest(SimpleTestCase):
    """ARCH-001 §11: Compliance check registration."""

    def test_check_registered(self):
        self.assertIn("architecture", ALL_CHECKS)

    def test_check_is_callable(self):
        fn, _category = ALL_CHECKS["architecture"]
        self.assertTrue(callable(fn))

    def test_check_returns_valid_structure(self):
        fn, _category = ALL_CHECKS["architecture"]
        result = fn()
        self.assertIn("status", result)
        self.assertIn("details", result)
        self.assertIn("soc2_controls", result)
        self.assertIn(result["status"], ("pass", "warning", "fail"))
