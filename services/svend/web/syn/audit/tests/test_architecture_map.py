"""
MAP-001 compliance tests: Architecture Map Standard.

Tests the standards registry table, module map table, table parsers,
phantom reference scanner, git integration, and check registration.

Standard: MAP-001
"""

from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.test import SimpleTestCase

from syn.audit.compliance import (
    _GIT_ROOT,
    _MAP_STANDARDS_DIR,
    ALL_CHECKS,
    _check_mapped_paths_exist,
    _find_unmapped_modules,
    _git_head_sha,
    _map_files_to_standards,
    _parse_map_table,
    _scan_phantom_references,
)


class RegistryTableTest(SimpleTestCase):
    """MAP-001 §4: Standards registry table."""

    def setUp(self):
        self.map_path = _MAP_STANDARDS_DIR / "MAP-001.md"
        self.registry = _parse_map_table(self.map_path, "standards-registry")

    def test_approved_standards_have_files(self):
        """Every APPROVED entry in registry has an existing .md file."""
        for entry in self.registry:
            if entry.get("status") == "APPROVED":
                file_path = _GIT_ROOT / entry["file"]
                self.assertTrue(
                    file_path.exists(),
                    f"APPROVED standard {entry['id']} file not found: {entry['file']}",
                )

    def test_registry_covers_all_md_files(self):
        """Every .md in docs/standards/ appears in the registry."""
        registry_files = {
            entry["file"] for entry in self.registry if entry.get("file") != "—"
        }
        for md_file in sorted(_MAP_STANDARDS_DIR.glob("*.md")):
            rel_path = f"docs/standards/{md_file.name}"
            self.assertIn(
                rel_path,
                registry_files,
                f"{md_file.name} exists on disk but is not in MAP-001 registry",
            )

    def test_no_duplicate_ids(self):
        """Registry has no duplicate standard IDs."""
        ids = [entry["id"] for entry in self.registry]
        self.assertEqual(
            len(ids),
            len(set(ids)),
            f"Duplicate IDs: {[i for i in ids if ids.count(i) > 1]}",
        )

    def test_phantom_entries_have_no_file(self):
        """PHANTOM entries have '—' as their file path."""
        for entry in self.registry:
            if entry.get("status") == "PHANTOM":
                self.assertEqual(
                    entry.get("file"),
                    "—",
                    f"PHANTOM standard {entry['id']} should have '—' as file, got: {entry.get('file')}",
                )

    def test_registry_has_minimum_entries(self):
        """Registry has at least 25 APPROVED entries and tracks deprecated IDs."""
        approved = sum(1 for e in self.registry if e.get("status") == "APPROVED")
        deprecated = sum(1 for e in self.registry if e.get("status") == "DEPRECATED")
        self.assertGreaterEqual(approved, 25, f"Expected ≥25 APPROVED, got {approved}")
        self.assertGreaterEqual(
            deprecated, 30, f"Expected ≥30 DEPRECATED, got {deprecated}"
        )

    def test_deprecated_entries_have_no_file(self):
        """DEPRECATED entries have '—' as their file path."""
        for entry in self.registry:
            if entry.get("status") == "DEPRECATED":
                self.assertEqual(
                    entry.get("file"),
                    "—",
                    f"DEPRECATED standard {entry['id']} should have '—' as file, got: {entry.get('file')}",
                )

    def test_planned_entries_have_no_file(self):
        """PLANNED entries have '—' as their file path."""
        for entry in self.registry:
            if entry.get("status") == "PLANNED":
                self.assertEqual(
                    entry.get("file"),
                    "—",
                    f"PLANNED standard {entry['id']} should have '—' as file, got: {entry.get('file')}",
                )


class ModuleMapTest(SimpleTestCase):
    """MAP-001 §5: Module architecture map."""

    def setUp(self):
        self.map_path = _MAP_STANDARDS_DIR / "MAP-001.md"
        self.module_map = _parse_map_table(self.map_path, "module-map")
        self.registry = _parse_map_table(self.map_path, "standards-registry")
        self.web_root = Path(settings.BASE_DIR)

    def test_all_syn_modules_mapped(self):
        """Every syn/ subdirectory with .py files appears in module map."""
        module_paths = [e.get("module", "") for e in self.module_map]
        unmapped = _find_unmapped_modules(module_paths, self.web_root)
        self.assertEqual(
            unmapped,
            [],
            f"syn/ directories not in module map: {unmapped}",
        )

    def test_mapped_paths_exist(self):
        """Every module map path exists on disk."""
        missing = _check_mapped_paths_exist(self.module_map, self.web_root)
        self.assertEqual(
            missing,
            [],
            f"Module map paths that don't exist: {[m['module_path'] for m in missing]}",
        )

    def test_governing_standards_in_registry(self):
        """All standard IDs referenced in module map exist in registry."""
        registry_ids = {e.get("id", "") for e in self.registry}
        for entry in self.module_map:
            standards_str = entry.get("standards", "")
            for std_id in [s.strip() for s in standards_str.split(",")]:
                if std_id and std_id != "—":
                    self.assertIn(
                        std_id,
                        registry_ids,
                        f"Module {entry['module']} references {std_id} not in registry",
                    )


class TableParserTest(SimpleTestCase):
    """MAP-001 §4: Table parsing helpers."""

    def test_parses_registry_table(self):
        """Parser extracts correct number of entries from MAP-001."""
        map_path = _MAP_STANDARDS_DIR / "MAP-001.md"
        registry = _parse_map_table(map_path, "standards-registry")
        self.assertGreater(
            len(registry), 30, f"Expected >30 registry entries, got {len(registry)}"
        )

    def test_parses_module_map(self):
        """Parser extracts module map entries with correct keys."""
        map_path = _MAP_STANDARDS_DIR / "MAP-001.md"
        module_map = _parse_map_table(map_path, "module-map")
        self.assertGreater(
            len(module_map),
            15,
            f"Expected >15 module map entries, got {len(module_map)}",
        )
        for entry in module_map[:3]:
            self.assertIn("module", entry)
            self.assertIn("standards", entry)

    def test_missing_file_returns_empty(self):
        """Returns empty list if MAP-001.md doesn't exist."""
        result = _parse_map_table(Path("/nonexistent/MAP-001.md"), "standards-registry")
        self.assertEqual(result, [])

    def test_missing_marker_returns_empty(self):
        """Returns empty list if table marker doesn't exist."""
        map_path = _MAP_STANDARDS_DIR / "MAP-001.md"
        result = _parse_map_table(map_path, "nonexistent-table")
        self.assertEqual(result, [])


class PhantomScannerTest(SimpleTestCase):
    """MAP-001 §7: Phantom reference detection."""

    def test_no_unregistered_phantoms(self):
        """All standard ID references in syn/ code are in the registry."""
        map_path = _MAP_STANDARDS_DIR / "MAP-001.md"
        registry = _parse_map_table(map_path, "standards-registry")
        registry_ids = {e.get("id", "") for e in registry}

        web_root = Path(settings.BASE_DIR)
        phantoms = _scan_phantom_references(registry_ids, web_root / "syn")

        if phantoms:
            # Group by ID for readable output
            by_id = {}
            for p in phantoms:
                by_id.setdefault(p["id"], []).append(p["file"])
            detail = "; ".join(f"{k}: {v[0]}" for k, v in sorted(by_id.items())[:5])
            self.fail(f"Unregistered phantom references found: {detail}")

    def test_registered_standards_not_flagged(self):
        """Standards in the registry are not flagged as phantoms."""
        registry_ids = {"ERR-001", "AUD-001", "SCH-001"}
        web_root = Path(settings.BASE_DIR)
        phantoms = _scan_phantom_references(registry_ids, web_root / "syn" / "err")
        phantom_ids = {p["id"] for p in phantoms}
        self.assertNotIn("ERR-001", phantom_ids)


class GitIntegrationTest(SimpleTestCase):
    """MAP-001 §6: Git-based change tracking."""

    def test_git_head_sha_format(self):
        """_git_head_sha returns a 40-char hex string or None."""
        sha = _git_head_sha()
        if sha is not None:
            self.assertEqual(len(sha), 40, f"SHA should be 40 chars, got {len(sha)}")
            self.assertTrue(all(c in "0123456789abcdef" for c in sha))

    def test_file_to_standard_mapping(self):
        """Changed files correctly map to affected standards via module map."""
        module_map = [
            {"module": "syn/audit/", "standards": "AUD-001, CMP-001"},
            {"module": "syn/err/", "standards": "ERR-001"},
        ]
        changed = [
            "services/svend/web/syn/audit/compliance.py",
            "services/svend/web/syn/err/exceptions.py",
        ]
        result = _map_files_to_standards(changed, module_map)
        self.assertIn("AUD-001", result)
        self.assertIn("CMP-001", result)
        self.assertIn("ERR-001", result)

    @patch("syn.audit.compliance.subprocess.run")
    def test_git_failure_graceful(self, mock_run):
        """Git command failure returns None gracefully."""
        mock_run.side_effect = Exception("git not found")
        sha = _git_head_sha()
        self.assertIsNone(sha)

    def test_unrelated_paths_not_mapped(self):
        """Files outside web/ are not mapped to any standard."""
        module_map = [
            {"module": "syn/audit/", "standards": "AUD-001"},
        ]
        changed = ["docs/standards/MAP-001.md", "README.md"]
        result = _map_files_to_standards(changed, module_map)
        self.assertEqual(result, {})


class CheckRegistrationTest(SimpleTestCase):
    """MAP-001: Check registered in ALL_CHECKS."""

    def test_check_registered(self):
        """'architecture_map' is registered in ALL_CHECKS."""
        self.assertIn("architecture_map", ALL_CHECKS)

    def test_check_is_callable(self):
        """The check function is callable."""
        entry = ALL_CHECKS["architecture_map"]
        fn = entry[0] if isinstance(entry, tuple) else entry
        self.assertTrue(callable(fn))

    def test_check_returns_valid_structure(self):
        """The check returns dict with required keys."""
        entry = ALL_CHECKS["architecture_map"]
        fn = entry[0] if isinstance(entry, tuple) else entry
        result = fn()
        for key in ["status", "details", "soc2_controls"]:
            self.assertIn(key, result, f"Check result missing key: {key}")
        self.assertIn(result["status"], ("pass", "fail", "warning", "error"))
