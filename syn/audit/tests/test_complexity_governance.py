"""
Complexity governance functional tests.

Behavioral tests for check_complexity_governance (CAL-001 §8, ARCH-001 §7).
Verifies exemption accuracy, threshold enforcement, and check correctness.

Standard: CAL-001 §8
INIT-012: NASA-Grade QMS Gap Closure
<!-- test: syn.audit.tests.test_complexity_governance -->
"""

from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from syn.audit.compliance import (
    _COMPLEXITY_EXEMPTIONS,
    check_complexity_governance,
)

WEB_ROOT = Path(settings.BASE_DIR)

# Thresholds from ARCH-001 §7 / CAL-001 §8.1
WARN_THRESHOLD = 2000
FAIL_THRESHOLD = 3000


def _get_source_files():
    """Yield (relative_path, line_count) for all non-test Python source files."""
    for py_file in sorted(WEB_ROOT.rglob("*.py")):
        rel = str(py_file.relative_to(WEB_ROOT))
        if any(
            skip in rel
            for skip in [
                "/migrations/",
                "/test",
                "/__pycache__/",
                "/staticfiles/",
                "_tests.py",
            ]
        ):
            continue
        try:
            line_count = sum(1 for _ in py_file.open())
        except OSError:
            continue
        yield rel, line_count


# ── Exemption accuracy ──


class ExemptionAccuracyTest(SimpleTestCase):
    """Every exemption corresponds to a real >3000-line file and vice versa."""

    def test_no_unexempted_oversize_files(self):
        """All files >3000 lines are in _COMPLEXITY_EXEMPTIONS."""
        unexempted = []
        for rel, lc in _get_source_files():
            if lc > FAIL_THRESHOLD:
                is_exempt = any(rel.endswith(path) for path in _COMPLEXITY_EXEMPTIONS)
                if not is_exempt:
                    unexempted.append(f"{rel} ({lc} lines)")
        self.assertEqual(
            unexempted,
            [],
            "Files >3000 lines without exemption:\n  " + "\n  ".join(unexempted),
        )

    def test_all_exemptions_still_needed(self):
        """Every entry in _COMPLEXITY_EXEMPTIONS corresponds to a file still >3000 lines."""
        actual_oversized = set()
        for rel, lc in _get_source_files():
            if lc > FAIL_THRESHOLD:
                actual_oversized.add(rel)

        stale = []
        for exempt_path in _COMPLEXITY_EXEMPTIONS:
            matched = any(f.endswith(exempt_path) for f in actual_oversized)
            if not matched:
                stale.append(exempt_path)
        self.assertEqual(
            stale,
            [],
            "Stale exemptions (file no longer >3000):\n  " + "\n  ".join(stale),
        )

    def test_exemption_count_matches_oversized_count(self):
        """Number of exemptions equals number of >3000-line files."""
        oversized_count = sum(1 for _, lc in _get_source_files() if lc > FAIL_THRESHOLD)
        self.assertEqual(
            len(_COMPLEXITY_EXEMPTIONS),
            oversized_count,
            f"Exemptions ({len(_COMPLEXITY_EXEMPTIONS)}) != oversized files ({oversized_count})",
        )


# ── Threshold enforcement ──


class ThresholdEnforcementTest(SimpleTestCase):
    """Check correctly classifies files by threshold."""

    def test_check_returns_zero_violations(self):
        """With all exemptions in place, no file triggers a violation (fail)."""
        result = check_complexity_governance()
        violations = result["details"]["violations"]
        self.assertEqual(
            violations,
            [],
            f"Unexpected violations: {violations}",
        )

    def test_warnings_are_between_thresholds(self):
        """Every warning entry is for a file between 2000 and 3000 lines."""
        result = check_complexity_governance()
        for w in result["details"]["warnings"]:
            self.assertGreater(
                w["lines"],
                WARN_THRESHOLD,
                f"{w['file']} has {w['lines']} lines — below warning threshold",
            )

    def test_exempted_files_not_in_violations(self):
        """Exempted files >3000 lines do not appear in violations list."""
        result = check_complexity_governance()
        violation_files = {v["file"] for v in result["details"]["violations"]}
        for exempt_path in _COMPLEXITY_EXEMPTIONS:
            matched = [f for f in violation_files if f.endswith(exempt_path)]
            self.assertEqual(
                matched,
                [],
                f"Exempted file {exempt_path} appeared in violations",
            )

    def test_check_status_is_pass_or_warning(self):
        """With all exemptions, check returns pass or warning — never fail."""
        result = check_complexity_governance()
        self.assertIn(
            result["status"],
            ("pass", "warning"),
            f"Expected pass/warning, got {result['status']}",
        )


# ── Check structure ──


class CheckStructureTest(SimpleTestCase):
    """Check returns well-formed output."""

    def test_files_checked_positive(self):
        """Check scans a meaningful number of files."""
        result = check_complexity_governance()
        self.assertGreater(result["details"]["files_checked"], 100)

    def test_soc2_control_mapped(self):
        """Check declares CC4.1 SOC 2 control."""
        result = check_complexity_governance()
        self.assertIn("CC4.1", result.get("soc2_controls", []))

    def test_exemptions_list_returned(self):
        """Check returns its exemption list in output for auditability."""
        result = check_complexity_governance()
        exemptions = result["details"].get("exemptions", [])
        self.assertEqual(
            len(exemptions),
            len(_COMPLEXITY_EXEMPTIONS),
            f"Exemptions in output ({len(exemptions)}) != defined ({len(_COMPLEXITY_EXEMPTIONS)})",
        )

    def test_warnings_include_file_and_lines(self):
        """Each warning entry has file, lines, and threshold fields."""
        result = check_complexity_governance()
        for w in result["details"]["warnings"]:
            self.assertIn("file", w)
            self.assertIn("lines", w)
            self.assertIn("threshold", w)
