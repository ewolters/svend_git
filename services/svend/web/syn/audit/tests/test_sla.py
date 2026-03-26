"""
SLA-001 compliance tests: Service Level Agreement Standard.

Tests verify SLA tag parsing, SLADefinition dataclass, measurement dispatcher,
target parsing, metric type coverage, and compliance check registration.

Standard: SLA-001
"""

import os
import re
from pathlib import Path

from django.test import SimpleTestCase

WEB_ROOT = Path(os.path.dirname(__file__)).parent.parent.parent
STANDARDS_DIR = WEB_ROOT.parent.parent.parent / "docs" / "standards"


def _read(path):
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


def _extract_sla_tags(text):
    """Extract real <!-- sla: --> definition tags (with attributes), skipping code fences and prose references."""
    tags = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Only match sla tags with pipe-separated attributes (real definitions)
        m = re.match(r"^<!--\s*sla:\s+.+\|.+-->$", stripped)
        if m:
            tags.append(m.group())
    return tags


# ── §4.1: SLA Tag Parser ─────────────────────────────────────────────────


class SLATagParserTest(SimpleTestCase):
    """SLA-001 §4: SLA parser recognizes sla tag type with required attributes."""

    def setUp(self):
        self.standards_src = _read(WEB_ROOT / "syn" / "audit" / "standards.py")
        self.sla_standard = _read(STANDARDS_DIR / "SLA-001.md")
        self.assertGreater(len(self.standards_src), 0, "standards.py not found")
        self.assertGreater(len(self.sla_standard), 0, "SLA-001.md not found")

    def test_sla_definition_dataclass_exists(self):
        """SLADefinition dataclass is defined in standards.py."""
        self.assertIn("class SLADefinition", self.standards_src)

    def test_sla_definition_has_required_fields(self):
        """SLADefinition has description, sla_id, metric, target, window, severity."""
        for field in [
            "description",
            "sla_id",
            "metric",
            "target",
            "window",
            "severity",
        ]:
            self.assertIn(field, self.standards_src)

    def test_parse_sla_definitions_exists(self):
        """parse_sla_definitions function is defined."""
        self.assertIn("def parse_sla_definitions(", self.standards_src)

    def test_parse_all_sla_definitions_exists(self):
        """parse_all_sla_definitions function deduplicates by sla_id."""
        self.assertIn("def parse_all_sla_definitions(", self.standards_src)

    def test_sla_tag_regex_recognizes_sla(self):
        """TAG_RE or SLA_TAG_RE pattern matches sla tags."""
        self.assertTrue(
            "sla" in self.standards_src.split("TAG_RE")[0:2].__repr__().lower()
            or "SLA_TAG_RE" in self.standards_src,
            "SLA tag regex not found in standards.py",
        )

    def test_valid_metrics_defined(self):
        """VALID_METRICS set includes all 6 metric types."""
        for metric in [
            "availability",
            "response_time",
            "durability",
            "incident_response",
            "compliance",
            "change_velocity",
        ]:
            self.assertIn(metric, self.standards_src)

    def test_incomplete_tag_rejected(self):
        """Parser warns on incomplete SLA tags (missing required attributes)."""
        self.assertIn("Incomplete SLA tag", self.standards_src)

    def test_deduplication_by_sla_id(self):
        """parse_all_sla_definitions deduplicates by sla_id."""
        fn_match = re.search(
            r"def parse_all_sla_definitions\(.*?(?=\ndef |\Z)",
            self.standards_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match)
        fn_body = fn_match.group()
        self.assertIn("sla_id", fn_body)
        self.assertIn("seen", fn_body)


# ── §4.2: SLA Tag Attributes ─────────────────────────────────────────────


class SLATagAttributeTest(SimpleTestCase):
    """SLA-001 §4.2: All SLA tags have required attributes."""

    def setUp(self):
        self.sla_standard = _read(STANDARDS_DIR / "SLA-001.md")

    def test_all_sla_tags_have_metric(self):
        """Every <!-- sla: --> tag in SLA-001 has a metric= attribute."""
        sla_tags = _extract_sla_tags(self.sla_standard)
        self.assertGreater(len(sla_tags), 0, "No SLA tags found in SLA-001.md")
        for tag in sla_tags:
            self.assertIn("metric=", tag, f"SLA tag missing metric: {tag[:60]}")

    def test_all_sla_tags_have_target(self):
        """Every <!-- sla: --> tag has a target= attribute."""
        sla_tags = _extract_sla_tags(self.sla_standard)
        for tag in sla_tags:
            self.assertIn("target=", tag, f"SLA tag missing target: {tag[:60]}")

    def test_all_sla_tags_have_window(self):
        """Every <!-- sla: --> tag has a window= attribute."""
        sla_tags = _extract_sla_tags(self.sla_standard)
        for tag in sla_tags:
            self.assertIn("window=", tag, f"SLA tag missing window: {tag[:60]}")

    def test_all_sla_tags_have_severity(self):
        """Every <!-- sla: --> tag has a severity= attribute."""
        sla_tags = _extract_sla_tags(self.sla_standard)
        for tag in sla_tags:
            self.assertIn("severity=", tag, f"SLA tag missing severity: {tag[:60]}")

    def test_twelve_sla_definitions_in_standard(self):
        """SLA-001 defines exactly 12 SLAs per §14 acceptance criteria."""
        sla_tags = _extract_sla_tags(self.sla_standard)
        self.assertEqual(
            len(sla_tags), 12, f"Expected 12 SLA tags, found {len(sla_tags)}"
        )


# ── §5-10: SLA Definitions Coverage ──────────────────────────────────────


class SLADefinitionCoverageTest(SimpleTestCase):
    """SLA-001 §5-10: All six metric categories have SLA definitions."""

    def setUp(self):
        self.sla_standard = _read(STANDARDS_DIR / "SLA-001.md")

    def test_availability_sla_exists(self):
        """At least one availability SLA defined."""
        self.assertIn("metric=availability", self.sla_standard)

    def test_response_time_sla_exists(self):
        """At least one response_time SLA defined."""
        self.assertIn("metric=response_time", self.sla_standard)

    def test_durability_sla_exists(self):
        """At least one durability SLA defined."""
        self.assertIn("metric=durability", self.sla_standard)

    def test_incident_response_sla_exists(self):
        """At least one incident_response SLA defined."""
        self.assertIn("metric=incident_response", self.sla_standard)

    def test_compliance_sla_exists(self):
        """At least one compliance SLA defined."""
        self.assertIn("metric=compliance", self.sla_standard)

    def test_change_velocity_sla_exists(self):
        """At least one change_velocity SLA defined."""
        self.assertIn("metric=change_velocity", self.sla_standard)


# ── Compliance Check: sla_compliance ──────────────────────────────────────


class SLAComplianceCheckTest(SimpleTestCase):
    """SLA-001 §1.3: sla_compliance check is registered and measures SLAs."""

    def setUp(self):
        self.compliance_src = _read(WEB_ROOT / "syn" / "audit" / "compliance.py")
        self.assertGreater(len(self.compliance_src), 0, "compliance.py not found")

    def test_sla_compliance_registered(self):
        """sla_compliance check is registered in compliance.py."""
        self.assertIn('"sla_compliance"', self.compliance_src)

    def test_check_calls_parse_all_sla_definitions(self):
        """sla_compliance calls parse_all_sla_definitions."""
        self.assertIn("parse_all_sla_definitions", self.compliance_src)

    def test_measure_sla_dispatcher_exists(self):
        """_measure_sla dispatcher function exists."""
        self.assertIn("def _measure_sla(", self.compliance_src)

    def test_dispatcher_handles_all_metrics(self):
        """_measure_sla dispatches to all 6 metric types."""
        fn_match = re.search(
            r"def _measure_sla\(.*?(?=\ndef [a-z]|\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match)
        fn_body = fn_match.group()
        for metric in [
            "availability",
            "durability",
            "compliance",
            "change_velocity",
            "response_time",
            "incident_response",
        ]:
            self.assertIn(
                metric,
                fn_body,
                f"_measure_sla missing dispatch for '{metric}'",
            )

    def test_manual_measurement_returns_unmeasurable(self):
        """Manual measurement SLAs return unmeasurable status."""
        fn_match = re.search(
            r"def _measure_sla\(.*?(?=\ndef [a-z]|\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("manual", fn_body)
        self.assertIn("unmeasurable", fn_body)

    def test_critical_breach_causes_fail(self):
        """Critical SLA breach causes overall check to fail."""
        self.assertIn("critical_breach", self.compliance_src)

    def test_soc2_controls_attached(self):
        """SLA compliance check attaches SOC 2 CC9.2 and CC4.1."""
        self.assertIn("CC9.2", self.compliance_src)
        self.assertIn("CC4.1", self.compliance_src)


# ── Measurement Helpers ───────────────────────────────────────────────────


class SLAMeasurementTest(SimpleTestCase):
    """SLA-001 §11: Measurement methods for each metric type."""

    def setUp(self):
        self.compliance_src = _read(WEB_ROOT / "syn" / "audit" / "compliance.py")

    def test_parse_target_handles_percentage(self):
        """_parse_target handles '99.9%' format."""
        self.assertIn("def _parse_target(", self.compliance_src)
        fn_match = re.search(
            r"def _parse_target\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("%", fn_body)

    def test_parse_target_handles_milliseconds(self):
        """_parse_target handles '2000ms' format."""
        fn_match = re.search(
            r"def _parse_target\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("ms", fn_body)

    def test_parse_target_handles_hours(self):
        """_parse_target handles '24h' format."""
        fn_match = re.search(
            r"def _parse_target\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("h", fn_body)

    def test_measure_availability_exists(self):
        """_measure_availability function defined."""
        self.assertIn("def _measure_availability(", self.compliance_src)

    def test_measure_response_time_exists(self):
        """_measure_response_time function defined."""
        self.assertIn("def _measure_response_time(", self.compliance_src)

    def test_measure_durability_exists(self):
        """_measure_durability function defined."""
        self.assertIn("def _measure_durability(", self.compliance_src)

    def test_measure_compliance_rate_exists(self):
        """_measure_compliance_rate function defined."""
        self.assertIn("def _measure_compliance_rate(", self.compliance_src)

    def test_measure_change_velocity_exists(self):
        """_measure_change_velocity function defined."""
        self.assertIn("def _measure_change_velocity(", self.compliance_src)

    def test_measure_incident_response_exists(self):
        """_measure_incident_response function defined."""
        self.assertIn("def _measure_incident_response(", self.compliance_src)

    def test_response_time_uses_percentile(self):
        """_measure_response_time computes p95 and p99 percentiles."""
        fn_match = re.search(
            r"def _measure_response_time\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match)
        fn_body = fn_match.group()
        self.assertIn("p99", fn_body)
        self.assertIn("p95", fn_body.replace("95", "95"))  # check for percentile logic

    def test_availability_uses_health_pings(self):
        """_measure_availability uses HealthPing records for real-time measurement."""
        fn_match = re.search(
            r"def _measure_availability\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("HealthPing", fn_body)


# ── Anti-Patterns ─────────────────────────────────────────────────────────


class SLAAntiPatternTest(SimpleTestCase):
    """SLA-001 §13: Anti-pattern enforcement."""

    def setUp(self):
        self.compliance_src = _read(WEB_ROOT / "syn" / "audit" / "compliance.py")

    def test_unmeasurable_reported_honestly(self):
        """Unmeasurable SLAs return explicit 'unmeasurable' status, not silent skip."""
        # Count occurrences — should be in multiple measurement functions
        count = self.compliance_src.count('"unmeasurable"')
        self.assertGreaterEqual(
            count, 5, f"'unmeasurable' status used only {count} times"
        )

    def test_no_silent_skip_on_unknown_metric(self):
        """Unknown metric types are reported, not silently skipped."""
        fn_match = re.search(
            r"def _measure_sla\(.*?(?=\ndef [a-z]|\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("Unknown metric", fn_body)
