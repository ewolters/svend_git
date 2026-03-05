"""
DOC-001 compliance tests: Standards structure & format enforcement.

Tests verify that all standards in docs/standards/ follow the mandatory
structure defined in DOC-001: metadata headers, required sections in correct
order, canonical section titles, valid standard codes, and machine-readable
hooks.

These tests are linked from DOC-001.md via <!-- test: --> hooks and verified
by the standards compliance runner.

Compliance: DOC-001 (Documentation Structure), ISO 9001:2015 §7.5
"""

import os
import re
from pathlib import Path

from django.test import SimpleTestCase

# All standards live here
STANDARDS_DIR = Path(__file__).resolve().parents[6] / "docs" / "standards"

# DOC-001 §5.3: Valid standard code prefixes
VALID_PREFIXES = {
    "DOC", "XRF", "API", "DAT", "SEC", "AUD", "LOG", "ERR",
    "MOD", "OPS", "TST", "FE", "CMP", "CHG", "SCH", "BILL",
    "SLA", "LLM", "QMS", "MAP", "STY", "DSW",
    "STAT", "ARCH", "JS", "TRN", "RDM", "CACHE", "QUAL",
}

# DOC-001 §1.3: Valid lifecycle statuses
VALID_STATUSES = {"DRAFT", "REVIEW", "APPROVED", "REVISION", "DEPRECATED", "SUPERSEDED"}

# DOC-001 §4.2: Canonical titles (forbidden variants listed for detection)
CANONICAL_TITLES = {
    "SCOPE AND PURPOSE",
    "NORMATIVE REFERENCES",
    "TERMINOLOGY",
    "ANTI-PATTERNS",
    "ACCEPTANCE CRITERIA",
    "COMPLIANCE MAPPING",
}

FORBIDDEN_TITLES = {
    "Introduction", "Purpose & Scope", "Purpose and Scope",
    "References", "Dependencies",
    "Definitions", "Glossary",
    "Don'ts", "Forbidden Patterns",
    "Success Criteria", "Done When",
    "Regulatory Mapping",
}

# DOC-001 §5.1: Required metadata fields
REQUIRED_METADATA = ["Version:", "Status:", "Date:", "Author:", "Compliance:"]


def _read_standard(filename):
    """Read a standard file and return its content."""
    path = STANDARDS_DIR / filename
    return path.read_text()


def _get_all_standards():
    """Return list of all .md files in standards directory."""
    return sorted(STANDARDS_DIR.glob("*.md"))


def _extract_sections(content):
    """Extract ## section headers with their titles and numbers."""
    sections = []
    for match in re.finditer(r'^## \*\*(\d+)\. (.+?)\*\*', content, re.MULTILINE):
        sections.append((int(match.group(1)), match.group(2)))
    return sections


def _extract_header(content):
    """Extract metadata from the standard header (before first ---)."""
    header = content.split("---")[0] if "---" in content else content[:500]
    return header


class StandardsDiscoveryTest(SimpleTestCase):
    """DOC-001 §1.2: Standards directory contains expected files."""

    def test_standards_directory_exists(self):
        """docs/standards/ directory exists."""
        self.assertTrue(STANDARDS_DIR.exists(),
                        f"Standards directory not found: {STANDARDS_DIR}")

    def test_minimum_standards_count(self):
        """At least 15 standards exist (current: 17)."""
        standards = _get_all_standards()
        self.assertGreaterEqual(len(standards), 15,
                                f"Expected ≥15 standards, found {len(standards)}")

    def test_foundation_standards_exist(self):
        """DOC-001 and XRF-001 foundation standards exist."""
        self.assertTrue((STANDARDS_DIR / "DOC-001.md").exists())
        self.assertTrue((STANDARDS_DIR / "XRF-001.md").exists())


class StandardCodeTest(SimpleTestCase):
    """DOC-001 §5.3: Standard codes follow [A-Z]{2,5}-[0-9]{3} pattern."""

    def test_all_filenames_match_code_pattern(self):
        """All .md filenames follow CODE-NNN.md pattern."""
        pattern = re.compile(r'^[A-Z]{2,5}-\d{3}\.md$')
        for path in _get_all_standards():
            self.assertRegex(path.name, pattern,
                             f"Invalid standard filename: {path.name}")

    def test_all_codes_use_valid_prefixes(self):
        """All standard codes use prefixes from DOC-001 §5.3."""
        for path in _get_all_standards():
            prefix = path.stem.split("-")[0]
            self.assertIn(prefix, VALID_PREFIXES,
                          f"Unknown prefix '{prefix}' in {path.name}. "
                          f"Register it in DOC-001 §5.3.")

    def test_title_matches_filename(self):
        """First line title code matches filename."""
        for path in _get_all_standards():
            content = path.read_text()
            first_line = content.strip().split("\n")[0]
            code = path.stem  # e.g. DOC-001
            self.assertIn(code, first_line,
                          f"{path.name}: title '{first_line}' doesn't contain code '{code}'")


class MetadataHeaderTest(SimpleTestCase):
    """DOC-001 §5.1: All standards have complete metadata headers."""

    def test_all_standards_have_version(self):
        """Every standard has a **Version:** field."""
        for path in _get_all_standards():
            header = _extract_header(path.read_text())
            self.assertIn("**Version:**", header,
                          f"{path.name} missing Version field")

    def test_all_standards_have_status(self):
        """Every standard has a **Status:** field."""
        for path in _get_all_standards():
            header = _extract_header(path.read_text())
            self.assertIn("**Status:**", header,
                          f"{path.name} missing Status field")

    def test_all_statuses_valid(self):
        """All Status values are from the valid lifecycle set."""
        for path in _get_all_standards():
            header = _extract_header(path.read_text())
            match = re.search(r'\*\*Status:\*\*\s+(\w+)', header)
            self.assertIsNotNone(match, f"{path.name}: can't parse Status")
            status = match.group(1)
            self.assertIn(status, VALID_STATUSES,
                          f"{path.name}: invalid status '{status}'")

    def test_all_standards_have_date(self):
        """Every standard has a **Date:** field in YYYY-MM-DD format."""
        for path in _get_all_standards():
            header = _extract_header(path.read_text())
            self.assertIn("**Date:**", header,
                          f"{path.name} missing Date field")
            match = re.search(r'\*\*Date:\*\*\s+(\d{4}-\d{2}-\d{2})', header)
            self.assertIsNotNone(match,
                                 f"{path.name}: Date not in YYYY-MM-DD format")

    def test_all_standards_have_author(self):
        """Every standard has an **Author:** field."""
        for path in _get_all_standards():
            header = _extract_header(path.read_text())
            self.assertIn("**Author:**", header,
                          f"{path.name} missing Author field")

    def test_all_standards_have_compliance(self):
        """Every standard has a **Compliance:** section."""
        for path in _get_all_standards():
            header = _extract_header(path.read_text())
            self.assertIn("**Compliance:**", header,
                          f"{path.name} missing Compliance field")

    def test_version_format(self):
        """Version numbers follow MAJOR.MINOR or MAJOR.MINOR.PATCH format."""
        pattern = re.compile(r'^\d+\.\d+(\.\d+)?$')
        for path in _get_all_standards():
            header = _extract_header(path.read_text())
            match = re.search(r'\*\*Version:\*\*\s+(\S+)', header)
            self.assertIsNotNone(match, f"{path.name}: can't parse Version")
            version = match.group(1)
            self.assertRegex(version, pattern,
                             f"{path.name}: invalid version '{version}'")


class MandatorySectionTest(SimpleTestCase):
    """DOC-001 §4.1: All standards have required sections in correct order."""

    def test_all_standards_have_scope_section(self):
        """§1 is SCOPE AND PURPOSE in every standard."""
        for path in _get_all_standards():
            sections = _extract_sections(path.read_text())
            sec_1 = [t for n, t in sections if n == 1]
            self.assertTrue(len(sec_1) > 0,
                            f"{path.name}: missing §1")
            self.assertEqual(sec_1[0], "SCOPE AND PURPOSE",
                             f"{path.name}: §1 is '{sec_1[0]}', expected 'SCOPE AND PURPOSE'")

    def test_all_standards_have_references_section(self):
        """§2 is NORMATIVE REFERENCES in every standard."""
        for path in _get_all_standards():
            sections = _extract_sections(path.read_text())
            sec_2 = [t for n, t in sections if n == 2]
            self.assertTrue(len(sec_2) > 0,
                            f"{path.name}: missing §2")
            self.assertEqual(sec_2[0], "NORMATIVE REFERENCES",
                             f"{path.name}: §2 is '{sec_2[0]}', expected 'NORMATIVE REFERENCES'")

    def test_all_standards_have_terminology_section(self):
        """§3 is TERMINOLOGY in every standard."""
        for path in _get_all_standards():
            sections = _extract_sections(path.read_text())
            sec_3 = [t for n, t in sections if n == 3]
            self.assertTrue(len(sec_3) > 0,
                            f"{path.name}: missing §3")
            self.assertEqual(sec_3[0], "TERMINOLOGY",
                             f"{path.name}: §3 is '{sec_3[0]}', expected 'TERMINOLOGY'")

    def test_all_standards_have_anti_patterns(self):
        """Third-to-last numbered section is ANTI-PATTERNS."""
        for path in _get_all_standards():
            sections = _extract_sections(path.read_text())
            if len(sections) < 4:
                continue  # Skip malformed
            # Last 3 must be: ANTI-PATTERNS, ACCEPTANCE CRITERIA, COMPLIANCE MAPPING
            self.assertEqual(sections[-3][1], "ANTI-PATTERNS",
                             f"{path.name}: §{sections[-3][0]} is '{sections[-3][1]}', "
                             f"expected 'ANTI-PATTERNS'")

    def test_all_standards_have_acceptance_criteria(self):
        """Second-to-last numbered section is ACCEPTANCE CRITERIA."""
        for path in _get_all_standards():
            sections = _extract_sections(path.read_text())
            if len(sections) < 4:
                continue
            self.assertEqual(sections[-2][1], "ACCEPTANCE CRITERIA",
                             f"{path.name}: §{sections[-2][0]} is '{sections[-2][1]}', "
                             f"expected 'ACCEPTANCE CRITERIA'")

    def test_all_standards_have_compliance_mapping(self):
        """Last numbered section is COMPLIANCE MAPPING."""
        for path in _get_all_standards():
            sections = _extract_sections(path.read_text())
            if len(sections) < 4:
                continue
            self.assertEqual(sections[-1][1], "COMPLIANCE MAPPING",
                             f"{path.name}: §{sections[-1][0]} is '{sections[-1][1]}', "
                             f"expected 'COMPLIANCE MAPPING'")

    def test_sections_numbered_sequentially(self):
        """Section numbers are sequential (1, 2, 3, ...)."""
        for path in _get_all_standards():
            sections = _extract_sections(path.read_text())
            numbers = [n for n, _ in sections]
            expected = list(range(1, len(numbers) + 1))
            self.assertEqual(numbers, expected,
                             f"{path.name}: non-sequential sections {numbers}")

    def test_no_forbidden_section_titles(self):
        """No standard uses forbidden title variants."""
        for path in _get_all_standards():
            content = path.read_text()
            for forbidden in FORBIDDEN_TITLES:
                # Only check in ## headers
                matches = re.findall(
                    rf'^## \*\*\d+\. {re.escape(forbidden)}\*\*',
                    content, re.MULTILINE
                )
                self.assertEqual(len(matches), 0,
                                 f"{path.name}: uses forbidden title '{forbidden}'")


class RevisionHistoryTest(SimpleTestCase):
    """DOC-001 §5.2: Standards have revision history."""

    def test_all_standards_have_revision_history(self):
        """Every standard has a REVISION HISTORY section."""
        for path in _get_all_standards():
            content = path.read_text()
            self.assertIn("REVISION HISTORY", content,
                          f"{path.name} missing REVISION HISTORY")

    def test_revision_history_has_table(self):
        """Revision history contains a markdown table."""
        for path in _get_all_standards():
            content = path.read_text()
            # Find everything after REVISION HISTORY
            idx = content.find("REVISION HISTORY")
            if idx == -1:
                continue
            after = content[idx:]
            self.assertIn("| Version |", after,
                          f"{path.name}: REVISION HISTORY missing table")


class MachineReadableHooksTest(SimpleTestCase):
    """DOC-001 §7: Standards with domain sections have machine-readable hooks."""

    HOOK_RE = re.compile(r'^<!--\s*(assert|impl|check|code|control|rule|table|test|sla):', re.MULTILINE)

    def test_domain_standards_have_assertions(self):
        """Standards with domain content (§4+) have at least one assert hook."""
        # Foundation standards (DOC-001, XRF-001) may have fewer assertions
        foundation = {"DOC-001.md", "XRF-001.md"}
        for path in _get_all_standards():
            if path.name in foundation:
                continue
            content = path.read_text()
            asserts = re.findall(r'<!-- assert:', content)
            self.assertGreater(len(asserts), 0,
                               f"{path.name}: no <!-- assert: --> hooks found. "
                               f"DOC-001 §7 requires machine-readable assertions.")

    def test_assertions_have_impl_links(self):
        """Every assert hook is followed by at least one impl hook."""
        for path in _get_all_standards():
            content = path.read_text()
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("<!-- assert:"):
                    # Look for impl in the next 5 lines
                    following = "\n".join(lines[i+1:i+6])
                    self.assertIn("<!-- impl:", following,
                                  f"{path.name} line {i+1}: assert without impl link")

    def test_hook_syntax_valid(self):
        """All hooks use correct <!-- keyword: value --> syntax."""
        bad_hooks = re.compile(r'<!--\s*(assert|impl|check|code|control|rule|table|test|sla)\s*[^:]')
        for path in _get_all_standards():
            content = path.read_text()
            for match in bad_hooks.finditer(content):
                self.fail(f"{path.name}: malformed hook at '{match.group()}'")


# ── DOC-001 §7.3-7.5: Hook Vocabulary & Parser Contract ──────────────────


class HookAttributeExtractionTest(SimpleTestCase):
    """DOC-001 §7.3: Hook attribute parsing with | key=value syntax."""

    def test_assert_hook_has_check_attribute(self):
        """Assert hooks use | check=id to define check IDs."""
        from syn.audit.standards import TAG_RE, ATTR_RE
        line = "<!-- assert: Test claim | check=test-id -->"
        m = TAG_RE.match(line)
        self.assertIsNotNone(m)
        attrs = dict(ATTR_RE.findall(m.group(2)))
        self.assertEqual(attrs["check"], "test-id")

    def test_check_hook_has_soc2_attribute(self):
        """Check hooks use | soc2=X for SOC 2 control mapping."""
        from syn.audit.standards import TAG_RE, ATTR_RE
        line = "<!-- check: aud-chain-integrity | soc2=CC7.2 | nist=AU-9 -->"
        m = TAG_RE.match(line)
        self.assertIsNotNone(m)
        attrs = dict(ATTR_RE.findall(m.group(2)))
        self.assertEqual(attrs["soc2"], "CC7.2")
        self.assertEqual(attrs["nist"], "AU-9")

    def test_sla_hook_has_metric_attribute(self):
        """SLA hooks use | metric=X | target=Y | window=Z | severity=S."""
        from syn.audit.standards import SLA_TAG_RE, ATTR_RE
        line = "<!-- sla: Platform availability | metric=availability | target=99.9% | window=monthly | severity=critical -->"
        m = SLA_TAG_RE.match(line)
        self.assertIsNotNone(m, "SLA_TAG_RE failed to match SLA tag")
        attrs = dict(ATTR_RE.findall(m.group(1)))
        self.assertEqual(attrs["metric"], "availability")
        self.assertEqual(attrs["target"], "99.9%")
        self.assertEqual(attrs["window"], "monthly")
        self.assertEqual(attrs["severity"], "critical")

    def test_attr_re_extracts_multiple_attributes(self):
        """ATTR_RE extracts all pipe-separated attributes."""
        from syn.audit.standards import ATTR_RE
        value = "Some desc | metric=availability | target=99.9% | window=monthly | severity=critical | check=sla-avail"
        attrs = dict(ATTR_RE.findall(value))
        self.assertEqual(len(attrs), 5)

    def test_tag_re_matches_sla_type(self):
        """TAG_RE recognizes 'sla' as a valid hook type."""
        from syn.audit.standards import TAG_RE
        line = "<!-- sla: Platform availability | metric=availability -->"
        m = TAG_RE.match(line)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "sla")


class ParserContractTest(SimpleTestCase):
    """DOC-001 §7.5: parse_standard produces complete Assertion objects."""

    def test_assertions_have_impl_links(self):
        """Parsed assertions include impl file paths."""
        from syn.audit.standards import parse_all_standards
        assertions = parse_all_standards()
        with_impls = [a for a in assertions if a.impls]
        self.assertGreater(len(with_impls), 50,
                           f"Only {len(with_impls)} assertions have impl links")

    def test_assertions_have_test_links(self):
        """Parsed assertions include test method paths."""
        from syn.audit.standards import parse_all_standards
        assertions = parse_all_standards()
        with_tests = [a for a in assertions if a.tests]
        self.assertGreater(len(with_tests), 50,
                           f"Only {len(with_tests)} assertions have test links")

    def test_assertions_have_controls(self):
        """Some assertions have SOC 2/NIST control mappings."""
        from syn.audit.standards import parse_all_standards
        assertions = parse_all_standards()
        with_controls = [a for a in assertions if a.controls]
        self.assertGreater(len(with_controls), 0,
                           "No assertions have control mappings")

    def test_assertions_have_rule_classification(self):
        """Some assertions have mandatory/recommended rule classification."""
        from syn.audit.standards import parse_all_standards
        assertions = parse_all_standards()
        with_rules = [a for a in assertions if a.rule]
        self.assertGreater(len(with_rules), 0,
                           "No assertions have rule classification")

    def test_assertions_span_multiple_standards(self):
        """Assertions come from at least 15 different standards."""
        from syn.audit.standards import parse_all_standards
        assertions = parse_all_standards()
        standards = {a.standard for a in assertions}
        self.assertGreaterEqual(len(standards), 15,
                                f"Only {len(standards)} standards have assertions")

    def test_code_correct_patterns_extracted(self):
        """Parser extracts <!-- code: correct --> blocks."""
        from syn.audit.standards import parse_all_standards
        assertions = parse_all_standards()
        with_correct = [a for a in assertions if a.code_correct]
        self.assertGreater(len(with_correct), 0,
                           "No assertions have code_correct patterns")

    def test_code_prohibited_patterns_extracted(self):
        """Parser extracts <!-- code: prohibited --> blocks."""
        from syn.audit.standards import parse_all_standards
        assertions = parse_all_standards()
        with_prohibited = [a for a in assertions if a.code_prohibited]
        self.assertGreater(len(with_prohibited), 0,
                           "No assertions have code_prohibited patterns")

    def test_section_numbers_populated(self):
        """Parsed assertions have section numbers like §4, §5.1."""
        from syn.audit.standards import parse_all_standards
        assertions = parse_all_standards()
        with_sections = [a for a in assertions if a.section]
        self.assertGreater(len(with_sections), 100,
                           f"Only {len(with_sections)} assertions have section numbers")


class SLAParserIntegrationTest(SimpleTestCase):
    """DOC-001 §7.4.8: SLA tag parsing via dedicated parser."""

    def test_parse_sla_definitions_returns_objects(self):
        """parse_sla_definitions returns SLADefinition objects from SLA-001."""
        from syn.audit.standards import parse_sla_definitions, SLADefinition
        sla_path = STANDARDS_DIR / "SLA-001.md"
        if sla_path.exists():
            slas = parse_sla_definitions(sla_path)
            self.assertGreater(len(slas), 0)
            self.assertIsInstance(slas[0], SLADefinition)

    def test_parse_all_sla_definitions_deduplicates(self):
        """parse_all_sla_definitions returns unique SLAs by sla_id."""
        from syn.audit.standards import parse_all_sla_definitions
        slas = parse_all_sla_definitions()
        ids = [s.sla_id for s in slas]
        self.assertEqual(len(ids), len(set(ids)),
                         "Duplicate SLA IDs found")

    def test_sla_definitions_have_required_fields(self):
        """All parsed SLAs have metric, target, window, severity."""
        from syn.audit.standards import parse_all_sla_definitions
        slas = parse_all_sla_definitions()
        for sla in slas:
            self.assertTrue(sla.metric, f"SLA {sla.sla_id} missing metric")
            self.assertTrue(sla.target, f"SLA {sla.sla_id} missing target")
            self.assertTrue(sla.window, f"SLA {sla.sla_id} missing window")
            self.assertTrue(sla.severity, f"SLA {sla.sla_id} missing severity")

    def test_sla_metric_values_valid(self):
        """All SLA metrics are from the valid set."""
        from syn.audit.standards import parse_all_sla_definitions, VALID_METRICS
        slas = parse_all_sla_definitions()
        for sla in slas:
            self.assertIn(sla.metric, VALID_METRICS,
                          f"SLA {sla.sla_id} has invalid metric '{sla.metric}'")
