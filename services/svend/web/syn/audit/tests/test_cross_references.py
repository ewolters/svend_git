"""
XRF-001 compliance tests: Cross-reference syntax & validation.

Tests verify that all standards in docs/standards/ follow XRF-001 rules:
§ notation for internal references, CODE-NNN for external references,
version requirements in Compliance headers, no circular dependencies,
and compliance comments in code follow the prescribed format.

These tests are linked from XRF-001.md via <!-- test: --> hooks and verified
by the standards compliance runner.

Compliance: XRF-001 (Cross-Reference Syntax), ISO 9001:2015 §7.5.3
"""

import re
from pathlib import Path

from django.test import SimpleTestCase

STANDARDS_DIR = Path(__file__).resolve().parents[6] / "docs" / "standards"
WEB_DIR = Path(__file__).resolve().parents[3]

# Standard code pattern (word boundaries prevent substring matches from longer prefixes)
CODE_RE = re.compile(r"\b[A-Z]{2,5}-\d{3}\b")

# Internal section reference pattern: §N or §N.M or §N.M.P
SECTION_REF_RE = re.compile(r"§(\d+(?:\.\d+)*)")

# External standard reference pattern: CODE-NNN §N
EXT_REF_RE = re.compile(r"([A-Z]{2,5}-\d{3})\s+§(\d+(?:\.\d+)*)")

# Compliance header dependency: CODE-NNN ≥ VERSION or CODE-NNN >= VERSION
COMPLIANCE_DEP_RE = re.compile(r"([A-Z]{2,5}-\d{3})\s*[≥>=]+\s*(\d+\.\d+(?:\.\d+)?)")

# Prose-style section references (forbidden by XRF-001 §7.2)
PROSE_SECTION_RE = re.compile(
    r"(?:section|Section|SECTION)\s+(\d+(?:\.\d+)*)\s+of\s+([A-Z]{2,5}-\d{3})",
    re.IGNORECASE,
)

# Code compliance comment pattern
CODE_COMPLIANCE_RE = re.compile(
    r"#\s*Compliance:\s*([A-Z]{2,5}-\d{3})\s+§(\d+(?:\.\d+)*)"
)


def _get_all_standards():
    """Return dict of standard code -> file path."""
    standards = {}
    for path in sorted(STANDARDS_DIR.glob("*.md")):
        code = path.stem  # e.g. DOC-001
        standards[code] = path
    return standards


def _extract_sections(content):
    """Extract section numbers from ## headers."""
    sections = set()
    for match in re.finditer(r"^## \*\*(\d+)\. ", content, re.MULTILINE):
        sections.add(match.group(1))
    # Also capture subsections from ### headers
    for match in re.finditer(r"^### \*\*(\d+\.\d+)", content, re.MULTILINE):
        sections.add(match.group(1))
    # And sub-subsections
    for match in re.finditer(r"^#### \*\*(\d+\.\d+\.\d+)", content, re.MULTILINE):
        sections.add(match.group(1))
    return sections


def _extract_compliance_deps(content):
    """Extract Compliance header dependencies (Kjerne standards only)."""
    header = content.split("---")[0] if "---" in content else content[:500]
    # Find the Compliance: block
    comp_idx = header.find("**Compliance:**")
    if comp_idx == -1:
        return []
    # Get lines after Compliance: until next ** field or ---
    comp_block = header[comp_idx:]
    lines = comp_block.split("\n")[1:]  # skip the Compliance: line itself
    deps = []
    for line in lines:
        line = line.strip()
        if not line.startswith("-"):
            break
        match = COMPLIANCE_DEP_RE.search(line)
        if match:
            deps.append(match.group(1))
    return deps


class InternalReferenceTest(SimpleTestCase):
    """XRF-001 §4.1: Internal §N references point to existing sections."""

    def test_internal_refs_valid(self):
        """All §N references within a standard point to existing sections."""
        for code, path in _get_all_standards().items():
            content = path.read_text()
            sections = _extract_sections(content)

            # Find all § references that are NOT preceded by a standard code
            # (those are external references)
            for match in SECTION_REF_RE.finditer(content):
                ref = match.group(1)
                pos = match.start()
                # Check if preceded by a standard code (external ref)
                prefix = content[max(0, pos - 10) : pos]
                if CODE_RE.search(prefix):
                    continue  # External ref, skip
                # Check the top-level section exists
                top_section = ref.split(".")[0]
                self.assertIn(
                    top_section,
                    sections,
                    f"{path.name}: internal ref §{ref} — §{top_section} not found. "
                    f"Available: {sorted(sections, key=lambda x: int(x.split('.')[0]))}",
                )


class ExternalReferenceTest(SimpleTestCase):
    """XRF-001 §4.2: External CODE-NNN references point to existing standards."""

    # Codes that match [A-Z]{2,5}-[0-9]{3} but are NOT Kjerne standards.
    # Technical terms, process docs outside docs/standards/, or planned standards.
    NON_STANDARD_CODES = {
        # Hash/crypto algorithms
        "SHA-256",
        "SHA-384",
        "SHA-512",
        "AES-256",
        "AES-128",
        "TLS-128",
        "RSA-256",
        "UTF-008",
        # HTTP status codes
        "HTTP-200",
        "HTTP-400",
        "HTTP-401",
        "HTTP-403",
        "HTTP-404",
        "HTTP-500",
        # Process docs (live outside docs/standards/)
        "DEBT-001",  # .kjerne/DEBT-001.md — debt closure process
        # Historical/superseded references
        "DOC-002",  # Was merged into XRF-001
        # Phantom standards — registered in MAP-001 §4 but no .md file yet
        "MOD-001",
        "ERR-002",
        "SBL-001",
        "SCH-002",
        "SCH-003",
        "SCH-004",
        "SCH-005",
        "SCH-006",
        "CONFIG-001",
        "SRX-001",
        "IO-001",
        "CGS-1001",
        "POL-002",
        "SEC-002",
        "ENC-001",
        "PCONF-001",
        "CEL-001",
        "API-002",
        "AUD-002",
        "CTG-001",
        "EVT-001",
        "EVT-002",
        "SDK-001",
        "GOV-001",
        "GOV-002",
        "LOG-002",
        "CLI-001",
        "COG-001",
        "SER-001",
        "TEL-001",
        "VAL-001",
        "USER-001",
        "DRF-001",
        "PRM-001",
        "ERM-001",
        "ORG-001",
        "ORG-002",
        "FORM-001",
        "FLD-001",
        "EVENTS-001",
        "SCHEMA-001",
        "REF-001",
        "STAT-001",  # Future planned standard (DSW-001 scope note)
        # Process/anti-pattern/error codes (not standards)
        "AP-003",
        "BOOT-001",
        "BOOT-005",
        "CFG-001",
        "DB-001",
        "ENC-002",
        "ENC-003",
        "ENC-004",
        "ENC-005",
        "ENC-006",
        "ENC-007",
        "ENC-008",
        "ENC-009",
        "ENC-010",
        "ENC-011",
        "SCH-101",
        "SCH-103",
        "SCH-201",
        "SCH-202",
        "SCH-501",
        "SYS-200",
        "INV-001",
        "INV-008",
        "INV-011",
        "XXX-001",
    }

    def test_external_standard_refs_exist(self):
        """All CODE-NNN references point to standards that exist."""
        all_standards = _get_all_standards()
        # Known external standards (ISO, SOC 2, NIST) — not in docs/standards/
        external_frameworks = {"ISO", "SOC", "NIST", "COSO"}

        for code, path in all_standards.items():
            content = path.read_text()
            for match in CODE_RE.finditer(content):
                ref_code = match.group()
                # Skip known non-standard technical codes
                if ref_code in self.NON_STANDARD_CODES:
                    continue
                # Skip if it's an external framework reference nearby
                context = content[max(0, match.start() - 20) : match.end() + 20]
                is_external = any(fw in context for fw in external_frameworks)
                if is_external:
                    continue
                # Skip self-references
                if ref_code == code:
                    continue
                # Skip code examples (inside ``` blocks)
                backtick_count = content[: match.start()].count("```")
                if backtick_count % 2 == 1:
                    continue
                # Skip HTML comments (hooks)
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.end())
                line = content[line_start : line_end if line_end != -1 else None]
                if line.strip().startswith("<!--"):
                    continue
                # Verify the referenced standard exists
                self.assertIn(
                    ref_code,
                    all_standards,
                    f"{path.name}: references {ref_code} which doesn't exist in "
                    f"docs/standards/. Available: {sorted(all_standards.keys())}",
                )


class ComplianceHeaderTest(SimpleTestCase):
    """XRF-001 §4.2, §5.3: Compliance headers have version requirements."""

    def test_kjerne_deps_have_version(self):
        """Kjerne standard dependencies in Compliance: headers include ≥ version."""
        all_standards = _get_all_standards()
        for code, path in all_standards.items():
            content = path.read_text()
            header = content.split("---")[0] if "---" in content else content[:500]

            comp_idx = header.find("**Compliance:**")
            if comp_idx == -1:
                continue

            comp_block = header[comp_idx:]
            lines = comp_block.split("\n")[1:]
            for line in lines:
                line = line.strip()
                if not line.startswith("-"):
                    break
                # Check if line references a Kjerne standard (not ISO/SOC/NIST)
                std_match = CODE_RE.search(line)
                if not std_match:
                    continue
                ref = std_match.group()
                if ref not in all_standards:
                    continue  # External standard
                # Must have version requirement
                has_version = bool(re.search(r"[≥>=]+\s*\d+\.\d+", line))
                self.assertTrue(
                    has_version,
                    f"{path.name}: Compliance dep '{ref}' missing version requirement. "
                    f"XRF-001 §5.3 requires '≥ VERSION'. Line: {line.strip()}",
                )

    def test_version_uses_proper_symbol(self):
        """Compliance headers use ≥ (not >=) for version requirements."""
        for code, path in _get_all_standards().items():
            content = path.read_text()
            header = content.split("---")[0] if "---" in content else content[:500]

            comp_idx = header.find("**Compliance:**")
            if comp_idx == -1:
                continue

            comp_block = header[comp_idx:]
            lines = comp_block.split("\n")[1:]
            for line in lines:
                line = line.strip()
                if not line.startswith("-"):
                    break
                # Check for ASCII >= instead of ≥
                if ">=" in line and "≥" not in line:
                    # Only flag if it's a Kjerne standard dep
                    std_match = CODE_RE.search(line)
                    if std_match and std_match.group() in _get_all_standards():
                        self.fail(
                            f"{path.name}: uses '>=' instead of '≥' in Compliance header. Line: {line.strip()}"
                        )


class NoDependencyCyclesTest(SimpleTestCase):
    """XRF-001 §5.2: No circular dependencies in Compliance headers."""

    def test_no_circular_dependencies(self):
        """Compliance dependency graph is acyclic (DAG)."""
        all_standards = _get_all_standards()
        # Build adjacency list: standard -> [dependencies]
        graph = {}
        for code, path in all_standards.items():
            content = path.read_text()
            deps = _extract_compliance_deps(content)
            graph[code] = deps

        # DFS cycle detection
        visited = set()
        in_stack = set()

        def has_cycle(node, path_trace):
            if node in in_stack:
                cycle = path_trace[path_trace.index(node) :]
                self.fail(f"Circular dependency detected: {' → '.join(cycle + [node])}")
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            path_trace.append(node)
            for dep in graph.get(node, []):
                if dep in graph:  # Only check Kjerne standards
                    has_cycle(dep, path_trace)
            path_trace.pop()
            in_stack.discard(node)
            return False

        for code in graph:
            if code not in visited:
                has_cycle(code, [])

    def test_doc001_is_root(self):
        """DOC-001 has no Kjerne standard dependencies (it's the root)."""
        all_standards = _get_all_standards()
        content = (STANDARDS_DIR / "DOC-001.md").read_text()
        deps = _extract_compliance_deps(content)
        kjerne_deps = [d for d in deps if d in all_standards]
        self.assertEqual(
            kjerne_deps,
            [],
            f"DOC-001 should be the root with no Kjerne deps, but has: {kjerne_deps}",
        )


class ProseReferenceTest(SimpleTestCase):
    """XRF-001 §7.2: No prose-style section references (use § notation)."""

    def test_no_prose_section_refs(self):
        """Standards don't use 'section N of CODE-NNN' (use 'CODE-NNN §N')."""
        for code, path in _get_all_standards().items():
            content = path.read_text()
            # Skip content inside code blocks
            # Simple approach: check outside of ``` blocks
            blocks = content.split("```")
            for i, block in enumerate(blocks):
                if i % 2 == 1:
                    continue  # Inside code block
                matches = PROSE_SECTION_RE.findall(block)
                self.assertEqual(
                    len(matches),
                    0,
                    (
                        f"{path.name}: prose-style ref found: 'section {matches[0][0]} of {matches[0][1]}'. "
                        f"Use '{matches[0][1]} §{matches[0][0]}' instead (XRF-001 §7.2)."
                        if matches
                        else ""
                    ),
                )


class EmDashConsistencyTest(SimpleTestCase):
    """XRF-001 formatting: Compliance headers use — (em dash) not -- (double hyphen)."""

    def test_compliance_lines_use_em_dash(self):
        """Compliance header lines use — not -- for descriptions."""
        for code, path in _get_all_standards().items():
            content = path.read_text()
            header = content.split("---")[0] if "---" in content else content[:500]

            comp_idx = header.find("**Compliance:**")
            if comp_idx == -1:
                continue

            comp_block = header[comp_idx:]
            lines = comp_block.split("\n")[1:]
            for line in lines:
                line = line.strip()
                if not line.startswith("-"):
                    break
                # Check for Kjerne standard deps with -- instead of —
                std_match = CODE_RE.search(line)
                if std_match and std_match.group() in _get_all_standards():
                    if " -- " in line or " --" in line.rstrip():
                        self.fail(
                            f"{path.name}: uses '--' instead of '—' in Compliance header. Line: {line.strip()}"
                        )


class CodeComplianceCommentTest(SimpleTestCase):
    """XRF-001 §4.4: Code compliance comments follow prescribed format."""

    def test_compliance_comments_in_python(self):
        """Python files with compliance comments use correct format."""
        # Check key infrastructure files
        key_files = [
            WEB_DIR / "syn" / "audit" / "models.py",
            WEB_DIR / "syn" / "audit" / "compliance.py",
            WEB_DIR / "syn" / "audit" / "standards.py",
            WEB_DIR / "syn" / "audit" / "utils.py",
        ]
        re.compile(
            r"#.*compliance:.*(?:aud|sec|api|dat|err|log|chg|cmp)", re.IGNORECASE
        )
        good_format = CODE_COMPLIANCE_RE

        for fpath in key_files:
            if not fpath.exists():
                continue
            content = fpath.read_text()
            # Find lines with "compliance" comments
            for i, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                if "compliance" in stripped.lower() and CODE_RE.search(stripped):
                    # It's a compliance comment — verify format
                    if good_format.search(stripped):
                        continue  # Correct format
                    # Check if it's just a docstring or general comment
                    if "Compliance:" not in stripped:
                        continue
                    # Malformed compliance comment
                    self.fail(
                        f"{fpath.name}:{i}: malformed compliance comment. "
                        f"Expected: '# Compliance: CODE-NNN §N.M — description'. "
                        f"Got: '{stripped}'"
                    )
