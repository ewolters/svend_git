"""
Standards-driven compliance testing.

Parses docs/standards/*.md for machine-readable assertion tags and verifies
implementations match the standards. Standards are the single source of truth —
update the standard, the tests follow automatically.

Tag vocabulary (DOC-001 §7.5):
    <!-- assert: [claim] | check=[id] -->   Testable compliance assertion
    <!-- impl: [path:symbol] -->            Implementation file link
    <!-- code: correct -->                  Canonical code pattern
    <!-- code: prohibited -->               Anti-pattern
    <!-- check: [id] | soc2=X | nist=Y --> Named check with control mappings
    <!-- control: [framework] [id] -->      External compliance control
    <!-- rule: mandatory|recommended -->    Enforcement classification
"""

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

STANDARDS_DIR = Path(settings.BASE_DIR).parent.parent.parent / "docs" / "standards"
WEB_ROOT = Path(settings.BASE_DIR)

# Tag regex: <!-- keyword: value -->
TAG_RE = re.compile(r"^<!--\s*(assert|impl|check|code|control|rule|table):\s*(.+?)\s*-->$")
# Attribute regex within tag value: | key=value
ATTR_RE = re.compile(r"\|\s*(\w+)=([^\s|]+)")
# Section header: ## **N. TITLE** or ### **N.M Title**
SECTION_RE = re.compile(r"^#{2,4}\s+\**(\d[\d.]*)\s*[\.\)]*\s*(.+?)\**\s*$")
# Fenced code block
FENCE_RE = re.compile(r"^```")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Assertion:
    text: str
    check_id: str
    impls: list = field(default_factory=list)
    code_correct: list = field(default_factory=list)
    code_prohibited: list = field(default_factory=list)
    controls: dict = field(default_factory=dict)
    rule: str = ""
    standard: str = ""
    section: str = ""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_standard(filepath: Path) -> list[Assertion]:
    """Parse a single standards markdown file, return Assertions."""
    lines = filepath.read_text().splitlines()
    standard_name = filepath.stem  # e.g. "AUD-001"

    assertions = []
    current: Assertion | None = None
    current_section = ""
    current_rule = ""

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Track section headers
        m = SECTION_RE.match(line)
        if m:
            current_section = f"§{m.group(1)}"
            i += 1
            continue

        # Parse tags
        m = TAG_RE.match(line)
        if m:
            tag_type, tag_value = m.group(1), m.group(2)

            if tag_type == "assert":
                # Start a new assertion
                attrs = dict(ATTR_RE.findall(tag_value))
                # Clean the assertion text (remove | attr=value parts)
                text = ATTR_RE.sub("", tag_value).strip().rstrip("|").strip()
                check_id = attrs.get("check", _slug(text))
                current = Assertion(
                    text=text,
                    check_id=check_id,
                    standard=standard_name,
                    section=current_section,
                    rule=current_rule,
                    controls={k: v for k, v in attrs.items() if k != "check"},
                )
                assertions.append(current)

            elif tag_type == "impl" and current:
                current.impls.append(tag_value.strip())

            elif tag_type == "check":
                # Standalone check tag — attach controls to nearest assertion
                attrs = dict(ATTR_RE.findall(tag_value))
                check_id = ATTR_RE.sub("", tag_value).strip().rstrip("|").strip()
                # Find matching assertion by check_id
                for a in assertions:
                    if a.check_id == check_id:
                        a.controls.update({k: v for k, v in attrs.items()})
                        break

            elif tag_type == "code" and current:
                code_type = tag_value.strip()  # "correct" or "prohibited"
                # Next non-empty line should be a fenced code block
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines) and FENCE_RE.match(lines[i].strip()):
                    code_lines = []
                    i += 1  # skip opening fence
                    while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                        code_lines.append(lines[i])
                        i += 1
                    code_block = "\n".join(code_lines)
                    if code_type == "correct":
                        current.code_correct.append(code_block)
                    elif code_type == "prohibited":
                        current.code_prohibited.append(code_block)

            elif tag_type == "control":
                # Section-level control mapping — attach to most recent assertion
                if current:
                    # Parse "SOC 2 CC7.2" or "NIST SP 800-53 AU-9"
                    cv = tag_value.strip()
                    if cv.startswith("SOC 2"):
                        current.controls["soc2"] = cv.replace("SOC 2 ", "")
                    elif "NIST" in cv:
                        current.controls["nist"] = cv.split()[-1]
                    elif "ISO" in cv:
                        current.controls["iso"] = cv.split()[-1]

            elif tag_type == "rule":
                current_rule = tag_value.strip()
                if current:
                    current.rule = current_rule

        i += 1

    return assertions


def parse_all_standards() -> list[Assertion]:
    """Parse all docs/standards/*.md files."""
    if not STANDARDS_DIR.exists():
        logger.warning(f"Standards directory not found: {STANDARDS_DIR}")
        return []

    all_assertions = []
    for md in sorted(STANDARDS_DIR.glob("*.md")):
        try:
            parsed = parse_standard(md)
            all_assertions.extend(parsed)
        except Exception as e:
            logger.warning(f"Failed to parse {md.name}: {e}")
    return all_assertions


def _slug(text: str) -> str:
    """Generate a kebab-case check ID from assertion text."""
    s = re.sub(r"[^a-z0-9\s]", "", text.lower())
    s = re.sub(r"\s+", "-", s.strip())
    return s[:60]


# ---------------------------------------------------------------------------
# Verification executors
# ---------------------------------------------------------------------------

def verify_impl_exists(impl_ref: str) -> tuple[bool, str]:
    """
    Verify an implementation reference points to an existing file/symbol.

    impl_ref format: "syn/audit/models.py:SysLogEntry.save"
    """
    parts = impl_ref.split(":", 1)
    file_path = WEB_ROOT / parts[0]

    if not file_path.exists():
        return False, f"File not found: {parts[0]}"

    if len(parts) == 1:
        return True, f"File exists: {parts[0]}"

    # Symbol check via AST
    symbol = parts[1]
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"Syntax error in {parts[0]}: {e}"

    # Split symbol: "ClassName.method_name" or "function_name"
    symbol_parts = symbol.split(".")
    target_name = symbol_parts[0]

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == target_name:
                if len(symbol_parts) == 1:
                    return True, f"Found {symbol} in {parts[0]}"
                # Look for method inside class
                method_name = symbol_parts[1]
                for child in ast.walk(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.name == method_name:
                            return True, f"Found {symbol} in {parts[0]}"
                return False, f"Method {method_name} not found in {target_name}"

    # Check for module-level constants/variables
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == target_name:
                    return True, f"Found {symbol} in {parts[0]}"
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == target_name:
                return True, f"Found {symbol} in {parts[0]}"

    return False, f"Symbol {symbol} not found in {parts[0]}"


def verify_code_pattern(impl_ref: str, code_block: str) -> tuple[bool, str]:
    """
    Verify that key patterns from a code block appear in the implementation.

    Does fuzzy matching: extracts significant identifiers and operations
    from the code block and checks they appear in the source file.
    """
    parts = impl_ref.split(":", 1)
    file_path = WEB_ROOT / parts[0]

    if not file_path.exists():
        return False, f"File not found: {parts[0]}"

    source = file_path.read_text()

    # Extract key lines from code block (skip comments, blank lines, imports)
    key_patterns = []
    for line in code_block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("..."):
            continue
        # Extract significant identifiers: variable assignments, function calls, dict keys
        # Normalize whitespace for comparison
        normalized = re.sub(r"\s+", " ", stripped)
        if len(normalized) > 10:  # skip trivial lines
            key_patterns.append(normalized)

    if not key_patterns:
        return True, "No significant patterns to verify"

    # Check how many key patterns appear in source
    source_normalized = re.sub(r"\s+", " ", source)
    matched = 0
    missing = []
    for pattern in key_patterns:
        # Check for key identifiers from the pattern
        # Extract quoted strings, variable names, function calls
        tokens = re.findall(r'[a-zA-Z_]\w+', pattern)
        significant_tokens = [t for t in tokens if len(t) > 3 and t not in (
            "self", "None", "True", "False", "return", "from", "import",
            "class", "def", "async", "await", "with", "for", "while",
        )]
        if significant_tokens:
            # At least half of significant tokens should appear in source
            found = sum(1 for t in significant_tokens if t in source)
            if found >= len(significant_tokens) * 0.5:
                matched += 1
            else:
                missing.append(pattern[:80])

    total = len(key_patterns)
    ratio = matched / total if total > 0 else 1.0

    if ratio >= 0.6:
        return True, f"Pattern match: {matched}/{total} key patterns found"
    return False, f"Pattern mismatch: {matched}/{total} found, missing: {missing[:3]}"


def verify_code_absent(impl_ref: str, code_block: str) -> tuple[bool, str]:
    """Verify that a prohibited code pattern does NOT appear in the implementation."""
    parts = impl_ref.split(":", 1)
    file_path = WEB_ROOT / parts[0]

    if not file_path.exists():
        return True, f"File not found (no violation): {parts[0]}"

    source = file_path.read_text()

    # Extract the core anti-pattern (usually short — 1-3 lines)
    pattern_lines = [l.strip() for l in code_block.splitlines() if l.strip() and not l.strip().startswith("#")]

    for pattern_line in pattern_lines:
        normalized = re.sub(r"\s+", "", pattern_line)
        source_no_ws = re.sub(r"\s+", "", source)
        if len(normalized) > 5 and normalized in source_no_ws:
            return False, f"Prohibited pattern found: {pattern_line[:80]}"

    return True, "No prohibited patterns detected"


# ---------------------------------------------------------------------------
# Assertion runner
# ---------------------------------------------------------------------------

def verify_assertion(assertion: Assertion) -> dict:
    """Run all applicable verifications for a single assertion."""
    results = {
        "check_id": assertion.check_id,
        "assertion": assertion.text,
        "standard": assertion.standard,
        "section": assertion.section,
        "rule": assertion.rule,
        "impl_checks": [],
        "code_checks": [],
        "status": "pass",
    }

    # 1. Verify all impls exist
    for impl_ref in assertion.impls:
        ok, msg = verify_impl_exists(impl_ref)
        results["impl_checks"].append({"impl": impl_ref, "ok": ok, "message": msg})
        if not ok:
            results["status"] = "fail"

    # 2. Verify code:correct patterns present in first impl file
    if assertion.code_correct and assertion.impls:
        primary_impl = assertion.impls[0]
        for code_block in assertion.code_correct:
            ok, msg = verify_code_pattern(primary_impl, code_block)
            results["code_checks"].append({"type": "correct", "ok": ok, "message": msg})
            if not ok and results["status"] == "pass":
                results["status"] = "warning"

    # 3. Verify code:prohibited patterns absent from all impls
    for code_block in assertion.code_prohibited:
        for impl_ref in assertion.impls:
            ok, msg = verify_code_absent(impl_ref, code_block)
            results["code_checks"].append({"type": "prohibited", "impl": impl_ref, "ok": ok, "message": msg})
            if not ok:
                results["status"] = "fail"

    return results


def run_standards_checks():
    """Parse all standards, execute all assertions, return per-assertion results."""
    from syn.audit.models import ComplianceCheck

    assertions = parse_all_standards()
    if not assertions:
        return []

    check_results = []
    for assertion in assertions:
        start = time.time()
        result = verify_assertion(assertion)
        duration_ms = (time.time() - start) * 1000

        # Collect SOC 2 controls
        soc2 = []
        if "soc2" in assertion.controls:
            soc2.append(assertion.controls["soc2"])

        check = ComplianceCheck.objects.create(
            check_name=assertion.check_id,
            category="standards_compliance",
            status=result["status"],
            details={
                "assertion": result["assertion"],
                "standard": result["standard"],
                "section": result["section"],
                "rule": result["rule"],
                "impl_checks": result["impl_checks"],
                "code_checks": result["code_checks"],
            },
            soc2_controls=soc2,
            duration_ms=duration_ms,
        )
        check_results.append(check)

    logger.info(
        "[COMPLIANCE] Standards checks: %d assertions, %d passed, %d failed",
        len(check_results),
        sum(1 for c in check_results if c.status == "pass"),
        sum(1 for c in check_results if c.status == "fail"),
    )
    return check_results
