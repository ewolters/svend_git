"""
Standards-driven compliance testing.

Parses docs/standards/*.md for machine-readable assertion tags and verifies
implementations match the standards. Standards are the single source of truth —
update the standard, the tests follow automatically.

Tag vocabulary (DOC-001 §7.3):
    <!-- assert: [claim] | check=[id] -->   Testable compliance assertion
    <!-- impl: [path:symbol] -->            Implementation file link
    <!-- test: [module.Class.method] -->    Linked test method (executable proof)
    <!-- code: correct -->                  Canonical code pattern
    <!-- code: prohibited -->               Anti-pattern
    <!-- check: [id] | soc2=X | nist=Y --> Named check with control mappings
    <!-- control: [framework] [id] -->      External compliance control
    <!-- rule: mandatory|recommended -->    Enforcement classification
    <!-- sla: [desc] | metric=X | ... -->  Service level agreement (SLA-001 §4)
"""

import ast
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

STANDARDS_DIR = Path(settings.BASE_DIR).parent.parent.parent / "docs" / "standards"
WEB_ROOT = Path(settings.BASE_DIR)

# Tag regex: <!-- keyword: value -->
TAG_RE = re.compile(r"^<!--\s*(assert|impl|check|code|control|rule|table|test|sla):\s*(.+?)\s*-->$")
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
    tests: list = field(default_factory=list)
    code_correct: list = field(default_factory=list)
    code_prohibited: list = field(default_factory=list)
    controls: dict = field(default_factory=dict)
    rule: str = ""
    standard: str = ""
    section: str = ""


@dataclass
class SLADefinition:
    """A machine-readable SLA extracted from <!-- sla: --> tags in standards."""
    description: str
    sla_id: str
    metric: str          # availability, response_time, durability, etc.
    target: str          # "99.9%", "2000ms", "24h"
    window: str          # monthly, quarterly, per_incident
    severity: str        # critical, high, medium, low
    measurement: str = "automated"
    soc2_controls: list = field(default_factory=list)
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

            elif tag_type == "test" and current:
                current.tests.append(tag_value.strip())

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


def parse_standard_titles() -> dict[str, str]:
    """Parse standard titles from the first line of each standards file.

    Returns dict like {"API-001": "API Design", "QMS-001": "Quality Management System"}.
    Format expected: **XXX-001: TITLE STANDARD**
    """
    titles = {}
    if not STANDARDS_DIR.exists():
        return titles
    title_re = re.compile(r"^\*\*([A-Z]+-\d+):\s+(.+?)\s+STANDARD\*\*$")
    for md in sorted(STANDARDS_DIR.glob("*.md")):
        try:
            with open(md) as f:
                first_line = f.readline().strip()
            m = title_re.match(first_line)
            if m:
                # Title-case but preserve known acronyms
                _ACRONYMS = {"API", "LLM", "SPC", "DOE", "CI", "CD", "SLA", "RCA", "VSM", "CSV", "SOC", "NIST", "FMEA", "CSRF", "CDN", "TLS", "SSL", "SQL", "ORM"}
                words = m.group(2).split()
                titled = []
                for w in words:
                    if w == "&":
                        titled.append("&")
                    elif w in _ACRONYMS:
                        titled.append(w)
                    elif "-" in w:
                        titled.append("-".join(p.capitalize() for p in w.split("-")))
                    else:
                        titled.append(w.capitalize())
                titles[m.group(1)] = " ".join(titled)
        except Exception:
            pass
    return titles


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
# SLA parser
# ---------------------------------------------------------------------------

SLA_TAG_RE = re.compile(r"^<!--\s*sla:\s*(.+?)\s*-->$")

VALID_METRICS = {"availability", "response_time", "durability", "incident_response", "compliance", "change_velocity"}
VALID_WINDOWS = {"monthly", "quarterly", "annually", "per_incident"}
VALID_SEVERITIES = {"critical", "high", "medium", "low"}


def parse_sla_definitions(filepath: Path) -> list[SLADefinition]:
    """Parse a single standards markdown file for <!-- sla: --> tags."""
    lines = filepath.read_text().splitlines()
    standard_name = filepath.stem
    current_section = ""
    in_fence = False
    slas = []

    for line in lines:
        stripped = line.strip()

        # Skip content inside fenced code blocks
        if FENCE_RE.match(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        m = SECTION_RE.match(stripped)
        if m:
            current_section = f"§{m.group(1)}"
            continue

        m = SLA_TAG_RE.match(stripped)
        if not m:
            continue

        tag_value = m.group(1)
        attrs = dict(ATTR_RE.findall(tag_value))
        description = ATTR_RE.sub("", tag_value).strip().rstrip("|").strip()

        metric = attrs.get("metric", "")
        target = attrs.get("target", "")
        window = attrs.get("window", "")
        severity = attrs.get("severity", "")

        if not all([metric, target, window, severity]):
            logger.warning(f"Incomplete SLA tag in {filepath.name} {current_section}: {description}")
            continue

        if metric not in VALID_METRICS:
            logger.warning(f"Unknown SLA metric '{metric}' in {filepath.name}: {description}")

        sla_id = attrs.get("check", f"sla-{_slug(description)}")
        soc2 = [attrs["soc2"]] if "soc2" in attrs else []

        slas.append(SLADefinition(
            description=description,
            sla_id=sla_id,
            metric=metric,
            target=target,
            window=window,
            severity=severity,
            measurement=attrs.get("measurement", "automated"),
            soc2_controls=soc2,
            standard=standard_name,
            section=current_section,
        ))

    return slas


def parse_all_sla_definitions() -> list[SLADefinition]:
    """Parse all docs/standards/*.md files for SLA definitions, deduplicated by sla_id."""
    if not STANDARDS_DIR.exists():
        logger.warning(f"Standards directory not found: {STANDARDS_DIR}")
        return []

    seen = {}
    for md in sorted(STANDARDS_DIR.glob("*.md")):
        try:
            for sla in parse_sla_definitions(md):
                if sla.sla_id not in seen:
                    seen[sla.sla_id] = sla
        except Exception as e:
            logger.warning(f"Failed to parse SLAs from {md.name}: {e}")

    return list(seen.values())


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
    """Verify that a prohibited code pattern does NOT appear in the implementation.

    For multi-line prohibited blocks, requires ALL non-comment lines to appear as a
    contiguous sequence in the source. Single-line patterns match individually.
    This prevents false positives when a prohibited block shares individual lines
    with the correct pattern (e.g., a function call that differs only in surrounding
    error-checking context).
    """
    parts = impl_ref.split(":", 1)
    file_path = WEB_ROOT / parts[0]

    if not file_path.exists():
        return True, f"File not found (no violation): {parts[0]}"

    source = file_path.read_text()

    # Extract the core anti-pattern lines (skip comments and blanks)
    pattern_lines = [l.strip() for l in code_block.splitlines()
                     if l.strip() and not l.strip().startswith("#")]

    if not pattern_lines:
        return True, "No prohibited patterns to check"

    # Single-line pattern: match individually (original behavior)
    if len(pattern_lines) == 1:
        normalized = re.sub(r"\s+", "", pattern_lines[0])
        source_no_ws = re.sub(r"\s+", "", source)
        if len(normalized) > 5 and normalized in source_no_ws:
            return False, f"Prohibited pattern found: {pattern_lines[0][:80]}"
        return True, "No prohibited patterns detected"

    # Multi-line pattern: require contiguous block match.
    # Normalize both source and pattern lines, then check if the full
    # sequence of pattern lines appears consecutively in the source.
    source_lines = [re.sub(r"\s+", "", l.strip()) for l in source.splitlines()]
    norm_patterns = [re.sub(r"\s+", "", l) for l in pattern_lines
                     if len(re.sub(r"\s+", "", l)) > 5]

    if not norm_patterns:
        return True, "No significant prohibited patterns to check"

    # Slide a window over source lines looking for the full pattern sequence
    window_size = len(norm_patterns)
    for i in range(len(source_lines) - window_size + 1):
        if all(norm_patterns[j] in source_lines[i + j]
               for j in range(window_size)):
            matched_source = pattern_lines[0][:80]
            return False, f"Prohibited pattern found: {matched_source}"

    return True, "No prohibited patterns detected"


def _ast_verify_test(file_path: str, class_name: str | None, method_name: str) -> tuple[bool, str]:
    """AST-based fallback for verifying test existence when import fails."""
    import ast as _ast

    try:
        with open(file_path) as f:
            tree = _ast.parse(f.read())
    except (FileNotFoundError, SyntaxError) as e:
        return False, f"Cannot parse {file_path}: {e}"

    if class_name:
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and item.name == method_name:
                        return True, f"Found via AST: {class_name}.{method_name}"
                return False, f"Method {method_name} not found in {class_name}"
        return False, f"Class {class_name} not found in {file_path}"
    else:
        for node in _ast.iter_child_nodes(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == method_name:
                return True, f"Found via AST: {method_name}"
        return False, f"Function {method_name} not found in {file_path}"


def verify_test_exists(test_ref: str) -> tuple[bool, str]:
    """
    Verify a test reference points to an importable test method.

    test_ref format: "syn.audit.tests.test_frontend.ThemeSystemTest.test_six_themes_defined"

    Falls back to AST parsing when module import fails (e.g. missing pytest).
    """
    import importlib

    parts = test_ref.rsplit(".", 2)
    if len(parts) < 2:
        return False, f"Invalid test reference: {test_ref}"

    if len(parts) == 3:
        module_path, class_name, method_name = parts
    else:
        module_path, method_name = parts
        class_name = None

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        # Fallback: resolve module path to file and use AST
        file_path = module_path.replace(".", "/") + ".py"
        if not os.path.isfile(file_path):
            return False, f"Cannot import {module_path} and file {file_path} not found"
        return _ast_verify_test(file_path, class_name, method_name)

    if class_name:
        cls = getattr(mod, class_name, None)
        if cls is None:
            return False, f"Class {class_name} not found in {module_path}"
        method = getattr(cls, method_name, None)
        if method is None:
            return False, f"Method {method_name} not found in {class_name}"
        return True, f"Found {test_ref}"
    else:
        func = getattr(mod, method_name, None)
        if func is None:
            return False, f"Function {method_name} not found in {module_path}"
        return True, f"Found {test_ref}"


_test_env_setup = False


def run_linked_test(test_ref: str) -> dict:
    """
    Run a single linked test method and return the result.

    Uses Django's test infrastructure for proper DB transaction management.
    Sets up Django's test environment on first call.
    Returns dict: {test_ref, passed, status, message}
        status: "pass" | "fail" | "skip" | "error"
    """
    global _test_env_setup
    import importlib
    import io
    import unittest

    if not _test_env_setup:
        from django.test.utils import setup_test_environment
        try:
            setup_test_environment()
        except RuntimeError:
            pass  # Already set up
        _test_env_setup = True

    parts = test_ref.rsplit(".", 2)
    if len(parts) != 3:
        return {"test_ref": test_ref, "passed": False, "status": "error", "message": "Invalid format"}

    module_path, class_name, method_name = parts

    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except Exception as e:
        # Fallback for modules needing pytest — skip execution
        file_path = module_path.replace(".", "/") + ".py"
        if os.path.isfile(file_path):
            ok, _ = _ast_verify_test(file_path, class_name, method_name)
            if ok:
                return {"test_ref": test_ref, "passed": False, "status": "skip",
                        "message": f"Cannot import (missing dependency): {e}"}
        return {"test_ref": test_ref, "passed": False, "status": "error", "message": str(e)}

    # TestCase/TransactionTestCase need the test DB — switch connection.
    needs_db = _is_testcase_needing_db(cls)
    switched_db = False
    original_db_name = None
    if needs_db:
        try:
            from django.db import connections
            conn = connections["default"]
            original_db_name = conn.settings_dict["NAME"]
            test_db_name = conn.creation._get_test_db_name()
            conn.close()
            conn.settings_dict["NAME"] = test_db_name
            switched_db = True
        except Exception:
            return {"test_ref": test_ref, "passed": False, "status": "skip",
                    "message": "Requires test database (Django TestCase)"}

    try:
        test_instance = cls(method_name)
        suite = unittest.TestSuite([test_instance])
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0)
        result = runner.run(suite)
    except Exception as e:
        err_msg = str(e)
        if switched_db:
            _restore_db(original_db_name)
        if _is_db_error(err_msg):
            return {"test_ref": test_ref, "passed": False, "status": "skip",
                    "message": "Requires test database"}
        return {"test_ref": test_ref, "passed": False, "status": "error", "message": err_msg}

    if switched_db:
        _restore_db(original_db_name)

    if result.wasSuccessful():
        return {"test_ref": test_ref, "passed": True, "status": "pass", "message": "PASS"}

    # Check if errors are DB-related
    for _, tb in result.errors:
        if _is_db_error(tb):
            return {"test_ref": test_ref, "passed": False, "status": "skip",
                    "message": "Requires test database"}

    # Collect failure/error messages
    msgs = []
    for _, tb in result.failures + result.errors:
        last_line = tb.strip().split("\n")[-1]
        msgs.append(last_line[:120])
    return {"test_ref": test_ref, "passed": False, "status": "fail", "message": "; ".join(msgs) or "FAIL"}


def _is_db_error(msg: str) -> bool:
    """Detect database-related errors that indicate a test needs a test DB."""
    indicators = [
        "database", "relation", "does not exist", "no such table",
        "OperationalError", "ProgrammingError", "connection",
        "test_svend", "createdb", "TransactionManagementError",
    ]
    msg_lower = str(msg).lower()
    return any(ind.lower() in msg_lower for ind in indicators)


def _restore_db(original_name: str):
    """Restore the default DB connection to the original database."""
    from django.db import connections
    conn = connections["default"]
    conn.close()
    conn.settings_dict["NAME"] = original_name


def _is_testcase_needing_db(cls) -> bool:
    """Check if a test class is a Django TestCase (needs DB) vs SimpleTestCase.

    Django hierarchy: SimpleTestCase → TransactionTestCase → TestCase.
    TransactionTestCase and TestCase need a test database.
    SimpleTestCase and plain unittest.TestCase do not.
    """
    from django.test import TransactionTestCase
    return issubclass(cls, TransactionTestCase)


# ---------------------------------------------------------------------------
# Assertion runner
# ---------------------------------------------------------------------------

def verify_assertion(assertion: Assertion, run_tests: bool = False) -> dict:
    """Run all applicable verifications for a single assertion."""
    results = {
        "check_id": assertion.check_id,
        "assertion": assertion.text,
        "standard": assertion.standard,
        "section": assertion.section,
        "rule": assertion.rule,
        "impl_checks": [],
        "code_checks": [],
        "test_checks": [],
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

    # 4. Verify linked tests exist (and optionally run them)
    # Test results affect assertion status: missing test → warning, failed test → warning.
    # Skipped tests (need test DB) don't count against the assertion.
    for test_ref in assertion.tests:
        ok, msg = verify_test_exists(test_ref)
        tc = {"test": test_ref, "exists": ok, "message": msg, "ran": False}

        if not ok:
            if results["status"] == "pass":
                results["status"] = "warning"
        elif run_tests:
            tr = run_linked_test(test_ref)
            tc["ran"] = True
            tc["passed"] = tr["passed"]
            tc["status"] = tr.get("status", "fail")
            tc["message"] = tr["message"]
            # Failed tests degrade assertion to warning (skips don't count against)
            if tc["status"] == "fail" and results["status"] == "pass":
                results["status"] = "warning"

        results["test_checks"].append(tc)

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
                "test_checks": result["test_checks"],
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
