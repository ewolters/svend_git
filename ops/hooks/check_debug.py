#!/usr/bin/env python3
"""Pre-commit hook: catch debug statements before they reach production.

OPS-001 enforcement — breakpoint(), pdb, and bare print() in
production code cause information leakage, performance degradation,
and stdout noise in Gunicorn logs.

Scans staged Python files for debug patterns. Pure regex — no
Django or DB needed.

Exit 0 = clean, Exit 1 = debug statements found.
"""

import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Debug patterns
# ---------------------------------------------------------------------------

DEBUG_PATTERNS = [
    (re.compile(r"^\s*breakpoint\(\)"), "breakpoint() call"),
    (re.compile(r"^\s*import pdb"), "pdb import"),
    (re.compile(r"^\s*pdb\.set_trace\(\)"), "pdb.set_trace() call"),
    (re.compile(r"^\s*import ipdb"), "ipdb import"),
    (re.compile(r"^\s*ipdb\.set_trace\(\)"), "ipdb.set_trace() call"),
    (re.compile(r"^\s*print\("), "bare print() statement"),
]

# Paths where print() is acceptable
PRINT_EXEMPT_PATHS = {
    # Management commands use print for console output
    "management/",
    # Hook scripts use print for pre-commit output
    "ops/hooks/",
    # Test files use print for debugging
    "tests",
    "test_",
    # conftest fixtures
    "conftest.py",
    # Scripts
    "scripts/",
    # Migrations use print for progress
    "migrations/",
    # Log handlers use stderr print as fallback when logging fails
    "syn/log/",
    # Synara engine debug output
    "synara/synara.py",
    # LLM service debug output
    "llm_service.py",
    # Bayesian module example output
    "bayesian.py",
    # Scheduler dashboard metrics examples
    "dashboard/metrics.py",
    # Agent example code
    "agents/experimenter/",
}

# All debug patterns (not just print) are blocked everywhere except tests
DEBUG_ONLY_EXEMPT = {
    "tests",
    "test_",
    "conftest.py",
}


def _get_staged_python_files():
    """Return staged .py files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return [f for f in files if f.endswith(".py")]
    except Exception:
        return []


def _is_exempt(filepath, pattern_desc):
    """Check if this file is exempt from the given pattern."""
    if pattern_desc == "bare print() statement":
        return any(exempt in filepath for exempt in PRINT_EXEMPT_PATHS)
    else:
        # breakpoint/pdb only exempt in test files
        return any(exempt in filepath for exempt in DEBUG_ONLY_EXEMPT)


def main():
    staged = _get_staged_python_files()
    if not staged:
        sys.exit(0)

    violations = []

    for filepath in staged:
        try:
            result = subprocess.run(
                ["git", "show", f":{filepath}"],
                capture_output=True,
                text=True,
            )
            content = result.stdout
        except Exception:
            continue

        for i, line in enumerate(content.splitlines(), 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern, description in DEBUG_PATTERNS:
                if pattern.search(line) and not _is_exempt(filepath, description):
                    violations.append((filepath, i, description, stripped[:80]))

    if violations:
        print()
        print("=" * 70)
        print("  DEBUG VIOLATION: Debug statement(s) detected in staged files")
        print()
        for filepath, line, description, content in violations:
            print(f"  {filepath}:{line}")
            print(f"    {description}: {content}")
        print()
        print("  Remove debug statements before committing to production.")
        print("  print() is allowed in management commands, hooks, and tests.")
        print("=" * 70)
        print()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
