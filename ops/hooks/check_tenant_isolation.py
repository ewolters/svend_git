#!/usr/bin/env python3
"""Pre-commit hook: catch unscoped Site/Project lookups in view files.

SEC-001 §6.6 enforcement — every Site lookup must use resolve_site(),
every Project lookup must use resolve_project(). Unscoped lookups
allow cross-tenant data linkage (23 vulns fixed in CR d43b7c34).

This hook does NOT require Django or DB access — it's a pure regex
grep over staged files, same patterns as the tenant_isolation_lint
compliance check but runs without the ORM.

Exit 0 = clean, Exit 1 = violations found.
"""

import re
import subprocess
import sys


def _get_staged_view_files():
    """Return staged *_views.py and dispatch.py files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return [
            f for f in files if f.endswith("_views.py") or f.endswith("dispatch.py")
        ]
    except Exception:
        return []


# Same patterns as syn/audit/compliance.py check_tenant_isolation_lint
SITE_PATTERN = re.compile(r"Site\.objects\.get\(id=(?!.*tenant)")
PROJECT_PATTERN = re.compile(r"Project\.objects\.get\(id=.*user=request\.user")

EXCLUDES = {"permissions.py", "hoshin_deep_tests.py"}


def main():
    staged = _get_staged_view_files()
    if not staged:
        sys.exit(0)

    violations = []

    for filepath in staged:
        filename = filepath.split("/")[-1]
        if filename in EXCLUDES:
            continue

        try:
            # Read the staged version, not the working copy
            result = subprocess.run(
                ["git", "show", f":{filepath}"],
                capture_output=True,
                text=True,
            )
            content = result.stdout
        except Exception:
            continue

        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                continue
            for pattern in [SITE_PATTERN, PROJECT_PATTERN]:
                if pattern.search(line):
                    violations.append((filepath, i, stripped[:100]))

    if violations:
        print()
        print("=" * 70)
        print("  SEC-001 VIOLATION: Unscoped Site/Project lookup(s) detected")
        print()
        for filepath, line, content in violations:
            print(f"  {filepath}:{line}")
            print(f"    {content}")
            print()
        print("  Fix: use resolve_site() or resolve_project() from")
        print("  agents_api/permissions.py instead of direct ORM lookups.")
        print("  See SEC-001 §6.5 for correct patterns.")
        print("=" * 70)
        print()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
