#!/usr/bin/env python3
"""Pre-commit hook: block commits unless a matching ChangeRequest is in_progress.

CHG-001 §7.1.1 enforcement — every code change must have a reason,
and that reason must be tracked in a ChangeRequest that covers the
files being committed.

Matching logic:
  1. Get staged files from git
  2. Find all CRs with status='in_progress'
  3. For each CR, check if any staged file matches CR.affected_files
     (glob-style matching: '*.py' matches any .py, 'agents_api/' matches subtree)
  4. If no CR covers the staged files, block the commit

Fail-closed: if Django can't load or DB is unreachable, commits are blocked.
This prevents bypass via infrastructure issues.

Exit 0 = allow commit, Exit 1 = block commit.
"""

import fnmatch
import os
import subprocess
import sys

# Add the Django project root to the path and chdir so pydantic-settings finds .env
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PROJECT_ROOT = os.path.normpath(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "svend.settings")

# Load env from /etc/svend/env if it exists (server hardening moved .env there)
_env_file = "/etc/svend/env"
if os.path.exists(_env_file):
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

# Suppress all Django startup logging
import logging

logging.disable(logging.CRITICAL)


def _get_staged_files():
    """Return list of staged file paths relative to repo root."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        return []


def _file_matches_cr(staged_files, affected_files):
    """Check if any staged file matches the CR's affected_files patterns.

    affected_files can be:
      - A JSON list: ["agents_api/views.py", "templates/*.html"]
      - A string: "agents_api/views.py, templates/*.html"
      - Glob patterns: "*.py", "agents_api/*", "docs/standards/"

    A staged file matches if:
      - Exact path match
      - Glob pattern match (fnmatch)
      - Staged file starts with an affected directory prefix
    """
    if not affected_files:
        return False

    # Normalize to list of patterns
    if isinstance(affected_files, str):
        patterns = [p.strip() for p in affected_files.split(",") if p.strip()]
    elif isinstance(affected_files, list):
        patterns = [str(p).strip() for p in affected_files if p]
    else:
        return False

    for staged in staged_files:
        rel = staged

        for pattern in patterns:
            # Exact match
            if rel == pattern or staged == pattern:
                return True
            # Glob match
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(staged, pattern):
                return True
            # Directory prefix match (pattern "agents_api/" covers "agents_api/views.py")
            if pattern.endswith("/") and rel.startswith(pattern):
                return True
            # Bare directory match without trailing slash
            if "/" not in pattern and "." not in pattern:
                if rel.startswith(pattern + "/"):
                    return True

    return False


try:
    import django

    django.setup()

    from syn.audit.models import ChangeRequest

    staged_files = _get_staged_files()

    if not staged_files:
        # Nothing staged — allow (e.g. amend with no new changes)
        sys.exit(0)

    active_crs = ChangeRequest.objects.filter(status="in_progress")

    if not active_crs.exists():
        print()
        print("=" * 70)
        print("  CHG-001 VIOLATION: No ChangeRequest in 'in_progress' status.")
        print()
        print("  Every code change requires a ChangeRequest. Before committing:")
        print()
        print("    1. Create a ChangeRequest with title, description, author")
        print("    2. Set justification, affected_files, implementation_plan")
        print("    3. Create a RiskAssessment (for code-touching types)")
        print("    4. Transition: draft → submitted → approved → in_progress")
        print("    5. Create ChangeLog entries at each step")
        print()
        print("  Then commit. See CLAUDE.md § Change Management for details.")
        print("=" * 70)
        print()
        sys.exit(1)

    # Check if any active CR covers the staged files
    matched = False
    for cr in active_crs:
        if _file_matches_cr(staged_files, cr.affected_files):
            matched = True
            break

    if not matched:
        print()
        print("=" * 70)
        print("  CHG-001 VIOLATION: No active CR covers the staged files.")
        print()
        print("  Staged files:")
        for f in staged_files[:10]:
            print(f"    - {f}")
        if len(staged_files) > 10:
            print(f"    ... and {len(staged_files) - 10} more")
        print()
        print("  Active CRs and their affected_files:")
        for cr in active_crs:
            title = cr.title[:60]
            af = cr.affected_files or "(none)"
            print(f"    [{str(cr.id)[:8]}] {title}")
            print(f"             affected_files: {af}")
        print()
        print("  Update the CR's affected_files to include the files you're")
        print("  changing, or create a new CR for this work.")
        print("=" * 70)
        print()
        sys.exit(1)

    # Matched — allow commit
    sys.exit(0)

except Exception as e:
    # Fail-closed: if Django/DB fails, block the commit
    print()
    print("=" * 70)
    print(f"  CHG-001 ENFORCEMENT ERROR: {e}")
    print()
    print("  The CR check could not run. Commits are blocked until resolved.")
    print("  If this is a Django/DB issue, fix it before committing.")
    print("=" * 70)
    print()
    sys.exit(1)
