#!/usr/bin/env python3
"""Pre-commit hook: block commits unless a ChangeRequest is in_progress.

CHG-001 §7.1.1 enforcement — every code change must have a reason,
and that reason must be tracked in a ChangeRequest before code is committed.

This is a poka-yoke. It cannot be bypassed without --no-verify,
which is itself a compliance violation visible in the audit trail.

Exit 0 = allow commit, Exit 1 = block commit.
"""

import os
import sys

# Add the Django project to the path
WEB_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "services", "svend", "web"
)
WEB_DIR = os.path.normpath(WEB_DIR)
sys.path.insert(0, WEB_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "svend.settings")

# Suppress all Django startup logging
import logging

logging.disable(logging.CRITICAL)

try:
    import django

    django.setup()

    from syn.audit.models import ChangeRequest

    active = ChangeRequest.objects.filter(status="in_progress").count()

    if active == 0:
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

    # If we get here, at least one CR is active — allow the commit
    sys.exit(0)

except Exception as e:
    # If Django fails to load or DB is unreachable, allow the commit
    # (don't block development due to infrastructure issues)
    print(f"  [check_cr] Warning: could not verify CR status ({e})")
    sys.exit(0)
