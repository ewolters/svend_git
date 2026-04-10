"""Shim — re-exports from qms_core.permissions during extraction.

Canonical home is now qms_core/permissions.py (CR-0.7, Phase 0).
This shim keeps all internal `from .permissions import ...` working.
Deleted at Phase 3 cutover.
"""

from qms_core.permissions import (  # noqa: F401
    check_site_read,
    check_site_write,
    get_accessible_sites,
    get_tenant,
    is_site_admin,
    qms_can_edit,
    qms_queryset,
    qms_set_ownership,
    require_tenant,
    resolve_project,
    resolve_site,
)
