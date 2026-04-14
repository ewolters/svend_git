"""QMS Core views — site picker for A3/FMEA forms."""

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid


@gated_paid
@require_http_methods(["GET"])
def qms_sites(request):
    """List sites accessible to the current user for QMS forms."""
    from qms_core.permissions import get_accessible_sites, get_tenant

    tenant = get_tenant(request.user)
    if not tenant:
        return JsonResponse({"sites": [], "has_org": False})

    sites, is_admin = get_accessible_sites(request.user, tenant)
    return JsonResponse(
        {
            "sites": [{"id": str(s.id), "name": s.name, "code": s.code} for s in sites.order_by("name")],
            "has_org": True,
            "is_org_admin": is_admin,
        }
    )
