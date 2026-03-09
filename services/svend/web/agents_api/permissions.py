"""Shared site/org permission helpers for multi-site views.

Extracted from hoshin_views.py per ORG-001 §11.3 step 1.
Used by hoshin_views, QMS views, xmatrix_views, and any view
that needs org/site permission checks.
"""

from django.http import JsonResponse

from core.models.tenant import Membership

from .models import Site, SiteAccess


def get_tenant(user):
    """Get the user's enterprise tenant, or None."""
    membership = Membership.objects.filter(user=user, is_active=True).select_related("tenant").first()
    return membership.tenant if membership else None


def require_tenant(user):
    """Get tenant or return error response.

    Returns (tenant, None) on success, (None, JsonResponse) on failure.
    """
    tenant = get_tenant(user)
    if not tenant:
        return None, JsonResponse(
            {"error": "No active tenant. Create or join an organization first."},
            status=400,
        )
    return tenant, None


def get_accessible_sites(user, tenant):
    """Return (queryset_of_sites, is_org_admin).

    Org owners/admins: all tenant sites.
    Others: only sites with a SiteAccess entry.
    """
    membership = Membership.objects.filter(
        user=user,
        tenant=tenant,
        is_active=True,
    ).first()
    if not membership:
        return Site.objects.none(), False

    if membership.can_admin:
        return Site.objects.filter(tenant=tenant), True

    accessible_ids = SiteAccess.objects.filter(
        user=user,
        site__tenant=tenant,
    ).values_list("site_id", flat=True)
    return Site.objects.filter(id__in=accessible_ids), False


def check_site_read(user, site, tenant):
    """Return True if user can view this site's resources."""
    membership = Membership.objects.filter(
        user=user,
        tenant=tenant,
        is_active=True,
    ).first()
    if not membership:
        return False
    if membership.can_admin:
        return True
    return SiteAccess.objects.filter(user=user, site=site).exists()


def check_site_write(user, site, tenant):
    """Return True if user can edit this site's resources."""
    membership = Membership.objects.filter(
        user=user,
        tenant=tenant,
        is_active=True,
    ).first()
    if not membership:
        return False
    if membership.can_admin:
        return True
    access = SiteAccess.objects.filter(user=user, site=site).first()
    return access is not None and access.role in ("member", "admin")


def is_site_admin(user, site, tenant):
    """Return True if user is org admin or site admin for this site."""
    membership = Membership.objects.filter(
        user=user,
        tenant=tenant,
        is_active=True,
    ).first()
    if not membership:
        return False
    if membership.can_admin:
        return True
    access = SiteAccess.objects.filter(user=user, site=site).first()
    return access is not None and access.role == "admin"
