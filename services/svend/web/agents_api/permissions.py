"""Shared site/org permission helpers for multi-site views.

Extracted from hoshin_views.py per ORG-001 §11.3 step 1.
Used by hoshin_views, QMS views, xmatrix_views, and any view
that needs org/site permission checks.
"""

from django.db.models import Q
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


# ---------------------------------------------------------------------------
# QMS site-aware query helpers (ORG-001 §5.3)
# ---------------------------------------------------------------------------


def qms_queryset(model, user):
    """Return a site-aware queryset for a QMS model per ORG-001 §5.3.

    - Individual user (no org): filter by owner=user
    - Org member: own records (created_by) + records at accessible sites
    - Org admin: all records in the org (all sites + unscoped tenant records)

    Returns (queryset, tenant_or_none, is_org_admin).
    """
    tenant = get_tenant(user)
    if not tenant:
        return model.objects.filter(owner=user), None, False

    accessible_sites, is_admin = get_accessible_sites(user, tenant)

    if is_admin:
        # Org admin: all site-scoped records in tenant + own unscoped records
        tenant_user_ids = Membership.objects.filter(tenant=tenant, is_active=True).values_list("user_id", flat=True)
        qs = model.objects.filter(Q(site__tenant=tenant) | Q(created_by__in=tenant_user_ids, site__isnull=True))
    else:
        # Org member: own records + records at accessible sites
        qs = model.objects.filter(Q(created_by=user) | Q(site__in=accessible_sites))

    return qs, tenant, is_admin


def qms_can_edit(user, record, tenant):
    """Check if user can edit a QMS record per ORG-001 §5.3.

    - Record without site (owner=user): only owner can edit
    - Record with site: need site write access
    - Org admin: can edit anything in the org
    """
    if not tenant:
        return record.owner_id == user.id

    membership = Membership.objects.filter(user=user, tenant=tenant, is_active=True).first()
    if not membership:
        return False
    if membership.can_admin:
        return True

    if record.site_id:
        return check_site_write(user, record.site, tenant)

    # Unscoped record — only owner can edit
    return record.owner_id == user.id


def qms_set_ownership(record, user, site=None):
    """Set ownership fields on a QMS record per ORG-001 §2.2/§5.2.

    Always sets created_by. If site is provided, clears owner (site-scoped).
    If no site, sets owner=user (individual-scoped).
    """
    record.created_by = user
    if site:
        record.site = site
        record.owner = None
    else:
        record.owner = user
        record.site = None
