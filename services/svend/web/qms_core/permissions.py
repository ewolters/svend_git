"""Shared site/org permission helpers for multi-site views.

Canonical home as of Phase 0 (CR-0.7). Previously agents_api/permissions.py.
Site and SiteAccess models remain in agents_api until Phase 4.

Standard:  ORG-001 §5.3, §11.3
"""

from django.db.models import Q
from django.http import JsonResponse

from agents_api.models import Site, SiteAccess
from core.models.tenant import Membership


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

    # Detect whether this model supports site-aware queries.
    # Models like SupplierRecord only have 'owner', not 'site'/'created_by'.
    field_names = {f.name for f in model._meta.get_fields()}
    has_site = "site" in field_names
    has_created_by = "created_by" in field_names

    if not has_site and not has_created_by:
        # Owner-only model (e.g. SupplierRecord) — no site scoping possible
        accessible_sites, is_admin = get_accessible_sites(user, tenant)
        if is_admin:
            tenant_user_ids = Membership.objects.filter(tenant=tenant, is_active=True).values_list("user_id", flat=True)
            return model.objects.filter(owner__in=tenant_user_ids), tenant, True
        else:
            return model.objects.filter(owner=user), tenant, False

    accessible_sites, is_admin = get_accessible_sites(user, tenant)

    if is_admin:
        # Org admin: all site-scoped records in tenant + unscoped records by tenant members
        # Include owner__in fallback for legacy records where created_by is NULL
        tenant_user_ids = Membership.objects.filter(tenant=tenant, is_active=True).values_list("user_id", flat=True)
        qs = model.objects.filter(
            Q(site__tenant=tenant)
            | Q(created_by__in=tenant_user_ids, site__isnull=True)
            | Q(owner__in=tenant_user_ids, site__isnull=True)
        )
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


def resolve_site(user, site_id):
    """Look up a Site with tenant isolation enforced.

    Returns (site, None) on success, (None, JsonResponse) on error.
    For individual users (no tenant): returns (None, None).
    """
    if not site_id:
        return None, None

    tenant = get_tenant(user)
    if not tenant:
        return None, None

    try:
        site = Site.objects.get(id=site_id, tenant=tenant)
    except Site.DoesNotExist:
        return None, JsonResponse({"error": "Site not found"}, status=404)

    return site, None


def resolve_project(user, project_id):
    """Look up a Project with tenant isolation enforced.

    Returns (project, None) on success, (None, JsonResponse) on error.
    Returns (None, None) if project_id is falsy.
    """
    if not project_id:
        return None, None

    from core.views import get_user_projects

    try:
        project = get_user_projects(user).get(id=project_id)
    except Exception:
        return None, JsonResponse({"error": "Project not found"}, status=404)

    return project, None


def qms_set_ownership(record, user, site=None):
    """Set ownership fields on a QMS record per ORG-001 §2.2/§5.2."""
    record.created_by = user
    if site:
        record.site = site
        record.owner = None
    else:
        record.owner = user
        record.site = None
