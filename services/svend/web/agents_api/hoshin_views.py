"""Hoshin Kanri CI project management API views.

Enterprise-only feature for managing continuous improvement project portfolios:
- Sites (manufacturing facilities within a tenant)
- HoshinProjects (CI initiatives with savings tracking)
- ActionItems (task tracking with Gantt-style dependencies)
- Dashboard (savings rollup by site, monthly trend, target vs actual)
- Calculation methods reference
"""

import json
import logging
from datetime import date
from decimal import Decimal, InvalidOperation

from django.db import transaction
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from accounts.permissions import require_feature
from .models import Site, SiteAccess, HoshinProject, ActionItem, ValueStreamMap
from .hoshin_calculations import (
    CALCULATION_METHODS,
    calculate_savings,
    aggregate_monthly_savings,
    evaluate_custom_formula,
    extract_formula_fields,
    normalize_formula,
)
from core.models.project import Project
from core.models.tenant import Tenant, Membership

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tenant(user):
    """Get the user's enterprise tenant, or None."""
    membership = Membership.objects.filter(
        user=user, is_active=True
    ).select_related("tenant").first()
    return membership.tenant if membership else None


def _require_tenant(user):
    """Get tenant or return error response."""
    tenant = _get_tenant(user)
    if not tenant:
        return None, JsonResponse({
            "error": "No active tenant. Create or join an organization first.",
        }, status=400)
    return tenant, None


def _get_accessible_sites(user, tenant):
    """Return (queryset_of_sites, is_org_admin).

    Org owners/admins: all tenant sites.
    Others: only sites with a SiteAccess entry.
    """
    membership = Membership.objects.filter(
        user=user, tenant=tenant, is_active=True,
    ).first()
    if not membership:
        return Site.objects.none(), False

    if membership.can_admin:
        return Site.objects.filter(tenant=tenant), True

    accessible_ids = SiteAccess.objects.filter(
        user=user, site__tenant=tenant,
    ).values_list("site_id", flat=True)
    return Site.objects.filter(id__in=accessible_ids), False


def _check_site_read(user, site, tenant):
    """Return True if user can view this site's projects."""
    membership = Membership.objects.filter(
        user=user, tenant=tenant, is_active=True,
    ).first()
    if not membership:
        return False
    if membership.can_admin:
        return True
    return SiteAccess.objects.filter(user=user, site=site).exists()


def _check_site_write(user, site, tenant):
    """Return True if user can edit this site's projects."""
    membership = Membership.objects.filter(
        user=user, tenant=tenant, is_active=True,
    ).first()
    if not membership:
        return False
    if membership.can_admin:
        return True
    access = SiteAccess.objects.filter(user=user, site=site).first()
    return access is not None and access.role in ("member", "admin")


def _is_site_admin(user, site, tenant):
    """Return True if user is org admin or site admin for this site."""
    membership = Membership.objects.filter(
        user=user, tenant=tenant, is_active=True,
    ).first()
    if not membership:
        return False
    if membership.can_admin:
        return True
    access = SiteAccess.objects.filter(user=user, site=site).first()
    return access is not None and access.role == "admin"


# ---------------------------------------------------------------------------
# Site CRUD
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def list_sites(request):
    """List manufacturing sites the user has access to."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    accessible_sites, is_admin = _get_accessible_sites(request.user, tenant)
    sites = accessible_sites
    if request.GET.get("active_only", "").lower() == "true":
        sites = sites.filter(is_active=True)

    return JsonResponse({
        "sites": [s.to_dict() for s in sites],
        "count": sites.count(),
        "is_org_admin": is_admin,
    })


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def create_site(request):
    """Create a new manufacturing site."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    data = json.loads(request.body)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "Site name is required"}, status=400)

    site = Site.objects.create(
        tenant=tenant,
        name=name,
        code=data.get("code", "").strip(),
        business_unit=data.get("business_unit", "").strip(),
        plant_manager=data.get("plant_manager", "").strip(),
        ci_leader=data.get("ci_leader", "").strip(),
        controller=data.get("controller", "").strip(),
        address=data.get("address", "").strip(),
    )

    return JsonResponse({"success": True, "site": site.to_dict()}, status=201)


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def get_site(request, site_id):
    """Get a single site with project summary."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    if not _check_site_read(request.user, site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    data = site.to_dict()

    # Add project summary
    projects = HoshinProject.objects.filter(site=site).exclude(hoshin_status="aborted")
    data["project_count"] = projects.count()
    data["active_count"] = projects.filter(hoshin_status="active").count()
    data["total_target"] = float(sum(p.annual_savings_target for p in projects))
    data["total_ytd"] = float(sum(p.ytd_savings for p in projects))

    return JsonResponse({"site": data})


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["PUT", "PATCH"])
def update_site(request, site_id):
    """Update a site."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    data = json.loads(request.body)

    for field in ["name", "code", "business_unit", "plant_manager", "ci_leader", "controller", "address"]:
        if field in data:
            setattr(site, field, data[field].strip() if isinstance(data[field], str) else data[field])
    if "is_active" in data:
        site.is_active = bool(data["is_active"])
    site.save()

    return JsonResponse({"success": True, "site": site.to_dict()})


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["DELETE"])
def delete_site(request, site_id):
    """Delete a site (cascades to hoshin projects)."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    project_count = site.hoshin_projects.count()
    site.delete()

    return JsonResponse({"success": True, "deleted_projects": project_count})


# ---------------------------------------------------------------------------
# HoshinProject CRUD
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def list_hoshin_projects(request):
    """List hoshin projects the user has access to."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    accessible_sites, _ = _get_accessible_sites(request.user, tenant)
    projects = HoshinProject.objects.filter(
        site__in=accessible_sites,
    ).select_related("project", "site")

    # Filters
    site_id = request.GET.get("site_id")
    if site_id:
        projects = projects.filter(site_id=site_id)

    status = request.GET.get("status")
    if status:
        projects = projects.filter(hoshin_status=status)

    fiscal_year = request.GET.get("fiscal_year")
    if fiscal_year:
        projects = projects.filter(fiscal_year=int(fiscal_year))

    project_type = request.GET.get("project_type")
    if project_type:
        projects = projects.filter(project_type=project_type)

    return JsonResponse({
        "projects": [p.to_dict() for p in projects],
        "count": projects.count(),
    })


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def create_hoshin_project(request):
    """Create a hoshin project (creates core.Project + HoshinProject in a transaction)."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    data = json.loads(request.body)
    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "Project title is required"}, status=400)

    site_id = data.get("site_id")
    site = None
    if site_id:
        site = get_object_or_404(Site, id=site_id, tenant=tenant)
        if not _check_site_write(request.user, site, tenant):
            return JsonResponse({"error": "Not found"}, status=404)

    try:
        savings_target = Decimal(str(data.get("annual_savings_target", 0)))
    except (InvalidOperation, ValueError):
        savings_target = Decimal("0")

    with transaction.atomic():
        # Create the core project
        core_project = Project.objects.create(
            user=request.user,
            tenant=tenant,
            title=title,
            status="ACTIVE",
            methodology=data.get("methodology", "PDCA"),
            current_phase="DEFINE",
            domain="manufacturing",
            goal_metric=data.get("goal_metric", ""),
            goal_baseline=data.get("goal_baseline"),
            goal_target=data.get("goal_target"),
            champion_name=data.get("champion_name", ""),
            leader_name=data.get("leader_name", ""),
            team_members=data.get("team_members", []),
        )

        # Create the hoshin wrapper
        hoshin = HoshinProject.objects.create(
            project=core_project,
            site=site,
            project_class=data.get("project_class", "project"),
            project_type=data.get("project_type", "material"),
            opportunity=data.get("opportunity", "budgeted_new"),
            hoshin_status=data.get("hoshin_status", "proposed"),
            fiscal_year=data.get("fiscal_year", date.today().year),
            annual_savings_target=savings_target,
            calculation_method=data.get("calculation_method", ""),
            custom_formula=data.get("custom_formula", ""),
            custom_formula_desc=data.get("custom_formula_desc", ""),
            kaizen_charter=data.get("kaizen_charter", {}),
            baseline_data=data.get("baseline_data", []),
            source_vsm_id=data.get("source_vsm_id"),
            source_burst_id=data.get("source_burst_id", ""),
        )

    return JsonResponse({"success": True, "project": hoshin.to_dict()}, status=201)


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def get_hoshin_project(request, hoshin_id):
    """Get a single hoshin project with full details."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("project", "site"),
        id=hoshin_id, site__tenant=tenant,
    )

    if not _check_site_read(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    data = hoshin.to_dict()
    data["savings_summary"] = aggregate_monthly_savings(hoshin.monthly_actuals or [])
    data["action_items"] = [
        a.to_dict() for a in ActionItem.objects.filter(project=hoshin.project)
    ]

    return JsonResponse({"project": data})


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["PUT", "PATCH"])
def update_hoshin_project(request, hoshin_id):
    """Update a hoshin project."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id, site__tenant=tenant,
    )
    if not _check_site_write(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    data = json.loads(request.body)

    # Hoshin-specific fields
    for field in ["project_class", "project_type", "opportunity", "hoshin_status",
                  "fiscal_year", "calculation_method", "custom_formula",
                  "custom_formula_desc", "kaizen_charter",
                  "monthly_actuals", "baseline_data", "source_burst_id"]:
        if field in data:
            setattr(hoshin, field, data[field])

    if "annual_savings_target" in data:
        try:
            hoshin.annual_savings_target = Decimal(str(data["annual_savings_target"]))
        except (InvalidOperation, ValueError):
            pass

    if "site_id" in data:
        if data["site_id"]:
            hoshin.site = get_object_or_404(Site, id=data["site_id"], tenant=tenant)
        else:
            hoshin.site = None

    hoshin.save()

    # Also update core.Project fields if provided
    core_fields = ["title", "champion_name", "leader_name", "team_members",
                   "methodology", "goal_metric"]
    core_changed = False
    for field in core_fields:
        if field in data:
            setattr(hoshin.project, field, data[field])
            core_changed = True
    if core_changed:
        hoshin.project.save()

    return JsonResponse({"success": True, "project": hoshin.to_dict()})


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["DELETE"])
def delete_hoshin_project(request, hoshin_id):
    """Delete a hoshin project (also deletes core.Project)."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id, site__tenant=tenant,
    )
    if not _check_site_write(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    core_project = hoshin.project
    hoshin.delete()
    core_project.delete()

    return JsonResponse({"success": True})


# ---------------------------------------------------------------------------
# Monthly savings tracking
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["PUT"])
def update_monthly_actual(request, hoshin_id, month):
    """Update one month's savings data."""
    if not 1 <= month <= 12:
        return JsonResponse({"error": "Month must be 1-12"}, status=400)

    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id, site__tenant=tenant,
    )
    if not _check_site_write(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    data = json.loads(request.body)

    actuals = list(hoshin.monthly_actuals or [])

    # Find existing month entry or create new one
    entry = None
    for m in actuals:
        if m.get("month") == month:
            entry = m
            break

    if entry is None:
        entry = {"month": month}
        actuals.append(entry)

    # Update fields
    for field in ["baseline", "actual", "volume", "cost_per_unit", "sales"]:
        if field in data:
            entry[field] = float(data[field]) if data[field] is not None else None

    # Store custom {{field}} variables
    if "custom_vars" in data and isinstance(data["custom_vars"], dict):
        entry["custom_vars"] = {
            k: float(v) if v is not None else None
            for k, v in data["custom_vars"].items()
        }

    # Calculate savings if we have the data
    baseline = entry.get("baseline")
    actual = entry.get("actual")
    volume = entry.get("volume", 1)
    cost = entry.get("cost_per_unit", 1)

    if baseline is not None and actual is not None and hoshin.calculation_method:
        extra_kwargs = {}
        if hoshin.calculation_method == "custom" and hoshin.custom_formula:
            extra_kwargs["formula"] = hoshin.custom_formula
            extra_kwargs["sales"] = entry.get("sales", 0) or 0
            # Pass any custom {{field}} variables stored on the entry
            custom_vars = entry.get("custom_vars", {})
            if custom_vars:
                extra_kwargs["custom_vars"] = custom_vars
        result = calculate_savings(
            hoshin.calculation_method, baseline, actual,
            volume or 1, cost or 1, **extra_kwargs,
        )
        entry["savings"] = result["savings"]
        entry["improvement_pct"] = result["improvement_pct"]
    elif "savings" in data:
        entry["savings"] = float(data["savings"])

    hoshin.monthly_actuals = sorted(actuals, key=lambda x: x.get("month", 0))
    hoshin.save()

    return JsonResponse({
        "success": True,
        "month": month,
        "entry": entry,
        "ytd_savings": hoshin.ytd_savings,
        "savings_pct": hoshin.savings_pct,
    })


# ---------------------------------------------------------------------------
# Batch create from VSM proposals
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def create_from_proposals(request):
    """Batch-create hoshin projects from approved VSM kaizen burst proposals.

    Request body:
    {
        "vsm_id": "...",
        "fiscal_year": 2026,
        "site_id": "...",
        "default_volume": 100000,
        "default_cost_per_unit": 50,
        "proposals": [
            {
                "burst_id": "abc",
                "title": "Reduce changeover on Assembly",
                "project_class": "kaizen",
                "project_type": "labor",
                "calculation_method": "time_reduction",
                "annual_savings_target": 25000,
                "approved": true
            }, ...
        ]
    }
    """
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    data = json.loads(request.body)
    proposals = data.get("proposals", [])
    approved = [p for p in proposals if p.get("approved", False)]

    if not approved:
        return JsonResponse({"error": "No approved proposals"}, status=400)

    vsm_id = data.get("vsm_id")
    vsm = None
    if vsm_id:
        vsm = ValueStreamMap.objects.select_related("project").filter(
            id=vsm_id, owner=request.user,
        ).first()
        # Drop if VSM belongs to a different tenant
        if vsm and vsm.project and vsm.project.tenant_id and vsm.project.tenant_id != tenant.id:
            vsm = None

    site_id = data.get("site_id")
    site = None
    if site_id:
        site = get_object_or_404(Site, id=site_id, tenant=tenant)
        if not _check_site_write(request.user, site, tenant):
            return JsonResponse({"error": "Not found"}, status=404)

    fiscal_year = data.get("fiscal_year", date.today().year)
    created = []

    with transaction.atomic():
        for prop in approved:
            title = prop.get("title", "Untitled CI Project").strip()

            try:
                target = Decimal(str(prop.get("annual_savings_target", 0)))
            except (InvalidOperation, ValueError):
                target = Decimal("0")

            core_project = Project.objects.create(
                user=request.user,
                tenant=tenant,
                title=title,
                status="ACTIVE",
                methodology="PDCA",
                current_phase="DEFINE",
                domain="manufacturing",
            )

            hoshin = HoshinProject.objects.create(
                project=core_project,
                site=site,
                project_class=prop.get("project_class", "project"),
                project_type=prop.get("project_type", "material"),
                opportunity="budgeted_new",
                hoshin_status="proposed",
                fiscal_year=fiscal_year,
                annual_savings_target=target,
                calculation_method=prop.get("calculation_method", ""),
                source_vsm=vsm,
                source_burst_id=prop.get("burst_id", ""),
            )
            created.append(hoshin.to_dict())

    return JsonResponse({
        "success": True,
        "created": created,
        "count": len(created),
    }, status=201)


# ---------------------------------------------------------------------------
# Action Items
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def list_action_items(request, hoshin_id):
    """List action items for a hoshin project."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id, site__tenant=tenant,
    )
    if not _check_site_read(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    items = ActionItem.objects.filter(project=hoshin.project)

    return JsonResponse({
        "action_items": [a.to_dict() for a in items],
        "count": items.count(),
    })


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def create_action_item(request, hoshin_id):
    """Create an action item for a hoshin project."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id, site__tenant=tenant,
    )
    if not _check_site_write(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    data = json.loads(request.body)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "Action item title is required"}, status=400)

    item = ActionItem.objects.create(
        project=hoshin.project,
        title=title,
        description=data.get("description", ""),
        owner_name=data.get("owner_name", ""),
        status=data.get("status", "not_started"),
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
        due_date=data.get("due_date"),
        progress=int(data.get("progress", 0)),
        sort_order=int(data.get("sort_order", 0)),
    )

    if data.get("depends_on_id"):
        dep = ActionItem.objects.filter(
            id=data["depends_on_id"], project=hoshin.project,
        ).first()
        if dep:
            item.depends_on = dep
            item.save()

    return JsonResponse({"success": True, "action_item": item.to_dict()}, status=201)


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["PUT", "PATCH"])
def update_action_item(request, action_id):
    """Update an action item."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    item = get_object_or_404(
        ActionItem.objects.select_related("project__hoshin__site"),
        id=action_id,
        project__hoshin__site__tenant=tenant,
    )

    if not _check_site_write(request.user, item.project.hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    data = json.loads(request.body)
    for field in ["title", "description", "owner_name", "status",
                  "start_date", "end_date", "due_date"]:
        if field in data:
            setattr(item, field, data[field])

    if "progress" in data:
        item.progress = max(0, min(100, int(data["progress"])))
    if "sort_order" in data:
        item.sort_order = int(data["sort_order"])
    if "depends_on_id" in data:
        if data["depends_on_id"]:
            item.depends_on = ActionItem.objects.filter(
                id=data["depends_on_id"], project=item.project,
            ).first()
        else:
            item.depends_on = None

    item.save()
    return JsonResponse({"success": True, "action_item": item.to_dict()})


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["DELETE"])
def delete_action_item(request, action_id):
    """Delete an action item."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    item = get_object_or_404(
        ActionItem.objects.select_related("project__hoshin__site"),
        id=action_id,
        project__hoshin__site__tenant=tenant,
    )

    if not _check_site_write(request.user, item.project.hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    item.delete()
    return JsonResponse({"success": True})


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def hoshin_dashboard(request):
    """Enterprise savings dashboard with rollup by site."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    fiscal_year = request.GET.get("fiscal_year", str(date.today().year))
    try:
        fiscal_year = int(fiscal_year)
    except ValueError:
        fiscal_year = date.today().year

    accessible_sites, _ = _get_accessible_sites(request.user, tenant)
    projects = HoshinProject.objects.filter(
        site__in=accessible_sites,
        fiscal_year=fiscal_year,
    ).exclude(hoshin_status="aborted").select_related("project", "site")

    # Totals
    total_target = float(sum(p.annual_savings_target for p in projects))
    total_ytd = float(sum(p.ytd_savings for p in projects))

    # By site
    site_data = {}
    for p in projects:
        site_name = p.site.name if p.site else "Unassigned"
        site_id = str(p.site_id) if p.site_id else "none"
        if site_id not in site_data:
            site_data[site_id] = {
                "site_id": site_id,
                "site_name": site_name,
                "target": 0, "ytd": 0, "project_count": 0,
                "active": 0, "completed": 0, "delayed": 0,
            }
        sd = site_data[site_id]
        sd["target"] += float(p.annual_savings_target)
        sd["ytd"] += float(p.ytd_savings)
        sd["project_count"] += 1
        if p.hoshin_status == "active":
            sd["active"] += 1
        elif p.hoshin_status == "completed":
            sd["completed"] += 1
        elif p.hoshin_status == "delayed":
            sd["delayed"] += 1

    # By type
    type_data = {}
    for p in projects:
        t = p.project_type
        if t not in type_data:
            type_data[t] = {"type": t, "target": 0, "ytd": 0, "count": 0}
        type_data[t]["target"] += float(p.annual_savings_target)
        type_data[t]["ytd"] += float(p.ytd_savings)
        type_data[t]["count"] += 1

    # Monthly trend (aggregate across all projects)
    monthly = {m: {"target": 0, "actual": 0} for m in range(1, 13)}
    for p in projects:
        monthly_target = float(p.annual_savings_target) / 12.0
        for m in range(1, 13):
            monthly[m]["target"] += monthly_target
        for entry in (p.monthly_actuals or []):
            mo = entry.get("month")
            if mo and 1 <= mo <= 12:
                monthly[mo]["actual"] += float(entry.get("savings", 0) or 0)

    monthly_trend = [
        {
            "month": m,
            "target": round(monthly[m]["target"], 2),
            "actual": round(monthly[m]["actual"], 2),
        }
        for m in range(1, 13)
    ]

    # Status breakdown
    status_counts = {}
    for p in projects:
        s = p.hoshin_status
        status_counts[s] = status_counts.get(s, 0) + 1

    return JsonResponse({
        "fiscal_year": fiscal_year,
        "total_target": round(total_target, 2),
        "total_ytd": round(total_ytd, 2),
        "variance": round(total_ytd - total_target, 2),
        "variance_pct": round((total_ytd / total_target * 100) if total_target else 0, 1),
        "project_count": projects.count(),
        "by_site": list(site_data.values()),
        "by_type": list(type_data.values()),
        "monthly_trend": monthly_trend,
        "status_counts": status_counts,
    })


# ---------------------------------------------------------------------------
# Site member management
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def list_site_members(request, site_id):
    """List users with access to a site."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    if not _check_site_read(request.user, site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    access_list = SiteAccess.objects.filter(
        site=site,
    ).select_related("user")

    return JsonResponse({
        "members": [a.to_dict() for a in access_list],
        "count": access_list.count(),
    })


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def grant_site_access(request, site_id):
    """Grant a user access to a site. Requires org admin or site admin."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    if not _is_site_admin(request.user, site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    data = json.loads(request.body)
    user_id = data.get("user_id")
    role = data.get("role", "member")
    if role not in ("viewer", "member", "admin"):
        return JsonResponse({"error": "Invalid role"}, status=400)

    if not user_id:
        return JsonResponse({"error": "user_id is required"}, status=400)

    # Verify user is a member of the same tenant
    if not Membership.objects.filter(
        user_id=user_id, tenant=tenant, is_active=True,
    ).exists():
        return JsonResponse({"error": "User is not a member of this organization"}, status=400)

    access, created = SiteAccess.objects.update_or_create(
        site=site, user_id=user_id,
        defaults={
            "role": role,
            "granted_by": request.user,
        },
    )

    return JsonResponse({
        "success": True,
        "access": access.to_dict(),
        "created": created,
    }, status=201 if created else 200)


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["DELETE"])
def revoke_site_access(request, site_id, access_id):
    """Revoke a user's access to a site. Requires org admin or site admin."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    if not _is_site_admin(request.user, site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    access = get_object_or_404(SiteAccess, id=access_id, site=site)
    access.delete()

    return JsonResponse({"success": True})


# ---------------------------------------------------------------------------
# Calendar view
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def hoshin_calendar_view(request):
    """Hoshin calendar: projects grouped by site with monthly target vs actual."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    fiscal_year = request.GET.get("fiscal_year", str(date.today().year))
    try:
        fiscal_year = int(fiscal_year)
    except ValueError:
        fiscal_year = date.today().year

    accessible_sites, _ = _get_accessible_sites(request.user, tenant)

    # Optional site filter
    site_id = request.GET.get("site_id")
    if site_id:
        accessible_sites = accessible_sites.filter(id=site_id)

    projects = HoshinProject.objects.filter(
        site__in=accessible_sites,
        fiscal_year=fiscal_year,
    ).exclude(
        hoshin_status="aborted",
    ).select_related("project", "site").order_by("site__name", "project__title")

    # Group by site
    sites_map = {}
    for p in projects:
        sid = str(p.site_id) if p.site_id else "none"
        if sid not in sites_map:
            sites_map[sid] = {
                "site_id": sid,
                "site_name": p.site.name if p.site else "Unassigned",
                "target": 0,
                "ytd": 0,
                "projects": [],
            }
        site_entry = sites_map[sid]
        target = float(p.annual_savings_target)
        site_entry["target"] += target
        site_entry["ytd"] += float(p.ytd_savings)

        monthly_target = target / 12.0
        actuals_by_month = {}
        for entry in (p.monthly_actuals or []):
            mo = entry.get("month")
            if mo and 1 <= mo <= 12:
                actuals_by_month[mo] = float(entry.get("savings", 0) or 0)

        months = []
        for m in range(1, 13):
            mt = round(monthly_target, 2)
            ma = round(actuals_by_month.get(m, 0), 2)
            pct = round(ma / mt * 100, 1) if mt else 0
            months.append({"month": m, "target": mt, "actual": ma, "pct": pct})

        site_entry["projects"].append({
            "id": str(p.id),
            "title": p.project.title,
            "status": p.hoshin_status,
            "type": p.project_type,
            "class": p.project_class,
            "target": target,
            "ytd": float(p.ytd_savings),
            "months": months,
        })

    # Build site-level monthly aggregates
    sites_list = []
    for site_entry in sites_map.values():
        site_months = []
        for m in range(1, 13):
            mt = sum(proj["months"][m - 1]["target"] for proj in site_entry["projects"])
            ma = sum(proj["months"][m - 1]["actual"] for proj in site_entry["projects"])
            pct = round(ma / mt * 100, 1) if mt else 0
            site_months.append({"month": m, "target": round(mt, 2), "actual": round(ma, 2), "pct": pct})
        site_entry["months"] = site_months
        site_entry["target"] = round(site_entry["target"], 2)
        site_entry["ytd"] = round(site_entry["ytd"], 2)
        sites_list.append(site_entry)

    return JsonResponse({
        "fiscal_year": fiscal_year,
        "sites": sites_list,
    })


# ---------------------------------------------------------------------------
# Calculation methods reference
# ---------------------------------------------------------------------------

@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def list_calculation_methods(request):
    """List available savings calculation methods."""
    methods = []
    for code, info in CALCULATION_METHODS.items():
        methods.append({
            "code": code,
            "name": info["name"],
            "category": info["category"],
            "description": info["description"],
            "formula": info["formula"],
            "variables": info["variables"],
        })
    return JsonResponse({"methods": methods})


@csrf_exempt
@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def test_formula(request):
    """Test a custom formula with sample variables.

    Supports {{fieldname}} syntax — brackets are stripped before evaluation.

    Request body: {"formula": "...", "variables": {"baseline": 10, ...}}
    Returns: {"result": float, "variables_used": {...}, "fields": [...]}
    """
    data = json.loads(request.body)
    formula = data.get("formula", "").strip()
    if not formula:
        return JsonResponse({"error": "Formula is required"}, status=400)

    # Extract {{field}} names so the frontend knows what inputs to show
    fields = extract_formula_fields(formula)

    variables = data.get("variables", {})
    # Ensure all provided variables are floats
    for k in list(variables.keys()):
        try:
            variables[k] = float(variables[k])
        except (ValueError, TypeError):
            variables[k] = 0

    # Add defaults for legacy built-in names if not already provided
    defaults = {"baseline": 0, "actual": 0, "volume": 1, "sales": 0, "rate": 1, "variance": 0}
    for k, v in defaults.items():
        if k not in variables:
            variables[k] = v

    # Auto-compute variance if not explicitly set
    if "variance" not in data.get("variables", {}):
        variables["variance"] = variables["baseline"] - variables["actual"]

    # Strip {{}} for evaluation
    eval_formula = normalize_formula(formula)

    try:
        result = evaluate_custom_formula(eval_formula, variables)
        return JsonResponse({
            "success": True,
            "result": round(float(result), 4),
            "variables_used": variables,
            "fields": fields,
        })
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return JsonResponse({"error": str(e), "fields": fields}, status=400)
