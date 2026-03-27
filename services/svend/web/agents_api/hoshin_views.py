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
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_feature
from core.models.project import Project
from core.models.tenant import Membership

from .hoshin_calculations import (
    CALCULATION_METHODS,
    aggregate_monthly_savings,
    calculate_savings,
    evaluate_custom_formula,
    extract_formula_fields,
    normalize_formula,
)
from .models import (
    ActionItem,
    Checklist,
    ChecklistExecution,
    Employee,
    HoshinProject,
    ProjectTemplate,
    ResourceCommitment,
    Site,
    SiteAccess,
    ValueStreamMap,
)
from .permissions import (
    check_site_read,
    check_site_write,
    get_accessible_sites,
    get_tenant,
    is_site_admin,
    require_tenant,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — delegated to agents_api.permissions per ORG-001 §11.3
# ---------------------------------------------------------------------------

# Legacy aliases for backward compatibility within this module
_get_tenant = get_tenant
_require_tenant = require_tenant
_get_accessible_sites = get_accessible_sites
_check_site_read = check_site_read
_check_site_write = check_site_write
_is_site_admin = is_site_admin


# ---------------------------------------------------------------------------
# Site CRUD
# ---------------------------------------------------------------------------


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

    return JsonResponse(
        {
            "sites": [s.to_dict() for s in sites],
            "count": sites.count(),
            "is_org_admin": is_admin,
        }
    )


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


@require_feature("hoshin_kanri")
@require_http_methods(["PUT", "PATCH"])
def update_site(request, site_id):
    """Update a site."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)

    if not _check_site_write(request.user, site, tenant):
        return JsonResponse({"error": "Write permission required"}, status=403)

    data = json.loads(request.body)

    for field in [
        "name",
        "code",
        "business_unit",
        "plant_manager",
        "ci_leader",
        "controller",
        "address",
    ]:
        if field in data:
            setattr(
                site,
                field,
                data[field].strip() if isinstance(data[field], str) else data[field],
            )
    if "is_active" in data:
        site.is_active = bool(data["is_active"])
    site.save()

    return JsonResponse({"success": True, "site": site.to_dict()})


@require_feature("hoshin_kanri")
@require_http_methods(["DELETE"])
def delete_site(request, site_id):
    """Delete a site (cascades to hoshin projects)."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)

    if not _is_site_admin(request.user, site, tenant):
        return JsonResponse({"error": "Site admin permission required to delete"}, status=403)

    project_count = site.hoshin_projects.count()
    site.delete()

    return JsonResponse({"success": True, "deleted_projects": project_count})


# ---------------------------------------------------------------------------
# HoshinProject CRUD
# ---------------------------------------------------------------------------


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

    return JsonResponse(
        {
            "projects": [p.to_dict() for p in projects],
            "count": projects.count(),
        }
    )


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
    if not site_id:
        return JsonResponse({"error": "site_id is required"}, status=400)
    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    if not _check_site_write(request.user, site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    try:
        savings_target = Decimal(str(data.get("annual_savings_target", 0)))
    except (InvalidOperation, ValueError):
        savings_target = Decimal("0")

    with transaction.atomic():
        # Create the core project (tenant-owned, not user-owned)
        core_project = Project.objects.create(
            tenant=tenant,
            title=title,
            status="ACTIVE",
            methodology=data.get("methodology", "PDCA"),
            current_phase="DEFINE",
            domain="manufacturing",
            goal_metric=data.get("goal_metric", ""),
            goal_baseline=data.get("goal_baseline", ""),
            goal_target=data.get("goal_target", ""),
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

    from .tool_events import tool_events

    tool_events.emit("hoshin.created", hoshin, user=request.user)

    return JsonResponse({"success": True, "project": hoshin.to_dict()}, status=201)


@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def get_hoshin_project(request, hoshin_id):
    """Get a single hoshin project with full details."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("project", "site"),
        id=hoshin_id,
        site__tenant=tenant,
    )

    if not _check_site_read(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    data = hoshin.to_dict()
    data["savings_summary"] = aggregate_monthly_savings(hoshin.monthly_actuals or [])
    data["action_items"] = [a.to_dict() for a in ActionItem.objects.filter(project=hoshin.project)]

    return JsonResponse({"project": data})


@require_feature("hoshin_kanri")
@require_http_methods(["PUT", "PATCH"])
def update_hoshin_project(request, hoshin_id):
    """Update a hoshin project."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id,
        site__tenant=tenant,
    )
    if not _check_site_write(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    data = json.loads(request.body)

    # Hoshin-specific fields
    for field in [
        "project_class",
        "project_type",
        "opportunity",
        "hoshin_status",
        "fiscal_year",
        "calculation_method",
        "custom_formula",
        "custom_formula_desc",
        "kaizen_charter",
        "monthly_actuals",
        "baseline_data",
        "source_burst_id",
    ]:
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
            # Prevent orphaning project from tenant scope (BUG-09)
            return JsonResponse(
                {"error": "Cannot remove site — project would be orphaned from tenant scope"},
                status=400,
            )

    hoshin.save()

    # Sync charter team_members → ResourceCommitments when charter changes
    if "kaizen_charter" in data:
        try:
            _sync_charter_commitments(hoshin, request.user)
        except Exception:
            logger.exception("Failed to sync charter commitments for %s", hoshin.id)

    # Also update core.Project fields if provided
    core_fields = [
        "title",
        "champion_name",
        "leader_name",
        "team_members",
        "methodology",
        "goal_metric",
    ]
    core_changed = False
    for field in core_fields:
        if field in data:
            setattr(hoshin.project, field, data[field])
            core_changed = True
    if core_changed:
        hoshin.project.save()

    from .tool_events import tool_events

    tool_events.emit("hoshin.updated", hoshin, user=request.user, data=data)

    return JsonResponse({"success": True, "project": hoshin.to_dict()})


# Role mapping: charter role strings → ResourceCommitment role keys
_CHARTER_ROLE_MAP = {
    "facilitator": "facilitator",
    "team_member": "team_member",
    "sponsor": "sponsor",
    "process_owner": "process_owner",
    "subject_expert": "subject_expert",
    # Legacy role names from old charter form
    "Member": "team_member",
    "Team Leader": "facilitator",
    "Facilitator": "facilitator",
    "SME": "subject_expert",
}


def _sync_charter_commitments(hoshin, requested_by):
    """Sync kaizen_charter.team_members with ResourceCommitments.

    For each team member with an employee_id, ensure a ResourceCommitment
    exists for this project. New members get a commitment + notification.
    Removed members get their commitment declined (if still requested).
    """
    from agents_api.commitment_notifications import notify_commitment_requested

    charter = hoshin.kaizen_charter or {}
    members = charter.get("team_members", [])
    tenant = hoshin.site.tenant if hoshin.site else None
    if not tenant:
        return

    # Get event dates for commitment range
    event_date = charter.get("event_date", "")
    end_date = charter.get("end_date", "") or event_date
    if not event_date:
        return

    try:
        start = date.fromisoformat(event_date)
        end = date.fromisoformat(end_date) if end_date else start + timedelta(days=5)
    except (ValueError, TypeError):
        return

    # Track which employees are in the charter
    charter_employee_ids = set()

    for member in members:
        emp_id = member.get("employee_id")
        if not emp_id:
            continue

        try:
            emp = Employee.objects.get(id=emp_id, tenant=tenant)
        except Employee.DoesNotExist:
            continue

        charter_employee_ids.add(str(emp.id))
        role = _CHARTER_ROLE_MAP.get(member.get("role", ""), "team_member")

        # Check if commitment already exists
        existing = (
            ResourceCommitment.objects.filter(
                employee=emp,
                project=hoshin,
            )
            .exclude(status="declined")
            .first()
        )

        if existing:
            # Update role if changed
            if existing.role != role:
                existing.role = role
                existing.save(update_fields=["role", "updated_at"])
            continue

        # Create new commitment
        commitment = ResourceCommitment.objects.create(
            employee=emp,
            project=hoshin,
            role=role,
            start_date=start,
            end_date=end,
            requested_by=requested_by,
        )
        notify_commitment_requested(commitment)

    # Decline commitments for employees removed from charter
    removed = ResourceCommitment.objects.filter(
        project=hoshin,
        status="requested",
    ).exclude(employee_id__in=charter_employee_ids)

    for commitment in removed:
        commitment.status = "declined"
        commitment.save(update_fields=["status", "updated_at"])


@require_feature("hoshin_kanri")
@require_http_methods(["DELETE"])
def delete_hoshin_project(request, hoshin_id):
    """Delete a hoshin project (also deletes core.Project)."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id,
        site__tenant=tenant,
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
        id=hoshin_id,
        site__tenant=tenant,
    )
    if not _check_site_write(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    data = json.loads(request.body)

    actuals = list(hoshin.monthly_actuals or [])

    # Dedup: remove any duplicate entries for the same month (data integrity guard)
    seen_months = set()
    deduped = []
    for m in actuals:
        mo = m.get("month")
        if mo not in seen_months:
            seen_months.add(mo)
            deduped.append(m)
    actuals = deduped

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
        entry["custom_vars"] = {k: float(v) if v is not None else None for k, v in data["custom_vars"].items()}

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
            hoshin.calculation_method,
            baseline,
            actual,
            volume or 1,
            cost or 1,
            **extra_kwargs,
        )
        entry["savings"] = result["savings"]
        entry["improvement_pct"] = result["improvement_pct"]
    elif "savings" in data:
        entry["savings"] = float(data["savings"])

    hoshin.monthly_actuals = sorted(actuals, key=lambda x: x.get("month", 0))
    hoshin.save()

    return JsonResponse(
        {
            "success": True,
            "month": month,
            "entry": entry,
            "ytd_savings": hoshin.ytd_savings,
            "savings_pct": hoshin.savings_pct,
        }
    )


# ---------------------------------------------------------------------------
# Batch create from VSM proposals
# ---------------------------------------------------------------------------


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
        vsm = (
            ValueStreamMap.objects.select_related("project")
            .filter(
                id=vsm_id,
                owner=request.user,
            )
            .first()
        )
        # Drop if VSM belongs to a different tenant (or has no project link)
        if vsm:
            vsm_tenant = getattr(vsm.project, "tenant_id", None) if vsm.project else None
            if vsm_tenant and vsm_tenant != tenant.id:
                vsm = None

    site_id = data.get("site_id")
    if not site_id:
        return JsonResponse({"error": "site_id is required"}, status=400)
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

    return JsonResponse(
        {
            "success": True,
            "created": created,
            "count": len(created),
        },
        status=201,
    )


# ---------------------------------------------------------------------------
# Action Items
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def list_action_items(request, hoshin_id):
    """List action items for a hoshin project."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id,
        site__tenant=tenant,
    )
    if not _check_site_read(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    items = ActionItem.objects.filter(project=hoshin.project)

    return JsonResponse(
        {
            "action_items": [a.to_dict() for a in items],
            "count": items.count(),
        }
    )


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def create_action_item(request, hoshin_id):
    """Create an action item for a hoshin project."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    hoshin = get_object_or_404(
        HoshinProject.objects.select_related("site"),
        id=hoshin_id,
        site__tenant=tenant,
    )
    if not _check_site_write(request.user, hoshin.site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)
    data = json.loads(request.body)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "Action item title is required"}, status=400)

    def _parse_date(val):
        if not val:
            return None
        if isinstance(val, str):
            return date.fromisoformat(val)
        return val

    item = ActionItem.objects.create(
        project=hoshin.project,
        title=title,
        description=data.get("description", ""),
        owner_name=data.get("owner_name", ""),
        status=data.get("status", "not_started"),
        start_date=_parse_date(data.get("start_date")),
        end_date=_parse_date(data.get("end_date")),
        due_date=_parse_date(data.get("due_date")),
        progress=int(data.get("progress", 0)),
        sort_order=int(data.get("sort_order", 0)),
        source_type="hoshin",
        source_id=hoshin.id,
    )

    if data.get("depends_on_id"):
        dep = ActionItem.objects.filter(
            id=data["depends_on_id"],
            project=hoshin.project,
        ).first()
        if dep:
            item.depends_on = dep
            item.save()

    return JsonResponse({"success": True, "action_item": item.to_dict()}, status=201)


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
    date_fields = {"start_date", "end_date", "due_date"}
    for field in [
        "title",
        "description",
        "owner_name",
        "status",
        "start_date",
        "end_date",
        "due_date",
    ]:
        if field in data:
            val = data[field]
            if field in date_fields and isinstance(val, str) and val:
                val = date.fromisoformat(val)
            setattr(item, field, val)

    if "progress" in data:
        item.progress = max(0, min(100, int(data["progress"])))
    if "sort_order" in data:
        item.sort_order = int(data["sort_order"])
    if "depends_on_id" in data:
        if data["depends_on_id"]:
            item.depends_on = ActionItem.objects.filter(
                id=data["depends_on_id"],
                project=item.project,
            ).first()
        else:
            item.depends_on = None

    item.save()
    return JsonResponse({"success": True, "action_item": item.to_dict()})


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
    projects = (
        HoshinProject.objects.filter(
            site__in=accessible_sites,
            fiscal_year=fiscal_year,
        )
        .exclude(hoshin_status="aborted")
        .select_related("project", "site")
    )

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
                "target": 0,
                "ytd": 0,
                "project_count": 0,
                "active": 0,
                "completed": 0,
                "delayed": 0,
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
        for entry in p.monthly_actuals or []:
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

    # Alignment metrics (X-matrix integration)
    from .models import AnnualObjective, XMatrixCorrelation

    annual_objs = AnnualObjective.objects.filter(tenant=tenant, fiscal_year=fiscal_year)
    linked_project_ids = set(
        str(uid)
        for uid in XMatrixCorrelation.objects.filter(
            tenant=tenant,
            fiscal_year=fiscal_year,
            pair_type="annual_project",
        )
        .values_list("col_id", flat=True)
        .distinct()
    )
    projects_linked = sum(1 for p in projects if str(p.id) in linked_project_ids)
    projects_unlinked = projects.count() - projects_linked

    return JsonResponse(
        {
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
            "alignment": {
                "projects_linked": projects_linked,
                "projects_unlinked": projects_unlinked,
                "annual_objectives_count": annual_objs.count(),
                "objectives_on_track": annual_objs.filter(status="on_track").count(),
                "objectives_at_risk": annual_objs.filter(status__in=["at_risk", "behind"]).count(),
            },
        }
    )


# =============================================================================
# Intelligence Layer — Phase 3: Strategy Alignment Analysis
# =============================================================================


@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def alignment_analysis(request, site_id):
    """Identify gaps in the strategy-to-execution chain.

    Finds unlinked objectives, projects, and KPIs via JOIN queries.
    No LLM required — pure database analysis.
    """
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    if not _check_site_read(request.user, site, tenant):
        return JsonResponse({"error": "Not found"}, status=404)

    fiscal_year = int(request.GET.get("fiscal_year", date.today().year))

    from .models import (
        AnnualObjective,
        HoshinKPI,
        HoshinProject,
        StrategicObjective,
        XMatrixCorrelation,
    )

    # All entities for this tenant and fiscal year
    projects = HoshinProject.objects.filter(site=site)
    annual_objs = AnnualObjective.objects.filter(tenant=tenant, fiscal_year=fiscal_year)
    strategic_objs = StrategicObjective.objects.filter(
        tenant=tenant,
        start_year__lte=fiscal_year,
        end_year__gte=fiscal_year,
    )
    kpis = HoshinKPI.objects.filter(tenant=tenant, fiscal_year=fiscal_year)

    # Correlations tell us what's linked
    correlations = XMatrixCorrelation.objects.filter(tenant=tenant, fiscal_year=fiscal_year)
    annual_project_links = set()  # annual objective IDs linked to projects
    project_linked_ids = set()  # project IDs linked to annual objectives
    strategic_annual_links = set()  # strategic objective IDs linked to annual
    project_kpi_links = set()  # project IDs linked to KPIs

    for c in correlations:
        if c.pair_type == "annual_project":
            annual_project_links.add(str(c.row_id))
            project_linked_ids.add(str(c.col_id))
        elif c.pair_type == "strategic_annual":
            strategic_annual_links.add(str(c.row_id))

    # KPI-project links via derived_from FK
    for kpi in kpis:
        if kpi.derived_from_id:
            project_kpi_links.add(str(kpi.derived_from_id))

    # Find gaps
    objectives_without_projects = [
        {"id": str(o.id), "title": o.title} for o in annual_objs if str(o.id) not in annual_project_links
    ]

    projects_without_objectives = [
        {"id": str(p.id), "title": p.project.title}
        for p in projects.select_related("project")
        if str(p.id) not in project_linked_ids
    ]

    projects_without_kpis = [
        {"id": str(p.id), "title": p.project.title}
        for p in projects.select_related("project")
        if p.hoshin_status in ("active", "budgeted") and str(p.id) not in project_kpi_links
    ]

    strategic_without_annual = [
        {"id": str(s.id), "title": s.title} for s in strategic_objs if str(s.id) not in strategic_annual_links
    ]

    # Compute alignment score
    total_entities = annual_objs.count() + projects.count() + strategic_objs.count()
    linked = len(annual_project_links) + len(project_linked_ids) + len(strategic_annual_links) + len(project_kpi_links)
    alignment_score = round(linked / max(total_entities, 1), 2)

    # Build recommendations
    recommendations = []
    if objectives_without_projects:
        recommendations.append(
            f"{len(objectives_without_projects)} annual objective(s) have no linked improvement projects"
        )
    if projects_without_objectives:
        recommendations.append(f"{len(projects_without_objectives)} project(s) are not linked to any annual objective")
    if projects_without_kpis:
        recommendations.append(
            f"{len(projects_without_kpis)} active project(s) have no KPI measurement — consider adding KPIs"
        )
    if strategic_without_annual:
        recommendations.append(f"{len(strategic_without_annual)} strategic objective(s) have no annual breakdowns")

    return JsonResponse(
        {
            "alignment_score": alignment_score,
            "gaps": {
                "objectives_without_projects": objectives_without_projects,
                "projects_without_objectives": projects_without_objectives,
                "projects_without_kpis": projects_without_kpis,
                "strategic_without_annual": strategic_without_annual,
            },
            "recommendations": recommendations,
            "fiscal_year": fiscal_year,
        }
    )


# ---------------------------------------------------------------------------
# Site member management
# ---------------------------------------------------------------------------


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

    return JsonResponse(
        {
            "members": [a.to_dict() for a in access_list],
            "count": access_list.count(),
        }
    )


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
        user_id=user_id,
        tenant=tenant,
        is_active=True,
    ).exists():
        return JsonResponse({"error": "User is not a member of this organization"}, status=400)

    access, created = SiteAccess.objects.update_or_create(
        site=site,
        user_id=user_id,
        defaults={
            "role": role,
            "granted_by": request.user,
        },
    )

    return JsonResponse(
        {
            "success": True,
            "access": access.to_dict(),
            "created": created,
        },
        status=201 if created else 200,
    )


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

    projects = (
        HoshinProject.objects.filter(
            site__in=accessible_sites,
            fiscal_year=fiscal_year,
        )
        .exclude(
            hoshin_status="aborted",
        )
        .select_related("project", "site")
        .order_by("site__name", "project__title")
    )

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
        for entry in p.monthly_actuals or []:
            mo = entry.get("month")
            if mo and 1 <= mo <= 12:
                actuals_by_month[mo] = float(entry.get("savings", 0) or 0)

        months = []
        for m in range(1, 13):
            mt = round(monthly_target, 2)
            ma = round(actuals_by_month.get(m, 0), 2)
            pct = round(ma / mt * 100, 1) if mt else 0
            months.append({"month": m, "target": mt, "actual": ma, "pct": pct})

        site_entry["projects"].append(
            {
                "id": str(p.id),
                "title": p.project.title,
                "status": p.hoshin_status,
                "type": p.project_type,
                "class": p.project_class,
                "target": target,
                "ytd": float(p.ytd_savings),
                "months": months,
            }
        )

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

    return JsonResponse(
        {
            "fiscal_year": fiscal_year,
            "sites": sites_list,
        }
    )


# ---------------------------------------------------------------------------
# Calculation methods reference
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def list_calculation_methods(request):
    """List available savings calculation methods."""
    methods = []
    for code, info in CALCULATION_METHODS.items():
        methods.append(
            {
                "code": code,
                "name": info["name"],
                "category": info["category"],
                "description": info["description"],
                "formula": info["formula"],
                "variables": info["variables"],
            }
        )
    return JsonResponse({"methods": methods})


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

    user_variables = data.get("variables", {})
    variables = dict(user_variables)
    # Ensure all provided variables are floats
    for k in list(variables.keys()):
        try:
            variables[k] = float(variables[k])
        except (ValueError, TypeError):
            variables[k] = 0

    # Add defaults for legacy built-in names if not already provided
    defaults = {
        "baseline": 0,
        "actual": 0,
        "volume": 1,
        "sales": 0,
        "rate": 1,
        "variance": 0,
    }
    for k, v in defaults.items():
        if k not in variables:
            variables[k] = v

    # Auto-compute variance if not explicitly set
    if "variance" not in user_variables:
        variables["variance"] = variables["baseline"] - variables["actual"]

    # Strip {{}} for evaluation
    eval_formula = normalize_formula(formula)

    try:
        result = evaluate_custom_formula(eval_formula, variables)
        return JsonResponse(
            {
                "success": True,
                "result": round(float(result), 4),
                "variables_used": variables,
                "fields": fields,
            }
        )
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return JsonResponse({"error": str(e), "fields": fields}, status=400)


# ---------------------------------------------------------------------------
# Employee CRUD (QMS-002 §2.1)
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "POST"])
def list_create_employees(request):
    """GET: list employees. POST: create employee."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if request.method == "GET":
        qs = Employee.objects.filter(tenant=tenant)
        site_id = request.GET.get("site")
        if site_id:
            qs = qs.filter(site_id=site_id)
        dept = request.GET.get("department")
        if dept:
            qs = qs.filter(department=dept)
        active = request.GET.get("is_active")
        if active is not None:
            qs = qs.filter(is_active=active.lower() in ("true", "1"))
        return JsonResponse([e.to_dict() for e in qs.order_by("name")], safe=False)

    # POST
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    if not name or not email:
        return JsonResponse({"error": "name and email are required"}, status=400)

    if Employee.objects.filter(tenant=tenant, email=email).exists():
        return JsonResponse({"error": "Employee with this email already exists"}, status=409)

    emp = Employee.objects.create(
        tenant=tenant,
        name=name,
        email=email,
        role=data.get("role", ""),
        department=data.get("department", ""),
        site_id=data.get("site_id"),
        user_link_id=data.get("user_link_id"),
    )
    return JsonResponse(emp.to_dict(), status=201)


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "PUT", "DELETE"])
def employee_detail(request, emp_id):
    """GET/PUT/DELETE a single employee."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    emp = get_object_or_404(Employee, pk=emp_id, tenant=tenant)

    if request.method == "GET":
        return JsonResponse(emp.to_dict())

    if request.method == "DELETE":
        emp.is_active = False
        emp.save(update_fields=["is_active"])
        return JsonResponse({"ok": True})

    # PUT
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    for field in ("name", "email", "role", "department"):
        if field in data:
            setattr(emp, field, data[field])
    if "site_id" in data:
        emp.site_id = data["site_id"]
    if "user_link_id" in data:
        emp.user_link_id = data["user_link_id"]
    if "is_active" in data:
        emp.is_active = data["is_active"]
    emp.save()
    return JsonResponse(emp.to_dict())


@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def employee_availability(request, emp_id):
    """Check availability for an employee over a date range."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    emp = get_object_or_404(Employee, pk=emp_id, tenant=tenant)
    start = request.GET.get("start")
    end = request.GET.get("end")
    if not start or not end:
        return JsonResponse({"error": "start and end query params required"}, status=400)

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError:
        return JsonResponse({"error": "Invalid date format (use YYYY-MM-DD)"}, status=400)

    conflicts = ResourceCommitment.check_availability(emp, start_date, end_date)
    return JsonResponse(
        {
            "employee_id": str(emp.id),
            "available": not conflicts.exists(),
            "conflicts": [c.to_dict() for c in conflicts],
        }
    )


# ---------------------------------------------------------------------------
# ResourceCommitment CRUD (QMS-002 §2.2)
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "POST"])
def list_create_commitments(request):
    """GET: list commitments. POST: create commitment."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if request.method == "GET":
        qs = ResourceCommitment.objects.filter(
            employee__tenant=tenant,
        ).select_related("employee", "project")
        project_id = request.GET.get("project")
        if project_id:
            qs = qs.filter(project_id=project_id)
        employee_id = request.GET.get("employee")
        if employee_id:
            qs = qs.filter(employee_id=employee_id)
        return JsonResponse([c.to_dict() for c in qs.order_by("-created_at")], safe=False)

    # POST
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    employee_id = data.get("employee_id")
    project_id = data.get("project_id")
    if not employee_id or not project_id:
        return JsonResponse({"error": "employee_id and project_id required"}, status=400)

    emp = get_object_or_404(Employee, pk=employee_id, tenant=tenant)
    project = get_object_or_404(HoshinProject, pk=project_id, site__tenant=tenant)

    try:
        start_date = date.fromisoformat(data["start_date"])
        end_date = date.fromisoformat(data["end_date"])
    except (KeyError, ValueError):
        return JsonResponse({"error": "start_date and end_date required (YYYY-MM-DD)"}, status=400)

    # Check availability
    conflicts = ResourceCommitment.check_availability(emp, start_date, end_date)
    conflict_data = [c.to_dict() for c in conflicts]

    commitment = ResourceCommitment.objects.create(
        employee=emp,
        project=project,
        role=data.get("role", "team_member"),
        start_date=start_date,
        end_date=end_date,
        hours_per_day=data.get("hours_per_day", 8),
        requested_by=request.user,
    )
    # Notify the assigned employee (NTF-001 / QMS-002)
    try:
        from agents_api.commitment_notifications import notify_commitment_requested

        notify_commitment_requested(commitment)
    except Exception:
        logger.exception("Failed to send commitment notification for %s", commitment.id)

    result = commitment.to_dict()
    if conflict_data:
        result["conflicts"] = conflict_data
    return JsonResponse(result, status=201)


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "PUT", "DELETE"])
def commitment_detail(request, commitment_id):
    """GET/PUT/DELETE a single commitment."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    commitment = get_object_or_404(
        ResourceCommitment,
        pk=commitment_id,
        employee__tenant=tenant,
    )

    if request.method == "GET":
        return JsonResponse(commitment.to_dict())

    if request.method == "DELETE":
        commitment.delete()
        return JsonResponse({"ok": True})

    # PUT — update status (with lifecycle enforcement) and/or fields
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    old_status = commitment.status

    if "status" in data:
        new_status = data["status"]
        valid = ResourceCommitment.VALID_TRANSITIONS.get(commitment.status, set())
        if new_status not in valid:
            return JsonResponse(
                {
                    "error": f"Cannot transition from '{commitment.status}' to '{new_status}'",
                    "valid_transitions": sorted(valid),
                },
                status=400,
            )
        commitment.status = new_status

    for field in ("role", "hours_per_day"):
        if field in data:
            setattr(commitment, field, data[field])
    if "start_date" in data:
        commitment.start_date = date.fromisoformat(data["start_date"])
    if "end_date" in data:
        commitment.end_date = date.fromisoformat(data["end_date"])

    commitment.save()

    # Notify requester on confirm/decline (NTF-001 / QMS-002)
    if commitment.status in ("confirmed", "declined") and old_status != commitment.status:
        try:
            from agents_api.commitment_notifications import notify_commitment_response

            notify_commitment_response(commitment, old_status)
        except Exception:
            logger.exception("Failed to send commitment response notification")

    return JsonResponse(commitment.to_dict())


# ---------------------------------------------------------------------------
# Bulk Employee Import (QMS-002 §3.1)
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def employees_import(request):
    """Bulk import employees from CSV.

    Accepts multipart file upload with columns: name, email (required),
    role, department, site_code (optional). Deduplicates by (tenant, email).
    """
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    uploaded = request.FILES["file"]
    if not uploaded.name.endswith(".csv"):
        return JsonResponse({"error": "Only CSV files are supported"}, status=400)

    import io

    import pandas as pd

    try:
        raw = uploaded.read()
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    except Exception as e:
        return JsonResponse({"error": f"Failed to parse CSV: {e}"}, status=400)

    # Validate required columns
    df.columns = [c.strip().lower() for c in df.columns]
    if "name" not in df.columns or "email" not in df.columns:
        return JsonResponse(
            {
                "error": "CSV must have 'name' and 'email' columns",
                "found_columns": list(df.columns),
            },
            status=400,
        )

    # Build site_code → site_id lookup
    site_map = {}
    if "site_code" in df.columns:
        sites = Site.objects.filter(tenant=tenant)
        site_map = {s.code.lower(): s.id for s in sites if s.code}

    created = 0
    updated = 0
    errors = []

    for idx, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        email = str(row.get("email", "")).strip()
        if not name or not email:
            errors.append({"row": idx + 2, "message": "Missing name or email"})
            continue

        defaults = {
            "name": name,
            "role": str(row.get("role", "")).strip(),
            "department": str(row.get("department", "")).strip(),
        }

        if "site_code" in df.columns:
            code = str(row.get("site_code", "")).strip().lower()
            if code and code in site_map:
                defaults["site_id"] = site_map[code]

        emp, was_created = Employee.objects.update_or_create(
            tenant=tenant,
            email=email,
            defaults=defaults,
        )
        if was_created:
            created += 1
        else:
            updated += 1

    return JsonResponse(
        {
            "created": created,
            "updated": updated,
            "errors": errors,
            "total_rows": len(df),
        }
    )


# ---------------------------------------------------------------------------
# Employee Timeline (QMS-002 §3.2)
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def employee_timeline(request, emp_id):
    """Return commitment timeline + capacity for an employee."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    emp = get_object_or_404(Employee, pk=emp_id, tenant=tenant)

    # Default to current fiscal year
    year = date.today().year
    start_str = request.GET.get("start", f"{year}-01-01")
    end_str = request.GET.get("end", f"{year}-12-31")
    try:
        start_date = date.fromisoformat(start_str)
        end_date = date.fromisoformat(end_str)
    except ValueError:
        return JsonResponse({"error": "Invalid date format (YYYY-MM-DD)"}, status=400)

    commitments = (
        ResourceCommitment.objects.filter(
            employee=emp,
            start_date__lt=end_date,
            end_date__gt=start_date,
        )
        .exclude(
            status__in=("completed", "declined"),
        )
        .select_related("project__project")
        .order_by("start_date")
    )

    # Calculate total committed hours across the date range
    total_hours = 0.0
    for c in commitments:
        overlap_start = max(c.start_date, start_date)
        overlap_end = min(c.end_date, end_date)
        days = (overlap_end - overlap_start).days
        if days > 0:
            total_hours += days * float(c.hours_per_day)

    return JsonResponse(
        {
            "employee": emp.to_dict(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "commitments": [c.to_dict() for c in commitments],
            "total_committed_hours": round(total_hours, 1),
        }
    )


# ---------------------------------------------------------------------------
# Facilitator Calendar (QMS-002 §3.4)
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def hoshin_calendar_facilitators(request):
    """Facilitator workload view with over-commitment detection."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    fiscal_year = request.GET.get("fiscal_year", str(date.today().year))
    try:
        fiscal_year = int(fiscal_year)
    except ValueError:
        fiscal_year = date.today().year

    fy_start = date(fiscal_year, 1, 1)
    fy_end = date(fiscal_year, 12, 31)

    # All facilitator commitments for this tenant in the fiscal year
    commitments = (
        ResourceCommitment.objects.filter(
            employee__tenant=tenant,
            role="facilitator",
            start_date__lt=fy_end,
            end_date__gt=fy_start,
        )
        .exclude(
            status__in=("completed", "declined"),
        )
        .select_related("employee", "project__project")
    )

    # Group by employee
    from collections import defaultdict

    emp_map = defaultdict(list)
    for c in commitments:
        emp_map[c.employee_id].append(c)

    facilitators = []
    for emp_id, comms in emp_map.items():
        emp = comms[0].employee

        # Detect over-commitment: find dates where total hours > 8
        over_committed_dates = []
        # Determine scan range (intersection of all commitment ranges with FY)
        all_starts = [max(c.start_date, fy_start) for c in comms]
        all_ends = [min(c.end_date, fy_end) for c in comms]
        scan_start = min(all_starts)
        scan_end = max(all_ends)

        from datetime import timedelta as _td

        current = scan_start
        while current < scan_end:
            day_hours = sum(float(c.hours_per_day) for c in comms if c.start_date <= current < c.end_date)
            if day_hours > 8:
                over_committed_dates.append(current.isoformat())
            current += _td(days=1)

        facilitators.append(
            {
                "employee": emp.to_dict(),
                "commitments": [c.to_dict() for c in comms],
                "over_committed": len(over_committed_dates) > 0,
                "over_committed_days": len(over_committed_dates),
                "over_committed_dates": over_committed_dates[:30],  # Cap for response size
            }
        )

    return JsonResponse(
        {
            "fiscal_year": fiscal_year,
            "facilitators": facilitators,
        }
    )


# =========================================================================
# Project Templates
# =========================================================================


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "POST"])
def template_list_create(request):
    """List or create project templates."""
    user = request.user

    if request.method == "GET":
        templates = ProjectTemplate.objects.filter(owner=user)
        return JsonResponse([t.to_dict() for t in templates], safe=False)

    data = json.loads(request.body)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "Name is required"}, status=400)

    tpl = ProjectTemplate(
        owner=user,
        name=name,
        description=data.get("description", ""),
        project_class=data.get("project_class", "project"),
        project_type=data.get("project_type", "material"),
        opportunity=data.get("opportunity", "budgeted_new"),
        calculation_method=data.get("calculation_method", ""),
        checklist_ids=data.get("checklist_ids", []),
        default_actions=data.get("default_actions", []),
    )
    if data.get("site_id"):
        try:
            tpl.site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            pass
    tpl.save()
    return JsonResponse(tpl.to_dict(), status=201)


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "PUT", "DELETE"])
def template_detail(request, tpl_id):
    """Get, update, or delete a project template."""
    try:
        tpl = ProjectTemplate.objects.get(id=tpl_id, owner=request.user)
    except ProjectTemplate.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(tpl.to_dict())

    if request.method == "DELETE":
        tpl.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in [
        "name",
        "description",
        "project_class",
        "project_type",
        "opportunity",
        "calculation_method",
        "checklist_ids",
        "default_actions",
    ]:
        if field in data:
            setattr(tpl, field, data[field])
    tpl.save()
    return JsonResponse(tpl.to_dict())


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def create_from_template(request):
    """Create a new Hoshin project from a template.

    POST body: {"template_id": "uuid", "title": "Project Title", "site_id": "uuid", ...overrides}

    Creates core.Project + HoshinProject with template defaults,
    pre-populates ActionItems, and creates ChecklistExecution stubs
    for each attached checklist.
    """
    data = json.loads(request.body)
    tpl_id = data.get("template_id")
    if not tpl_id:
        return JsonResponse({"error": "template_id required"}, status=400)

    try:
        tpl = ProjectTemplate.objects.get(id=tpl_id, owner=request.user)
    except ProjectTemplate.DoesNotExist:
        return JsonResponse({"error": "Template not found"}, status=404)

    title = data.get("title", "").strip() or f"{tpl.name} — New Project"

    # Create core.Project
    project = Project.objects.create(
        title=title,
        user=request.user,
    )

    # Create HoshinProject with template defaults
    site = None
    if data.get("site_id"):
        try:
            site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            pass
    elif tpl.site:
        site = tpl.site

    hoshin = HoshinProject.objects.create(
        project=project,
        site=site,
        project_class=data.get("project_class", tpl.project_class),
        project_type=data.get("project_type", tpl.project_type),
        opportunity=data.get("opportunity", tpl.opportunity),
        calculation_method=data.get("calculation_method", tpl.calculation_method),
        annual_savings_target=data.get("annual_savings_target", 0),
    )

    # Pre-populate ActionItems from template
    for i, action_def in enumerate(tpl.default_actions or []):
        ActionItem.objects.create(
            project=project,
            title=action_def.get("title", f"Action {i + 1}"),
            description=action_def.get("description", ""),
            sort_order=action_def.get("sort_order", i),
            source_type="template",
            source_id=tpl.id,
        )

    # Create ChecklistExecution stubs for each attached checklist
    checklists_attached = []
    for cl_id in tpl.checklist_ids or []:
        try:
            cl = Checklist.objects.get(id=cl_id)
            ChecklistExecution.objects.create(
                checklist=cl,
                executor=request.user,
                entity_type="project",
                entity_id=project.id,
                status="not_started",
            )
            checklists_attached.append(cl.name)
        except Checklist.DoesNotExist:
            pass

    return JsonResponse(
        {
            "success": True,
            "project_id": str(project.id),
            "hoshin_id": str(hoshin.id),
            "template_name": tpl.name,
            "actions_created": len(tpl.default_actions or []),
            "checklists_attached": checklists_attached,
        },
        status=201,
    )
