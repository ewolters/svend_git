"""X-Matrix strategy deployment API views.

Provides CRUD for the four X-matrix quadrants:
- Strategic Objectives (south) — 3-5 year breakthroughs
- Annual Objectives (west) — this FY's targets
- HoshinProjects (north) — improvement initiatives (existing model)
- HoshinKPIs (east) — measurable indicators

Plus correlation management, auto-suggestion engine, VSM lifecycle
promotion, and fiscal year rollover.

Enterprise-only — all views gated by @require_feature("hoshin_kanri").
"""

import json
import logging
from datetime import date
from decimal import Decimal, InvalidOperation

from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_feature

from .models import (
    AnnualObjective,
    HoshinKPI,
    HoshinProject,
    Site,
    StrategicObjective,
    ValueStreamMap,
    XMatrixCorrelation,
)
from .permissions import get_accessible_sites as _get_accessible_sites
from .permissions import require_tenant as _require_tenant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# X-Matrix data endpoint (the core)
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET"])
def get_xmatrix_data(request):
    """Return all four quadrants, correlations, and dollar rollup for a FY."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    fiscal_year = request.GET.get("fiscal_year", str(date.today().year))
    try:
        fiscal_year = int(fiscal_year)
    except ValueError:
        fiscal_year = date.today().year

    accessible_sites, _ = _get_accessible_sites(request.user, tenant)

    # Run auto-suggestion before returning
    _auto_suggest_correlations(tenant, fiscal_year, accessible_sites)

    # Strategic objectives active during this FY
    strategic = StrategicObjective.objects.filter(
        tenant=tenant,
        start_year__lte=fiscal_year,
        end_year__gte=fiscal_year,
    ).exclude(status="deferred")

    # Annual objectives for this FY
    annual = AnnualObjective.objects.filter(
        tenant=tenant,
        fiscal_year=fiscal_year,
    ).select_related("site", "strategic_objective")

    # Hoshin projects for this FY
    projects = (
        HoshinProject.objects.filter(
            site__in=accessible_sites,
            fiscal_year=fiscal_year,
        )
        .exclude(hoshin_status="aborted")
        .select_related("project", "site")
    )

    # KPIs for this FY
    kpis = HoshinKPI.objects.filter(
        tenant=tenant,
        fiscal_year=fiscal_year,
    ).select_related("derived_from")

    # Correlations for this FY
    correlations = XMatrixCorrelation.objects.filter(
        tenant=tenant,
        fiscal_year=fiscal_year,
    )

    # Build valid ID sets for filtering out orphaned correlations
    strategic_ids = set(str(s.id) for s in strategic)
    annual_ids = set(str(a.id) for a in annual)
    project_ids = set(str(p.id) for p in projects)
    kpi_ids = set(str(k.id) for k in kpis)

    valid_ids_by_pair = {
        "strategic_annual": (strategic_ids, annual_ids),
        "annual_project": (annual_ids, project_ids),
        "project_kpi": (project_ids, kpi_ids),
        "kpi_strategic": (kpi_ids, strategic_ids),
    }

    # Group correlations by pair_type, filtering orphans
    corr_grouped = {
        "strategic_annual": [],
        "annual_project": [],
        "project_kpi": [],
        "kpi_strategic": [],
    }
    for c in correlations:
        valid = valid_ids_by_pair.get(c.pair_type)
        if valid:
            row_set, col_set = valid
            if str(c.row_id) in row_set and str(c.col_id) in col_set:
                corr_grouped[c.pair_type].append(c.to_dict())

    # Dollar rollup: strategic → annual → projects
    project_map = {str(p.id): p for p in projects}
    annual_project_links = [c for c in correlations if c.pair_type == "annual_project"]

    rollup_by_strategic = []
    for so in strategic:
        so_meta = HoshinKPI.METRIC_CATALOG.get(so.target_metric, {})
        so_agg = so_meta.get("aggregation", "sum")
        so_unit = so_meta.get("unit", so.target_unit or "$")

        linked_annual = [
            a for a in annual if str(a.strategic_objective_id) == str(so.id)
        ]
        linked_project_ids = set()
        for ao in linked_annual:
            for link in annual_project_links:
                if str(link.row_id) == str(ao.id) and str(link.col_id) in project_map:
                    linked_project_ids.add(str(link.col_id))

        if so_agg == "weighted_avg":
            # Volume-weighted average of raw metric across linked projects
            total_w, total_v = 0.0, 0.0
            for pid in linked_project_ids:
                proj = project_map.get(pid)
                if not proj:
                    continue
                for entry in proj.monthly_actuals or []:
                    actual = entry.get("actual")
                    volume = entry.get("volume")
                    if actual is not None and volume:
                        total_w += float(actual) * float(volume)
                        total_v += float(volume)
            linked_actual = round(total_w / total_v, 4) if total_v > 0 else None
            linked_target_val = float(so.target_value) if so.target_value else None
            rollup_by_strategic.append(
                {
                    "id": str(so.id),
                    "title": so.title,
                    "unit": so_unit,
                    "aggregation": so_agg,
                    "direction": so_meta.get("direction", "up"),
                    "linked_target": linked_target_val,
                    "linked_ytd": linked_actual,
                    "total_volume": round(total_v, 2) if total_v > 0 else None,
                }
            )
        else:
            # Default: sum dollar savings
            linked_target = sum(
                float(project_map[pid].annual_savings_target)
                for pid in linked_project_ids
                if pid in project_map
            )
            linked_ytd = sum(
                float(project_map[pid].ytd_savings)
                for pid in linked_project_ids
                if pid in project_map
            )
            rollup_by_strategic.append(
                {
                    "id": str(so.id),
                    "title": so.title,
                    "unit": so_unit,
                    "aggregation": so_agg,
                    "direction": so_meta.get("direction", "up"),
                    "linked_target": round(linked_target, 2),
                    "linked_ytd": round(linked_ytd, 2),
                }
            )

    total_target = float(sum(p.annual_savings_target for p in projects))
    total_ytd = float(sum(p.ytd_savings for p in projects))

    # KPI rollup — aggregate across correlated projects per KPI
    {str(k.id): k for k in kpis}
    project_kpi_links = [c for c in correlations if c.pair_type == "project_kpi"]
    kpi_rollup = []
    for kpi in kpis:
        kpi_id = str(kpi.id)
        correlated_pids = set()
        for link in project_kpi_links:
            if str(link.col_id) == kpi_id and str(link.row_id) in project_map:
                correlated_pids.add(str(link.row_id))

        agg = getattr(kpi, "aggregation", "sum")
        rollup_entry = {
            "id": kpi_id,
            "name": kpi.name,
            "unit": kpi.unit,
            "aggregation": agg,
            "direction": kpi.direction,
            "target_value": float(kpi.target_value) if kpi.target_value else None,
            "project_count": len(correlated_pids),
        }

        if agg == "sum":
            # Sum dollar savings across correlated projects
            agg_value = sum(
                float(project_map[pid].ytd_savings)
                for pid in correlated_pids
                if pid in project_map
            )
            rollup_entry["aggregated_value"] = round(agg_value, 2)

        elif agg == "weighted_avg":
            # Volume-weighted average of raw metric across correlated projects
            total_weighted = 0.0
            total_volume = 0.0
            for pid in correlated_pids:
                proj = project_map.get(pid)
                if not proj:
                    continue
                for entry in proj.monthly_actuals or []:
                    actual = entry.get("actual")
                    volume = entry.get("volume")
                    if actual is not None and volume:
                        total_weighted += float(actual) * float(volume)
                        total_volume += float(volume)
            if total_volume > 0:
                rollup_entry["aggregated_value"] = round(
                    total_weighted / total_volume, 4
                )
                rollup_entry["total_volume"] = round(total_volume, 2)
            else:
                rollup_entry["aggregated_value"] = None

        elif agg == "latest":
            # Not aggregatable — use the KPI's own effective_actual
            rollup_entry["aggregated_value"] = kpi.effective_actual

        else:
            # manual
            rollup_entry["aggregated_value"] = kpi.effective_actual

        kpi_rollup.append(rollup_entry)

    # Expose metric catalog to frontend for dropdown rendering
    metric_catalog = {
        k: {ck: cv for ck, cv in v.items()} for k, v in HoshinKPI.METRIC_CATALOG.items()
    }

    return JsonResponse(
        {
            "fiscal_year": fiscal_year,
            "strategic_objectives": [s.to_dict() for s in strategic],
            "annual_objectives": [a.to_dict() for a in annual],
            "projects": [p.to_dict() for p in projects],
            "kpis": [k.to_dict() for k in kpis],
            "correlations": corr_grouped,
            "metric_catalog": metric_catalog,
            "rollup": {
                "total_target": round(total_target, 2),
                "total_ytd": round(total_ytd, 2),
                "by_strategic_objective": rollup_by_strategic,
                "by_kpi": kpi_rollup,
            },
        }
    )


# ---------------------------------------------------------------------------
# Auto-suggestion engine
# ---------------------------------------------------------------------------


def _auto_suggest_correlations(tenant, fiscal_year, accessible_sites):
    """Pre-compute suggested correlations from data lineage.

    Only creates if no existing correlation (manual or auto) for that pair.
    Never overwrites human decisions.
    """
    annual_objs = list(
        AnnualObjective.objects.filter(
            tenant=tenant,
            fiscal_year=fiscal_year,
        ).select_related("site", "strategic_objective")
    )

    projects = list(
        HoshinProject.objects.filter(
            site__in=accessible_sites,
            fiscal_year=fiscal_year,
        )
        .exclude(hoshin_status="aborted")
        .select_related("site")
    )

    kpis = list(
        HoshinKPI.objects.filter(
            tenant=tenant,
            fiscal_year=fiscal_year,
        )
    )

    strategic_objs = list(
        StrategicObjective.objects.filter(
            tenant=tenant,
            start_year__lte=fiscal_year,
            end_year__gte=fiscal_year,
        ).exclude(status="deferred")
    )

    suggestions = []

    # strategic_annual: AnnualObjective has FK to StrategicObjective
    for ao in annual_objs:
        if ao.strategic_objective_id:
            suggestions.append(
                (
                    "strategic_annual",
                    ao.strategic_objective_id,
                    ao.id,
                    "strong",
                )
            )

    # annual_project: same site + same fiscal year
    for ao in annual_objs:
        for p in projects:
            if ao.site_id and p.site_id and str(ao.site_id) == str(p.site_id):
                suggestions.append(
                    (
                        "annual_project",
                        ao.id,
                        p.id,
                        "moderate",
                    )
                )

    # project_kpi: KPI derived_from == project (strong)
    #   + metric catalog filter_method match (moderate)
    for kpi in kpis:
        if kpi.derived_from_id:
            suggestions.append(
                (
                    "project_kpi",
                    kpi.derived_from_id,
                    kpi.id,
                    "strong",
                )
            )
        # Auto-suggest based on matching calculation_method
        meta = HoshinKPI.METRIC_CATALOG.get(kpi.calculator_result_type, {})
        filter_method = meta.get("filter_method")
        if filter_method:
            for p in projects:
                if p.calculation_method == filter_method:
                    suggestions.append(
                        (
                            "project_kpi",
                            p.id,
                            kpi.id,
                            "moderate",
                        )
                    )

    # kpi_strategic: matching metric catalog key
    for kpi in kpis:
        kpi_metric = kpi.calculator_result_type or ""
        for so in strategic_objs:
            so_metric = so.target_metric or ""
            if so_metric and kpi_metric and so_metric == kpi_metric:
                # Exact catalog key match → strong
                suggestions.append(
                    (
                        "kpi_strategic",
                        kpi.id,
                        so.id,
                        "strong",
                    )
                )
            elif so_metric and kpi_metric:
                # Same group match (e.g. both dollar savings)
                so_group = HoshinKPI.METRIC_CATALOG.get(so_metric, {}).get("group")
                kpi_group = HoshinKPI.METRIC_CATALOG.get(kpi_metric, {}).get("group")
                if so_group and so_group == kpi_group:
                    suggestions.append(
                        (
                            "kpi_strategic",
                            kpi.id,
                            so.id,
                            "moderate",
                        )
                    )

    # Batch create — skip existing
    for pair_type, row_id, col_id, strength in suggestions:
        XMatrixCorrelation.objects.get_or_create(
            tenant=tenant,
            pair_type=pair_type,
            row_id=row_id,
            col_id=col_id,
            defaults={
                "fiscal_year": fiscal_year,
                "strength": strength,
                "source": "auto",
                "is_confirmed": False,
            },
        )


# ---------------------------------------------------------------------------
# Correlation CRUD
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def update_correlation(request):
    """Upsert or delete a single correlation dot."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    pair_type = data.get("pair_type")
    row_id = data.get("row_id")
    col_id = data.get("col_id")
    strength = data.get("strength")
    confirm = data.get("confirm", False)

    if not pair_type or not row_id or not col_id:
        return JsonResponse({"error": "pair_type, row_id, col_id required"}, status=400)

    valid_pairs = ["strategic_annual", "annual_project", "project_kpi", "kpi_strategic"]
    if pair_type not in valid_pairs:
        return JsonResponse(
            {"error": f"Invalid pair_type. Must be one of {valid_pairs}"}, status=400
        )

    # Delete if strength is null/empty
    if not strength:
        deleted, _ = XMatrixCorrelation.objects.filter(
            tenant=tenant,
            pair_type=pair_type,
            row_id=row_id,
            col_id=col_id,
        ).delete()
        return JsonResponse({"success": True, "deleted": deleted > 0})

    valid_strengths = ["strong", "moderate", "weak"]
    if strength not in valid_strengths:
        return JsonResponse(
            {"error": f"Invalid strength. Must be one of {valid_strengths}"}, status=400
        )

    fiscal_year = data.get("fiscal_year", date.today().year)

    corr, created = XMatrixCorrelation.objects.update_or_create(
        tenant=tenant,
        pair_type=pair_type,
        row_id=row_id,
        col_id=col_id,
        defaults={
            "fiscal_year": fiscal_year,
            "strength": strength,
            "source": "manual" if not confirm else XMatrixCorrelation.Source.AUTO,
            "is_confirmed": True,
        },
    )

    # If confirming an auto-suggestion, keep source as auto
    if confirm and not created:
        corr.is_confirmed = True
        corr.save(update_fields=["is_confirmed", "updated_at"])

    return JsonResponse({"success": True, "correlation": corr.to_dict()})


# ---------------------------------------------------------------------------
# Strategic Objectives CRUD
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "POST"])
def list_create_strategic_objectives(request):
    """List (GET) or create (POST) strategic objectives."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if request.method == "GET":
        fiscal_year = request.GET.get("fiscal_year")
        qs = StrategicObjective.objects.filter(tenant=tenant)
        if fiscal_year:
            try:
                fy = int(fiscal_year)
                qs = qs.filter(start_year__lte=fy, end_year__gte=fy)
            except ValueError:
                pass
        return JsonResponse(
            {
                "strategic_objectives": [s.to_dict() for s in qs],
            }
        )

    # POST
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    current_year = date.today().year
    metric_key = data.get("target_metric", "")
    meta = HoshinKPI.METRIC_CATALOG.get(metric_key, {})

    obj = StrategicObjective.objects.create(
        tenant=tenant,
        title=title,
        description=data.get("description", ""),
        owner_name=data.get("owner_name", ""),
        start_year=data.get("start_year", current_year),
        end_year=data.get("end_year", current_year + 3),
        target_metric=metric_key,
        target_value=_decimal_or_none(data.get("target_value")),
        target_unit=meta.get("unit", data.get("target_unit", "")),
        status=data.get("status", "draft"),
        sort_order=data.get("sort_order", 0),
    )
    return JsonResponse(
        {"success": True, "strategic_objective": obj.to_dict()}, status=201
    )


@require_feature("hoshin_kanri")
@require_http_methods(["PUT", "DELETE"])
def update_delete_strategic_objective(request, obj_id):
    """Update (PUT) or delete (DELETE) a strategic objective."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    obj = get_object_or_404(StrategicObjective, id=obj_id, tenant=tenant)

    if request.method == "DELETE":
        obj.delete()
        return JsonResponse({"success": True})

    # PUT
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # If target_metric changed, auto-fill target_unit from catalog
    if "target_metric" in data:
        metric_key = data["target_metric"]
        meta = HoshinKPI.METRIC_CATALOG.get(metric_key, {})
        obj.target_metric = metric_key
        obj.target_unit = meta.get("unit", "")

    for field in ["title", "description", "owner_name", "status"]:
        if field in data:
            setattr(obj, field, data[field])
    for field in ["start_year", "end_year", "sort_order"]:
        if field in data:
            setattr(obj, field, int(data[field]))
    if "target_value" in data:
        obj.target_value = _decimal_or_none(data["target_value"])
    obj.save()
    return JsonResponse({"success": True, "strategic_objective": obj.to_dict()})


# ---------------------------------------------------------------------------
# Annual Objectives CRUD
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "POST"])
def list_create_annual_objectives(request):
    """List (GET) or create (POST) annual objectives."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if request.method == "GET":
        fiscal_year = request.GET.get("fiscal_year", str(date.today().year))
        try:
            fiscal_year = int(fiscal_year)
        except ValueError:
            fiscal_year = date.today().year

        qs = AnnualObjective.objects.filter(
            tenant=tenant,
            fiscal_year=fiscal_year,
        ).select_related("site", "strategic_objective")
        return JsonResponse(
            {
                "annual_objectives": [a.to_dict() for a in qs],
            }
        )

    # POST
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    site = None
    if data.get("site_id"):
        site = get_object_or_404(Site, id=data["site_id"], tenant=tenant)

    strategic = None
    if data.get("strategic_objective_id"):
        strategic = get_object_or_404(
            StrategicObjective,
            id=data["strategic_objective_id"],
            tenant=tenant,
        )

    obj = AnnualObjective.objects.create(
        tenant=tenant,
        strategic_objective=strategic,
        site=site,
        fiscal_year=data.get("fiscal_year", date.today().year),
        title=title,
        description=data.get("description", ""),
        owner_name=data.get("owner_name", ""),
        target_value=_decimal_or_none(data.get("target_value")),
        target_unit=data.get("target_unit", ""),
        status=data.get("status", "on_track"),
        sort_order=data.get("sort_order", 0),
    )
    return JsonResponse(
        {"success": True, "annual_objective": obj.to_dict()}, status=201
    )


@require_feature("hoshin_kanri")
@require_http_methods(["PUT", "DELETE"])
def update_delete_annual_objective(request, obj_id):
    """Update (PUT) or delete (DELETE) an annual objective."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    obj = get_object_or_404(AnnualObjective, id=obj_id, tenant=tenant)

    if request.method == "DELETE":
        obj.delete()
        return JsonResponse({"success": True})

    # PUT
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    for field in ["title", "description", "owner_name", "target_unit", "status"]:
        if field in data:
            setattr(obj, field, data[field])
    if "fiscal_year" in data:
        obj.fiscal_year = int(data["fiscal_year"])
    if "sort_order" in data:
        obj.sort_order = int(data["sort_order"])
    if "target_value" in data:
        obj.target_value = _decimal_or_none(data["target_value"])
    if "actual_value" in data:
        obj.actual_value = _decimal_or_none(data["actual_value"])
    if "site_id" in data:
        obj.site = (
            get_object_or_404(Site, id=data["site_id"], tenant=tenant)
            if data["site_id"]
            else None
        )
    if "strategic_objective_id" in data:
        obj.strategic_objective = (
            get_object_or_404(
                StrategicObjective, id=data["strategic_objective_id"], tenant=tenant
            )
            if data["strategic_objective_id"]
            else None
        )
    obj.save()
    return JsonResponse({"success": True, "annual_objective": obj.to_dict()})


# ---------------------------------------------------------------------------
# KPIs CRUD
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["GET", "POST"])
def list_create_kpis(request):
    """List (GET) or create (POST) KPIs."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if request.method == "GET":
        fiscal_year = request.GET.get("fiscal_year", str(date.today().year))
        try:
            fiscal_year = int(fiscal_year)
        except ValueError:
            fiscal_year = date.today().year

        qs = HoshinKPI.objects.filter(
            tenant=tenant,
            fiscal_year=fiscal_year,
        ).select_related("derived_from")
        return JsonResponse(
            {
                "kpis": [k.to_dict() for k in qs],
            }
        )

    # POST
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)

    # Resolve metric_type from catalog → auto-fill unit, direction, aggregation, etc.
    metric_type = data.get("metric_type", "manual")
    meta = HoshinKPI.METRIC_CATALOG.get(metric_type, HoshinKPI.METRIC_CATALOG["manual"])

    derived_from = None
    if data.get("derived_from_id"):
        derived_from = get_object_or_404(HoshinProject, id=data["derived_from_id"])

    obj = HoshinKPI.objects.create(
        tenant=tenant,
        fiscal_year=data.get("fiscal_year", date.today().year),
        name=name,
        description=data.get("description", ""),
        target_value=_decimal_or_none(data.get("target_value")),
        actual_value=_decimal_or_none(data.get("actual_value")),
        unit=meta.get("unit", data.get("unit", "")),
        frequency=data.get("frequency", "monthly"),
        direction=meta.get("direction", data.get("direction", "up")),
        aggregation=meta.get("aggregation", data.get("aggregation", "sum")),
        derived_from=derived_from,
        derived_field=meta.get(
            "derived_field", data.get("derived_field", "ytd_savings")
        ),
        calculator_result_type=metric_type,
        calculator_field=meta.get("calculator_field", data.get("calculator_field", "")),
        sort_order=data.get("sort_order", 0),
    )
    return JsonResponse({"success": True, "kpi": obj.to_dict()}, status=201)


@require_feature("hoshin_kanri")
@require_http_methods(["PUT", "DELETE"])
def update_delete_kpi(request, kpi_id):
    """Update (PUT) or delete (DELETE) a KPI."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    obj = get_object_or_404(HoshinKPI, id=kpi_id, tenant=tenant)

    if request.method == "DELETE":
        obj.delete()
        return JsonResponse({"success": True})

    # PUT
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # If metric_type changed, re-derive all catalog fields
    if "metric_type" in data:
        metric_type = data["metric_type"]
        meta = HoshinKPI.METRIC_CATALOG.get(
            metric_type, HoshinKPI.METRIC_CATALOG["manual"]
        )
        obj.calculator_result_type = metric_type
        obj.unit = meta.get("unit", "")
        obj.direction = meta.get("direction", "up")
        obj.aggregation = meta.get("aggregation", "sum")
        obj.derived_field = meta.get("derived_field", "ytd_savings")
        obj.calculator_field = meta.get("calculator_field", "")

    for field in ["name", "description", "frequency"]:
        if field in data:
            setattr(obj, field, data[field])
    if "fiscal_year" in data:
        obj.fiscal_year = int(data["fiscal_year"])
    if "sort_order" in data:
        obj.sort_order = int(data["sort_order"])
    if "target_value" in data:
        obj.target_value = _decimal_or_none(data["target_value"])
    if "actual_value" in data:
        obj.actual_value = _decimal_or_none(data["actual_value"])
    if "derived_from_id" in data:
        obj.derived_from = (
            get_object_or_404(HoshinProject, id=data["derived_from_id"])
            if data["derived_from_id"]
            else None
        )
    obj.save()
    return JsonResponse({"success": True, "kpi": obj.to_dict()})


# ---------------------------------------------------------------------------
# VSM Lifecycle — Promotion
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def promote_vsm(request, vsm_id):
    """Promote a future-state VSM to current state.

    - Future → Current
    - Old Current → Archived
    - Metric snapshots carry forward
    - Hoshin realized savings written back to kaizen bursts
    """
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    if vsm.status != "future":
        return JsonResponse(
            {"error": "Only future-state maps can be promoted"},
            status=400,
        )

    if not vsm.paired_with_id:
        return JsonResponse(
            {"error": "No paired current-state map found"},
            status=400,
        )

    old_current = vsm.paired_with

    with transaction.atomic():
        # Archive the old current state
        old_current.status = "archived"
        old_current.paired_with = None
        old_current.save(update_fields=["status", "paired_with", "updated_at"])

        # Promote future to current
        vsm.status = "current"
        vsm.paired_with = None

        # Carry forward metric snapshots
        if old_current.metric_snapshots:
            snapshots = list(vsm.metric_snapshots or [])
            snapshots.extend(old_current.metric_snapshots)
            vsm.metric_snapshots = snapshots

        vsm.save()

        # Write back realized savings from linked hoshin projects
        _writeback_hoshin_savings(vsm, tenant)

    return JsonResponse({"success": True, "vsm": vsm.to_dict()})


def _writeback_hoshin_savings(vsm, tenant):
    """Annotate VSM kaizen bursts with realized savings from linked projects."""
    projects = HoshinProject.objects.filter(
        source_vsm=vsm,
        site__tenant=tenant,
    ).select_related("project")

    bursts = list(vsm.kaizen_bursts or [])
    burst_map = {b.get("id"): b for b in bursts}
    changed = False

    for hp in projects:
        if hp.source_burst_id and hp.source_burst_id in burst_map:
            burst = burst_map[hp.source_burst_id]
            burst["realized_savings"] = float(hp.ytd_savings)
            burst["savings_pct"] = float(hp.savings_pct)
            burst["project_status"] = hp.hoshin_status
            changed = True

    if changed:
        vsm.kaizen_bursts = bursts
        vsm.save(update_fields=["kaizen_bursts"])


# ---------------------------------------------------------------------------
# Fiscal Year Rollover
# ---------------------------------------------------------------------------


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def rollover_fiscal_year(request):
    """Roll forward annual objectives and KPIs to a new fiscal year.

    Strategic objectives persist naturally (multi-year span).
    Annual objectives clone with actuals reset.
    KPIs clone with derived_from=null (carryover projects get new UUIDs).
    Strategic↔annual correlations recreate as unconfirmed.
    Project-level correlations start fresh.
    """
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    from_year = data.get("from_year")
    to_year = data.get("to_year")
    if not from_year or not to_year:
        return JsonResponse({"error": "from_year and to_year required"}, status=400)

    from_year = int(from_year)
    to_year = int(to_year)

    if to_year <= from_year:
        return JsonResponse(
            {"error": "to_year must be greater than from_year"}, status=400
        )

    # Idempotency check
    existing = AnnualObjective.objects.filter(
        tenant=tenant,
        fiscal_year=to_year,
    ).count()
    if existing > 0:
        return JsonResponse(
            {
                "error": f"FY{to_year} already has {existing} annual objectives. Rollover aborted."
            },
            status=400,
        )

    # Get source objects
    source_annuals = list(
        AnnualObjective.objects.filter(
            tenant=tenant,
            fiscal_year=from_year,
        ).select_related("strategic_objective")
    )

    source_kpis = list(
        HoshinKPI.objects.filter(
            tenant=tenant,
            fiscal_year=from_year,
        )
    )

    old_to_new_annual = {}

    with transaction.atomic():
        # Clone annual objectives
        for ao in source_annuals:
            # Skip if parent strategic objective is achieved/deferred
            if ao.strategic_objective and ao.strategic_objective.status in (
                "achieved",
                "deferred",
            ):
                continue

            old_id = ao.id
            ao.pk = None  # Force new instance
            ao.id = None
            ao.fiscal_year = to_year
            ao.actual_value = None
            ao.status = "on_track"
            ao.save()
            old_to_new_annual[str(old_id)] = ao

        # Clone KPIs
        for kpi in source_kpis:
            kpi.pk = None
            kpi.id = None
            kpi.fiscal_year = to_year
            kpi.actual_value = None
            kpi.derived_from = None  # Carryover projects get new UUIDs
            kpi.save()

        # Recreate strategic_annual correlations using old→new mapping
        old_corrs = XMatrixCorrelation.objects.filter(
            tenant=tenant,
            fiscal_year=from_year,
            pair_type="strategic_annual",
        )
        for c in old_corrs:
            new_annual = old_to_new_annual.get(str(c.col_id))
            if new_annual:
                XMatrixCorrelation.objects.get_or_create(
                    tenant=tenant,
                    pair_type="strategic_annual",
                    row_id=c.row_id,
                    col_id=new_annual.id,
                    defaults={
                        "fiscal_year": to_year,
                        "strength": c.strength,
                        "source": "auto",
                        "is_confirmed": False,
                    },
                )

    return JsonResponse(
        {
            "success": True,
            "rolled_over": {
                "annual_objectives": len(old_to_new_annual),
                "kpis": len(source_kpis),
                "from_year": from_year,
                "to_year": to_year,
            },
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decimal_or_none(val):
    """Convert a value to Decimal, or return None."""
    if val is None or val == "" or val == "null":
        return None
    try:
        return Decimal(str(val))
    except (InvalidOperation, ValueError):
        return None
