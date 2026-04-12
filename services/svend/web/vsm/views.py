"""Value Stream Map API views — extracted from agents_api."""

import copy
import json
import logging
import math
import time as _time
import uuid

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid, require_feature

from .models import ValueStreamMap

logger = logging.getLogger(__name__)


# =============================================================================
# SOCKETS — integration points for other apps
# =============================================================================


def get_vsm_metrics(vsm_id):
    """Socket: get VSM metrics for embedding in other tools."""
    vsm = ValueStreamMap.objects.get(id=vsm_id)
    return {
        "id": str(vsm.id),
        "name": vsm.name,
        "total_lead_time": vsm.total_lead_time,
        "total_process_time": vsm.total_process_time,
        "pce": vsm.pce,
        "step_count": len(vsm.process_steps or []),
    }


# =============================================================================
# HANGING WIRES
# =============================================================================


def _resolve_project(user, project_id):
    """Hanging wire: resolve project. Reconnect to core."""
    if not project_id:
        return None, None
    try:
        from core.models import Project

        project = Project.objects.get(id=project_id, user=user)
        return project, None
    except Exception:
        return None, None


def _emit_event(event_name, vsm, user, **kwargs):
    """Hanging wire: emit tool event. Reconnect to event system."""
    pass


def _estimate_savings(current_step, future_step, **kwargs):
    """Hanging wire: Monte Carlo savings estimate. Reconnect to simulators/hoshin."""
    ct_delta = (current_step.get("cycle_time", 0) or 0) - (future_step.get("cycle_time", 0) or 0)
    co_delta = (current_step.get("changeover_time", 0) or 0) - (future_step.get("changeover_time", 0) or 0)
    ut_delta = (future_step.get("uptime", 100) or 100) - (current_step.get("uptime", 100) or 100)
    op_delta = (current_step.get("operators", 0) or 0) - (future_step.get("operators", 0) or 0)
    return {
        "cycle_time_delta": ct_delta,
        "changeover_delta": co_delta,
        "uptime_delta": ut_delta,
        "operators_delta": op_delta,
        "estimated_annual_savings": 0,
        "suggested_method": "direct",
        "improvement_pct": 0,
        "median_savings": 0,
        "lower_5": 0,
        "upper_95": 0,
        "lower_25": 0,
        "upper_75": 0,
        "p_positive": 0,
        "mean_savings": 0,
        "std_savings": 0,
        "deterministic": 0,
    }


# =============================================================================
# BOTTLENECK DETECTION
# =============================================================================


def detect_bottleneck(vsm):
    """Identify bottleneck step and set flags on process_steps."""
    steps = vsm.process_steps or []
    if not steps:
        return None

    wc_steps = {}
    standalone = []

    for step in steps:
        ct = step.get("cycle_time", 0) or 0
        wc_id = step.get("work_center_id")
        if wc_id:
            wc_steps.setdefault(wc_id, []).append((step, ct))
        else:
            standalone.append((step, ct))

    effective = []
    for step, ct in standalone:
        effective.append((step, ct))
    for wc_id, members in wc_steps.items():
        rate_sum = sum(1.0 / ct for _, ct in members if ct > 0)
        eff_ct = (1.0 / rate_sum) if rate_sum > 0 else 0
        effective.append((members[0][0], eff_ct))

    valid = [(s, ct) for s, ct in effective if ct > 0]
    if not valid:
        for step in steps:
            step["flags"] = {}
        return None

    max_ct = max(ct for _, ct in valid)
    bottleneck_step = next(s for s, ct in valid if ct == max_ct)
    throughput = 3600.0 / max_ct if max_ct > 0 else 0
    takt = vsm.takt_time

    for step in steps:
        step_ct = step.get("cycle_time", 0) or 0
        flags = {}
        flags["is_bottleneck"] = step.get("id") == bottleneck_step.get("id")
        if takt and takt > 0 and step_ct > 0:
            flags["takt_ratio"] = round(step_ct / takt, 2)
            flags["exceeds_takt"] = step_ct > takt
        step["flags"] = flags

    return {
        "bottleneck_step_id": bottleneck_step.get("id", ""),
        "bottleneck_step_name": bottleneck_step.get("name", ""),
        "bottleneck_ct": max_ct,
        "theoretical_throughput": round(throughput, 1),
    }


# =============================================================================
# CRUD
# =============================================================================


@gated_paid
@require_http_methods(["GET"])
def list_vsm(request):
    """List user's value stream maps."""
    maps = ValueStreamMap.objects.filter(owner=request.user).select_related("project")

    project_id = request.GET.get("project_id")
    if project_id:
        maps = maps.filter(project_id=project_id)

    status = request.GET.get("status")
    if status:
        maps = maps.filter(status=status)

    return JsonResponse({"maps": [m.to_dict() for m in maps[:50]]})


@gated_paid
@require_http_methods(["POST"])
def create_vsm(request):
    """Create a new value stream map."""
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    project, _ = _resolve_project(request.user, data.get("project_id"))

    vsm = ValueStreamMap.objects.create(
        owner=request.user,
        project=project,
        name=data.get("name", "Untitled VSM"),
        product_family=data.get("product_family", ""),
        customer_name=data.get("customer_name", "Customer"),
        customer_demand=data.get("customer_demand", ""),
        supplier_name=data.get("supplier_name", "Supplier"),
        supply_frequency=data.get("supply_frequency", ""),
    )

    _emit_event("vsm.created", vsm, user=request.user)

    return JsonResponse({"id": str(vsm.id), "vsm": vsm.to_dict()})


@gated_paid
@require_http_methods(["GET"])
def get_vsm(request, vsm_id):
    """Get a single VSM with full details."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)
    bottleneck_info = detect_bottleneck(vsm)
    return JsonResponse({"vsm": vsm.to_dict(), "bottleneck": bottleneck_info})


@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_vsm(request, vsm_id):
    """Update a VSM."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "takt_time" in data and data["takt_time"] is not None:
        try:
            tt = float(data["takt_time"])
            if tt <= 0:
                return JsonResponse({"error": "Takt time must be positive"}, status=400)
            data["takt_time"] = tt
        except (ValueError, TypeError):
            return JsonResponse({"error": "Takt time must be a number"}, status=400)

    for field in [
        "name",
        "status",
        "product_family",
        "customer_name",
        "customer_demand",
        "takt_time",
        "supplier_name",
        "supply_frequency",
        "zoom",
        "pan_x",
        "pan_y",
    ]:
        if field in data:
            setattr(vsm, field, data[field])

    for field in [
        "process_steps",
        "inventory",
        "information_flow",
        "material_flow",
        "kaizen_bursts",
        "customers",
        "suppliers",
        "work_centers",
    ]:
        if field in data:
            setattr(vsm, field, data[field])

    if "project_id" in data:
        if data["project_id"]:
            proj, _ = _resolve_project(request.user, data["project_id"])
            if proj:
                vsm.project = proj
        else:
            vsm.project = None

    auto_kaizen = data.get("auto_kaizen")
    if auto_kaizen and isinstance(auto_kaizen, dict):
        text = auto_kaizen.get("text", "").strip()
        near_step = auto_kaizen.get("near_step", "")
        priority = auto_kaizen.get("priority", "medium")
        if text:
            bursts = vsm.kaizen_bursts or []
            if not any(b.get("text") == text for b in bursts):
                x, y = 200, 50
                for i, step in enumerate(vsm.process_steps or []):
                    if step.get("name", "").lower() == near_step.lower():
                        x = step.get("x", 200 + i * 200)
                        y = max(0, step.get("y", 100) - 60)
                        break
                bursts.append(
                    {
                        "id": f"kaizen_{len(bursts) + 1}_{int(_time.time())}",
                        "text": text,
                        "x": x,
                        "y": y,
                        "priority": priority,
                    }
                )
                vsm.kaizen_bursts = bursts

    vsm.calculate_metrics()
    bottleneck_info = detect_bottleneck(vsm)
    vsm.save()

    _emit_event("vsm.updated", vsm, user=request.user)

    return JsonResponse({"success": True, "vsm": vsm.to_dict(), "bottleneck": bottleneck_info})


@gated_paid
@require_http_methods(["DELETE"])
def delete_vsm(request, vsm_id):
    """Delete a VSM."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)
    vsm.delete()
    return JsonResponse({"success": True})


# =============================================================================
# STRUCTURED ADDITIONS
# =============================================================================


@gated_paid
@require_http_methods(["POST"])
def add_process_step(request, vsm_id):
    """Add a process step to the VSM."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    step = {
        "id": str(uuid.uuid4())[:8],
        "name": data.get("name", "Process"),
        "x": data.get("x", 100),
        "y": data.get("y", 300),
        "cycle_time": data.get("cycle_time"),
        "changeover_time": data.get("changeover_time"),
        "uptime": data.get("uptime", 100),
        "operators": data.get("operators", 1),
        "shifts": data.get("shifts", 1),
        "batch_size": data.get("batch_size"),
        "notes": data.get("notes", ""),
    }

    vsm.process_steps.append(step)
    vsm.calculate_metrics()
    vsm.save()

    return JsonResponse({"success": True, "step": step, "vsm": vsm.to_dict()})


@gated_paid
@require_http_methods(["POST"])
def add_inventory(request, vsm_id):
    """Add inventory triangle between process steps."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    inv = {
        "id": str(uuid.uuid4())[:8],
        "before_step_id": data.get("before_step_id"),
        "quantity": data.get("quantity"),
        "days_of_supply": data.get("days_of_supply"),
        "x": data.get("x", 100),
        "y": data.get("y", 350),
    }

    vsm.inventory.append(inv)
    vsm.calculate_metrics()
    vsm.save()

    return JsonResponse({"success": True, "inventory": inv, "vsm": vsm.to_dict()})


@gated_paid
@require_http_methods(["POST"])
def add_kaizen_burst(request, vsm_id):
    """Add a kaizen burst (improvement opportunity)."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    burst = {
        "id": str(uuid.uuid4())[:8],
        "x": data.get("x", 100),
        "y": data.get("y", 100),
        "text": data.get("text", ""),
        "priority": data.get("priority", "medium"),
    }

    vsm.kaizen_bursts.append(burst)
    vsm.save()

    return JsonResponse({"success": True, "burst": burst, "vsm": vsm.to_dict()})


# =============================================================================
# FUTURE STATE & COMPARISON
# =============================================================================


@gated_paid
@require_http_methods(["POST"])
def create_future_state(request, vsm_id):
    """Create a future state VSM from the current state."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    future = ValueStreamMap.objects.create(
        owner=request.user,
        project=vsm.project,
        name=f"{vsm.name} (Future State)",
        status=ValueStreamMap.Status.FUTURE,
        fiscal_year=vsm.fiscal_year,
        product_family=vsm.product_family,
        customer_name=vsm.customer_name,
        customer_demand=vsm.customer_demand,
        takt_time=vsm.takt_time,
        supplier_name=vsm.supplier_name,
        supply_frequency=vsm.supply_frequency,
        process_steps=copy.deepcopy(vsm.process_steps),
        inventory=copy.deepcopy(vsm.inventory),
        information_flow=copy.deepcopy(vsm.information_flow),
        material_flow=copy.deepcopy(vsm.material_flow),
        kaizen_bursts=copy.deepcopy(vsm.kaizen_bursts),
        work_centers=copy.deepcopy(vsm.work_centers),
        zoom=vsm.zoom,
        pan_x=vsm.pan_x,
        pan_y=vsm.pan_y,
    )

    future.calculate_metrics()
    future.save()
    future.paired_with = vsm
    future.save(update_fields=["paired_with"])

    _emit_event("vsm.future_state_created", future, user=request.user)

    return JsonResponse({"success": True, "future_state": future.to_dict()})


@gated_paid
@require_http_methods(["GET"])
def compare_vsm(request, vsm_id):
    """Compare current and future state VSMs."""
    current = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    future = None
    if current.project:
        future = (
            ValueStreamMap.objects.filter(
                owner=request.user,
                project=current.project,
                status=ValueStreamMap.Status.FUTURE,
            )
            .exclude(id=current.id)
            .first()
        )

    if not future:
        return JsonResponse({"current": current.to_dict(), "future": None, "comparison": None})

    comparison = {
        "lead_time": {
            "current": current.total_lead_time,
            "future": future.total_lead_time,
            "improvement": (
                ((current.total_lead_time or 0) - (future.total_lead_time or 0)) / (current.total_lead_time or 1) * 100
            )
            if current.total_lead_time
            else 0,
        },
        "process_time": {
            "current": current.total_process_time,
            "future": future.total_process_time,
            "improvement": (
                ((current.total_process_time or 0) - (future.total_process_time or 0))
                / (current.total_process_time or 1)
                * 100
            )
            if current.total_process_time
            else 0,
        },
        "pce": {
            "current": current.pce,
            "future": future.pce,
            "improvement": (future.pce or 0) - (current.pce or 0),
        },
        "inventory_reduction": {
            "current_count": len(current.inventory),
            "future_count": len(future.inventory),
        },
        "process_steps": {
            "current_count": len(current.process_steps),
            "future_count": len(future.process_steps),
        },
    }

    return JsonResponse({"current": current.to_dict(), "future": future.to_dict(), "comparison": comparison})


# =============================================================================
# WASTE ANALYSIS (TIMWOODS)
# =============================================================================


@gated_paid
@require_http_methods(["GET"])
def waste_analysis(request, vsm_id):
    """Classify TIMWOODS waste categories from VSM process metrics."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    steps = vsm.process_steps or []
    inventory = vsm.inventory or []
    material_flow = vsm.material_flow or []
    info_flow = vsm.information_flow or []
    takt_time = vsm.takt_time or 0

    waste = {
        "transport": [],
        "inventory": [],
        "motion": [],
        "waiting": [],
        "overproduction": [],
        "overprocessing": [],
        "defects": [],
        "skills": [],
    }

    for inv in inventory:
        dos = inv.get("days_of_supply", 0)
        if dos > 5:
            severity = "high" if dos > 15 else ("medium" if dos > 10 else "low")
            waste["inventory"].append(
                {
                    "location": inv.get("name", inv.get("id", "unknown")),
                    "detail": f"{dos} days supply",
                    "severity": severity,
                }
            )

    for step in steps:
        ct = step.get("cycle_time", 0) or 0
        changeover = step.get("changeover_time", 0) or 0
        uptime = step.get("uptime", 100) or 100
        batch_size = step.get("batch_size", 1) or 1
        name = step.get("name", "Unknown")

        if ct > 0 and changeover > 10 * ct:
            waste["waiting"].append(
                {
                    "step": name,
                    "detail": f"changeover {changeover}s vs cycle time {ct}s (ratio {changeover / ct:.0f}x)",
                    "severity": "high",
                    "suggested_kaizen": "SMED changeover reduction",
                }
            )

        if takt_time > 0 and ct > 2 * takt_time:
            waste["waiting"].append(
                {
                    "step": name,
                    "detail": f"cycle time {ct}s exceeds 2x takt ({takt_time}s)",
                    "severity": "high",
                }
            )

        if uptime < 85:
            waste["defects"].append(
                {
                    "step": name,
                    "detail": f"uptime {uptime}%",
                    "severity": "high" if uptime < 70 else "medium",
                    "suggested_kaizen": "TPM (Total Productive Maintenance)",
                }
            )

        if batch_size > 50:
            waste["overproduction"].append(
                {
                    "step": name,
                    "detail": f"batch size {batch_size}",
                    "severity": "medium" if batch_size < 200 else "high",
                }
            )

    push_count = sum(1 for mf in material_flow if mf.get("type") == "push")
    if push_count > 2:
        waste["overproduction"].append(
            {
                "step": "Material flow",
                "detail": f"{push_count} push connections (consider pull/kanban)",
                "severity": "medium",
            }
        )

    manual_count = sum(1 for inf in info_flow if inf.get("type") == "manual")
    if manual_count > 0:
        waste["motion"].append(
            {
                "step": "Information flow",
                "detail": f"{manual_count} manual information flows",
                "severity": "low" if manual_count < 3 else "medium",
            }
        )

    pce = vsm.pce or 0
    if pce > 0 and pce < 5:
        waste["overprocessing"].append(
            {
                "step": "Overall",
                "detail": f"PCE {pce:.1f}% — less than 5% of lead time is value-adding",
                "severity": "high",
            }
        )

    total = sum(len(items) for items in waste.values())
    top_opportunities = []
    for category, items in waste.items():
        for item in items:
            if item.get("severity") in ("high", "medium"):
                top_opportunities.append(
                    {
                        "category": category,
                        "step": item.get("step", ""),
                        "suggested_kaizen": item.get("suggested_kaizen", ""),
                    }
                )

    return JsonResponse(
        {
            "waste_categories": waste,
            "total_waste_items": total,
            "top_opportunities": top_opportunities[:10],
        }
    )


# =============================================================================
# HOSHIN INTEGRATION (hanging wires)
# =============================================================================


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def generate_proposals(request, vsm_id):
    """Auto-propose CI projects from VSM kaizen bursts.
    Hanging wire: uses stub savings estimator until simulators app is wired.
    """
    current = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    future = None
    if current.project:
        future = (
            ValueStreamMap.objects.filter(
                owner=request.user,
                project=current.project,
                status=ValueStreamMap.Status.FUTURE,
            )
            .exclude(id=current.id)
            .first()
        )

    if not future:
        return JsonResponse({"error": "No future state VSM found. Create a future state first."}, status=400)

    bursts = future.kaizen_bursts or []
    if not bursts:
        return JsonResponse({"error": "No kaizen bursts on the future state VSM."}, status=400)

    data = json.loads(request.body) if request.body else {}
    annual_volume = float(data.get("annual_volume", 100000))
    cost_per_unit = float(data.get("cost_per_unit", 50.0))
    labor_rate = float(data.get("labor_rate", 35.0))

    current_steps = {s.get("name", ""): s for s in (current.process_steps or [])}
    future_steps = future.process_steps or []

    proposals = []
    for burst in bursts:
        burst_id = burst.get("id", "")
        burst_x = float(burst.get("x", 0))
        burst_y = float(burst.get("y", 0))
        burst_text = burst.get("text", "Improvement")
        burst_priority = burst.get("priority", "medium")

        nearest_step = None
        min_dist = float("inf")
        for step in future_steps:
            dist = math.sqrt((burst_x - float(step.get("x", 0))) ** 2 + (burst_y - float(step.get("y", 0))) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_step = step

        if not nearest_step:
            continue

        step_name = nearest_step.get("name", "Unknown")
        current_step = current_steps.get(step_name, {})

        estimate = (
            _estimate_savings(
                current_step,
                nearest_step,
                annual_volume=annual_volume,
                cost_per_unit=cost_per_unit,
                labor_rate=labor_rate,
            )
            if current_step
            else {
                "cycle_time_delta": 0,
                "changeover_delta": 0,
                "uptime_delta": 0,
                "operators_delta": 0,
                "estimated_annual_savings": 0,
                "suggested_method": "direct",
                "improvement_pct": 0,
                "median_savings": 0,
                "lower_5": 0,
                "upper_95": 0,
                "p_positive": 0,
            }
        )

        proposals.append(
            {
                "burst_id": burst_id,
                "burst_text": burst_text,
                "priority": burst_priority,
                "process_step": step_name,
                "has_current_match": bool(current_step),
                "metric_deltas": {
                    "cycle_time": estimate["cycle_time_delta"],
                    "changeover": estimate["changeover_delta"],
                    "uptime": estimate["uptime_delta"],
                    "operators": estimate["operators_delta"],
                },
                "estimated_annual_savings": estimate.get("median_savings", 0),
                "suggested_method": estimate["suggested_method"],
                "improvement_pct": estimate["improvement_pct"],
                "median_savings": estimate.get("median_savings", 0),
                "lower_5": estimate.get("lower_5", 0),
                "upper_95": estimate.get("upper_95", 0),
                "p_positive": estimate.get("p_positive", 0),
                "suggested_title": f"{burst_text} — {step_name}",
            }
        )

    return JsonResponse(
        {
            "vsm_id": str(vsm_id),
            "vsm_name": current.name,
            "future_vsm_id": str(future.id),
            "proposals": proposals,
            "count": len(proposals),
            "defaults": {"annual_volume": annual_volume, "cost_per_unit": cost_per_unit, "labor_rate": labor_rate},
        }
    )


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def approve_proposal(request, vsm_id):
    """Approve a VSM proposal -> create HoshinProject.
    Hanging wire: HoshinProject creation stubbed until hoshin app is wired.
    """
    return JsonResponse(
        {"error": "Hoshin integration not yet wired. Approve proposals from the hoshin app."},
        status=501,
    )
