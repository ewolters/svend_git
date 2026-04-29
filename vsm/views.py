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
    """Estimate savings from current→future state step deltas.

    Uses deterministic calculation with ±20% spread for confidence bounds.
    """
    annual_volume = kwargs.get("annual_volume", 100000)
    cost_per_unit = kwargs.get("cost_per_unit", 50.0)
    labor_rate = kwargs.get("labor_rate", 35.0)  # $/hr

    cur_ct = current_step.get("cycle_time", 0) or 0
    fut_ct = future_step.get("cycle_time", 0) or 0
    ct_delta = cur_ct - fut_ct

    cur_co = current_step.get("changeover_time", 0) or 0
    fut_co = future_step.get("changeover_time", 0) or 0
    co_delta = cur_co - fut_co

    cur_ut = current_step.get("uptime", 100) or 100
    fut_ut = future_step.get("uptime", 100) or 100
    ut_delta = fut_ut - cur_ut

    cur_ops = current_step.get("operators", 0) or 0
    fut_ops = future_step.get("operators", 0) or 0
    op_delta = cur_ops - fut_ops

    cur_batch = current_step.get("batch_size") or 0
    fut_batch = future_step.get("batch_size") or 0

    # Determine dominant savings method and estimate
    savings = 0.0
    method = "direct"

    # Cycle time reduction → throughput gain
    if ct_delta > 0 and cur_ct > 0:
        time_saved_hrs = (ct_delta * annual_volume) / 3600
        savings += time_saved_hrs * labor_rate
        method = "time_reduction"

    # Changeover reduction → capacity recovery
    if co_delta > 0:
        # Assume 1 changeover per batch or per day if no batch
        changeovers_per_year = annual_volume / max(cur_batch, 100) if cur_batch > 0 else 250
        co_hours_saved = (co_delta * changeovers_per_year) / 3600
        savings += co_hours_saved * labor_rate

    # Headcount reduction
    if op_delta > 0:
        annual_labor_cost = 2080 * labor_rate  # 2080 hrs/yr
        savings += op_delta * annual_labor_cost
        if op_delta >= 1:
            method = "headcount"

    # Uptime improvement → capacity gain
    if ut_delta > 0 and cur_ut < 100:
        capacity_pct_gain = ut_delta / 100
        savings += capacity_pct_gain * annual_volume * cost_per_unit * 0.05  # 5% margin recovery

    # Batch size reduction → inventory savings
    if cur_batch > 0 and fut_batch > 0 and cur_batch > fut_batch:
        avg_wip_reduction = (cur_batch - fut_batch) / 2
        holding_cost = cost_per_unit * 0.25  # 25% annual holding
        savings += avg_wip_reduction * holding_cost

    improvement_pct = 0.0
    if cur_ct > 0:
        improvement_pct = round((ct_delta / cur_ct) * 100, 1)

    deterministic = round(savings, 2)
    # Spread: ±20% for confidence bounds
    lower_5 = round(savings * 0.6, 2)
    upper_95 = round(savings * 1.4, 2)
    lower_25 = round(savings * 0.8, 2)
    upper_75 = round(savings * 1.2, 2)
    p_positive = 0.85 if savings > 0 else 0.15

    return {
        "cycle_time_delta": ct_delta,
        "changeover_delta": co_delta,
        "uptime_delta": ut_delta,
        "operators_delta": op_delta,
        "estimated_annual_savings": deterministic,
        "suggested_method": method,
        "improvement_pct": improvement_pct,
        "median_savings": deterministic,
        "lower_5": lower_5,
        "upper_95": upper_95,
        "lower_25": lower_25,
        "upper_75": upper_75,
        "p_positive": p_positive,
        "mean_savings": deterministic,
        "std_savings": round(savings * 0.2, 2),
        "deterministic": deterministic,
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
    """Delete a VSM (with pull contract friction)."""
    from qms_core.pull_views import check_delete_friction

    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)
    force = request.GET.get("force", "").lower() == "true"
    ok, err_resp, _tombstoned = check_delete_friction("vsm", "ValueStreamMap", vsm_id, force=force)
    if not ok:
        return err_resp
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
        "batch_process": data.get("batch_process", False),
        "demand_rate": data.get("demand_rate"),
        "demand_unit": data.get("demand_unit", ""),
        "pack_size": data.get("pack_size"),
        "pitch": data.get("pitch"),
        "epei": data.get("epei"),
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
# WASTE ANALYSIS (DOWNTIME)
# Defects, Overproduction, Waiting, NVA Processing, Transport,
# Inventory, Motion, Employee Intellect/Passion
# =============================================================================


@gated_paid
@require_http_methods(["GET"])
def waste_analysis(request, vsm_id):
    """Classify DOWNTIME waste categories from VSM process metrics."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    steps = vsm.process_steps or []
    inventory = vsm.inventory or []
    material_flow = vsm.material_flow or []
    info_flow = vsm.information_flow or []
    takt_time = vsm.takt_time or 0

    waste = {
        "defects": [],
        "overproduction": [],
        "waiting": [],
        "nva_processing": [],
        "transport": [],
        "inventory": [],
        "motion": [],
        "employee_intellect": [],
    }

    # Build step lookup for adjacency analysis
    step_ids = [s.get("id") for s in steps]
    step_by_id = {s.get("id"): s for s in steps}

    # --- DEFECTS ---
    # Low uptime = quality/reliability issues producing defects and rework
    for step in steps:
        uptime = step.get("uptime", 100) or 100
        name = step.get("name", "Unknown")

        if uptime < 85:
            waste["defects"].append(
                {
                    "step": name,
                    "detail": f"uptime {uptime}% — equipment/process reliability issue",
                    "severity": "high" if uptime < 70 else "medium",
                    "suggested_kaizen": "TPM (Total Productive Maintenance)",
                }
            )

        # Quality field if present
        quality = step.get("quality", step.get("yield", step.get("fpy")))
        if quality is not None and quality < 95:
            waste["defects"].append(
                {
                    "step": name,
                    "detail": f"first pass yield {quality}%",
                    "severity": "high" if quality < 85 else "medium",
                    "suggested_kaizen": "Root cause analysis + error proofing (poka-yoke)",
                }
            )

    # --- OVERPRODUCTION ---
    # Large batch sizes, push systems, producing ahead of demand
    for step in steps:
        batch_size = step.get("batch_size") or 0
        name = step.get("name", "Unknown")
        changeover = step.get("changeover_time", 0) or 0
        ct = step.get("cycle_time", 0) or 0

        if batch_size > 50:
            waste["overproduction"].append(
                {
                    "step": name,
                    "detail": f"batch size {batch_size} — producing in large lots ahead of demand",
                    "severity": "medium" if batch_size < 200 else "high",
                    "suggested_kaizen": "Reduce lot size via SMED, implement pull system",
                }
            )

        # Batch exceeds pitch-optimal quantity
        if batch_size > 0 and pitch > 0 and ct > 0:
            pitch_qty = pitch / ct  # units per pitch interval
            if batch_size > pitch_qty * 3:
                waste["overproduction"].append(
                    {
                        "step": name,
                        "detail": f"batch {batch_size} is {batch_size / pitch_qty:.0f}x the pitch quantity ({pitch_qty:.0f}) — overproducing vs withdrawal rhythm",
                        "severity": "high",
                        "suggested_kaizen": "Reduce lot to pitch quantity or 2x pitch max. Run SMED to enable smaller batches.",
                    }
                )

        # Changeover too long relative to cycle time = batch pressure
        if changeover > 0 and ct > 0 and changeover > 20 * ct and batch_size == 0:
            waste["overproduction"].append(
                {
                    "step": name,
                    "detail": f"changeover {changeover}s is {changeover / ct:.0f}x cycle time — creates pressure for large batches",
                    "severity": "medium",
                    "suggested_kaizen": "SMED to reduce changeover. Current ratio forces large lots to amortize setup.",
                }
            )

    push_count = sum(1 for mf in material_flow if mf.get("type") == "push")
    if push_count > 2:
        waste["overproduction"].append(
            {
                "step": "Material flow",
                "detail": f"{push_count} push connections — schedule-driven, not demand-driven",
                "severity": "medium" if push_count < 5 else "high",
                "suggested_kaizen": "Convert to pull/kanban between operations",
            }
        )

    # --- WAITING ---
    # Long changeovers, cycle time exceeding takt, idle time between steps
    for step in steps:
        ct = step.get("cycle_time", 0) or 0
        changeover = step.get("changeover_time", 0) or 0
        name = step.get("name", "Unknown")

        if ct > 0 and changeover > 10 * ct:
            waste["waiting"].append(
                {
                    "step": name,
                    "detail": f"changeover {changeover}s vs cycle time {ct}s ({changeover / ct:.0f}x ratio) — line waits during setup",
                    "severity": "high",
                    "suggested_kaizen": "SMED changeover reduction",
                }
            )

        if takt_time > 0 and ct > 2 * takt_time:
            waste["waiting"].append(
                {
                    "step": name,
                    "detail": f"cycle time {ct}s exceeds 2x takt ({takt_time}s) — downstream starved",
                    "severity": "high",
                    "suggested_kaizen": "Balance work content, add parallel capacity, or kaizen cycle time",
                }
            )

    # Inventory between steps = waiting (WIP sitting = product waiting)
    for inv in inventory:
        dos = inv.get("days_of_supply", 0)
        if dos > 5:
            location = inv.get("name", inv.get("before_step", "unknown"))
            waste["waiting"].append(
                {
                    "step": str(location),
                    "detail": f"{dos} days of supply sitting — product waiting in queue",
                    "severity": "high" if dos > 15 else ("medium" if dos > 10 else "low"),
                }
            )

    # --- NVA PROCESSING ---
    # Low PCE = most of lead time is non-value-adding processing/handling
    pce = vsm.pce or 0
    if pce > 0 and pce < 5:
        waste["nva_processing"].append(
            {
                "step": "Overall value stream",
                "detail": f"PCE {pce:.1f}% — {100 - pce:.1f}% of lead time is non-value-adding",
                "severity": "high",
                "suggested_kaizen": "Map value-add vs NVA at each step, eliminate NVA handling and inspection",
            }
        )

    # Steps with very long cycle times relative to neighbors may include NVA work
    if len(steps) >= 3:
        cycle_times = [(s.get("name", "?"), s.get("cycle_time", 0) or 0) for s in steps]
        valid_cts = [ct for _, ct in cycle_times if ct > 0]
        if valid_cts:
            avg_ct = sum(valid_cts) / len(valid_cts)
            for name, ct in cycle_times:
                if ct > 3 * avg_ct and ct > 0:
                    waste["nva_processing"].append(
                        {
                            "step": name,
                            "detail": f"cycle time {ct}s is {ct / avg_ct:.1f}x the average ({avg_ct:.0f}s) — likely contains NVA work content",
                            "severity": "medium",
                            "suggested_kaizen": "Time study: separate value-add from NVA elements",
                        }
                    )

    # --- TRANSPORT ---
    # Material movements between non-adjacent steps, long flow paths
    connections = {(mf.get("from_id"), mf.get("to_id")) for mf in material_flow}
    for from_id, to_id in connections:
        if from_id in step_ids and to_id in step_ids:
            from_idx = step_ids.index(from_id)
            to_idx = step_ids.index(to_id)
            gap = abs(to_idx - from_idx)
            if gap > 1:
                from_name = step_by_id.get(from_id, {}).get("name", "?")
                to_name = step_by_id.get(to_id, {}).get("name", "?")
                waste["transport"].append(
                    {
                        "step": f"{from_name} → {to_name}",
                        "detail": f"material skips {gap - 1} steps — non-adjacent flow, likely physical transport",
                        "severity": "medium" if gap < 4 else "high",
                        "suggested_kaizen": "Relocate operations closer, create cells, reduce material handling",
                    }
                )

    # Multiple inventory points = multiple transport legs
    if len(inventory) > 3:
        waste["transport"].append(
            {
                "step": "Value stream",
                "detail": f"{len(inventory)} WIP locations — each requires material movement",
                "severity": "medium" if len(inventory) < 6 else "high",
                "suggested_kaizen": "Reduce WIP locations by creating flow between operations",
            }
        )

    # --- INVENTORY ---
    # WIP, raw material buffers, finished goods sitting
    for inv in inventory:
        dos = inv.get("days_of_supply", 0)
        qty = inv.get("quantity", 0)
        if dos > 5 or qty > 500:
            location = inv.get("name", inv.get("before_step", "unknown"))
            severity = "high" if dos > 15 else ("medium" if dos > 10 else "low")
            detail_parts = []
            if dos:
                detail_parts.append(f"{dos} days supply")
            if qty:
                detail_parts.append(f"{qty} units")
            waste["inventory"].append(
                {
                    "step": str(location),
                    "detail": " / ".join(detail_parts) + " — ties up cash, hides problems",
                    "severity": severity,
                    "suggested_kaizen": "Reduce lot sizes, implement FIFO lanes, establish kanban pull",
                }
            )

    # Total lead time vs process time ratio
    lead_time = vsm.total_lead_time or 0
    process_time = vsm.total_process_time or 0
    if lead_time > 0 and process_time > 0 and lead_time > 10 * process_time:
        ratio = lead_time / process_time
        waste["inventory"].append(
            {
                "step": "Overall value stream",
                "detail": f"lead time is {ratio:.0f}x process time — {ratio - 1:.0f}x is inventory wait",
                "severity": "high",
                "suggested_kaizen": "Attack largest WIP buffers first, create flow between operations",
            }
        )

    # --- MOTION ---
    # Manual information flows, operator walking between stations
    manual_count = sum(1 for inf in info_flow if inf.get("type") == "manual")
    if manual_count > 0:
        waste["motion"].append(
            {
                "step": "Information flow",
                "detail": f"{manual_count} manual information flows — people walking paper, checking schedules",
                "severity": "low" if manual_count < 3 else "medium",
                "suggested_kaizen": "Digitize scheduling signals, implement visual management",
            }
        )

    # Steps with multiple operators may have motion waste
    for step in steps:
        operators = step.get("operators", 1) or 1
        name = step.get("name", "Unknown")
        if operators > 3:
            waste["motion"].append(
                {
                    "step": name,
                    "detail": f"{operators} operators at one station — likely walking, reaching, searching",
                    "severity": "low" if operators < 5 else "medium",
                    "suggested_kaizen": "Yamazumi balance, 5S workplace organization, cell redesign",
                }
            )

    # --- EMPLOYEE INTELLECT/PASSION ---
    # Underutilized capability: manual repetitive work, no problem-solving structure
    for step in steps:
        operators = step.get("operators", 1) or 1
        ct = step.get("cycle_time", 0) or 0
        changeover = step.get("changeover_time", 0) or 0
        name = step.get("name", "Unknown")

        # Operators stuck watching machines = wasted intellect
        if operators >= 1 and ct > 120 and changeover == 0:
            waste["employee_intellect"].append(
                {
                    "step": name,
                    "detail": f"{operators} operator(s) with {ct}s cycle, no changeover — likely machine-watching",
                    "severity": "low" if operators == 1 else "medium",
                    "suggested_kaizen": "Multi-machine operation, autonomous maintenance (jidoka), cross-training",
                }
            )

    # No kaizen bursts = people not being asked for ideas
    kaizen_bursts = vsm.kaizen_bursts or []
    if not kaizen_bursts and len(steps) > 3:
        waste["employee_intellect"].append(
            {
                "step": "Value stream",
                "detail": "no kaizen bursts on future state — team ideas not captured",
                "severity": "medium",
                "suggested_kaizen": "Facilitate kaizen workshop to capture operator improvement ideas",
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
                        "detail": item.get("detail", ""),
                        "suggested_kaizen": item.get("suggested_kaizen", ""),
                        "severity": item["severity"],
                    }
                )

    # Sort: high first, then medium
    top_opportunities.sort(key=lambda x: 0 if x["severity"] == "high" else 1)

    return JsonResponse(
        {
            "framework": "DOWNTIME",
            "categories": {
                "D": "Defects",
                "O": "Overproduction",
                "W": "Waiting",
                "N": "NVA Processing",
                "T": "Transport",
                "I": "Inventory",
                "M": "Motion",
                "E": "Employee Intellect/Passion",
            },
            "waste": waste,
            "total_waste_items": total,
            "top_opportunities": top_opportunities[:15],
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
    """Approve VSM proposals -> create HoshinProjects via hoshin app."""
    from hoshin.views import create_from_proposals

    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Inject vsm_id into the payload and forward to hoshin's create_from_proposals
    data["vsm_id"] = str(vsm.id)

    # Build a fake request body with the merged data
    from django.test import RequestFactory

    factory = RequestFactory()
    forwarded = factory.post(
        "/api/hoshin/projects/from-proposals/",
        data=json.dumps(data),
        content_type="application/json",
    )
    forwarded.user = request.user
    forwarded.META = request.META.copy()

    return create_from_proposals(forwarded)


# =============================================================================
# LOT SIZE RECOMMENDATION (regime-detecting)
# =============================================================================


def _detect_regime(demand_per_day, mix_count, changeover_sec, cycle_time_sec, available_sec):
    """Detect production regime from step/VSM metrics.

    Returns dict with regime scores (0-1) for each regime.
    """
    if available_sec <= 0:
        available_sec = 28800  # 8hr default

    # Schedule intensity: fraction of capacity consumed by production alone
    schedule_intensity = (demand_per_day * cycle_time_sec) / available_sec if demand_per_day > 0 else 0

    # Mix pressure: fraction of capacity consumed if every part runs daily
    mix_pressure = (mix_count * changeover_sec) / available_sec if mix_count > 0 else 0

    # Changeover ratio: how long is changeover vs cycle time
    co_ratio = changeover_sec / cycle_time_sec if cycle_time_sec > 0 else 0

    scores = {}

    # Schedule-paced: very low demand rate, or CT dominates available time
    if demand_per_day < 1:
        scores["schedule_paced"] = 0.9
    elif demand_per_day < 5 and cycle_time_sec > 3600:
        scores["schedule_paced"] = 0.7
    elif schedule_intensity < 0.1:
        scores["schedule_paced"] = 0.5
    else:
        scores["schedule_paced"] = max(0, 0.3 - schedule_intensity)

    # Mix-constrained: changeover burden from mix is the binding constraint
    if mix_pressure > 0.5:
        scores["mix_constrained"] = 0.9
    elif mix_pressure > 0.3:
        scores["mix_constrained"] = 0.7
    elif mix_count > 20:
        scores["mix_constrained"] = 0.6
    else:
        scores["mix_constrained"] = min(0.4, mix_pressure * 2)

    # Capacity-budget: running hard, changeovers eat into output
    if schedule_intensity > 0.5 and mix_pressure < 0.2:
        scores["capacity_budget"] = 0.8
    elif schedule_intensity > 0.3:
        scores["capacity_budget"] = 0.5 + schedule_intensity * 0.3
    else:
        scores["capacity_budget"] = max(0, schedule_intensity)

    # Normalize: pick winner
    best = max(scores, key=scores.get)
    return {
        "regime": best,
        "confidence": round(scores[best], 2),
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "indicators": {
            "schedule_intensity": round(schedule_intensity, 3),
            "mix_pressure": round(mix_pressure, 3),
            "changeover_ratio": round(co_ratio, 1),
        },
    }


def _lot_recommendation(step, vsm):
    """Compute lot size recommendation for a single process step."""
    ct = step.get("cycle_time", 0) or 0
    co = step.get("changeover_time", 0) or 0
    batch = step.get("batch_size") or 0
    batch_process = step.get("batch_process", False)  # oven, furnace, plating tank, etc.
    uptime = (step.get("uptime", 100) or 100) / 100
    shifts = step.get("shifts", 1) or 1

    # Detect display unit from cycle time — all time displays normalized to this
    time_unit, time_div = _detect_time_unit(ct)

    # Step-level demand takes priority over VSM-level
    step_demand = step.get("demand_rate") or 0
    step_demand_unit = (step.get("demand_unit") or "").lower()
    demand_per_day = 0

    if step_demand:
        demand_per_day = float(step_demand)
        if "month" in step_demand_unit:
            demand_per_day /= 22
        elif "week" in step_demand_unit:
            demand_per_day /= 5
        elif "year" in step_demand_unit or "annual" in step_demand_unit:
            demand_per_day /= 260
    else:
        # Fall back to VSM-level customer demand
        demand_str = vsm.customer_demand or ""
        try:
            import re

            nums = re.findall(r"[\d.]+", demand_str)
            if nums:
                demand_per_day = float(nums[0])
                low = demand_str.lower()
                if "/month" in low or "month" in low:
                    demand_per_day /= 22
                elif "/week" in low or "week" in low:
                    demand_per_day /= 5
                elif "/year" in low or "annual" in low:
                    demand_per_day /= 260
        except (ValueError, IndexError):
            pass

    takt = vsm.takt_time or 0
    available_sec = 28800 * shifts  # 8hr × shifts

    # Count parts in this value stream as proxy for mix
    steps = vsm.process_steps or []
    # Use number of unique batch_size values or default
    mix_count = max(1, len(steps))

    # Detect regime
    regime = _detect_regime(demand_per_day, mix_count, co, ct, available_sec)

    result = {
        "step_name": step.get("name", ""),
        "regime": regime,
        "demand_per_day": round(demand_per_day, 2),
        "takt_time": takt,
        "cycle_time": ct,
        "changeover_time": co,
        "current_batch": batch,
    }

    r = regime["regime"]

    # Batch process analysis — applies to ALL regimes
    if batch_process and batch > 0:
        effective_ct = ct / batch if batch > 0 else ct
        if demand_per_day <= 0:
            # No demand data — just report the batch as the lot
            result["recommendation"] = {
                "lot_size": batch,
                "reasoning": (
                    f"Batch process — equipment processes {batch} units in {_fmt_time(ct)}. "
                    f"Effective throughput: {_fmt_time(effective_ct)}/unit. "
                    f"Set step demand to enable scenario analysis."
                ),
                "takt_vs_ct": _build_takt_vs_ct(takt, ct, effective_ct, time_unit, time_div),
            }
            return result

        # Full scenario analysis with demand data
        holding_cost = 10.0 if ct > 3600 else 0.10 if ct > 60 else 0.001
        setup_cost = (co / 3600) * 50  # changeover hours × $50/hr

        # Analyze scenarios: full batch, partial batches, optimal
        scenarios = []
        # Scenario range: from demand-matched small batch up to full capacity
        # Minimum viable batch: need at least 1 day of demand or 1 unit
        min_batch = max(1, math.ceil(demand_per_day))
        test_sizes = sorted(
            set(
                [
                    min_batch,
                    max(1, round(demand_per_day * 5)),  # ~weekly
                    max(1, round(demand_per_day * 10)),  # ~biweekly
                    max(1, round(demand_per_day * 22)),  # ~monthly
                    batch,  # current/full capacity
                ]
            )
        )
        # Add EPQ if meaningful
        if setup_cost > 0 and holding_cost > 0:
            epq_lot = max(1, round(math.sqrt((2 * demand_per_day * setup_cost) / holding_cost)))
            test_sizes = sorted(set(test_sizes + [epq_lot]))

        for lot in test_sizes:
            if lot < 1:
                continue
            batches_per_month = (demand_per_day * 22) / lot
            days_of_supply = lot / demand_per_day
            avg_wip = lot / 2
            daily_hold = avg_wip * holding_cost
            daily_setup = (demand_per_day / lot) * setup_cost
            daily_total = daily_hold + daily_setup
            utilization = (batches_per_month * (ct + co)) / (available_sec * 22) * 100

            scenarios.append(
                {
                    "lot_size": lot,
                    "is_current": lot == batch,
                    "is_epq": lot == epq_lot if setup_cost > 0 else False,
                    "label": "full capacity"
                    if lot == batch
                    else "EPQ optimal"
                    if setup_cost > 0 and lot == epq_lot
                    else f"~{days_of_supply:.0f} day supply",
                    "days_of_supply": round(days_of_supply, 1),
                    "batches_per_month": round(batches_per_month, 1),
                    "avg_wip": round(avg_wip),
                    "daily_cost": round(daily_total, 2),
                    "daily_holding": round(daily_hold, 2),
                    "daily_setup": round(daily_setup, 2),
                    "equipment_utilization_pct": round(utilization, 1),
                }
            )

        # Find lowest-cost scenario
        best = min(scenarios, key=lambda s: s["daily_cost"])
        current = next((s for s in scenarios if s["is_current"]), scenarios[-1])

        savings_vs_current = round(current["daily_cost"] - best["daily_cost"], 2)
        wip_reduction = current["avg_wip"] - best["avg_wip"]

        result["recommendation"] = {
            "lot_size": best["lot_size"],
            "reasoning": (
                f"Optimal batch: {best['lot_size']} units ({best['days_of_supply']:.0f} days supply, "
                f"{best['batches_per_month']:.1f} runs/month). "
                f"Current: {batch} units ({current['days_of_supply']:.0f} days supply). "
                + (
                    f"Switching saves ${savings_vs_current:.0f}/day and reduces avg WIP by {wip_reduction} units."
                    if savings_vs_current > 0.5
                    else "Current batch is near-optimal for this cost structure."
                )
            ),
            "takt_vs_ct": _build_takt_vs_ct(takt, ct, effective_ct, time_unit, time_div),
            "scenarios": scenarios,
            "best_scenario": best,
            "current_scenario": current,
            "savings_per_day": savings_vs_current,
            "wip_reduction": wip_reduction,
        }

        # Kanban for recommended lot
        rec_lot = best["lot_size"]
        replenishment_sec = ct + co
        replenishment_days = replenishment_sec / available_sec if available_sec > 0 else 0
        safety_factor = 0.2 if uptime > 0.9 else 0.5
        kanban_qty = math.ceil(demand_per_day * replenishment_days * (1 + safety_factor))
        kanban_cards = max(1, math.ceil(kanban_qty / rec_lot))
        result["kanban"] = {
            "container_size": rec_lot,
            "kanban_cards": kanban_cards,
            "total_units_in_loop": kanban_cards * rec_lot,
            "replenishment_time": _fmt_time(replenishment_sec),
            "safety_factor": safety_factor,
            "reasoning": (
                f"Batch of {rec_lot} takes {_fmt_time(replenishment_sec)} to process + changeover. "
                f"At {demand_per_day:.1f}/day, need {kanban_cards} kanban card(s) "
                f"({kanban_cards * rec_lot} units in loop)."
            ),
        }

        return result

    if r == "schedule_paced":
        # Lot = 1 for flow processes (batch_process handled by early return above)
        result["recommendation"] = {
            "lot_size": 1,
            "reasoning": "Schedule-paced regime — demand is low enough to build one-piece flow or in contract quantities.",
            "takt_vs_ct": _build_takt_vs_ct(takt, ct, None, time_unit, time_div),
            "pitch": {
                "value": round(takt / 60, 1) if takt else None,
                "unit": "min",
                "meaning": "Material withdrawal interval = takt (one piece at a time)",
            },
            "epei": "N/A — single-piece flow, no batching",
        }
        if batch and batch > 1:
            result["recommendation"]["batch_warning"] = (
                f"Current batch size is {batch}. In a schedule-paced environment, "
                f"this adds {batch - 1} units of WIP without value. "
                f"Target lot size: 1."
            )

    elif r == "mix_constrained":
        # EPEI-first: target EPEI, derive lot size
        # Default: runners daily, repeaters every 3 days
        target_epei_days = 1.0 if demand_per_day > 10 else 3.0
        lot = max(1, round(demand_per_day * target_epei_days))
        co_per_day = demand_per_day / lot if lot > 0 else 0
        daily_co_min = co_per_day * (co / 60)
        co_pct = (daily_co_min / (available_sec / 60)) * 100

        # Feasibility check
        prod_min = (demand_per_day * ct) / 60
        total_min = prod_min + daily_co_min
        feasible = total_min <= available_sec / 60

        result["recommendation"] = {
            "lot_size": lot,
            "reasoning": f"Mix-constrained regime — {mix_count} parts cycling. EPEI target: {target_epei_days} days.",
            "epei_days": target_epei_days,
            "changeovers_per_day": round(co_per_day, 1),
            "changeover_pct": round(co_pct, 1),
            "feasible": feasible,
        }
        if not feasible:
            # Compute SMED target for feasibility
            available_for_co = (available_sec / 60) - prod_min
            if available_for_co > 0 and co_per_day > 0:
                max_co_min = available_for_co / co_per_day
                result["recommendation"]["smed_target"] = {
                    "current_co_min": round(co / 60, 1),
                    "required_co_min": round(max_co_min, 1),
                    "reduction_pct": round((1 - max_co_min / (co / 60)) * 100, 0) if co > 0 else 0,
                }
            # Fallback EPEI
            fallback_epei = 5.0
            fallback_lot = max(1, round(demand_per_day * fallback_epei))
            fb_co_per_day = demand_per_day / fallback_lot
            fb_co_min = fb_co_per_day * (co / 60)
            fb_feasible = (prod_min + fb_co_min) <= available_sec / 60
            result["recommendation"]["fallback"] = {
                "epei_days": fallback_epei,
                "lot_size": fallback_lot,
                "feasible": fb_feasible,
            }

    elif r == "capacity_budget":
        # Budget-first: allocate changeover budget, derive lot size
        budget_pct = 0.10  # 10% default
        budget_min = (available_sec / 60) * budget_pct
        max_co = budget_min / (co / 60) if co > 0 else float("inf")
        lot = max(1, round(demand_per_day / max_co)) if max_co > 0 else max(1, round(demand_per_day))
        actual_co = demand_per_day / lot if lot > 0 else 0
        actual_co_min = actual_co * (co / 60)
        actual_pct = (actual_co_min / (available_sec / 60)) * 100
        epei_days = (lot / demand_per_day * mix_count) if demand_per_day > 0 else 0
        avg_wip = lot / 2

        # SMED impact
        half_co = co / 2
        half_max_co = budget_min / (half_co / 60) if half_co > 0 else float("inf")
        half_lot = max(1, round(demand_per_day / half_max_co)) if half_max_co > 0 else 1

        result["recommendation"] = {
            "lot_size": lot,
            "reasoning": f"Capacity-budget regime — {budget_pct * 100:.0f}% changeover budget.",
            "changeover_budget_pct": budget_pct * 100,
            "changeovers_per_day": round(actual_co, 1),
            "changeover_pct_actual": round(actual_pct, 1),
            "epei_days": round(epei_days, 1),
            "avg_wip": round(avg_wip),
            "smed_impact": {
                "current_co_sec": co,
                "if_halved": {
                    "lot_size": half_lot,
                    "wip_reduction_pct": round((1 - half_lot / lot) * 100) if lot > 0 else 0,
                },
            },
        }

    # Kanban lot sizing: units in the pull loop between this step and downstream
    if takt and ct and demand_per_day > 0:
        # Replenishment time = time to produce one lot + changeover
        lot_for_kanban = result.get("recommendation", {}).get("lot_size", 1)
        replenishment_sec = (lot_for_kanban * ct) + co
        replenishment_days = replenishment_sec / available_sec if available_sec > 0 else 0
        safety_factor = 0.2 if uptime > 0.9 else 0.5  # more safety for less reliable
        # Kanban qty = demand_during_replenishment × (1 + safety)
        kanban_qty = math.ceil(demand_per_day * replenishment_days * (1 + safety_factor))
        container_size = lot_for_kanban or 1
        kanban_cards = math.ceil(kanban_qty / container_size) if container_size > 0 else 1

        result["kanban"] = {
            "container_size": container_size,
            "kanban_cards": max(1, kanban_cards),
            "total_units_in_loop": kanban_cards * container_size,
            "replenishment_time": _fmt_time(replenishment_sec),
            "safety_factor": safety_factor,
            "reasoning": (
                f"Lot of {container_size} takes {_fmt_time(replenishment_sec)} to replenish. "
                f"At {demand_per_day:.1f}/day, need {kanban_cards} kanban cards "
                f"({kanban_cards * container_size} units in loop) with {safety_factor:.0%} safety."
            ),
        }

    # EPQ reference (always include for comparison)
    holding_cost = 0.01  # default $/unit/day — conservative
    if ct > 3600:
        holding_cost = 10.0  # aerospace / heavy mfg
    elif ct > 60:
        holding_cost = 0.10  # medium
    setup_cost = (co / 3600) * 50  # rough: changeover hours × $50/hr
    if demand_per_day > 0 and holding_cost > 0 and setup_cost > 0:
        epq = math.sqrt((2 * demand_per_day * setup_cost) / holding_cost)
        result["epq_reference"] = {
            "lot_size": max(1, round(epq)),
            "note": "Classic EPQ — shown for reference. May not be appropriate for this regime.",
        }

    return result


def _fmt_time(seconds):
    """Format seconds to human-readable, auto-detecting unit."""
    if not seconds:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hrs"


def _detect_time_unit(seconds):
    """Detect the natural display unit for a value in seconds."""
    if not seconds:
        return "sec", 1
    if seconds >= 3600:
        return "hrs", 3600
    if seconds >= 60:
        return "min", 60
    return "sec", 1


def _fmt_time_in(seconds, unit, divisor):
    """Format seconds into a specific unit."""
    if not seconds:
        return "-"
    val = seconds / divisor
    if val >= 100:
        return f"{val:.0f} {unit}"
    if val >= 10:
        return f"{val:.1f} {unit}"
    return f"{val:.2f} {unit}"


def _build_takt_vs_ct(takt, ct, effective_ct, time_unit, time_div):
    """Build takt-vs-CT display dict with all values in the step's time unit.

    ct = actual cycle time (e.g. 6 hrs for oven)
    effective_ct = throughput rate per unit (e.g. 7.5 min/unit for 48 gears in 6 hrs)
    Ratio uses effective_ct for takt comparison (can this step keep up?)
    Display shows both: CT as labeled, effective rate separately if different.
    """
    compare_ct = effective_ct if effective_ct and effective_ct != ct else ct
    ratio = round(compare_ct / takt, 4) if takt and compare_ct else None

    result = {
        "takt_sec": takt,
        "takt_display": _fmt_time_in(takt, time_unit, time_div) if takt else "not set",
        "ct_sec": ct,
        "ct_display": _fmt_time_in(ct, time_unit, time_div),
        "ratio": ratio,
        "unit": time_unit,
        "assessment": _takt_ct_assessment(takt, compare_ct),
    }
    if effective_ct and effective_ct != ct:
        eff_unit, eff_div = _detect_time_unit(effective_ct)
        result["effective_ct_sec"] = effective_ct
        result["effective_ct_display"] = _fmt_time_in(effective_ct, eff_unit, eff_div) + "/unit"
    return result


def _takt_ct_assessment(takt, ct):
    """Assess takt vs cycle time relationship."""
    if not takt or not ct:
        return "Set takt time and cycle time to assess."
    ratio = ct / takt
    if ratio < 0.5:
        return (
            f"CT is {ratio:.1%} of takt — significant free capacity. Consider multi-machine operation or rebalancing."
        )
    if ratio < 0.85:
        return f"CT is {ratio:.1%} of takt — healthy margin for variation."
    if ratio < 1.0:
        return f"CT is {ratio:.1%} of takt — tight. Monitor for variation-induced misses."
    if ratio < 1.2:
        return (
            f"CT exceeds takt by {(ratio - 1) * 100:.0f}% — cannot meet demand without overtime or parallel capacity."
        )
    return f"CT is {ratio:.1f}× takt — major capacity gap. Need {math.ceil(ratio)} parallel stations or fundamental process redesign."


@gated_paid
@require_http_methods(["GET"])
def lot_recommendation(request, vsm_id, step_id):
    """Get lot size recommendation for a specific process step."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    steps = vsm.process_steps or []
    step = next((s for s in steps if s.get("id") == step_id), None)
    if not step:
        return JsonResponse({"error": "Step not found"}, status=404)

    result = _lot_recommendation(step, vsm)
    return JsonResponse(result)
