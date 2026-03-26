"""Value Stream Map API views.

VSM is a lean manufacturing tool for visualizing process flow.
Unlike a general whiteboard, VSM has structured elements:
- Process steps with cycle time, changeover time, uptime
- Inventory triangles between steps
- Information and material flow
- Timeline showing value-add vs non-value-add time
"""

import copy
import json
import logging
import time as _time

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid, require_feature
from core.models import Project

from .hoshin_calculations import estimate_savings_monte_carlo
from .models import HoshinProject, ValueStreamMap

logger = logging.getLogger(__name__)


def detect_bottleneck(vsm):
    """Identify bottleneck step and set flags on process_steps.

    Sets per-step flags: is_bottleneck, takt_ratio, exceeds_takt.
    Returns bottleneck summary dict or None if no steps.
    """
    steps = vsm.process_steps or []
    if not steps:
        return None

    wc_steps = {}  # work_center_id -> [(step_dict, ct)]
    standalone = []

    for step in steps:
        ct = step.get("cycle_time", 0) or 0
        wc_id = step.get("work_center_id")
        if wc_id:
            wc_steps.setdefault(wc_id, []).append((step, ct))
        else:
            standalone.append((step, ct))

    # Build effective station list: (step_dict, effective_ct)
    effective = []
    for step, ct in standalone:
        effective.append((step, ct))
    for wc_id, members in wc_steps.items():
        rate_sum = sum(1.0 / ct for _, ct in members if ct > 0)
        eff_ct = (1.0 / rate_sum) if rate_sum > 0 else 0
        # Attribute to first member for display
        effective.append((members[0][0], eff_ct))

    valid = [(s, ct) for s, ct in effective if ct > 0]
    if not valid:
        # Clear flags if no valid cycle times
        for step in steps:
            step["flags"] = {}
        return None

    max_ct = max(ct for _, ct in valid)
    bottleneck_step = next(s for s, ct in valid if ct == max_ct)
    throughput = 3600.0 / max_ct if max_ct > 0 else 0
    takt = vsm.takt_time

    # Set flags on every step
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


@gated_paid
@require_http_methods(["GET"])
def list_vsm(request):
    """List user's value stream maps.

    Query params:
    - project_id: filter by project
    - status: filter by status (current/future/archived)
    """
    maps = ValueStreamMap.objects.filter(owner=request.user).select_related("project")

    project_id = request.GET.get("project_id")
    if project_id:
        maps = maps.filter(project_id=project_id)

    status = request.GET.get("status")
    if status:
        maps = maps.filter(status=status)

    return JsonResponse(
        {
            "maps": [m.to_dict() for m in maps[:50]],
        }
    )


@gated_paid
@require_http_methods(["POST"])
def create_vsm(request):
    """Create a new value stream map."""
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Optional project link
    project = None
    project_id = data.get("project_id")
    if project_id:
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found"}, status=404)

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

    return JsonResponse(
        {
            "id": str(vsm.id),
            "vsm": vsm.to_dict(),
        }
    )


@gated_paid
@require_http_methods(["GET"])
def get_vsm(request, vsm_id):
    """Get a single VSM with full details."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)
    bottleneck_info = detect_bottleneck(vsm)

    return JsonResponse(
        {
            "vsm": vsm.to_dict(),
            "bottleneck": bottleneck_info,
        }
    )


@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_vsm(request, vsm_id):
    """Update a VSM."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Validate takt_time if provided
    if "takt_time" in data and data["takt_time"] is not None:
        try:
            tt = float(data["takt_time"])
            if tt <= 0:
                return JsonResponse({"error": "Takt time must be positive"}, status=400)
            data["takt_time"] = tt
        except (ValueError, TypeError):
            return JsonResponse({"error": "Takt time must be a number"}, status=400)

    # Update simple fields
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

    # Update structured data
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

    # Update project link
    if "project_id" in data:
        project_id = data["project_id"]
        if project_id:
            try:
                project = Project.objects.get(id=project_id, user=request.user)
                vsm.project = project
            except Project.DoesNotExist:
                pass
        else:
            vsm.project = None

    # Handle auto_kaizen from calculator exports
    auto_kaizen = data.get("auto_kaizen")
    if auto_kaizen and isinstance(auto_kaizen, dict):
        text = auto_kaizen.get("text", "").strip()
        near_step = auto_kaizen.get("near_step", "")
        priority = auto_kaizen.get("priority", "medium")
        if text:
            bursts = vsm.kaizen_bursts or []
            # Dedup: skip if burst with same text already exists
            if not any(b.get("text") == text for b in bursts):
                # Find position near the named step
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

    # Recalculate metrics and detect bottleneck
    vsm.calculate_metrics()
    bottleneck_info = detect_bottleneck(vsm)
    vsm.save()

    return JsonResponse(
        {
            "success": True,
            "vsm": vsm.to_dict(),
            "bottleneck": bottleneck_info,
        }
    )


@gated_paid
@require_http_methods(["DELETE"])
def delete_vsm(request, vsm_id):
    """Delete a VSM."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)
    vsm.delete()

    return JsonResponse({"success": True})


@gated_paid
@require_http_methods(["POST"])
def add_process_step(request, vsm_id):
    """Add a process step to the VSM.

    Request body:
    {
        "name": "Assembly",
        "x": 200,
        "y": 300,
        "cycle_time": 45,       // seconds
        "changeover_time": 600, // seconds
        "uptime": 95,           // percent
        "operators": 2,
        "shifts": 2,
        "batch_size": 10
    }
    """
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    import uuid

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

    return JsonResponse(
        {
            "success": True,
            "step": step,
            "vsm": vsm.to_dict(),
        }
    )


@gated_paid
@require_http_methods(["POST"])
def add_inventory(request, vsm_id):
    """Add inventory triangle between process steps.

    Request body:
    {
        "before_step_id": "abc123",  // Process step this inventory feeds
        "quantity": 500,
        "days_of_supply": 2.5,
        "x": 150,
        "y": 350
    }
    """
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    import uuid

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

    return JsonResponse(
        {
            "success": True,
            "inventory": inv,
            "vsm": vsm.to_dict(),
        }
    )


@gated_paid
@require_http_methods(["POST"])
def add_kaizen_burst(request, vsm_id):
    """Add a kaizen burst (improvement opportunity).

    Request body:
    {
        "x": 300,
        "y": 200,
        "text": "Reduce changeover time",
        "priority": "high"  // high, medium, low
    }
    """
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    import uuid

    burst = {
        "id": str(uuid.uuid4())[:8],
        "x": data.get("x", 100),
        "y": data.get("y", 100),
        "text": data.get("text", ""),
        "priority": data.get("priority", "medium"),
    }

    vsm.kaizen_bursts.append(burst)
    vsm.save()

    return JsonResponse(
        {
            "success": True,
            "burst": burst,
            "vsm": vsm.to_dict(),
        }
    )


@gated_paid
@require_http_methods(["POST"])
def create_future_state(request, vsm_id):
    """Create a future state VSM from the current state.

    Copies all elements to a new VSM marked as 'future' status.
    """
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    # Create copy with future state status
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

    # Set pairing: future points to current (one-to-one, one direction)
    future.paired_with = vsm
    future.save(update_fields=["paired_with"])

    return JsonResponse(
        {
            "success": True,
            "future_state": future.to_dict(),
        }
    )


@gated_paid
@require_http_methods(["GET"])
def compare_vsm(request, vsm_id):
    """Compare current and future state VSMs.

    Returns metrics comparison if a future state exists.
    """
    current = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    # Find related future state (same project, same product family)
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
        return JsonResponse(
            {
                "current": current.to_dict(),
                "future": None,
                "comparison": None,
            }
        )

    # Calculate improvements
    comparison = {
        "lead_time": {
            "current": current.total_lead_time,
            "future": future.total_lead_time,
            "improvement": (
                (
                    ((current.total_lead_time or 0) - (future.total_lead_time or 0))
                    / (current.total_lead_time or 1)
                    * 100
                )
                if current.total_lead_time
                else 0
            ),
        },
        "process_time": {
            "current": current.total_process_time,
            "future": future.total_process_time,
            "improvement": (
                (
                    (
                        (current.total_process_time or 0)
                        - (future.total_process_time or 0)
                    )
                    / (current.total_process_time or 1)
                    * 100
                )
                if current.total_process_time
                else 0
            ),
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

    return JsonResponse(
        {
            "current": current.to_dict(),
            "future": future.to_dict(),
            "comparison": comparison,
        }
    )


# =============================================================================
# Intelligence Layer — Phase 3: TIMWOODS Waste Analysis
# =============================================================================


@gated_paid
@require_http_methods(["GET"])
def waste_analysis(request, vsm_id):
    """Classify TIMWOODS waste categories from VSM process metrics.

    Rule-based analysis — no LLM required. Uses configurable thresholds.
    """
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    steps = vsm.process_steps or []
    inventory = vsm.inventory or []
    material_flow = vsm.material_flow or []
    info_flow = vsm.information_flow or []

    # Calculate takt time if available
    takt_time = getattr(vsm, "takt_time", None) or 0

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

    # --- Inventory waste: high days of supply ---
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

        # --- Waiting: high changeover relative to cycle time ---
        if ct > 0 and changeover > 10 * ct:
            waste["waiting"].append(
                {
                    "step": name,
                    "detail": f"changeover {changeover}s vs cycle time {ct}s (ratio {changeover / ct:.0f}x)",
                    "severity": "high",
                    "suggested_kaizen": "SMED changeover reduction",
                }
            )

        # --- Waiting: CT exceeds takt time ---
        if takt_time > 0 and ct > 2 * takt_time:
            waste["waiting"].append(
                {
                    "step": name,
                    "detail": f"cycle time {ct}s exceeds 2× takt ({takt_time}s)",
                    "severity": "high",
                }
            )

        # --- Defects: low uptime ---
        if uptime < 85:
            severity = "high" if uptime < 70 else "medium"
            waste["defects"].append(
                {
                    "step": name,
                    "detail": f"uptime {uptime}%",
                    "severity": severity,
                    "suggested_kaizen": "TPM (Total Productive Maintenance)",
                }
            )

        # --- Overproduction: large batch sizes ---
        if batch_size > 50:
            waste["overproduction"].append(
                {
                    "step": name,
                    "detail": f"batch size {batch_size}",
                    "severity": "medium" if batch_size < 200 else "high",
                }
            )

    # --- Overproduction: too many push flows ---
    push_count = sum(1 for mf in material_flow if mf.get("type") == "push")
    if push_count > 2:
        waste["overproduction"].append(
            {
                "step": "Material flow",
                "detail": f"{push_count} push connections (consider pull/kanban)",
                "severity": "medium",
            }
        )

    # --- Motion/Transport: manual information flows ---
    manual_count = sum(1 for inf in info_flow if inf.get("type") == "manual")
    if manual_count > 0:
        waste["motion"].append(
            {
                "step": "Information flow",
                "detail": f"{manual_count} manual information flows",
                "severity": "low" if manual_count < 3 else "medium",
            }
        )

    # --- Overprocessing: very low PCE ---
    pce = vsm.pce or 0
    if pce > 0 and pce < 5:
        waste["overprocessing"].append(
            {
                "step": "Overall",
                "detail": f"PCE {pce:.1f}% — less than 5% of lead time is value-adding",
                "severity": "high",
            }
        )

    # Count total and build top opportunities
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


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def generate_proposals(request, vsm_id):
    """Auto-propose CI projects from VSM kaizen bursts.

    Diffs current vs future state VSM, matches kaizen bursts to process
    steps, and estimates savings from metric deltas.

    Request body (optional):
    {
        "annual_volume": 100000,
        "cost_per_unit": 50.0
    }

    Returns list of proposals for user review before creating projects.
    """
    import math

    current = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    # Find future state
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
        return JsonResponse(
            {
                "error": "No future state VSM found. Create a future state first.",
            },
            status=400,
        )

    bursts = future.kaizen_bursts or []
    if not bursts:
        return JsonResponse(
            {
                "error": "No kaizen bursts on the future state VSM.",
            },
            status=400,
        )

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

        # Find nearest future-state process step by distance
        nearest_step = None
        min_dist = float("inf")
        for step in future_steps:
            sx = float(step.get("x", 0))
            sy = float(step.get("y", 0))
            dist = math.sqrt((burst_x - sx) ** 2 + (burst_y - sy) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_step = step

        if not nearest_step:
            continue

        step_name = nearest_step.get("name", "Unknown")

        # Match to current-state step by name
        current_step = current_steps.get(step_name, {})

        # Estimate savings from metric deltas (Monte Carlo for confidence intervals)
        if current_step:
            estimate = estimate_savings_monte_carlo(
                current_step,
                nearest_step,
                annual_volume=annual_volume,
                cost_per_unit=cost_per_unit,
                labor_rate=labor_rate,
            )
            estimate["estimated_annual_savings"] = estimate["median_savings"]
        else:
            estimate = {
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
                "lower_25": 0,
                "upper_75": 0,
                "p_positive": 0,
                "mean_savings": 0,
                "std_savings": 0,
                "deterministic": 0,
            }

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
                "estimated_annual_savings": estimate["estimated_annual_savings"],
                "suggested_method": estimate["suggested_method"],
                "improvement_pct": estimate["improvement_pct"],
                "median_savings": estimate.get("median_savings", 0),
                "lower_5": estimate.get("lower_5", 0),
                "upper_95": estimate.get("upper_95", 0),
                "p_positive": estimate.get("p_positive", 0),
                "suggested_title": f"{burst_text} — {step_name}",
                "suggested_type": (
                    "labor"
                    if estimate["suggested_method"] in ("time_reduction", "headcount")
                    else "material"
                ),
            }
        )

    return JsonResponse(
        {
            "vsm_id": str(vsm_id),
            "vsm_name": current.name,
            "future_vsm_id": str(future.id),
            "proposals": proposals,
            "count": len(proposals),
            "defaults": {
                "annual_volume": annual_volume,
                "cost_per_unit": cost_per_unit,
                "labor_rate": labor_rate,
            },
        }
    )


@require_feature("hoshin_kanri")
@require_http_methods(["POST"])
def approve_proposal(request, vsm_id):
    """Approve a VSM DOWNTIME waste proposal → auto-create HoshinProject with pre-calculated ROI.

    POST body: {
        "burst_id": "string",
        "title": "optional override",
        "project_type": "material|labor|quality|...",
        "annual_savings_target": float,
        "calculation_method": "direct|time_reduction|headcount",
    }
    """
    from core.models.project import Project

    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    data = json.loads(request.body) if request.body else {}
    burst_id = data.get("burst_id", "")

    # Find the burst in VSM kaizen_bursts
    burst = None
    for b in vsm.kaizen_bursts or []:
        if str(b.get("id", "")) == burst_id:
            burst = b
            break

    # Duplicate check — prevent approving same burst twice
    if burst_id:
        existing = HoshinProject.objects.filter(
            source_vsm=vsm, source_burst_id=burst_id
        ).first()
        if existing:
            return JsonResponse(
                {
                    "error": "This proposal has already been approved",
                    "hoshin_id": str(existing.id),
                },
                status=409,
            )

    title = data.get("title", "").strip()
    if not title:
        burst_text = (
            burst.get("text", "VSM Improvement") if burst else "VSM Improvement"
        )
        title = f"CI — {burst_text}"[:300]

    project = Project.objects.create(title=title, user=request.user)

    # VSM may not have a site field — derive from project context
    site = getattr(vsm, "site", None)

    hoshin = HoshinProject.objects.create(
        project=project,
        site=site,
        project_class=data.get("project_class", "kaizen"),
        project_type=data.get("project_type", "material"),
        opportunity="budgeted_new",
        hoshin_status="proposed",
        annual_savings_target=data.get("annual_savings_target", 0),
        calculation_method=data.get("calculation_method", "direct"),
        source_vsm=vsm,
        source_burst_id=burst_id,
    )

    logger.info(
        "VSM %s: proposal approved → HoshinProject %s (savings=$%s)",
        vsm.id,
        hoshin.id,
        hoshin.annual_savings_target,
    )

    return JsonResponse(
        {
            "success": True,
            "project_id": str(project.id),
            "hoshin_id": str(hoshin.id),
            "title": title,
            "annual_savings_target": float(hoshin.annual_savings_target),
        },
        status=201,
    )
