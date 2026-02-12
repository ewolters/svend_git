"""Value Stream Map API views.

VSM is a lean manufacturing tool for visualizing process flow.
Unlike a general whiteboard, VSM has structured elements:
- Process steps with cycle time, changeover time, uptime
- Inventory triangles between steps
- Information and material flow
- Timeline showing value-add vs non-value-add time
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from accounts.permissions import gated_paid, require_feature
from .models import ValueStreamMap
from .hoshin_calculations import estimate_savings_from_vsm_delta
from core.models import Project

logger = logging.getLogger(__name__)


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def list_vsm(request):
    """List user's value stream maps.

    Query params:
    - project_id: filter by project
    - status: filter by status (current/future/archived)
    """
    maps = ValueStreamMap.objects.filter(owner=request.user).select_related('project')

    project_id = request.GET.get("project_id")
    if project_id:
        maps = maps.filter(project_id=project_id)

    status = request.GET.get("status")
    if status:
        maps = maps.filter(status=status)

    return JsonResponse({
        "maps": [m.to_dict() for m in maps[:50]],
    })


@csrf_exempt
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
            return JsonResponse({"error": "Study not found"}, status=404)

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

    return JsonResponse({
        "id": str(vsm.id),
        "vsm": vsm.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def get_vsm(request, vsm_id):
    """Get a single VSM with full details."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    return JsonResponse({
        "vsm": vsm.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_vsm(request, vsm_id):
    """Update a VSM."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Update simple fields
    for field in ['name', 'status', 'product_family', 'customer_name', 'customer_demand',
                  'takt_time', 'supplier_name', 'supply_frequency', 'zoom', 'pan_x', 'pan_y']:
        if field in data:
            setattr(vsm, field, data[field])

    # Update structured data
    for field in ['process_steps', 'inventory', 'information_flow', 'material_flow', 'kaizen_bursts',
                  'customers', 'suppliers', 'work_centers']:
        if field in data:
            setattr(vsm, field, data[field])

    # Update project link
    if 'project_id' in data:
        project_id = data['project_id']
        if project_id:
            try:
                project = Project.objects.get(id=project_id, user=request.user)
                vsm.project = project
            except Project.DoesNotExist:
                pass
        else:
            vsm.project = None

    # Recalculate metrics
    vsm.calculate_metrics()
    vsm.save()

    return JsonResponse({
        "success": True,
        "vsm": vsm.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def delete_vsm(request, vsm_id):
    """Delete a VSM."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)
    vsm.delete()

    return JsonResponse({"success": True})


@csrf_exempt
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

    return JsonResponse({
        "success": True,
        "step": step,
        "vsm": vsm.to_dict(),
    })


@csrf_exempt
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

    return JsonResponse({
        "success": True,
        "inventory": inv,
        "vsm": vsm.to_dict(),
    })


@csrf_exempt
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

    return JsonResponse({
        "success": True,
        "burst": burst,
        "vsm": vsm.to_dict(),
    })


@csrf_exempt
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
        product_family=vsm.product_family,
        customer_name=vsm.customer_name,
        customer_demand=vsm.customer_demand,
        takt_time=vsm.takt_time,
        supplier_name=vsm.supplier_name,
        supply_frequency=vsm.supply_frequency,
        process_steps=vsm.process_steps.copy(),
        inventory=vsm.inventory.copy(),
        information_flow=vsm.information_flow.copy(),
        material_flow=vsm.material_flow.copy(),
        kaizen_bursts=vsm.kaizen_bursts.copy(),
        work_centers=vsm.work_centers.copy(),
        zoom=vsm.zoom,
        pan_x=vsm.pan_x,
        pan_y=vsm.pan_y,
    )

    future.calculate_metrics()
    future.save()

    return JsonResponse({
        "success": True,
        "future_state": future.to_dict(),
    })


@csrf_exempt
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
        future = ValueStreamMap.objects.filter(
            owner=request.user,
            project=current.project,
            status=ValueStreamMap.Status.FUTURE,
        ).exclude(id=current.id).first()

    if not future:
        return JsonResponse({
            "current": current.to_dict(),
            "future": None,
            "comparison": None,
        })

    # Calculate improvements
    comparison = {
        "lead_time": {
            "current": current.total_lead_time,
            "future": future.total_lead_time,
            "improvement": (
                ((current.total_lead_time or 0) - (future.total_lead_time or 0)) /
                (current.total_lead_time or 1) * 100
            ) if current.total_lead_time else 0,
        },
        "process_time": {
            "current": current.total_process_time,
            "future": future.total_process_time,
            "improvement": (
                ((current.total_process_time or 0) - (future.total_process_time or 0)) /
                (current.total_process_time or 1) * 100
            ) if current.total_process_time else 0,
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

    return JsonResponse({
        "current": current.to_dict(),
        "future": future.to_dict(),
        "comparison": comparison,
    })


@csrf_exempt
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
        future = ValueStreamMap.objects.filter(
            owner=request.user,
            project=current.project,
            status=ValueStreamMap.Status.FUTURE,
        ).exclude(id=current.id).first()

    if not future:
        return JsonResponse({
            "error": "No future state VSM found. Create a future state first.",
        }, status=400)

    bursts = future.kaizen_bursts or []
    if not bursts:
        return JsonResponse({
            "error": "No kaizen bursts on the future state VSM.",
        }, status=400)

    data = json.loads(request.body) if request.body else {}
    annual_volume = float(data.get("annual_volume", 100000))
    cost_per_unit = float(data.get("cost_per_unit", 50.0))

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

        # Estimate savings from metric deltas
        if current_step:
            estimate = estimate_savings_from_vsm_delta(
                current_step, nearest_step,
                annual_volume=annual_volume,
                cost_per_unit=cost_per_unit,
            )
        else:
            estimate = {
                "cycle_time_delta": 0, "changeover_delta": 0,
                "uptime_delta": 0, "operators_delta": 0,
                "estimated_annual_savings": 0,
                "suggested_method": "direct",
                "improvement_pct": 0,
            }

        proposals.append({
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
            "suggested_title": f"{burst_text} — {step_name}",
            "suggested_type": "labor" if estimate["suggested_method"] in ("time_reduction", "headcount") else "material",
        })

    return JsonResponse({
        "vsm_id": str(vsm_id),
        "vsm_name": current.name,
        "future_vsm_id": str(future.id),
        "proposals": proposals,
        "count": len(proposals),
        "defaults": {
            "annual_volume": annual_volume,
            "cost_per_unit": cost_per_unit,
        },
    })
