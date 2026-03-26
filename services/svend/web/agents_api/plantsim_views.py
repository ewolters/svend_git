"""Plant Simulator API views.

Discrete-event simulation of factory/plant layouts. Users build layouts
with machines, buffers, sources, and sinks. The DES engine runs client-side;
the server stores layouts and results for persistence and comparison.
"""

import json
import logging

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from core.models import Evidence, Project

from .models import PlantSimulation, ValueStreamMap

logger = logging.getLogger(__name__)

MAX_RESULTS = 20  # Cap stored simulation runs per layout


@gated_paid
@require_http_methods(["GET"])
def list_simulations(request):
    """List user's plant simulations.

    Query params:
    - project_id: filter by linked project
    """
    sims = PlantSimulation.objects.filter(owner=request.user)

    project_id = request.GET.get("project_id")
    if project_id:
        sims = sims.filter(project_id=project_id)

    return JsonResponse(
        {
            "simulations": [s.to_dict() for s in sims[:50]],
        }
    )


@gated_paid
@require_http_methods(["POST"])
def create_simulation(request):
    """Create a new plant simulation."""
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    project = None
    project_id = data.get("project_id")
    if project_id:
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found"}, status=404)

    sim = PlantSimulation.objects.create(
        owner=request.user,
        project=project,
        name=data.get("name", "Untitled Plant"),
        description=data.get("description", ""),
        simulation_config=data.get(
            "simulation_config",
            {
                "warmup_time": 300,
                "run_time": 3600,
                "speed_factor": 10,
                "random_seed": None,
            },
        ),
    )

    return JsonResponse(
        {
            "id": str(sim.id),
            "simulation": sim.to_dict(),
        }
    )


@gated_paid
@require_http_methods(["GET"])
def get_simulation(request, sim_id):
    """Get a single simulation with full layout and results."""
    sim = get_object_or_404(PlantSimulation, id=sim_id, owner=request.user)

    return JsonResponse(
        {
            "simulation": sim.to_dict(),
        }
    )


@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_simulation(request, sim_id):
    """Update simulation layout, config, or canvas state.

    Accepts partial updates — only provided fields are changed.
    """
    sim = get_object_or_404(PlantSimulation, id=sim_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Simple fields
    for field in ["name", "description", "zoom", "pan_x", "pan_y"]:
        if field in data:
            setattr(sim, field, data[field])

    # Structured layout data
    for field in [
        "stations",
        "connections",
        "sources",
        "sinks",
        "work_centers",
        "simulation_config",
    ]:
        if field in data:
            setattr(sim, field, data[field])

    # Project link
    if "project_id" in data:
        project_id = data["project_id"]
        if project_id:
            try:
                project = Project.objects.get(id=project_id, user=request.user)
                sim.project = project
            except Project.DoesNotExist:
                pass
        else:
            sim.project = None

    sim.save()

    return JsonResponse(
        {
            "simulation": sim.to_dict(),
        }
    )


@gated_paid
@require_http_methods(["DELETE"])
def delete_simulation(request, sim_id):
    """Delete a plant simulation."""
    sim = get_object_or_404(PlantSimulation, id=sim_id, owner=request.user)
    sim.delete()

    return JsonResponse({"success": True})


@gated_paid
@require_http_methods(["POST"])
def save_results(request, sim_id):
    """Save simulation run results from the client.

    Client POSTs the complete results object after a run completes.
    Server appends to simulation_results (capped at MAX_RESULTS)
    and adds a metric snapshot.
    """
    sim = get_object_or_404(PlantSimulation, id=sim_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    results = data.get("results")
    if not results:
        return JsonResponse({"error": "No results provided"}, status=400)

    # Append to results list, cap at MAX_RESULTS
    sim_results = sim.simulation_results or []
    sim_results.append(results)
    if len(sim_results) > MAX_RESULTS:
        sim_results = sim_results[-MAX_RESULTS:]
    sim.simulation_results = sim_results

    # Add metric snapshot
    snapshots = sim.metric_snapshots or []
    snapshots.append(
        {
            "run_index": len(sim_results) - 1,
            "throughput": results.get("throughput"),
            "avg_wip": results.get("avg_wip"),
            "avg_lead_time": results.get("avg_lead_time"),
            "bottleneck": results.get("bottleneck_station_name"),
            "station_count": len(sim.stations or []),
        }
    )
    if len(snapshots) > 100:
        snapshots = snapshots[-100:]
    sim.metric_snapshots = snapshots

    sim.save()

    return JsonResponse(
        {
            "success": True,
            "run_index": len(sim_results) - 1,
            "total_runs": len(sim_results),
        }
    )


@gated_paid
@require_http_methods(["POST"])
def import_from_vsm(request, sim_id):
    """Import layout from an existing VSM.

    Converts VSM process_steps to stations, inventory to buffer constraints,
    material_flow to connections. Auto-creates source and sink.
    """
    sim = get_object_or_404(PlantSimulation, id=sim_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    vsm_id = data.get("vsm_id")
    if not vsm_id:
        return JsonResponse({"error": "vsm_id required"}, status=400)

    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)

    # Convert process steps to stations
    stations = []
    step_ids = []
    for i, step in enumerate(vsm.process_steps or []):
        station = {
            "id": step.get("id", f"stn-{i}"),
            "type": "single",
            "name": step.get("name", f"Process {i + 1}"),
            "x": step.get("x", 200 + i * 200),
            "y": step.get("y", 300),
            "cycle_time": step.get("cycle_time", 30),
            "cycle_time_cv": 0.15,
            "changeover_time": step.get("changeover_time", 0),
            "changeover_frequency": 0,
            "uptime": step.get("uptime", 100),
            "mtbf": None,
            "mttr": None,
            "operators": step.get("operators", 1),
            "batch_size": step.get("batch_size", 1),
            "setup_time": 0,
            "work_center_id": step.get("work_center_id"),
        }
        stations.append(station)
        step_ids.append(station["id"])

    # Auto-create source
    source = {
        "id": "source-1",
        "name": vsm.supplier_name or "Source",
        "x": 50,
        "y": 300,
        "arrival_distribution": "exponential",
        "arrival_rate": vsm.takt_time or 60,
        "arrival_cv": 0.0,
        "batch_size": 1,
    }

    # Auto-create sink
    sink = {
        "id": "sink-1",
        "name": vsm.customer_name or "Sink",
        "x": 200 + len(stations) * 200,
        "y": 300,
    }

    # Build connections: source → stn[0] → stn[1] → ... → sink
    connections = []
    if step_ids:
        connections.append(
            {
                "id": "conn-source",
                "from_id": "source-1",
                "to_id": step_ids[0],
                "buffer_capacity": None,
            }
        )
        for i in range(len(step_ids) - 1):
            # Check if there's inventory between these steps
            buffer_cap = None
            for inv in vsm.inventory or []:
                if inv.get("before_step_id") == step_ids[i + 1]:
                    buffer_cap = inv.get("quantity")
                    break
            connections.append(
                {
                    "id": f"conn-{i}",
                    "from_id": step_ids[i],
                    "to_id": step_ids[i + 1],
                    "buffer_capacity": buffer_cap,
                }
            )
        connections.append(
            {
                "id": "conn-sink",
                "from_id": step_ids[-1],
                "to_id": "sink-1",
                "buffer_capacity": None,
            }
        )

    # Import work centers
    work_centers = []
    for wc in vsm.work_centers or []:
        work_centers.append(
            {
                "id": wc.get("id"),
                "name": wc.get("name", "Work Center"),
                "x": wc.get("x", 200),
                "y": wc.get("y", 200),
                "width": wc.get("width", 200),
                "height": wc.get("height", 150),
            }
        )

    sim.stations = stations
    sim.connections = connections
    sim.sources = [source]
    sim.sinks = [sink]
    sim.work_centers = work_centers
    sim.source_vsm = vsm
    sim.name = f"Sim: {vsm.name}"
    sim.save()

    return JsonResponse(
        {
            "simulation": sim.to_dict(),
            "imported_stations": len(stations),
        }
    )


@gated_paid
@require_http_methods(["POST"])
def export_to_project(request, sim_id):
    """Export simulation results as evidence to a core.Project."""
    sim = get_object_or_404(PlantSimulation, id=sim_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    project_id = data.get("project_id")
    if not project_id:
        return JsonResponse({"error": "project_id required"}, status=400)

    project = get_object_or_404(Project, id=project_id, user=request.user)

    results = sim.simulation_results[-1] if sim.simulation_results else None
    if not results:
        return JsonResponse({"error": "No simulation results to export"}, status=400)

    throughput = results.get("throughput", 0)
    bottleneck = results.get("bottleneck_station_name", "N/A")
    avg_lt = results.get("avg_lead_time", 0)
    avg_wip = results.get("avg_wip", 0)

    evidence = Evidence.objects.create(
        project=project,
        source_type="simulation",
        result_type="quantitative",
        title=f"Plant Simulation: {sim.name}",
        summary=(
            f"Throughput: {throughput:.1f}/hr, "
            f"Bottleneck: {bottleneck}, "
            f"Avg Lead Time: {avg_lt:.1f}s, "
            f"Avg WIP: {avg_wip:.1f}"
        ),
        raw_data=results,
        confidence=0.7,
    )

    return JsonResponse(
        {
            "success": True,
            "evidence_id": str(evidence.id),
        }
    )
