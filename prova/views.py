"""PROVA API views — operating graph, hypotheses, trials, conflicts, signals."""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth
from qms_core.permissions import require_tenant

from . import engine, integrations
from .bridge import FindingSpec, integrate
from .models import (
    Conflict,
    ConflictStatus,
    GraphEdge,
    GraphEdit,
    GraphNode,
    GraphVersion,
    PropagationSignal,
    ProvaHypothesis,
    Trial,
    TrialStatus,
    WorkingGraph,
)

logger = logging.getLogger("prova.views")


def _json_body(request):
    try:
        return json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return {}


# =============================================================================
# OPERATING GRAPH
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def operating_graph_detail(request):
    """Get the tenant's operating graph with nodes, edges, and stats."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)

    nodes = list(
        GraphNode.objects.filter(operating_graph=og).values(
            "id",
            "label",
            "node_type",
            "alpha",
            "beta",
            "metadata",
        )
    )
    edges = list(
        GraphEdge.objects.filter(operating_graph=og)
        .select_related(
            "source",
            "target",
        )
        .values(
            "id",
            "source_id",
            "target_id",
            "edge_type",
            "weight",
            "confidence",
            "conditions",
            "truth_frequency",
            "status",
            "cycle_length",
        )
    )

    # Annotate node confidence
    for n in nodes:
        total = n["alpha"] + n["beta"]
        n["confidence"] = n["alpha"] / total if total > 0 else 0.5
        n["id"] = str(n["id"])

    for e in edges:
        e["id"] = str(e["id"])
        e["source_id"] = str(e["source_id"])
        e["target_id"] = str(e["target_id"])

    # Active conflicts and signals
    conflicts = Conflict.objects.filter(
        operating_graph=og,
        status__in=[ConflictStatus.OPEN, ConflictStatus.INVESTIGATING],
    ).count()
    signals = PropagationSignal.objects.filter(
        edge__operating_graph=og,
        status="active",
    ).count()

    return JsonResponse(
        {
            "id": str(og.id),
            "version": og.current_version,
            "predictive_score": og.predictive_score,
            "last_evaluated": og.last_evaluated.isoformat() if og.last_evaluated else None,
            "nodes": nodes,
            "edges": edges,
            "active_conflicts": conflicts,
            "active_signals": signals,
        }
    )


@require_http_methods(["GET"])
@require_auth
def graph_versions(request):
    """List version history of the operating graph."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    versions = (
        GraphVersion.objects.filter(
            operating_graph=og,
        )
        .order_by("-version_number")
        .values(
            "id",
            "version_number",
            "promoted_by_trial_id",
            "notes",
            "created_at",
        )[:50]
    )

    result = []
    for v in versions:
        v["id"] = str(v["id"])
        v["promoted_by_trial_id"] = str(v["promoted_by_trial_id"]) if v["promoted_by_trial_id"] else None
        v["created_at"] = v["created_at"].isoformat() if v["created_at"] else None
        result.append(v)

    return JsonResponse({"versions": result, "current_version": og.current_version})


@require_http_methods(["POST"])
@require_auth
def graph_rollback(request, version_id):
    """Rollback to a previous graph version. Creates a new version pointing back."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    try:
        version = GraphVersion.objects.get(id=version_id, operating_graph=og)
    except GraphVersion.DoesNotExist:
        return JsonResponse({"error": "Version not found."}, status=404)

    # TODO: Implement full rollback from snapshot
    # For now, return the snapshot for manual review
    return JsonResponse(
        {
            "version": version.version_number,
            "snapshot_keys": list(version.snapshot.keys()) if version.snapshot else [],
            "message": "Rollback review — full restore not yet implemented.",
        }
    )


# =============================================================================
# NODES
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def node_create(request):
    """Add a node to the operating graph."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    data = _json_body(request)

    label = data.get("label", "").strip()
    if not label:
        return JsonResponse({"error": "Label is required."}, status=400)

    node = GraphNode.objects.create(
        operating_graph=og,
        label=label,
        node_type=data.get("node_type", "factor"),
        alpha=data.get("alpha", 1.0),
        beta=data.get("beta", 1.0),
        metadata=data.get("metadata", {}),
    )

    return JsonResponse(
        {
            "id": str(node.id),
            "label": node.label,
            "node_type": node.node_type,
            "confidence": node.confidence,
        },
        status=201,
    )


@require_http_methods(["GET"])
@require_auth
def node_list(request):
    """List all nodes in the operating graph."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    node_type = request.GET.get("type")

    qs = GraphNode.objects.filter(operating_graph=og)
    if node_type:
        qs = qs.filter(node_type=node_type)

    nodes = []
    for n in qs.order_by("label"):
        nodes.append(
            {
                "id": str(n.id),
                "label": n.label,
                "node_type": n.node_type,
                "confidence": n.confidence,
                "uncertainty": n.uncertainty,
                "alpha": n.alpha,
                "beta": n.beta,
            }
        )

    return JsonResponse({"nodes": nodes})


# =============================================================================
# EDGES
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def edge_create(request):
    """Add an edge to the operating graph."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    data = _json_body(request)

    source_id = data.get("source_id")
    target_id = data.get("target_id")
    if not source_id or not target_id:
        return JsonResponse({"error": "source_id and target_id required."}, status=400)

    try:
        source = GraphNode.objects.get(id=source_id, operating_graph=og)
        target = GraphNode.objects.get(id=target_id, operating_graph=og)
    except GraphNode.DoesNotExist:
        return JsonResponse({"error": "Source or target node not found."}, status=404)

    if source_id == target_id:
        return JsonResponse({"error": "Self-loops not allowed."}, status=400)

    edge = GraphEdge.objects.create(
        operating_graph=og,
        source=source,
        target=target,
        edge_type=data.get("edge_type", "causes"),
        weight=data.get("weight", 0.5),
        confidence=data.get("confidence", 0.5),
        conditions=data.get("conditions", []),
        cycle_length=data.get("cycle_length"),
    )

    return JsonResponse(
        {
            "id": str(edge.id),
            "source": source.label,
            "target": target.label,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
            "confidence": edge.confidence,
        },
        status=201,
    )


# =============================================================================
# WORKING GRAPH
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def working_graph_list(request):
    """List user's working graphs."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    graphs = WorkingGraph.objects.filter(
        owner=request.user,
        tenant=tenant,
        is_deleted=False,
    ).select_related("project", "operating_graph")

    result = []
    for wg in graphs:
        hyp_count = ProvaHypothesis.objects.filter(working_graph=wg, is_deleted=False).count()
        result.append(
            {
                "id": str(wg.id),
                "project": str(wg.project_id) if wg.project_id else None,
                "project_title": wg.project.title if wg.project else None,
                "operating_graph_version": wg.operating_graph.current_version,
                "hypothesis_count": hyp_count,
                "created_at": wg.created_at.isoformat(),
            }
        )

    return JsonResponse({"working_graphs": result})


@require_http_methods(["POST"])
@require_auth
def working_graph_create(request):
    """Create a new working graph (scratchpad)."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    data = _json_body(request)

    project_id = data.get("project_id")
    project = None
    if project_id:
        from core.views import get_user_projects

        try:
            project = get_user_projects(request.user).get(id=project_id)
        except Exception:
            return JsonResponse({"error": "Project not found."}, status=404)

    wg = WorkingGraph.objects.create(
        tenant=tenant,
        owner=request.user,
        project=project,
        operating_graph=og,
    )

    return JsonResponse(
        {
            "id": str(wg.id),
            "project_id": str(project.id) if project else None,
            "operating_graph_version": og.current_version,
        },
        status=201,
    )


# =============================================================================
# HYPOTHESES
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def hypothesis_list(request, working_graph_id):
    """List hypotheses in a working graph, grouped by curation tier."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        wg = WorkingGraph.objects.get(id=working_graph_id, tenant=tenant, is_deleted=False)
    except WorkingGraph.DoesNotExist:
        return JsonResponse({"error": "Working graph not found."}, status=404)

    tier_filter = request.GET.get("tier")  # active, curated, noise

    hypotheses = ProvaHypothesis.objects.filter(
        working_graph=wg,
        is_deleted=False,
    ).order_by("-posterior")

    result = []
    for h in hypotheses:
        if tier_filter and h.curation_tier != tier_filter:
            continue
        result.append(
            {
                "id": str(h.id),
                "description": h.description,
                "status": h.status,
                "prior": h.prior,
                "posterior": h.posterior,
                "outcome_label": h.outcome_label,
                "trial_commitment_date": h.trial_commitment_date.isoformat() if h.trial_commitment_date else None,
                "curation_tier": h.curation_tier,
                "parent_id": str(h.parent_id) if h.parent_id else None,
                "edit_count": h.edits.count(),
                "trial_count": h.trials.count(),
            }
        )

    return JsonResponse({"hypotheses": result})


@require_http_methods(["POST"])
@require_auth
def hypothesis_create(request, working_graph_id):
    """Create a hypothesis (proposed graph edit) in a working graph.

    Built through the structured builder — this endpoint receives the
    machine-parseable contract, not free text.
    """
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        wg = WorkingGraph.objects.get(id=working_graph_id, tenant=tenant, is_deleted=False)
    except WorkingGraph.DoesNotExist:
        return JsonResponse({"error": "Working graph not found."}, status=404)

    data = _json_body(request)
    description = data.get("description", "").strip()
    if not description:
        return JsonResponse({"error": "Description is required."}, status=400)

    parent_id = data.get("parent_id")
    parent = None
    if parent_id:
        try:
            parent = ProvaHypothesis.objects.get(id=parent_id, working_graph=wg)
        except ProvaHypothesis.DoesNotExist:
            return JsonResponse({"error": "Parent hypothesis not found."}, status=404)

    hypothesis = ProvaHypothesis.objects.create(
        working_graph=wg,
        parent=parent,
        description=description,
        outcome_label=data.get("outcome_label", ""),
        trial_commitment_date=data.get("trial_commitment_date"),
        prior=data.get("prior", 0.5),
        created_by=str(request.user.id),
    )

    # Create graph edits
    edits = data.get("edits", [])
    for edit_data in edits:
        GraphEdit.objects.create(
            hypothesis=hypothesis,
            operation=edit_data.get("operation", "add_edge"),
            target_edge_id=edit_data.get("target_edge_id"),
            target_node_id=edit_data.get("target_node_id"),
            params=edit_data.get("params", {}),
        )

    return JsonResponse(
        {
            "id": str(hypothesis.id),
            "description": hypothesis.description,
            "curation_tier": hypothesis.curation_tier,
            "edit_count": hypothesis.edits.count(),
        },
        status=201,
    )


@require_http_methods(["GET"])
@require_auth
def hypothesis_detail(request, hypothesis_id):
    """Get full hypothesis detail with edits and trials."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        h = ProvaHypothesis.objects.get(
            id=hypothesis_id,
            working_graph__tenant=tenant,
            is_deleted=False,
        )
    except ProvaHypothesis.DoesNotExist:
        return JsonResponse({"error": "Hypothesis not found."}, status=404)

    edits = []
    for e in h.edits.all().select_related("target_edge", "target_node"):
        edits.append(
            {
                "id": str(e.id),
                "operation": e.operation,
                "target_edge_id": str(e.target_edge_id) if e.target_edge_id else None,
                "target_node_id": str(e.target_node_id) if e.target_node_id else None,
                "target_label": (
                    str(e.target_edge) if e.target_edge else (e.target_node.label if e.target_node else None)
                ),
                "params": e.params,
            }
        )

    trials = []
    for t in h.trials.all().order_by("-created_at"):
        trials.append(
            {
                "id": str(t.id),
                "status": t.status,
                "complexity_tier": t.complexity_tier,
                "meets_minimum_validity": t.meets_minimum_validity,
                "evaluation": t.evaluation,
                "created_at": t.created_at.isoformat(),
            }
        )

    refinements = []
    for r in h.refinements.filter(is_deleted=False):
        refinements.append(
            {
                "id": str(r.id),
                "description": r.description,
                "status": r.status,
                "posterior": r.posterior,
            }
        )

    return JsonResponse(
        {
            "id": str(h.id),
            "description": h.description,
            "status": h.status,
            "prior": h.prior,
            "posterior": h.posterior,
            "outcome_label": h.outcome_label,
            "trial_commitment_date": h.trial_commitment_date.isoformat() if h.trial_commitment_date else None,
            "curation_tier": h.curation_tier,
            "parent_id": str(h.parent_id) if h.parent_id else None,
            "edits": edits,
            "trials": trials,
            "refinements": refinements,
        }
    )


# =============================================================================
# TRIALS
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def trial_create(request, hypothesis_id):
    """Create a trial for a hypothesis. Can be from proposal or custom."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        hypothesis = ProvaHypothesis.objects.get(
            id=hypothesis_id,
            working_graph__tenant=tenant,
            is_deleted=False,
        )
    except ProvaHypothesis.DoesNotExist:
        return JsonResponse({"error": "Hypothesis not found."}, status=404)

    data = _json_body(request)

    edge_id = data.get("edge_id")
    edge = None
    if edge_id:
        try:
            edge = GraphEdge.objects.get(
                id=edge_id,
                operating_graph=hypothesis.working_graph.operating_graph,
            )
        except GraphEdge.DoesNotExist:
            pass

    trial = Trial.objects.create(
        hypothesis=hypothesis,
        edge=edge,
        doe_spec=data.get("doe_spec", {}),
        complexity_tier=data.get("complexity_tier", "green"),
        status=TrialStatus.PLANNED,
        created_by=str(request.user.id),
    )

    # Update hypothesis status
    hypothesis.status = "testing"
    hypothesis.save(update_fields=["status", "updated_at"])

    return JsonResponse(
        {
            "id": str(trial.id),
            "status": trial.status,
            "complexity_tier": trial.complexity_tier,
            "doe_spec": trial.doe_spec,
        },
        status=201,
    )


@require_http_methods(["POST"])
@require_auth
def trial_submit_data(request, trial_id):
    """Submit trial data (CSV-parsed or form input)."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        trial = Trial.objects.get(
            id=trial_id,
            hypothesis__working_graph__tenant=tenant,
        )
    except Trial.DoesNotExist:
        return JsonResponse({"error": "Trial not found."}, status=404)

    data = _json_body(request)
    raw_data = data.get("data", {})

    if not raw_data:
        return JsonResponse({"error": "No data provided."}, status=400)

    # Validate minimum DOE shape
    rows = raw_data.get("rows", [])
    columns = raw_data.get("columns", [])
    is_valid = len(rows) >= 2 and len(columns) >= 2

    trial.raw_data = raw_data
    trial.meets_minimum_validity = is_valid
    trial.status = TrialStatus.COMPLETED if is_valid else TrialStatus.DRAFT
    trial.save(update_fields=["raw_data", "meets_minimum_validity", "status", "updated_at"])

    if not is_valid:
        return JsonResponse(
            {
                "id": str(trial.id),
                "status": trial.status,
                "meets_minimum_validity": False,
                "message": "Minimum DOE shape not met. Need at least 2 rows and 2 columns (factor + outcome).",
            }
        )

    return JsonResponse(
        {
            "id": str(trial.id),
            "status": trial.status,
            "meets_minimum_validity": True,
            "row_count": len(rows),
            "column_count": len(columns),
        }
    )


@require_http_methods(["POST"])
@require_auth
def trial_evaluate(request, trial_id):
    """Evaluate a completed trial — compute discriminating power and graph delta."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        trial = Trial.objects.get(
            id=trial_id,
            hypothesis__working_graph__tenant=tenant,
        )
    except Trial.DoesNotExist:
        return JsonResponse({"error": "Trial not found."}, status=404)

    if trial.status != TrialStatus.COMPLETED:
        return JsonResponse({"error": "Trial must be completed before evaluation."}, status=400)

    if not trial.meets_minimum_validity:
        return JsonResponse({"error": "Trial does not meet minimum DOE validity."}, status=400)

    og = trial.hypothesis.working_graph.operating_graph
    evidence_weight = engine.compute_discriminating_power(
        operating_graph=og,
        hypothesis=trial.hypothesis,
        trial_data=trial.raw_data,
    )

    trial.evaluation = {
        "theoretical_weight": evidence_weight.theoretical_weight,
        "graph_delta": evidence_weight.graph_delta,
        "is_meaningful": evidence_weight.is_meaningful,
        "reliability": evidence_weight.reliability,
        "discriminating_power": evidence_weight.discriminating_power,
    }
    trial.save(update_fields=["evaluation", "updated_at"])

    return JsonResponse(
        {
            "id": str(trial.id),
            "evaluation": trial.evaluation,
            "can_promote": evidence_weight.is_meaningful,
            "message": (
                "Evidence is meaningful — ready for promotion."
                if evidence_weight.is_meaningful
                else "Evidence did not produce a meaningful graph delta. H0 stands."
            ),
        }
    )


@require_http_methods(["POST"])
@require_auth
def trial_promote(request, trial_id):
    """Promote a trial's findings to the operating graph."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        trial = Trial.objects.get(
            id=trial_id,
            hypothesis__working_graph__tenant=tenant,
        )
    except Trial.DoesNotExist:
        return JsonResponse({"error": "Trial not found."}, status=404)

    result = engine.promote_trial(trial)

    return JsonResponse(
        {
            "success": result.success,
            "version_before": result.version_before,
            "version_after": result.version_after,
            "graph_delta": result.graph_delta,
            "signals_count": len(result.signals),
            "conflicts_count": len(result.conflicts),
            "message": result.message,
        }
    )


@require_http_methods(["GET"])
@require_auth
def trial_proposals(request, hypothesis_id):
    """Get ranked trial proposals for a hypothesis."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        hypothesis = ProvaHypothesis.objects.get(
            id=hypothesis_id,
            working_graph__tenant=tenant,
            is_deleted=False,
        )
    except ProvaHypothesis.DoesNotExist:
        return JsonResponse({"error": "Hypothesis not found."}, status=404)

    tier_filter = request.GET.get("tier")  # green, blue, purple
    proposals = engine.propose_trials(hypothesis)

    if tier_filter:
        proposals = [p for p in proposals if p.complexity_tier == tier_filter]

    result = []
    for p in proposals:
        result.append(
            {
                "description": p.description,
                "doe_spec": p.doe_spec,
                "complexity_tier": p.complexity_tier,
                "discriminating_power": p.discriminating_power,
                "estimated_cost": p.estimated_cost,
                "factors": p.factors,
                "outcome": p.outcome,
                "suggested_n": p.suggested_n,
            }
        )

    return JsonResponse({"proposals": result})


# =============================================================================
# CONFLICTS
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def conflict_list(request):
    """List active conflicts in the operating graph."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    status_filter = request.GET.get("status", "open")

    qs = Conflict.objects.filter(
        operating_graph=og,
    ).select_related("edge__source", "edge__target")

    if status_filter != "all":
        qs = qs.filter(status=status_filter)

    conflicts = []
    for c in qs.order_by("-evaluation_cost"):
        conflicts.append(
            {
                "id": str(c.id),
                "edge_id": str(c.edge_id),
                "edge_label": str(c.edge),
                "magnitude": c.magnitude,
                "evaluation_cost": c.evaluation_cost,
                "status": c.status,
                "proposed_resolutions": c.proposed_resolutions,
                "created_at": c.created_at.isoformat(),
            }
        )

    return JsonResponse({"conflicts": conflicts})


@require_http_methods(["POST"])
@require_auth
def conflict_update(request, conflict_id):
    """Update conflict status (acknowledge, accept, resolve)."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        conflict = Conflict.objects.get(
            id=conflict_id,
            operating_graph__tenant=tenant,
        )
    except Conflict.DoesNotExist:
        return JsonResponse({"error": "Conflict not found."}, status=404)

    data = _json_body(request)
    new_status = data.get("status")

    valid_transitions = {
        "open": ["investigating", "accepted"],
        "investigating": ["resolved", "accepted", "open"],
    }

    if new_status not in valid_transitions.get(conflict.status, []):
        return JsonResponse(
            {
                "error": f"Cannot transition from {conflict.status} to {new_status}.",
            },
            status=400,
        )

    conflict.status = new_status

    if new_status == "resolved":
        trial_id = data.get("resolved_by_trial_id")
        if trial_id:
            try:
                conflict.resolved_by_trial = Trial.objects.get(id=trial_id)
            except Trial.DoesNotExist:
                pass

        # Restore the edge
        edge = conflict.edge
        if edge.status == "broken":
            edge.status = "active"
            edge.save(update_fields=["status", "updated_at"])

    conflict.save(update_fields=["status", "resolved_by_trial", "updated_at"])

    return JsonResponse(
        {
            "id": str(conflict.id),
            "status": conflict.status,
        }
    )


# =============================================================================
# PROPAGATION SIGNALS
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def signal_list(request):
    """List active propagation signals."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    status_filter = request.GET.get("status", "active")

    qs = PropagationSignal.objects.filter(
        edge__operating_graph=og,
    ).select_related("edge__source", "edge__target")

    if status_filter != "all":
        qs = qs.filter(status=status_filter)

    signals = []
    for s in qs.order_by("-magnitude"):
        signals.append(
            {
                "id": str(s.id),
                "edge_id": str(s.edge_id),
                "edge_label": str(s.edge),
                "signal_type": s.signal_type,
                "magnitude": s.magnitude,
                "status": s.status,
                "triggered_by_trial_id": str(s.triggered_by_trial_id) if s.triggered_by_trial_id else None,
                "created_at": s.created_at.isoformat(),
            }
        )

    return JsonResponse({"signals": signals})


@require_http_methods(["POST"])
@require_auth
def signal_acknowledge(request, signal_id):
    """Acknowledge or resolve a propagation signal."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        signal = PropagationSignal.objects.get(
            id=signal_id,
            edge__operating_graph__tenant=tenant,
        )
    except PropagationSignal.DoesNotExist:
        return JsonResponse({"error": "Signal not found."}, status=404)

    data = _json_body(request)
    new_status = data.get("status", "acknowledged")

    if new_status not in ("acknowledged", "resolved"):
        return JsonResponse({"error": "Status must be 'acknowledged' or 'resolved'."}, status=400)

    signal.status = new_status
    signal.save(update_fields=["status", "updated_at"])

    # If resolved and edge was dark, restore it
    if new_status == "resolved" and signal.edge.status == "dark":
        signal.edge.status = "active"
        signal.edge.save(update_fields=["status", "updated_at"])

    return JsonResponse(
        {
            "id": str(signal.id),
            "status": signal.status,
        }
    )


# =============================================================================
# EVALUATE — on-demand graph evaluation
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def evaluate_graph(request):
    """Re-evaluate the operating graph's predictive score."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    og = engine.get_or_create_operating_graph(tenant)
    score = engine._evaluate_graph_score(og)

    og.predictive_score = score
    og.last_evaluated = engine.timezone.now()
    og.save(update_fields=["predictive_score", "last_evaluated", "updated_at"])

    return JsonResponse(
        {
            "predictive_score": score,
            "version": og.current_version,
            "node_count": GraphNode.objects.filter(operating_graph=og).count(),
            "edge_count": GraphEdge.objects.filter(operating_graph=og, status="active").count(),
            "conflict_count": Conflict.objects.filter(operating_graph=og, status="open").count(),
        }
    )


# =============================================================================
# BRIDGE — universal tool integration endpoint
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def bridge_integrate(request):
    """Universal integration endpoint: tool output → PROVA graph.

    Any SVEND tool calls this to submit findings. Routes by tool_type
    to the appropriate handler (information, inference, intent, report).

    Body: {
        "summary": "Finding description",
        "tool_type": "spc",  // rca, ishikawa, dsw, doe_results, etc.
        "source_id": "uuid",  // tool output ID
        "source_field": "field_name",  // optional
        // Inference tools:
        "sample_size": 30,
        "p_value": 0.01,
        "effect_size": 1.2,
        "edge_mappings": [{"edge_id": "uuid", "direction": "supports"}],
        "raw_output": {},
        // Information tools:
        "hypotheses": [{"description": "...", "outcome_label": "...", "edits": [...]}],
        // Intent tools:
        "target_hypothesis_ids": ["uuid"],
        "design_metadata": {},
        // Context:
        "project_id": "uuid",
        "investigation_id": "uuid",
    }
    """
    tenant, err = require_tenant(request.user)
    if err:
        return err

    data = _json_body(request)

    summary = data.get("summary", "").strip()
    tool_type = data.get("tool_type", "").strip()

    if not summary or not tool_type:
        return JsonResponse(
            {"error": "summary and tool_type are required."},
            status=400,
        )

    spec = FindingSpec(
        summary=summary,
        tool_type=tool_type,
        source_id=data.get("source_id", ""),
        source_field=data.get("source_field", ""),
        sample_size=data.get("sample_size"),
        measurement_system_id=data.get("measurement_system_id"),
        study_quality_factors=data.get("study_quality_factors"),
        p_value=data.get("p_value"),
        effect_size=data.get("effect_size"),
        raw_output=data.get("raw_output", {}),
        edge_mappings=data.get("edge_mappings", []),
        hypotheses=data.get("hypotheses", []),
        target_hypothesis_ids=data.get("target_hypothesis_ids", []),
        design_metadata=data.get("design_metadata", {}),
        project_id=data.get("project_id"),
        investigation_id=data.get("investigation_id"),
    )

    # tool_output model instance — look up if source_id provided
    tool_output = None  # TODO: resolve from content_type + source_id if needed

    result = integrate(
        user=request.user,
        tenant=tenant,
        tool_output=tool_output,
        spec=spec,
    )

    return JsonResponse(
        {
            "success": result.success,
            "tool_function": result.tool_function,
            "evidence_ids": result.evidence_ids,
            "evidence_weight": result.evidence_weight,
            "hypotheses_created": result.hypotheses_created,
            "edges_updated": result.edges_updated,
            "conflicts_detected": result.conflicts_detected,
            "superseded_evidence_id": result.superseded_evidence_id,
            "message": result.message,
        }
    )


# =============================================================================
# INVESTIGATE OOC — SPC → PROVA → RCA
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def investigate_ooc(request):
    """Investigate an out-of-control point from SPC.

    PROVA creates a hypothesis (expectation violation) in the working graph,
    then spawns an RCA session linked to that hypothesis. The frontend
    opens the RCA session for the user to build the causal chain.

    When the RCA completes, on_rca_root_cause() refines the hypothesis.

    Body: {
        "dsw_result_id": "dsw_xxx",
        "observation_index": 5,
        "value": 42.5,
        "rules_violated": "Rule 1: Beyond 3σ",
        "measurement": "thickness",  // optional
        "project_id": "uuid",  // optional
    }
    """
    tenant, err = require_tenant(request.user)
    if err:
        return err

    data = _json_body(request)

    obs_index = data.get("observation_index")
    value = data.get("value")
    rules = data.get("rules_violated", "")

    if obs_index is None or value is None:
        return JsonResponse(
            {"error": "observation_index and value are required."},
            status=400,
        )

    result = integrations.on_spc_investigate_ooc(
        user=request.user,
        tenant=tenant,
        dsw_result_id=data.get("dsw_result_id", ""),
        observation_index=int(obs_index),
        value=float(value),
        rules_violated=rules,
        measurement_col=data.get("measurement", ""),
        project_id=data.get("project_id"),
    )

    return JsonResponse(
        {
            "hypothesis_id": result["hypothesis_id"],
            "rca_session_id": result["rca_session_id"],
            "working_graph_id": result["working_graph_id"],
            "rca_url": f"/app/rca/?session={result['rca_session_id']}",
        },
        status=201,
    )
