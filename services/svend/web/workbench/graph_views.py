"""
Knowledge Graph API Views.

CRUD operations for nodes, edges, and Bayesian updates.
"""

import json
import logging

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth

from .models import EpistemicLog, KnowledgeGraph, Workbench

logger = logging.getLogger(__name__)


def get_or_create_graph(workbench_id: str, user) -> KnowledgeGraph:
    """Get or create knowledge graph for workbench."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=user)

    graph, created = KnowledgeGraph.objects.get_or_create(
        workbench=workbench, user=user, defaults={"title": f"Graph: {workbench.title}"}
    )

    if created:
        EpistemicLog.log(
            user=user,
            event_type=EpistemicLog.EventType.INQUIRY_STARTED,
            event_data={"workbench_id": str(workbench_id), "graph_id": str(graph.id)},
            workbench=workbench,
            knowledge_graph=graph,
            source="system",
        )

    return graph


# =============================================================================
# Graph CRUD
# =============================================================================


@require_auth
@require_http_methods(["GET"])
def get_graph(request, workbench_id: str):
    """Get the knowledge graph for a workbench."""
    graph = get_or_create_graph(workbench_id, request.user)
    return JsonResponse(graph.to_dict())


@require_auth
@require_http_methods(["DELETE"])
def clear_graph(request, workbench_id: str):
    """Clear all nodes and edges from the graph."""
    graph = get_or_create_graph(workbench_id, request.user)
    graph.nodes = []
    graph.edges = []
    graph.expansion_signals = []
    graph.save()

    return JsonResponse({"success": True})


# =============================================================================
# Node operations
# =============================================================================


@require_auth
@require_http_methods(["POST"])
def add_node(request, workbench_id: str):
    """
    Add a node to the knowledge graph.

    Request body:
    {
        "type": "hypothesis" | "cause" | "effect" | "observation",
        "label": "Temperature causes defects",
        "artifact_id": "optional-uuid",
        "metadata": {}
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    graph = get_or_create_graph(workbench_id, request.user)

    node = graph.add_node(
        node_type=body.get("type", "hypothesis"),
        label=body.get("label", ""),
        artifact_id=body.get("artifact_id"),
        metadata=body.get("metadata", {}),
    )

    # Log the event
    EpistemicLog.log(
        user=request.user,
        event_type=EpistemicLog.EventType.NODE_ADDED,
        event_data=node,
        knowledge_graph=graph,
        workbench=graph.workbench,
        source="user",
    )

    return JsonResponse({"success": True, "node": node})


@require_auth
@require_http_methods(["DELETE"])
def remove_node(request, workbench_id: str, node_id: str):
    """Remove a node and all connected edges."""
    graph = get_or_create_graph(workbench_id, request.user)

    node = graph.get_node(node_id)
    if not node:
        return JsonResponse({"error": "Node not found"}, status=404)

    graph.remove_node(node_id)

    return JsonResponse({"success": True})


@require_auth
@require_http_methods(["GET"])
def get_nodes(request, workbench_id: str):
    """Get all nodes in the graph."""
    graph = get_or_create_graph(workbench_id, request.user)

    return JsonResponse(
        {
            "nodes": graph.nodes,
            "count": len(graph.nodes),
        }
    )


# =============================================================================
# Edge operations (causal vectors)
# =============================================================================


@require_auth
@require_http_methods(["POST"])
def add_edge(request, workbench_id: str):
    """
    Add a causal edge between nodes.

    Request body:
    {
        "from_node": "node_id",
        "to_node": "node_id",
        "weight": 0.7,
        "mechanism": "Temperature affects viscosity which causes defects",
        "metadata": {}
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    graph = get_or_create_graph(workbench_id, request.user)

    # Validate nodes exist
    from_node = graph.get_node(body.get("from_node"))
    to_node = graph.get_node(body.get("to_node"))

    if not from_node:
        return JsonResponse({"error": "Source node not found"}, status=400)
    if not to_node:
        return JsonResponse({"error": "Target node not found"}, status=400)

    edge = graph.add_edge(
        from_node=body["from_node"],
        to_node=body["to_node"],
        weight=body.get("weight", 0.5),
        mechanism=body.get("mechanism", ""),
        metadata=body.get("metadata", {}),
    )

    # Log the event
    EpistemicLog.log(
        user=request.user,
        event_type=EpistemicLog.EventType.EDGE_ADDED,
        event_data=edge,
        knowledge_graph=graph,
        workbench=graph.workbench,
        source="user",
    )

    return JsonResponse({"success": True, "edge": edge})


@require_auth
@require_http_methods(["POST"])
def update_edge_weight(request, workbench_id: str, edge_id: str):
    """
    Update an edge's weight.

    Request body:
    {
        "weight": 0.8,
        "evidence_id": "optional-evidence-reference"
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    graph = get_or_create_graph(workbench_id, request.user)

    edge = graph.get_edge(edge_id)
    if not edge:
        return JsonResponse({"error": "Edge not found"}, status=404)

    old_weight = edge["weight"]
    new_weight = body.get("weight", old_weight)

    graph.update_edge_weight(
        edge_id=edge_id,
        new_weight=new_weight,
        evidence_id=body.get("evidence_id"),
    )

    # Log the belief update
    EpistemicLog.log_belief_update(
        user=request.user,
        knowledge_graph=graph,
        edge_id=edge_id,
        old_weight=old_weight,
        new_weight=new_weight,
        evidence_id=body.get("evidence_id"),
    )

    return JsonResponse(
        {
            "success": True,
            "edge_id": edge_id,
            "old_weight": old_weight,
            "new_weight": new_weight,
        }
    )


@require_auth
@require_http_methods(["GET"])
def get_edges(request, workbench_id: str):
    """Get all edges in the graph."""
    graph = get_or_create_graph(workbench_id, request.user)

    return JsonResponse(
        {
            "edges": graph.edges,
            "count": len(graph.edges),
        }
    )


# =============================================================================
# Evidence & Bayesian updates
# =============================================================================


@require_auth
@require_http_methods(["POST"])
def apply_evidence(request, workbench_id: str):
    """
    Apply evidence to update edge weights via Bayesian inference.

    Request body:
    {
        "evidence_id": "ev_123",
        "supports": [
            {"edge_id": "e1", "likelihood": 0.8},
            {"edge_id": "e2", "likelihood": 0.6}
        ],
        "weakens": [
            {"edge_id": "e3", "likelihood": 0.7}
        ]
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    graph = get_or_create_graph(workbench_id, request.user)

    evidence_id = body.get("evidence_id", "")
    supports = [(s["edge_id"], s["likelihood"]) for s in body.get("supports", [])]
    weakens = [(w["edge_id"], w["likelihood"]) for w in body.get("weakens", [])]

    # Get old weights for logging
    updates = []
    for edge_id, _ in supports + weakens:
        edge = graph.get_edge(edge_id)
        if edge:
            updates.append({"edge_id": edge_id, "old_weight": edge["weight"]})

    # Apply Bayesian update
    graph.update_from_evidence(evidence_id, supports, weakens)

    # Get new weights
    for update in updates:
        edge = graph.get_edge(update["edge_id"])
        if edge:
            update["new_weight"] = edge["weight"]

            # Log each belief update
            EpistemicLog.log_belief_update(
                user=request.user,
                knowledge_graph=graph,
                edge_id=update["edge_id"],
                old_weight=update["old_weight"],
                new_weight=update["new_weight"],
                evidence_id=evidence_id,
            )

    return JsonResponse(
        {
            "success": True,
            "evidence_id": evidence_id,
            "updates": updates,
        }
    )


@require_auth
@require_http_methods(["POST"])
def check_expansion(request, workbench_id: str):
    """
    Check if evidence triggers an expansion signal.

    Request body:
    {
        "likelihoods": {
            "edge_id_1": 0.1,
            "edge_id_2": 0.05,
            "edge_id_3": 0.02
        }
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    graph = get_or_create_graph(workbench_id, request.user)

    likelihoods = body.get("likelihoods", {})
    signal = graph.check_expansion(likelihoods)

    if signal:
        # Log the expansion signal
        EpistemicLog.log_expansion_signal(
            user=request.user,
            knowledge_graph=graph,
            signal_data=signal,
        )

        return JsonResponse(
            {
                "expansion_needed": True,
                "signal": signal,
            }
        )

    return JsonResponse(
        {
            "expansion_needed": False,
            "max_likelihood": max(likelihoods.values()) if likelihoods else 0,
        }
    )


# =============================================================================
# Graph traversal
# =============================================================================


@require_auth
@require_http_methods(["GET"])
def get_causal_chain(request, workbench_id: str, from_node: str, to_node: str):
    """Get all causal chains between two nodes."""
    graph = get_or_create_graph(workbench_id, request.user)

    chains = graph.get_causal_chain(from_node, to_node)

    # Log the traversal
    EpistemicLog.log(
        user=request.user,
        event_type=EpistemicLog.EventType.CAUSAL_CHAIN_TRACED,
        event_data={"from": from_node, "to": to_node, "chains_found": len(chains)},
        knowledge_graph=graph,
        source="user",
    )

    return JsonResponse(
        {
            "from_node": from_node,
            "to_node": to_node,
            "chains": chains,
            "count": len(chains),
        }
    )


@require_auth
@require_http_methods(["GET"])
def get_upstream_causes(request, workbench_id: str, node_id: str):
    """Get all causes upstream of a node."""
    graph = get_or_create_graph(workbench_id, request.user)

    depth = int(request.GET.get("depth", 3))
    causes = graph.get_upstream_causes(node_id, depth)

    return JsonResponse(
        {
            "node_id": node_id,
            "depth": depth,
            "causes": causes,
            "count": len(causes),
        }
    )


# =============================================================================
# Expansion signals
# =============================================================================


@require_auth
@require_http_methods(["GET"])
def get_expansion_signals(request, workbench_id: str):
    """Get pending expansion signals."""
    graph = get_or_create_graph(workbench_id, request.user)

    pending = [s for s in graph.expansion_signals if s.get("status") == "pending"]

    return JsonResponse(
        {
            "signals": pending,
            "count": len(pending),
        }
    )


@require_auth
@require_http_methods(["POST"])
def resolve_expansion(request, workbench_id: str, signal_id: str):
    """
    Resolve an expansion signal.

    Request body:
    {
        "resolution": "new_hypothesis" | "expanded_hypothesis" | "dismissed",
        "new_node": {...}  // if adding new hypothesis
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    graph = get_or_create_graph(workbench_id, request.user)

    # Find and update the signal
    for signal in graph.expansion_signals:
        if signal.get("id") == signal_id:
            signal["status"] = "resolved"
            signal["resolution"] = body.get("resolution", "dismissed")

            # If adding new hypothesis, create the node
            new_node = None
            if body.get("resolution") == "new_hypothesis" and body.get("new_node"):
                node_data = body["new_node"]
                new_node = graph.add_node(
                    node_type=node_data.get("type", "hypothesis"),
                    label=node_data.get("label", ""),
                    metadata=node_data.get("metadata", {}),
                )

            graph.save()

            # Log resolution
            EpistemicLog.log(
                user=request.user,
                event_type=EpistemicLog.EventType.EXPANSION_RESOLVED,
                event_data={
                    "signal_id": signal_id,
                    "resolution": body.get("resolution"),
                    "new_node": new_node,
                },
                knowledge_graph=graph,
                source="user",
            )

            return JsonResponse(
                {
                    "success": True,
                    "resolution": body.get("resolution"),
                    "new_node": new_node,
                }
            )

    return JsonResponse({"error": "Signal not found"}, status=404)


# =============================================================================
# Epistemic log
# =============================================================================


@require_auth
@require_http_methods(["GET"])
def get_epistemic_log(request, workbench_id: str):
    """Get the epistemic log for a workbench."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    limit = int(request.GET.get("limit", 50))
    event_type = request.GET.get("type")

    logs = EpistemicLog.objects.filter(workbench=workbench).order_by("-created_at")

    if event_type:
        logs = logs.filter(event_type=event_type)

    logs = logs[:limit]

    return JsonResponse(
        {
            "logs": [
                {
                    "id": str(log.id),
                    "event_type": log.event_type,
                    "event_data": log.event_data,
                    "source": log.source,
                    "led_to_insight": log.has_led_to_insight,
                    "led_to_dead_end": log.has_led_to_dead_end,
                    "created_at": log.created_at.isoformat(),
                }
                for log in logs
            ],
            "count": len(logs),
        }
    )


@require_auth
@require_http_methods(["POST"])
def mark_log_outcome(request, workbench_id: str, log_id: str):
    """
    Mark the outcome of an epistemic event retrospectively.

    Request body:
    {
        "led_to_insight": true,
        "led_to_dead_end": false
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    log = get_object_or_404(EpistemicLog, id=log_id, user=request.user)

    log.mark_outcome(
        has_led_to_insight=body.get("led_to_insight"),
        has_led_to_dead_end=body.get("led_to_dead_end"),
    )

    return JsonResponse(
        {
            "success": True,
            "log_id": str(log_id),
            "led_to_insight": log.has_led_to_insight,
            "led_to_dead_end": log.has_led_to_dead_end,
        }
    )
