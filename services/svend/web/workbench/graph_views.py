"""
Knowledge Graph API Views.

CRUD operations for nodes, edges, and Bayesian updates.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from .models import Workbench, KnowledgeGraph, EpistemicLog, Project, Hypothesis

logger = logging.getLogger(__name__)


def get_or_create_graph(workbench_id: str, user) -> KnowledgeGraph:
    """Get or create knowledge graph for workbench."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=user)

    graph, created = KnowledgeGraph.objects.get_or_create(
        workbench=workbench,
        user=user,
        defaults={"title": f"Graph: {workbench.title}"}
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


def get_or_create_project_graph(project_id: str, user) -> KnowledgeGraph:
    """Get or create knowledge graph for a project."""
    project = get_object_or_404(Project, id=project_id, user=user)

    graph, created = KnowledgeGraph.objects.get_or_create(
        project=project,
        user=user,
        defaults={"title": f"Graph: {project.title}"}
    )

    if created:
        EpistemicLog.log(
            user=user,
            event_type=EpistemicLog.EventType.INQUIRY_STARTED,
            event_data={"project_id": str(project_id), "graph_id": str(graph.id)},
            project=project,
            knowledge_graph=graph,
            source="system",
        )

    return graph


# =============================================================================
# Graph CRUD
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
def get_graph(request, workbench_id: str):
    """Get the knowledge graph for a workbench."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    graph = get_or_create_graph(workbench_id, request.user)
    return JsonResponse(graph.to_dict())


@csrf_exempt
@require_http_methods(["DELETE"])
def clear_graph(request, workbench_id: str):
    """Clear all nodes and edges from the graph."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    graph = get_or_create_graph(workbench_id, request.user)
    graph.nodes = []
    graph.edges = []
    graph.expansion_signals = []
    graph.save()

    return JsonResponse({"success": True})


# =============================================================================
# Node operations
# =============================================================================

@csrf_exempt
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
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

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


@csrf_exempt
@require_http_methods(["DELETE"])
def remove_node(request, workbench_id: str, node_id: str):
    """Remove a node and all connected edges."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    graph = get_or_create_graph(workbench_id, request.user)

    node = graph.get_node(node_id)
    if not node:
        return JsonResponse({"error": "Node not found"}, status=404)

    graph.remove_node(node_id)

    return JsonResponse({"success": True})


@csrf_exempt
@require_http_methods(["GET"])
def get_nodes(request, workbench_id: str):
    """Get all nodes in the graph."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    graph = get_or_create_graph(workbench_id, request.user)

    return JsonResponse({
        "nodes": graph.nodes,
        "count": len(graph.nodes),
    })


# =============================================================================
# Edge operations (causal vectors)
# =============================================================================

@csrf_exempt
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
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

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


@csrf_exempt
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
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

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

    return JsonResponse({
        "success": True,
        "edge_id": edge_id,
        "old_weight": old_weight,
        "new_weight": new_weight,
    })


@csrf_exempt
@require_http_methods(["GET"])
def get_edges(request, workbench_id: str):
    """Get all edges in the graph."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    graph = get_or_create_graph(workbench_id, request.user)

    return JsonResponse({
        "edges": graph.edges,
        "count": len(graph.edges),
    })


# =============================================================================
# Evidence & Bayesian updates
# =============================================================================

@csrf_exempt
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
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

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

    return JsonResponse({
        "success": True,
        "evidence_id": evidence_id,
        "updates": updates,
    })


@csrf_exempt
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
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

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

        return JsonResponse({
            "expansion_needed": True,
            "signal": signal,
        })

    return JsonResponse({
        "expansion_needed": False,
        "max_likelihood": max(likelihoods.values()) if likelihoods else 0,
    })


# =============================================================================
# Graph traversal
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
def get_causal_chain(request, workbench_id: str, from_node: str, to_node: str):
    """Get all causal chains between two nodes."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

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

    return JsonResponse({
        "from_node": from_node,
        "to_node": to_node,
        "chains": chains,
        "count": len(chains),
    })


@csrf_exempt
@require_http_methods(["GET"])
def get_upstream_causes(request, workbench_id: str, node_id: str):
    """Get all causes upstream of a node."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    graph = get_or_create_graph(workbench_id, request.user)

    depth = int(request.GET.get("depth", 3))
    causes = graph.get_upstream_causes(node_id, depth)

    return JsonResponse({
        "node_id": node_id,
        "depth": depth,
        "causes": causes,
        "count": len(causes),
    })


# =============================================================================
# Expansion signals
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
def get_expansion_signals(request, workbench_id: str):
    """Get pending expansion signals."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    graph = get_or_create_graph(workbench_id, request.user)

    pending = [s for s in graph.expansion_signals if s.get("status") == "pending"]

    return JsonResponse({
        "signals": pending,
        "count": len(pending),
    })


@csrf_exempt
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
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

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

            return JsonResponse({
                "success": True,
                "resolution": body.get("resolution"),
                "new_node": new_node,
            })

    return JsonResponse({"error": "Signal not found"}, status=404)


# =============================================================================
# Epistemic log
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
def get_epistemic_log(request, workbench_id: str):
    """Get the epistemic log for a workbench."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    limit = int(request.GET.get("limit", 50))
    event_type = request.GET.get("type")

    logs = EpistemicLog.objects.filter(workbench=workbench).order_by("-created_at")

    if event_type:
        logs = logs.filter(event_type=event_type)

    logs = logs[:limit]

    return JsonResponse({
        "logs": [
            {
                "id": str(log.id),
                "event_type": log.event_type,
                "event_data": log.event_data,
                "source": log.source,
                "led_to_insight": log.led_to_insight,
                "led_to_dead_end": log.led_to_dead_end,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
        "count": len(logs),
    })


@csrf_exempt
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
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    log = get_object_or_404(EpistemicLog, id=log_id, user=request.user)

    log.mark_outcome(
        led_to_insight=body.get("led_to_insight"),
        led_to_dead_end=body.get("led_to_dead_end"),
    )

    return JsonResponse({
        "success": True,
        "log_id": str(log_id),
        "led_to_insight": log.led_to_insight,
        "led_to_dead_end": log.led_to_dead_end,
    })


# =============================================================================
# Project-level Knowledge Graph (for connecting hypotheses)
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
def get_project_graph(request, project_id: str):
    """Get the knowledge graph for a project."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    graph = get_or_create_project_graph(project_id, request.user)
    return JsonResponse(graph.to_dict())


@csrf_exempt
@require_http_methods(["POST"])
def add_hypothesis_to_graph(request, project_id: str, hypothesis_id: str):
    """
    Add a hypothesis as a node in the project's knowledge graph.

    Optionally connect it to other hypotheses.

    Request body:
    {
        "metadata": {},
        "connect_to": [  // optional: create edges to these nodes
            {"node_id": "abc123", "weight": 0.7, "mechanism": "causes"}
        ],
        "connect_from": [  // optional: edges from these nodes to this one
            {"node_id": "def456", "weight": 0.6, "mechanism": "influences"}
        ]
    }
    """
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    try:
        body = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        body = {}

    graph = get_or_create_project_graph(project_id, request.user)

    # Check if hypothesis is already in graph
    existing = next(
        (n for n in graph.nodes if n.get("hypothesis_id") == str(hypothesis_id)),
        None
    )

    if existing:
        return JsonResponse({
            "success": True,
            "node": existing,
            "already_exists": True
        })

    # Add hypothesis as node
    node = graph.add_node(
        node_type="hypothesis",
        label=hypothesis.statement[:100],
        artifact_id=str(hypothesis.id),
        metadata={
            "hypothesis_id": str(hypothesis.id),
            "probability": hypothesis.current_probability,
            "status": hypothesis.status,
            **(body.get("metadata", {}))
        },
    )

    # Create edges to other nodes
    edges_created = []
    for conn in body.get("connect_to", []):
        edge = graph.add_edge(
            from_node=node["id"],
            to_node=conn["node_id"],
            weight=conn.get("weight", 0.5),
            mechanism=conn.get("mechanism", ""),
        )
        edges_created.append(edge)

    for conn in body.get("connect_from", []):
        edge = graph.add_edge(
            from_node=conn["node_id"],
            to_node=node["id"],
            weight=conn.get("weight", 0.5),
            mechanism=conn.get("mechanism", ""),
        )
        edges_created.append(edge)

    # Log
    EpistemicLog.log(
        user=request.user,
        event_type=EpistemicLog.EventType.NODE_ADDED,
        event_data={"node": node, "edges": edges_created},
        project=project,
        knowledge_graph=graph,
        source="user",
    )

    return JsonResponse({
        "success": True,
        "node": node,
        "edges_created": edges_created,
    })


@csrf_exempt
@require_http_methods(["POST"])
def connect_hypotheses(request, project_id: str):
    """
    Create a causal connection between two hypotheses in the project graph.

    Request body:
    {
        "from_hypothesis_id": "uuid",
        "to_hypothesis_id": "uuid",
        "weight": 0.7,
        "mechanism": "High temperature causes material degradation"
    }
    """
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    project = get_object_or_404(Project, id=project_id, user=request.user)
    graph = get_or_create_project_graph(project_id, request.user)

    from_hyp_id = body.get("from_hypothesis_id")
    to_hyp_id = body.get("to_hypothesis_id")

    if not from_hyp_id or not to_hyp_id:
        return JsonResponse({"error": "Both hypothesis IDs are required"}, status=400)

    # Find or create nodes for both hypotheses
    from_node = next(
        (n for n in graph.nodes if n.get("metadata", {}).get("hypothesis_id") == from_hyp_id),
        None
    )
    to_node = next(
        (n for n in graph.nodes if n.get("metadata", {}).get("hypothesis_id") == to_hyp_id),
        None
    )

    # Auto-add hypotheses to graph if not present
    if not from_node:
        from_hyp = get_object_or_404(Hypothesis, id=from_hyp_id, project=project)
        from_node = graph.add_node(
            node_type="hypothesis",
            label=from_hyp.statement[:100],
            artifact_id=str(from_hyp.id),
            metadata={
                "hypothesis_id": str(from_hyp.id),
                "probability": from_hyp.current_probability,
                "status": from_hyp.status,
            },
        )

    if not to_node:
        to_hyp = get_object_or_404(Hypothesis, id=to_hyp_id, project=project)
        to_node = graph.add_node(
            node_type="hypothesis",
            label=to_hyp.statement[:100],
            artifact_id=str(to_hyp.id),
            metadata={
                "hypothesis_id": str(to_hyp.id),
                "probability": to_hyp.current_probability,
                "status": to_hyp.status,
            },
        )

    # Create the edge
    edge = graph.add_edge(
        from_node=from_node["id"],
        to_node=to_node["id"],
        weight=body.get("weight", 0.5),
        mechanism=body.get("mechanism", ""),
    )

    # Log
    EpistemicLog.log(
        user=request.user,
        event_type=EpistemicLog.EventType.EDGE_ADDED,
        event_data={
            "edge": edge,
            "from_hypothesis": from_hyp_id,
            "to_hypothesis": to_hyp_id,
        },
        project=project,
        knowledge_graph=graph,
        source="user",
    )

    return JsonResponse({
        "success": True,
        "edge": edge,
        "from_node": from_node,
        "to_node": to_node,
    })


@csrf_exempt
@require_http_methods(["GET"])
def get_hypothesis_connections(request, project_id: str, hypothesis_id: str):
    """Get all connections to/from a hypothesis in the knowledge graph."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    graph = get_or_create_project_graph(project_id, request.user)

    # Find the node for this hypothesis
    node = next(
        (n for n in graph.nodes if n.get("metadata", {}).get("hypothesis_id") == str(hypothesis_id)),
        None
    )

    if not node:
        return JsonResponse({
            "hypothesis_id": str(hypothesis_id),
            "in_graph": False,
            "causes": [],
            "effects": [],
        })

    # Get edges from/to this node
    effects = []  # This hypothesis causes...
    causes = []   # This hypothesis is caused by...

    for edge in graph.edges:
        if edge["from_node"] == node["id"]:
            target_node = graph.get_node(edge["to_node"])
            effects.append({
                "edge": edge,
                "node": target_node,
            })
        elif edge["to_node"] == node["id"]:
            source_node = graph.get_node(edge["from_node"])
            causes.append({
                "edge": edge,
                "node": source_node,
            })

    return JsonResponse({
        "hypothesis_id": str(hypothesis_id),
        "in_graph": True,
        "node": node,
        "causes": causes,
        "effects": effects,
    })
