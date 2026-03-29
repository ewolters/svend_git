"""
Graph API views — GRAPH-001 §11, §15.

Endpoints for the graph navigator frontend. All endpoints enforce
tenant isolation via request.user.membership.tenant.
"""

import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth

from .models import ProcessGraph
from .service import GraphService

logger = logging.getLogger("svend.graph.views")


def _get_tenant_id(request):
    """Extract tenant_id from authenticated user."""
    membership = getattr(request.user, "active_membership", None)
    if membership:
        return membership.tenant_id
    memberships = request.user.memberships.all()
    if memberships.exists():
        return memberships.first().tenant_id
    return None


@require_auth
@require_http_methods(["GET"])
def graph_data(request):
    """Return full graph data for Cytoscape.js rendering.

    Query params:
        graph_id: optional UUID. If omitted, uses org's primary graph.
        lens: optional filter. "process_map" (default), "fmea", "gap", "control"
    """
    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    graph_id = request.GET.get("graph_id")
    lens = request.GET.get("lens", "process_map")

    if graph_id:
        try:
            graph = GraphService.get_graph(tenant_id, graph_id)
        except ProcessGraph.DoesNotExist:
            return JsonResponse({"error": "Graph not found"}, status=404)
    else:
        graph = GraphService.get_org_graph(tenant_id)
        if not graph:
            return JsonResponse(
                {
                    "nodes": [],
                    "edges": [],
                    "stats": {"node_count": 0, "edge_count": 0},
                    "empty": True,
                }
            )

    nodes_qs = graph.nodes.all()
    edges_qs = graph.edges.select_related("source", "target").all()

    # Apply lens filters
    if lens == "fmea":
        failure_nodes = set(nodes_qs.filter(node_type="failure_mode").values_list("id", flat=True))
        # Include failure mode nodes + their upstream causes + downstream effects
        upstream_ids = set()
        downstream_ids = set()
        for edge in edges_qs:
            if edge.target_id in failure_nodes:
                upstream_ids.add(edge.source_id)
            if edge.source_id in failure_nodes:
                downstream_ids.add(edge.target_id)
        visible_ids = failure_nodes | upstream_ids | downstream_ids
        nodes_qs = nodes_qs.filter(id__in=visible_ids)
        edges_qs = edges_qs.filter(source_id__in=visible_ids, target_id__in=visible_ids)

    elif lens == "gap":
        report = GraphService.gap_report(tenant_id, graph.id)
        gap_edge_ids = set()
        gap_node_ids = set()
        for e in (
            report.uncalibrated_edges + report.stale_edges + report.contradicted_edges + report.low_confidence_edges
        ):
            gap_edge_ids.add(e.id)
            gap_node_ids.add(e.source_id)
            gap_node_ids.add(e.target_id)
        for n in report.measurement_gaps:
            gap_node_ids.add(n.id)
        if gap_node_ids:
            nodes_qs = nodes_qs.filter(id__in=gap_node_ids)
            edges_qs = edges_qs.filter(id__in=gap_edge_ids)
        else:
            nodes_qs = nodes_qs.none()
            edges_qs = edges_qs.none()

    elif lens == "control":
        nodes_qs = nodes_qs.filter(linked_spc_chart__isnull=False)
        controlled_ids = set(nodes_qs.values_list("id", flat=True))
        edges_qs = edges_qs.filter(source_id__in=controlled_ids) | edges_qs.filter(target_id__in=controlled_ids)

    # Serialize for Cytoscape.js
    nodes = list(nodes_qs)
    edges = list(edges_qs)

    cy_nodes = []
    for n in nodes:
        cy_nodes.append(
            {
                "data": {
                    "id": str(n.id),
                    "label": n.name,
                    "node_type": n.node_type,
                    "controllability": n.controllability,
                    "unit": n.unit,
                    "has_spc": n.linked_spc_chart is not None,
                    "has_equipment": bool(n.linked_equipment),
                    "distribution": n.distribution,
                    "provenance": n.provenance,
                },
            }
        )

    cy_edges = []
    for e in edges:
        cy_edges.append(
            {
                "data": {
                    "id": str(e.id),
                    "source": str(e.source_id),
                    "target": str(e.target_id),
                    "label": e.relation_type,
                    "relation_type": e.relation_type,
                    "posterior": round(e.posterior_strength, 3),
                    "effect_size": e.effect_size,
                    "is_calibrated": e.is_calibrated,
                    "is_stale": e.is_stale,
                    "is_contradicted": e.is_contradicted,
                    "evidence_count": e.evidence_count,
                    "provenance": e.provenance,
                },
            }
        )

    return JsonResponse(
        {
            "nodes": cy_nodes,
            "edges": cy_edges,
            "stats": {
                "node_count": len(cy_nodes),
                "edge_count": len(cy_edges),
                "calibrated_edges": sum(1 for e in edges if e.is_calibrated),
                "stale_edges": sum(1 for e in edges if e.is_stale),
                "contradicted_edges": sum(1 for e in edges if e.is_contradicted),
            },
            "graph_id": str(graph.id),
            "graph_name": graph.name,
            "lens": lens,
            "empty": False,
        }
    )


@require_auth
@require_http_methods(["GET"])
def node_detail(request, node_id):
    """Return detailed node data including evidence on connected edges."""
    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    try:
        node = GraphService.get_node(tenant_id, node_id)
    except Exception:
        return JsonResponse({"error": "Node not found"}, status=404)

    upstream = GraphService.get_upstream(tenant_id, node.id, depth=1)
    downstream = GraphService.get_downstream(tenant_id, node.id, depth=1)

    return JsonResponse(
        {
            "id": str(node.id),
            "name": node.name,
            "node_type": node.node_type,
            "description": node.description,
            "unit": node.unit,
            "controllability": node.controllability,
            "control_method": node.control_method,
            "distribution": node.distribution,
            "spec_limits": node.spec_limits,
            "control_limits": node.control_limits,
            "has_spc": node.linked_spc_chart is not None,
            "has_equipment": bool(node.linked_equipment),
            "provenance": node.provenance,
            "upstream": [{"id": str(n.id), "name": n.name} for n in upstream],
            "downstream": [{"id": str(n.id), "name": n.name} for n in downstream],
        }
    )


@require_auth
@require_http_methods(["GET"])
def edge_detail(request, edge_id):
    """Return detailed edge data with full evidence stack."""
    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    try:
        explanation = GraphService.explain_edge(tenant_id, edge_id)
    except Exception:
        return JsonResponse({"error": "Edge not found"}, status=404)

    return JsonResponse(explanation)


@require_auth
@require_http_methods(["GET"])
def gap_report(request):
    """Return gap analysis for the org's graph."""
    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    graph_id = request.GET.get("graph_id")
    if not graph_id:
        graph = GraphService.get_org_graph(tenant_id)
        if not graph:
            return JsonResponse({"total_gaps": 0, "empty": True})
        graph_id = graph.id

    report = GraphService.gap_report(tenant_id, graph_id)
    return JsonResponse(report.to_dict())
