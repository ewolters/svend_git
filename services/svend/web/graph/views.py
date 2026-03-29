"""
Graph API views — GRAPH-001 §11, §15.

Endpoints for the graph navigator frontend. All endpoints enforce
tenant isolation via request.user.membership.tenant.
"""

import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth

from .models import ProcessGraph, ProcessNode
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


@require_auth
@require_http_methods(["POST"])
def seed_from_fmis(request):
    """Propose graph structure from FMIS rows.

    POST body: {"fmis_id": uuid} — seeds from all rows in the FMIS document.
    Returns proposals for user confirmation.
    """
    import json

    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    data = json.loads(request.body)
    fmis_id = data.get("fmis_id")
    if not fmis_id:
        return JsonResponse({"error": "fmis_id required"}, status=400)

    # Get or create the org graph
    graph = GraphService.get_or_create_org_graph(tenant_id, created_by=request.user)

    # Load FMIS rows
    from loop.models import FMISRow

    rows = list(FMISRow.objects.filter(fmis_id=fmis_id, fmis__tenant=tenant_id))
    if not rows:
        return JsonResponse({"error": "No FMIS rows found"}, status=404)

    proposals = GraphService.seed_from_fmis(tenant_id, graph.id, rows)
    return JsonResponse({"proposals": proposals, "graph_id": str(graph.id)})


@require_auth
@require_http_methods(["POST"])
def confirm_seed(request):
    """Confirm proposed graph structure from FMIS seeding.

    POST body: {"graph_id": uuid, "proposals": [...]}
    """
    import json

    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    data = json.loads(request.body)
    graph_id = data.get("graph_id")
    proposals = data.get("proposals", [])

    if not graph_id or not proposals:
        return JsonResponse({"error": "graph_id and proposals required"}, status=400)

    result = GraphService.confirm_seed(tenant_id, graph_id, proposals, confirmed_by=request.user)

    return JsonResponse(
        {
            "created_nodes": len(result["created_nodes"]),
            "created_edges": len(result["created_edges"]),
            "node_names": [n.name for n in result["created_nodes"]],
        }
    )


# =============================================================================
# S1-3: Unified Search Endpoint (for command palette)
# =============================================================================


@require_auth
@require_http_methods(["GET"])
def search(request):
    """Unified search across all entity types for command palette.

    GET /api/graph/search/?q=<query>&types=signal,commitment,...
    Returns top 10 results ranked by relevance.
    """
    from django.db.models import Q

    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    q = request.GET.get("q", "").strip()
    if len(q) < 2:
        return JsonResponse({"results": []})

    results = []

    # Search ProcessNodes
    for node in ProcessNode.objects.filter(graph__tenant_id=tenant_id, name__icontains=q)[:5]:
        results.append(
            {
                "type": "node",
                "id": str(node.id),
                "title": node.name,
                "subtitle": node.get_node_type_display(),
                "url": f"/app/process-map/?node={node.id}",
            }
        )

    # Search Signals
    from loop.models import Commitment, Signal, SupplierClaim

    for sig in Signal.objects.filter(tenant_id=tenant_id).filter(Q(title__icontains=q) | Q(description__icontains=q))[
        :5
    ]:
        results.append(
            {
                "type": "signal",
                "id": str(sig.id),
                "title": sig.title,
                "subtitle": f"{sig.severity} — {sig.triage_state}",
                "url": f"/app/loop/detect/signals/?id={sig.id}",
                "status": sig.triage_state,
            }
        )

    # Search Commitments
    for cmt in Commitment.objects.filter(tenant_id=tenant_id).filter(
        Q(title__icontains=q) | Q(description__icontains=q)
    )[:5]:
        results.append(
            {
                "type": "commitment",
                "id": str(cmt.id),
                "title": cmt.title,
                "subtitle": f"{cmt.status} — due {cmt.due_date}" if cmt.due_date else cmt.status,
                "url": f"/app/loop/standardize/commitments/?id={cmt.id}",
                "status": cmt.status,
            }
        )

    # Search Supplier Claims
    for claim in (
        SupplierClaim.objects.filter(tenant_id=tenant_id)
        .filter(Q(title__icontains=q) | Q(description__icontains=q))
        .select_related("supplier")[:5]
    ):
        results.append(
            {
                "type": "claim",
                "id": str(claim.id),
                "title": claim.title,
                "subtitle": f"{claim.supplier.name} — {claim.status}",
                "url": f"/app/loop/detect/supplier/?claim={claim.id}",
                "status": claim.status,
            }
        )

    # Search Suppliers
    from agents_api.models import ControlledDocument, NonconformanceRecord, SupplierRecord

    for sup in SupplierRecord.objects.filter(owner__memberships__tenant_id=tenant_id).filter(
        Q(name__icontains=q) | Q(contact_email__icontains=q)
    )[:5]:
        results.append(
            {
                "type": "supplier",
                "id": str(sup.id),
                "title": sup.name,
                "subtitle": f"Supplier — {sup.status}",
                "url": f"/app/loop/detect/supplier/?supplier={sup.id}",
            }
        )

    # Search NCRs
    for ncr in NonconformanceRecord.objects.filter(owner__memberships__tenant_id=tenant_id).filter(
        Q(title__icontains=q) | Q(description__icontains=q)
    )[:5]:
        results.append(
            {
                "type": "ncr",
                "id": str(ncr.id),
                "title": ncr.title,
                "subtitle": f"NCR — {ncr.status}",
                "url": f"/app/loop/standardize/ncr/?id={ncr.id}",
                "status": ncr.status,
            }
        )

    # Search Documents
    for doc in ControlledDocument.objects.filter(owner__memberships__tenant_id=tenant_id).filter(
        Q(title__icontains=q) | Q(document_number__icontains=q)
    )[:5]:
        results.append(
            {
                "type": "document",
                "id": str(doc.id),
                "title": doc.title,
                "subtitle": f"Doc {doc.document_number} — {doc.status}",
                "url": f"/app/loop/standardize/documents/?id={doc.id}",
            }
        )

    # Search Investigations
    from core.models.investigation import Investigation

    for inv in Investigation.objects.filter(tenant_id=tenant_id).filter(
        Q(title__icontains=q) | Q(description__icontains=q)
    )[:5]:
        results.append(
            {
                "type": "investigation",
                "id": str(inv.id),
                "title": inv.title,
                "subtitle": f"Investigation — {inv.status}",
                "url": f"/app/loop/investigations/{inv.id}/",
            }
        )

    # Sort by relevance (exact match > starts with > contains)
    q_lower = q.lower()
    for r in results:
        title_lower = r["title"].lower()
        if title_lower == q_lower:
            r["_score"] = 3
        elif title_lower.startswith(q_lower):
            r["_score"] = 2
        else:
            r["_score"] = 1

    results.sort(key=lambda r: r["_score"], reverse=True)

    # Remove internal score, cap at 10
    for r in results:
        r.pop("_score", None)

    return JsonResponse({"results": results[:10], "query": q})


# =============================================================================
# S1-4: Knowledge Health Metrics
# =============================================================================


@require_auth
@require_http_methods(["GET"])
def knowledge_health(request):
    """Compute and return knowledge health metrics for the org's graph."""
    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    graph = GraphService.get_org_graph(tenant_id)
    if not graph:
        return JsonResponse({"empty": True, "maturity_level": 0})

    health = GraphService.compute_knowledge_health(tenant_id, graph.id)
    return JsonResponse(health)


# =============================================================================
# S1-6: Activity Feed Endpoint
# =============================================================================


@require_auth
@require_http_methods(["GET"])
def activity_feed(request, entity_type, entity_id):
    """Unified activity feed for any entity.

    Returns chronological events from all sources:
    status changes, notes, evidence, resource assignments, graph events.
    """
    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    events = []

    if entity_type == "commitment":
        from loop.models import Commitment

        try:
            cmt = Commitment.objects.get(id=entity_id, tenant_id=tenant_id)
        except Commitment.DoesNotExist:
            return JsonResponse({"error": "Not found"}, status=404)

        for note in cmt.notes.all().order_by("-created_at")[:20]:
            events.append(
                {
                    "type": "note",
                    "timestamp": note.created_at.isoformat(),
                    "author": note.author.get_full_name() or note.author.email if note.author_id else "System",
                    "description": note.text[:200],
                    "is_system": note.is_system_note if hasattr(note, "is_system_note") else False,
                }
            )

        for res in cmt.resources.all().order_by("-created_at")[:10]:
            events.append(
                {
                    "type": "resource_assignment",
                    "timestamp": res.created_at.isoformat(),
                    "author": "",
                    "description": f"{res.employee.name} assigned as {res.role} ({res.status})",
                    "is_system": True,
                }
            )

    elif entity_type == "signal":
        from loop.models import Signal

        try:
            sig = Signal.objects.get(id=entity_id, tenant_id=tenant_id)
        except Signal.DoesNotExist:
            return JsonResponse({"error": "Not found"}, status=404)

        if sig.triaged_at:
            events.append(
                {
                    "type": "triage",
                    "timestamp": sig.triaged_at.isoformat(),
                    "author": "",
                    "description": f"Triaged → {sig.triage_state}",
                    "is_system": True,
                }
            )
        if sig.resolved_at:
            events.append(
                {
                    "type": "resolution",
                    "timestamp": sig.resolved_at.isoformat(),
                    "author": "",
                    "description": f"Resolved via investigation {sig.resolved_by_investigation_id}",
                    "is_system": True,
                }
            )

    elif entity_type == "claim":
        from loop.models import SupplierClaim

        try:
            claim = SupplierClaim.objects.get(id=entity_id, tenant_id=tenant_id)
        except SupplierClaim.DoesNotExist:
            return JsonResponse({"error": "Not found"}, status=404)

        for resp in claim.responses.all().order_by("revision"):
            events.append(
                {
                    "type": "response",
                    "timestamp": resp.created_at.isoformat(),
                    "author": "Supplier",
                    "description": f"Response #{resp.revision} — quality score: {resp.response_quality_score or '—'}",
                    "is_system": False,
                }
            )
            if resp.reviewed_at:
                events.append(
                    {
                        "type": "review",
                        "timestamp": resp.reviewed_at.isoformat(),
                        "author": resp.reviewer.get_full_name() if resp.reviewer else "",
                        "description": f"{'Accepted' if resp.accepted else 'Rejected'}: {resp.reviewer_notes[:100]}",
                        "is_system": False,
                    }
                )

        for ver in claim.verifications.all().order_by("-verified_at"):
            events.append(
                {
                    "type": "verification",
                    "timestamp": ver.verified_at.isoformat(),
                    "author": ver.verified_by.get_full_name() if ver.verified_by else "",
                    "description": f"Verification ({ver.verification_type}): {ver.result}",
                    "is_system": False,
                }
            )

    elif entity_type == "edge":
        from graph.models import EdgeEvidence

        for ev in EdgeEvidence.objects.filter(edge_id=entity_id, edge__graph__tenant_id=tenant_id).order_by(
            "-observed_at"
        )[:20]:
            events.append(
                {
                    "type": "evidence",
                    "timestamp": ev.observed_at.isoformat(),
                    "author": ev.created_by.get_full_name() if ev.created_by else "",
                    "description": f"{ev.source_type}: {ev.source_description[:150]}",
                    "is_system": False,
                    "retracted": ev.retracted,
                }
            )

    events.sort(key=lambda e: e["timestamp"], reverse=True)
    return JsonResponse({"events": events[:50], "entity_type": entity_type, "entity_id": str(entity_id)})


# =============================================================================
# S1-7: Gates Endpoint
# =============================================================================


@require_auth
@require_http_methods(["GET"])
def workflow_gates(request, entity_type, entity_id):
    """Return workflow gates (prerequisites) for lifecycle transitions.

    Shows what actions are available and what blocks each one.
    """
    tenant_id = _get_tenant_id(request)
    if not tenant_id:
        return JsonResponse({"error": "No tenant"}, status=400)

    gates = []

    if entity_type == "signal":
        from loop.models import Signal

        try:
            sig = Signal.objects.get(id=entity_id, tenant_id=tenant_id)
        except Signal.DoesNotExist:
            return JsonResponse({"error": "Not found"}, status=404)

        if sig.triage_state == "untriaged":
            gates.append({"action": "acknowledge", "met": True, "description": "Acknowledge this signal"})
            gates.append({"action": "investigate", "met": True, "description": "Open investigation"})
            gates.append({"action": "dismiss", "met": True, "description": "Dismiss with reason"})

    elif entity_type == "commitment":
        from loop.models import Commitment

        try:
            cmt = Commitment.objects.get(id=entity_id, tenant_id=tenant_id)
        except Commitment.DoesNotExist:
            return JsonResponse({"error": "Not found"}, status=404)

        if cmt.status == "open":
            gates.append({"action": "start", "met": True, "description": "Begin work"})
        elif cmt.status == "in_progress":
            has_artifacts = bool(cmt.linked_artifacts)
            gates.append(
                {
                    "action": "fulfill",
                    "met": has_artifacts,
                    "description": "Fulfill commitment"
                    if has_artifacts
                    else "Link at least one artifact before fulfilling",
                }
            )
            gates.append({"action": "break", "met": True, "description": "Mark as broken (with reason)"})

    elif entity_type == "claim":
        from loop.models import SupplierClaim

        try:
            claim = SupplierClaim.objects.get(id=entity_id, tenant_id=tenant_id)
        except SupplierClaim.DoesNotExist:
            return JsonResponse({"error": "Not found"}, status=404)

        valid = SupplierClaim.VALID_TRANSITIONS.get(claim.status, set())
        for target in valid:
            met = True
            desc = f"Transition to {target}"

            if target == "verified":
                has_accepted = claim.responses.filter(accepted=True).exists()
                met = has_accepted
                desc = "Verify — accepted response required" if not has_accepted else "Verify claim resolution"
            elif target == "closed":
                has_verification = claim.verifications.exists()
                met = has_verification
                desc = "Close — verification record required" if not has_verification else "Close claim"

            gates.append({"action": target, "met": met, "description": desc})

    return JsonResponse({"gates": gates, "entity_type": entity_type, "entity_id": str(entity_id)})
