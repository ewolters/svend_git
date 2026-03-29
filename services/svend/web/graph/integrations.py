"""
Graph integrations — GRAPH-001 §11.2, Phase 2 wiring.

Bridge functions that connect existing tools to GraphService.add_evidence().
Each function extracts the right fields from a tool's output and writes
evidence to the appropriate graph edge.

All functions are graceful — if no graph or edge exists, they log and return None.
Tool behavior is never blocked by graph integration.
"""

import logging
import math
from uuid import UUID

from django.utils import timezone

from .models import ProcessEdge, ProcessGraph, ProcessNode
from .service import GraphService

logger = logging.getLogger("svend.graph.integrations")


def _find_tenant_graph(tenant_id: UUID) -> ProcessGraph | None:
    """Find the org's primary process graph. Returns None if none exists."""
    return GraphService.get_org_graph(tenant_id)


def _find_edges_for_fmis_row(graph: ProcessGraph, fmis_row_id: UUID) -> list[ProcessEdge]:
    """Find graph edges that were seeded from a specific FMIS row.

    Checks both source and target nodes' linked_fmis_rows JSON arrays.
    """
    edges = []
    nodes_with_row = ProcessNode.objects.filter(
        graph=graph,
        linked_fmis_rows__contains=[str(fmis_row_id)],
    )
    node_ids = set(nodes_with_row.values_list("id", flat=True))
    if node_ids:
        edges = list(
            ProcessEdge.objects.filter(
                graph=graph,
                source_id__in=node_ids,
            ).union(
                ProcessEdge.objects.filter(
                    graph=graph,
                    target_id__in=node_ids,
                )
            )
        )
    return edges


def _find_edges_for_node(graph: ProcessGraph, node_id: UUID) -> list[ProcessEdge]:
    """Find all edges touching a node (as source or target)."""
    from django.db import models as db_models

    return list(
        ProcessEdge.objects.filter(graph=graph).filter(db_models.Q(source_id=node_id) | db_models.Q(target_id=node_id))
    )


def _find_node_by_spc_chart(graph: ProcessGraph, chart_id: UUID) -> ProcessNode | None:
    """Find graph node linked to an SPC chart."""
    return graph.nodes.filter(linked_spc_chart=chart_id).first()


# =============================================================================
# DOE → Graph Evidence (Phase 2, item 14)
# =============================================================================


def doe_to_graph_evidence(
    tenant_id: UUID,
    coefficients: list[dict],
    sample_size: int,
    response_name: str = "",
    experiment_id: UUID | None = None,
    user=None,
) -> list:
    """Write DOE analysis results as graph edge evidence.

    Each significant coefficient becomes evidence on the edge between
    the factor node and the response node.

    Args:
        coefficients: from DOE analyze_results, each has:
            term, effect, coefficient, se_coef, t_value, p_value, significant
        sample_size: total observations in the DOE
        response_name: name of the response variable
        experiment_id: UUID of the ExperimentDesign
        user: requesting user

    Returns list of created EdgeEvidence records (may be empty).
    """
    graph = _find_tenant_graph(tenant_id)
    if not graph:
        logger.debug("DOE→graph: no graph for tenant %s", tenant_id)
        return []

    created = []
    for coef in coefficients:
        if not coef.get("significant"):
            continue

        term = coef.get("term", "")
        effect = coef.get("effect")
        p_value = coef.get("p_value")
        se_coef = coef.get("se_coef")

        if not term or effect is None:
            continue

        # Find source node (factor) and target node (response) by name
        source_node = graph.nodes.filter(name__iexact=term).first()
        target_node = graph.nodes.filter(name__iexact=response_name).first() if response_name else None

        if not source_node or not target_node:
            logger.debug(
                "DOE→graph: no nodes for %s → %s in graph %s",
                term,
                response_name,
                graph.id,
            )
            continue

        # Find or skip edge
        edge = ProcessEdge.objects.filter(graph=graph, source=source_node, target=target_node).first()
        if not edge:
            logger.debug(
                "DOE→graph: no edge %s → %s in graph %s",
                source_node.name,
                target_node.name,
                graph.id,
            )
            continue

        # Compute CI from SE if available
        ci = None
        if se_coef and se_coef > 0:
            t_crit = 1.96  # approximate for large n
            ci = {
                "lower": effect - t_crit * se_coef * 2,  # effect = 2*coef
                "upper": effect + t_crit * se_coef * 2,
            }

        ev = GraphService.add_evidence(
            tenant_id=tenant_id,
            edge_id=edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            source_description=f"DOE: {term} → {response_name}, effect={effect:.4f}, p={p_value:.4f}"
            if p_value
            else f"DOE: {term} → {response_name}, effect={effect:.4f}",
            effect_size=effect,
            confidence_interval=ci,
            sample_size=sample_size,
            p_value=p_value,
            strength=0.95,  # DOE is high-quality evidence
            source_id=experiment_id,
            created_by=user,
        )
        created.append(ev)
        logger.info("DOE→graph: evidence on %s → %s (effect=%.4f)", source_node.name, target_node.name, effect)

    return created


# =============================================================================
# SPC → Graph Node Distribution + Staleness (Phase 2, item 15)
# =============================================================================


def spc_to_graph(
    tenant_id: UUID,
    chart_id: UUID | None = None,
    node_name: str | None = None,
    center_line: float | None = None,
    ucl: float | None = None,
    lcl: float | None = None,
    in_control: bool = True,
    out_of_control_count: int = 0,
    data_points: list | None = None,
    chart_type: str = "",
    user=None,
) -> dict:
    """Update graph from SPC analysis.

    Two operations:
    1. Update node distribution (always, if node found)
    2. Flag stale edges (only if out of control, per conference D8 debounce)

    Returns dict with what was done.
    """
    graph = _find_tenant_graph(tenant_id)
    if not graph:
        return {"status": "no_graph"}

    # Find the node
    node = None
    if chart_id:
        node = _find_node_by_spc_chart(graph, chart_id)
    if not node and node_name:
        node = graph.nodes.filter(name__iexact=node_name).first()

    if not node:
        return {"status": "no_node"}

    result = {"status": "ok", "node": str(node.id), "actions": []}

    # 1. Update node distribution
    if data_points and center_line is not None:
        n = len(data_points)
        std = (sum((x - center_line) ** 2 for x in data_points) / max(n - 1, 1)) ** 0.5 if n > 1 else 0
        GraphService.update_node_distribution(
            tenant_id,
            node.id,
            {
                "mean": center_line,
                "std": round(std, 6),
                "n": n,
                "source": f"spc_{chart_type}",
                "as_of": timezone.now().isoformat(),
            },
        )
        result["actions"].append("distribution_updated")

    # Update control limits on node
    if ucl is not None or lcl is not None:
        GraphService.update_node(
            tenant_id,
            node.id,
            control_limits={"ucl": ucl, "lcl": lcl, "cl": center_line},
        )
        result["actions"].append("control_limits_updated")

    # 2. Flag stale edges if out of control
    if not in_control and out_of_control_count > 0:
        stale_edges = GraphService.flag_stale_edges(
            tenant_id, node.id, reason=f"spc_shift_{chart_type}_{out_of_control_count}_ooc"
        )
        result["actions"].append(f"flagged_{len(stale_edges)}_stale_edges")
        logger.info(
            "SPC→graph: %d edges flagged stale for node %s (%d OOC points)",
            len(stale_edges),
            node.name,
            out_of_control_count,
        )

    return result


# =============================================================================
# FFT → Graph Evidence (Phase 2, item 19)
# =============================================================================


def fft_to_graph_evidence(
    tenant_id: UUID,
    fft_id: UUID,
    fmis_row_id: UUID | None,
    detection_count: int,
    injection_count: int,
    result_code: str,
    control_being_tested: str = "",
    conducted_at=None,
    user=None,
) -> list:
    """Write ForcedFailureTest result as graph edge evidence.

    Detection rate becomes evidence on edges linked to the FMIS row.
    """
    graph = _find_tenant_graph(tenant_id)
    if not graph:
        return []

    if injection_count == 0:
        return []

    detection_rate = detection_count / injection_count

    # Map result code to evidence strength
    strength_map = {
        "detected": 0.95,
        "partially_detected": 0.6,
        "not_detected": 0.3,
    }
    strength = strength_map.get(result_code, 0.5)

    # Find edges via FMIS row linkage
    edges = []
    if fmis_row_id:
        edges = _find_edges_for_fmis_row(graph, fmis_row_id)

    if not edges:
        logger.debug("FFT→graph: no edges for FMIS row %s", fmis_row_id)
        return []

    created = []
    for edge in edges:
        # Binomial CI on detection rate
        if injection_count >= 2:
            se = math.sqrt(detection_rate * (1 - detection_rate) / injection_count)
            ci = {
                "lower": max(0, detection_rate - 1.96 * se),
                "upper": min(1, detection_rate + 1.96 * se),
            }
        else:
            ci = None

        ev = GraphService.add_evidence(
            tenant_id=tenant_id,
            edge_id=edge.id,
            source_type="forced_failure_test",
            observed_at=conducted_at or timezone.now(),
            source_description=f"FFT: {detection_count}/{injection_count} detected. Control: {control_being_tested}",
            effect_size=detection_rate,
            confidence_interval=ci,
            sample_size=injection_count,
            strength=strength,
            source_id=fft_id,
            created_by=user,
        )
        created.append(ev)

    logger.info("FFT→graph: %d evidence records from FFT %s", len(created), fft_id)
    return created


# =============================================================================
# PC → Graph Evidence (Phase 2, item 18)
# =============================================================================


def pc_to_graph_evidence(
    tenant_id: UUID,
    pc_id: UUID,
    diagnosis: str,
    pass_rate: float | None,
    observation_count: int,
    process_area: str = "",
    controlled_document_name: str = "",
    linked_node_ids: list[UUID] | None = None,
    created_at=None,
    user=None,
) -> list:
    """Write ProcessConfirmation result as graph edge evidence.

    Diagnosis maps to evidence on edges connected to linked nodes.
    """
    graph = _find_tenant_graph(tenant_id)
    if not graph:
        return []

    # Map diagnosis to effect size and strength
    diagnosis_map = {
        "system_works": {"effect_size": 1.0, "strength": 0.7, "p_approx": 1.0},
        "standard_unclear": {"effect_size": -0.3, "strength": 0.5, "p_approx": 0.1},
        "process_gap": {"effect_size": -1.0, "strength": 0.9, "p_approx": 0.001},
        "incomplete": None,
    }

    mapping = diagnosis_map.get(diagnosis)
    if not mapping:
        return []

    # Find edges via linked nodes
    edges = []
    if linked_node_ids:
        for node_id in linked_node_ids:
            edges.extend(_find_edges_for_node(graph, node_id))
    # Deduplicate
    seen = set()
    unique_edges = []
    for e in edges:
        if e.id not in seen:
            seen.add(e.id)
            unique_edges.append(e)

    if not unique_edges:
        logger.debug("PC→graph: no edges for linked nodes %s", linked_node_ids)
        return []

    created = []
    for edge in unique_edges:
        ci = None
        if pass_rate is not None and observation_count >= 2:
            se = math.sqrt(pass_rate * (1 - pass_rate) / max(observation_count, 1))
            ci = {
                "lower": max(0, pass_rate - 1.96 * se),
                "upper": min(1, pass_rate + 1.96 * se),
            }

        ev = GraphService.add_evidence(
            tenant_id=tenant_id,
            edge_id=edge.id,
            source_type="process_confirmation",
            observed_at=created_at or timezone.now(),
            source_description=f"PC: {diagnosis}. Pass rate: {pass_rate:.0%}. Doc: {controlled_document_name}"
            if pass_rate is not None
            else f"PC: {diagnosis}. Doc: {controlled_document_name}",
            effect_size=mapping["effect_size"],
            confidence_interval=ci,
            sample_size=observation_count,
            p_value=mapping["p_approx"],
            strength=mapping["strength"],
            source_id=pc_id,
            created_by=user,
        )
        created.append(ev)

    logger.info("PC→graph: %d evidence records from PC %s (%s)", len(created), pc_id, diagnosis)
    return created


# =============================================================================
# Investigation → Graph (Phase 2, items 16-17)
# =============================================================================


def scope_investigation_from_graph(
    tenant_id: UUID,
    graph_id: UUID,
    node_ids: list[UUID],
    include_neighbors: bool = True,
) -> dict:
    """Create a subgraph snapshot for an investigation.

    Returns a dict with nodes, edges, and their current state.
    The investigation stores this as its starting context.
    """
    graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)

    selected_nodes = set(node_ids)

    if include_neighbors:
        for node_id in list(selected_nodes):
            # Add 1-hop neighbors
            for edge in ProcessEdge.objects.filter(graph=graph, source_id=node_id):
                selected_nodes.add(edge.target_id)
            for edge in ProcessEdge.objects.filter(graph=graph, target_id=node_id):
                selected_nodes.add(edge.source_id)

    nodes = list(ProcessNode.objects.filter(id__in=selected_nodes, graph=graph))
    edges = list(
        ProcessEdge.objects.filter(
            graph=graph,
            source_id__in=selected_nodes,
            target_id__in=selected_nodes,
        ).select_related("source", "target")
    )

    return {
        "graph_id": str(graph_id),
        "scoped_at": timezone.now().isoformat(),
        "nodes": [
            {
                "id": str(n.id),
                "name": n.name,
                "node_type": n.node_type,
                "distribution": n.distribution,
            }
            for n in nodes
        ],
        "edges": [
            {
                "id": str(e.id),
                "source": str(e.source_id),
                "target": str(e.target_id),
                "source_name": e.source.name,
                "target_name": e.target.name,
                "relation_type": e.relation_type,
                "posterior_strength": e.posterior_strength,
                "effect_size": e.effect_size,
                "is_calibrated": e.is_calibrated,
                "is_stale": e.is_stale,
                "evidence_count": e.evidence_count,
            }
            for e in edges
        ],
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def write_back_from_investigation(
    tenant_id: UUID,
    graph_id: UUID,
    proposed_changes: list[dict],
    confirmed_by=None,
) -> dict:
    """Write investigation findings back to the graph.

    Proposed changes:
    - {"type": "new_node", "name": ..., "node_type": ...}
    - {"type": "new_edge", "source_name": ..., "target_name": ..., "relation_type": ...}
    - {"type": "new_evidence", "edge_id": ..., "source_type": ..., "effect_size": ..., ...}

    Returns summary of what was created.
    """
    from .synara_adapter import SynaraAdapter

    graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)
    result = {"created_nodes": [], "created_edges": [], "added_evidence": [], "contradictions": []}

    node_map = {n.name.lower(): n for n in graph.nodes.all()}

    for change in proposed_changes:
        change_type = change.get("type")

        if change_type == "new_node":
            name = change.get("name", "")
            if name.lower() not in node_map:
                node = GraphService.add_node(
                    tenant_id,
                    graph_id,
                    name=name,
                    node_type=change.get("node_type", "process_parameter"),
                    provenance="investigation",
                    created_by=confirmed_by,
                )
                node_map[name.lower()] = node
                result["created_nodes"].append(str(node.id))

        elif change_type == "new_edge":
            source_name = change.get("source_name", "")
            target_name = change.get("target_name", "")
            source = node_map.get(source_name.lower())
            target = node_map.get(target_name.lower())
            if source and target:
                exists = ProcessEdge.objects.filter(graph=graph, source=source, target=target).exists()
                if not exists:
                    edge = GraphService.add_edge(
                        tenant_id,
                        graph_id,
                        source.id,
                        target.id,
                        relation_type=change.get("relation_type", "causal"),
                        provenance="investigation",
                        created_by=confirmed_by,
                    )
                    result["created_edges"].append(str(edge.id))

        elif change_type == "new_evidence":
            edge_id = change.get("edge_id")
            if edge_id:
                ev = GraphService.add_evidence(
                    tenant_id=tenant_id,
                    edge_id=UUID(edge_id) if isinstance(edge_id, str) else edge_id,
                    source_type=change.get("source_type", "investigation"),
                    observed_at=timezone.now(),
                    source_description=change.get("description", ""),
                    effect_size=change.get("effect_size"),
                    confidence_interval=change.get("confidence_interval"),
                    sample_size=change.get("sample_size"),
                    p_value=change.get("p_value"),
                    strength=change.get("strength", 0.8),
                    source_id=change.get("investigation_id"),
                    created_by=confirmed_by,
                )
                result["added_evidence"].append(str(ev.id))

                # Check for contradiction
                contradiction = SynaraAdapter.check_edge_contradiction(
                    tenant_id,
                    ev.edge_id,
                    ev,
                )
                if contradiction:
                    result["contradictions"].append(contradiction)

    logger.info(
        "Investigation writeback: %d nodes, %d edges, %d evidence, %d contradictions",
        len(result["created_nodes"]),
        len(result["created_edges"]),
        len(result["added_evidence"]),
        len(result["contradictions"]),
    )
    return result


# =============================================================================
# QMS → Graph Bridges (Systems 1-7 alignment)
# =============================================================================


def equipment_calibration_to_graph(
    tenant_id: UUID,
    equipment_id: UUID,
    calibration_result: str = "pass",
    calibration_date=None,
    certificate_number: str = "",
    user=None,
) -> list:
    """Record a calibration event as graph evidence on measurement edges.

    Equipment must have linked_process_node set. The calibration result
    becomes evidence on all edges where this equipment's node is source or target.
    """
    from agents_api.models import MeasurementEquipment

    graph = _find_tenant_graph(tenant_id)
    if not graph:
        return []

    try:
        equip = MeasurementEquipment.objects.get(id=equipment_id)
    except MeasurementEquipment.DoesNotExist:
        return []

    if not equip.linked_process_node_id:
        logger.debug("Equipment %s has no linked ProcessNode", equipment_id)
        return []

    edges = _find_edges_for_node(graph, equip.linked_process_node_id)
    if not edges:
        return []

    effect = 1.0 if calibration_result == "pass" else -0.5
    strength = 0.85 if calibration_result == "pass" else 0.9

    created = []
    for edge in edges:
        ev = GraphService.add_evidence(
            tenant_id=tenant_id,
            edge_id=edge.id,
            source_type="gage_rr",
            observed_at=calibration_date or timezone.now(),
            source_description=f"Calibration {'PASS' if calibration_result == 'pass' else 'FAIL'}: {equip.name}. Cert: {certificate_number}",
            effect_size=effect,
            strength=strength,
            source_id=equipment_id,
            created_by=user,
        )
        created.append(ev)

    logger.info("Equipment calibration→graph: %d evidence records for %s", len(created), equip.name)
    return created


def ncr_to_graph_context(
    tenant_id: UUID,
    ncr_id: UUID,
    linked_node_ids: list[UUID],
) -> dict:
    """Link an NCR to graph nodes for traceability.

    Does NOT create evidence — NCRs are reactive events, not calibration.
    Returns context about which edges are affected for investigation scoping.
    """
    graph = _find_tenant_graph(tenant_id)
    if not graph:
        return {"status": "no_graph", "affected_edges": []}

    affected_edges = []
    for node_id in linked_node_ids:
        edges = _find_edges_for_node(graph, node_id)
        affected_edges.extend(edges)

    # Deduplicate
    seen = set()
    unique = []
    for e in affected_edges:
        if e.id not in seen:
            seen.add(e.id)
            unique.append(e)

    return {
        "status": "ok",
        "ncr_id": str(ncr_id),
        "affected_edges": [
            {
                "id": str(e.id),
                "source": e.source.name if hasattr(e, "source") else str(e.source_id),
                "target": e.target.name if hasattr(e, "target") else str(e.target_id),
                "posterior": e.posterior_strength,
                "is_calibrated": e.is_calibrated,
            }
            for e in unique
        ],
        "suggested_investigation_nodes": [str(n) for n in linked_node_ids],
    }


def audit_finding_to_graph_context(
    tenant_id: UUID,
    audit_id: UUID,
    finding_severity: str,
    linked_node_ids: list[UUID],
    finding_description: str = "",
) -> dict:
    """Link an audit finding to graph context.

    Major/critical findings suggest investigation scoping.
    Does NOT auto-create Signals — that's the user's decision (HiTL).
    """
    graph = _find_tenant_graph(tenant_id)
    if not graph:
        return {"status": "no_graph"}

    affected_edges = []
    for node_id in linked_node_ids:
        affected_edges.extend(_find_edges_for_node(graph, node_id))

    seen = set()
    unique = [e for e in affected_edges if e.id not in seen and not seen.add(e.id)]

    return {
        "status": "ok",
        "audit_id": str(audit_id),
        "severity": finding_severity,
        "affected_edges": len(unique),
        "suggested_action": "investigate" if finding_severity in ("major", "critical") else "monitor",
        "suggested_investigation_nodes": [str(n) for n in linked_node_ids],
    }


def training_gaps_from_graph(
    tenant_id: UUID,
    graph_id: UUID | None = None,
) -> list[dict]:
    """Identify training needs from graph gaps (System 6 — reader).

    Returns list of gap-derived training suggestions:
    - Uncalibrated edges → "Training needed on measurement/DOE for this relationship"
    - Measurement gaps → "No measurement system — competency needed"
    - Stale edges → "Recalibration training needed"
    """
    graph = _find_tenant_graph(tenant_id) if not graph_id else None
    if graph_id:
        try:
            graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)
        except ProcessGraph.DoesNotExist:
            return []
    if not graph:
        return []

    report = GraphService.gap_report(tenant_id, graph.id)
    suggestions = []

    for edge in report.uncalibrated_edges:
        suggestions.append(
            {
                "type": "uncalibrated_edge",
                "description": f"Relationship {edge.source.name} → {edge.target.name} has no empirical evidence. Consider DOE training.",
                "priority": "medium",
                "node_ids": [str(edge.source_id), str(edge.target_id)],
            }
        )

    for node in report.measurement_gaps:
        suggestions.append(
            {
                "type": "measurement_gap",
                "description": f"Node '{node.name}' has no linked measurement system. Gage R&R competency needed.",
                "priority": "high",
                "node_ids": [str(node.id)],
            }
        )

    for edge in report.stale_edges:
        suggestions.append(
            {
                "type": "stale_edge",
                "description": f"Relationship {edge.source.name} → {edge.target.name} is stale. Recalibration investigation needed.",
                "priority": "high",
                "node_ids": [str(edge.source_id), str(edge.target_id)],
            }
        )

    return suggestions


def management_review_graph_input(
    tenant_id: UUID,
    graph_id: UUID | None = None,
) -> dict:
    """Generate graph health metrics as management review input (System 7 — reader).

    Returns summary of graph state for inclusion in review agenda.
    """
    graph = _find_tenant_graph(tenant_id) if not graph_id else None
    if graph_id:
        try:
            graph = ProcessGraph.objects.get(id=graph_id, tenant_id=tenant_id)
        except ProcessGraph.DoesNotExist:
            return {"status": "no_graph"}
    if not graph:
        return {"status": "no_graph"}

    report = GraphService.gap_report(tenant_id, graph.id)
    total_edges = graph.edges.count()
    total_nodes = graph.nodes.count()
    calibrated = graph.edges.filter(evidence_count__gt=0).exclude(provenance="fmea_assertion").count()

    return {
        "status": "ok",
        "graph_name": graph.name,
        "summary": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "calibrated_edges": calibrated,
            "calibration_rate": round(calibrated / total_edges * 100, 1) if total_edges else 0,
            "uncalibrated_edges": len(report.uncalibrated_edges),
            "stale_edges": len(report.stale_edges),
            "contradicted_edges": len(report.contradicted_edges),
            "measurement_gaps": len(report.measurement_gaps),
            "total_gaps": report.total_gaps,
        },
        "top_priorities": [
            {
                "type": "contradicted",
                "count": len(report.contradicted_edges),
                "action": "Investigate edge contradictions — evidence conflicts with current model",
            },
            {
                "type": "stale",
                "count": len(report.stale_edges),
                "action": "Recalibrate stale edges — process may have changed since last evidence",
            },
            {
                "type": "uncalibrated",
                "count": len(report.uncalibrated_edges),
                "action": "Run DOEs to calibrate FMEA assertions with empirical data",
            },
        ],
    }
