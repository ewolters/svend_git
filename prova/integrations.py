"""
PROVA Integrations — tool-specific handlers for the bridge.

Each integration function knows how to interpret a specific tool's output
and translate it into PROVA's language (evidence, hypotheses, graph edits).

These are called from the tool's own views after the tool produces results.
They are NOT patches on existing code — they are PROVA's native understanding
of what each tool's output means for the knowledge graph.
"""

from __future__ import annotations

import logging
from typing import Optional

from . import engine
from .bridge import FindingSpec, IntegrationResult, integrate
from .models import (
    EdgeStatus,
    GraphEdge,
    ProvaHypothesis,
    WorkingGraph,
)

logger = logging.getLogger("prova.integrations")


# =============================================================================
# SPC INTEGRATION
# =============================================================================


def on_spc_result(
    user,
    tenant,
    dsw_result,
    result_data: dict,
    analysis_id: str,
) -> IntegrationResult:
    """Called when an SPC analysis completes.

    SPC is an inference tool — it produces evidence about process behavior.
    OOC signals are expectation violations that should appear in the graph.

    Args:
        user: Request user
        tenant: User's tenant
        dsw_result: The DSWResult model instance
        result_data: The full analysis result dict
        analysis_id: "imr", "xbar_r", "xbar_s", "p_chart", etc.
    """
    stats = result_data.get("statistics", {})
    n_ooc = stats.get("n_ooc", 0)
    ooc_indices = result_data.get("what_if_data", {}).get("ooc_indices", [])

    # Build summary from the SPC result
    summary_parts = [f"SPC {analysis_id.upper()} analysis"]
    if stats.get("grand_mean") is not None:
        summary_parts.append(f"mean={stats['grand_mean']:.4f}")
    if n_ooc > 0:
        summary_parts.append(f"{n_ooc} out-of-control signal(s)")
    else:
        summary_parts.append("process in statistical control")
    summary = " — ".join(summary_parts)

    # Build edge mappings: look for graph edges that involve the measured variable
    # The config should tell us what column was measured
    config = result_data.get("config", {})
    measurement_col = config.get("measurement", config.get("column", ""))

    edge_mappings = []
    if measurement_col and tenant:
        og = engine.get_or_create_operating_graph(tenant)
        # Find edges where source or target matches the measurement
        matching_edges = (
            GraphEdge.objects.filter(
                operating_graph=og,
                status=EdgeStatus.ACTIVE,
            )
            .select_related("source", "target")
            .filter(
                # Match by label (case-insensitive)
                target__label__iexact=measurement_col,
            )
        )
        for edge in matching_edges:
            direction = "supports" if n_ooc == 0 else "weakens"
            edge_mappings.append(
                {
                    "edge_id": str(edge.id),
                    "field": measurement_col,
                    "direction": direction,
                }
            )

    spec = FindingSpec(
        summary=summary,
        tool_type=f"spc_{analysis_id}" if analysis_id else "spc",
        source_id=str(dsw_result.id) if dsw_result else "",
        source_field=measurement_col,
        sample_size=stats.get("n"),
        raw_output={
            "statistics": stats,
            "summary": result_data.get("summary", "")[:500],
            "n_ooc": n_ooc,
            "ooc_indices": ooc_indices[:50],  # cap for storage
            "analysis_id": analysis_id,
        },
        edge_mappings=edge_mappings,
    )

    result = integrate(
        user=user,
        tenant=tenant,
        tool_output=dsw_result,
        spec=spec,
    )

    logger.info(
        "prova.spc.integrated",
        extra={
            "analysis_id": analysis_id,
            "n_ooc": n_ooc,
            "evidence_weight": result.evidence_weight,
            "edges_updated": result.edges_updated,
        },
    )

    return result


def on_spc_investigate_ooc(
    user,
    tenant,
    dsw_result_id: str,
    observation_index: int,
    value: float,
    rules_violated: str,
    measurement_col: str = "",
    project_id: Optional[str] = None,
) -> dict:
    """Called when user clicks "Investigate" on an OOC point.

    This is the critical handshake: PROVA creates a hypothesis in the
    working graph (the expectation violation), THEN spawns an RCA session
    linked to that hypothesis.

    Returns dict with hypothesis_id and rca_session_id for the frontend.
    """
    from rca.models import RCASession

    og = engine.get_or_create_operating_graph(tenant)

    # Create or get working graph for this user
    wg, _ = WorkingGraph.objects.get_or_create(
        tenant=tenant,
        owner=user,
        operating_graph=og,
        project_id=project_id,
        is_deleted=False,
        defaults={"state": {}},
    )

    # Build the hypothesis: "this OOC signal indicates an expectation violation"
    description = (
        f"Out-of-control signal at observation #{observation_index} (value: {value}). Rules violated: {rules_violated}."
    )
    if measurement_col:
        description += f" Measurement: {measurement_col}."

    # Find the relevant graph edge (if the measurement maps to one)
    target_edge = None
    if measurement_col:
        target_edge = GraphEdge.objects.filter(
            operating_graph=og,
            status=EdgeStatus.ACTIVE,
            target__label__iexact=measurement_col,
        ).first()

    hypothesis = ProvaHypothesis.objects.create(
        working_graph=wg,
        description=description,
        outcome_label=measurement_col or "process output",
        prior=0.5,
        status="proposed",
        created_by=str(user.id),
    )

    # If we found a matching edge, create a CHALLENGE_EDGE edit
    if target_edge:
        from .models import GraphEdit

        GraphEdit.objects.create(
            hypothesis=hypothesis,
            operation="challenge_edge",
            target_edge=target_edge,
            params={
                "reason": "OOC signal detected",
                "observation_index": observation_index,
                "value": value,
                "rules": rules_violated,
            },
        )

    # Spawn RCA session linked to this hypothesis
    event_text = (
        f"Out-of-control point at observation #{observation_index} (value: {value}). Rules violated: {rules_violated}"
    )

    rca_session = RCASession.objects.create(
        title=f"Investigate OOC #{observation_index}",
        event=event_text,
        status="draft",
        owner=user,
        created_by=user,
    )
    # Store PROVA context in RCA session metadata so the RCA→PROVA backflow knows
    # which hypothesis to refine
    rca_session.metadata = rca_session.metadata or {}
    rca_session.metadata["prova_hypothesis_id"] = str(hypothesis.id)
    rca_session.metadata["prova_working_graph_id"] = str(wg.id)
    rca_session.metadata["dsw_result_id"] = dsw_result_id
    rca_session.save(update_fields=["metadata", "updated_at"])

    # Generate embedding for similarity search
    try:
        rca_session.generate_embedding()
    except Exception:
        pass  # Embedding is nice-to-have, not critical

    logger.info(
        "prova.ooc.investigated",
        extra={
            "hypothesis_id": str(hypothesis.id),
            "rca_session_id": str(rca_session.id),
            "observation_index": observation_index,
            "rules": rules_violated,
        },
    )

    return {
        "hypothesis_id": str(hypothesis.id),
        "rca_session_id": str(rca_session.id),
        "working_graph_id": str(wg.id),
    }


# =============================================================================
# RCA INTEGRATION
# =============================================================================


def on_rca_root_cause(
    user,
    tenant,
    rca_session,
    root_cause: str,
    chain: list[dict],
) -> IntegrationResult:
    """Called when RCA identifies a root cause.

    RCA is an information tool — it proposes hypotheses about causation.
    The root cause becomes a hypothesis (or refines an existing one).
    Accepted chain steps become supporting hypotheses.

    If the RCA was spawned from a PROVA OOC investigation, this refines
    the original hypothesis rather than creating a new one.
    """
    meta = rca_session.metadata or {}
    prova_hypothesis_id = meta.get("prova_hypothesis_id")

    # Build hypothesis list
    hypotheses = []

    if prova_hypothesis_id:
        # This RCA was spawned from PROVA — refine the existing hypothesis
        try:
            parent = ProvaHypothesis.objects.get(id=prova_hypothesis_id)
            # Update the parent description with the root cause
            parent.description = f"Root cause: {root_cause[:500]}"
            parent.status = "testing"  # identified but not yet trialed
            parent.save(update_fields=["description", "status", "updated_at"])

            # Add chain steps as refinements
            for step in chain:
                if step.get("accepted") and step.get("claim"):
                    hypotheses.append(
                        {
                            "description": f"Causal factor: {step['claim'][:300]}",
                            "outcome_label": parent.outcome_label,
                            "prior": 0.5,
                        }
                    )

            logger.info(
                "prova.rca.refined_hypothesis",
                extra={
                    "hypothesis_id": prova_hypothesis_id,
                    "root_cause": root_cause[:100],
                    "chain_steps": len(chain),
                },
            )
        except ProvaHypothesis.DoesNotExist:
            prova_hypothesis_id = None  # Fall through to create new

    if not prova_hypothesis_id:
        # Standalone RCA — create new hypothesis
        hypotheses.append(
            {
                "description": f"Root cause: {root_cause[:500]}",
                "outcome_label": "",
                "prior": 0.6,
            }
        )
        for step in chain:
            if step.get("accepted") and step.get("claim"):
                hypotheses.append(
                    {
                        "description": f"Causal factor: {step['claim'][:300]}",
                        "prior": 0.5,
                    }
                )

    spec = FindingSpec(
        summary=f"RCA root cause: {root_cause[:200]}",
        tool_type="rca",
        source_id=str(rca_session.id),
        source_field="root_cause",
        hypotheses=hypotheses,
        project_id=str(rca_session.project_id) if rca_session.project_id else None,
    )

    result = integrate(
        user=user,
        tenant=tenant,
        tool_output=rca_session,
        spec=spec,
    )

    logger.info(
        "prova.rca.integrated",
        extra={
            "rca_session_id": str(rca_session.id),
            "root_cause": root_cause[:100],
            "hypotheses_created": result.hypotheses_created,
            "refined_existing": bool(prova_hypothesis_id),
        },
    )

    return result


# =============================================================================
# GENERIC ANALYSIS (DSW) INTEGRATION
# =============================================================================


def on_analysis_result(
    user,
    tenant,
    dsw_result,
    result_data: dict,
    analysis_type: str,
    analysis_id: str,
) -> Optional[IntegrationResult]:
    """Called when any analysis completes (DSW, DOE, Forecast, etc.).

    Routes to the appropriate handler based on analysis type.
    SPC has its own handler; everything else uses the generic inference path.
    """
    if analysis_type == "spc":
        return on_spc_result(user, tenant, dsw_result, result_data, analysis_id)

    # Generic analysis — inference tool
    stats = result_data.get("statistics", {})
    summary = result_data.get("summary", f"{analysis_type}/{analysis_id} result")[:200]

    spec = FindingSpec(
        summary=summary,
        tool_type="dsw",
        source_id=str(dsw_result.id) if dsw_result else "",
        sample_size=stats.get("n") or stats.get("n_obs") or stats.get("sample_size"),
        p_value=stats.get("p_value") or stats.get("p"),
        effect_size=stats.get("effect_size") or stats.get("cohens_d"),
        raw_output={
            "analysis_type": analysis_type,
            "analysis_id": analysis_id,
            "statistics": stats,
            "summary": summary,
        },
    )

    return integrate(
        user=user,
        tenant=tenant,
        tool_output=dsw_result,
        spec=spec,
    )
