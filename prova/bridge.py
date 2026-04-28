"""
PROVA Bridge — unified tool→graph integration.

Replaces agents_api.evidence_bridge.create_tool_evidence() and
agents_api.investigation_bridge.connect_tool() with a single entry point.

Every tool in SVEND calls bridge.integrate() to submit findings.
The bridge handles:
  - Evidence creation (core.Evidence)
  - CANON-002 evidence weighting (source rank × sample × MSA × quality)
  - Field-level mapping to graph edges
  - Working graph hypothesis creation (information tools)
  - Operating graph edge evidence attachment (inference tools)
  - Supersession detection
  - Conflict detection when new evidence contradicts existing edges

Tool classification (CANON-002 §11.1):
  - Information: RCA, Ishikawa, CE Matrix, FMEA, NCR, CAPA → working graph hypotheses
  - Inference: SPC, DSW, DOE results, ML, Forecast → evidence on graph edges
  - Intent: DOE design → annotates hypotheses with trial designs
  - Report: A3, 8D → reads graph, produces no evidence
  - Null: Triage, VSM → no graph interaction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from django.contrib.contenttypes.models import ContentType

from agents_api.evidence_weights import (
    TOOL_FUNCTIONS,
)
from agents_api.evidence_weights import (
    compute_evidence_weight as canon_evidence_weight,
)
from core.models import Evidence
from core.models.investigation import InvestigationToolLink

from . import engine
from .models import (
    ConflictStatus,
    EdgeStatus,
    GraphEdge,
    GraphEdit,
    OperatingGraph,
    ProvaHypothesis,
    WorkingGraph,
)

logger = logging.getLogger("prova.bridge")


# =============================================================================
# SPECS — what tools pass to integrate()
# =============================================================================


@dataclass
class FindingSpec:
    """Universal spec for any tool finding flowing into PROVA.

    Tools fill in the fields relevant to their function. The bridge
    routes based on tool_type → TOOL_FUNCTIONS mapping.
    """

    # Required
    summary: str
    tool_type: str  # "spc", "rca", "ishikawa", "dsw", "doe_results", etc.

    # Source identification (for supersession detection)
    source_id: str = ""  # tool output model ID
    source_field: str = ""  # specific field within tool output

    # For inference tools (SPC, DSW, DOE, etc.)
    sample_size: Optional[int] = None
    measurement_system_id: Optional[str] = None
    study_quality_factors: Optional[dict] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    raw_output: dict = field(default_factory=dict)

    # For field-level mapping to graph edges
    edge_mappings: list[dict] = field(default_factory=list)
    # Each: {"edge_id": uuid, "field": "column_name", "direction": "supports|weakens"}

    # For information tools (RCA, Ishikawa, etc.)
    hypotheses: list[dict] = field(default_factory=list)
    # Each: {"description": str, "outcome_label": str, "prior": float, "edits": [...]}

    # For intent tools (DOE design)
    target_hypothesis_ids: list[str] = field(default_factory=list)
    design_metadata: dict = field(default_factory=dict)

    # Optional context
    project_id: Optional[str] = None
    investigation_id: Optional[str] = None


@dataclass
class IntegrationResult:
    """What the bridge returns to the calling tool."""

    success: bool
    tool_function: Optional[str]  # information, inference, intent, report, None
    evidence_ids: list[str]  # core.Evidence IDs created
    evidence_weight: float  # CANON-002 composite weight
    hypotheses_created: int  # working graph hypotheses added
    edges_updated: int  # operating graph edges that received evidence
    conflicts_detected: int
    superseded_evidence_id: Optional[str]  # if supersession occurred
    message: str


# =============================================================================
# integrate() — the single entry point
# =============================================================================


def integrate(
    user,
    tenant,
    tool_output,  # Django model instance (RCASession, DSWResult, etc.)
    spec: FindingSpec,
) -> IntegrationResult:
    """
    Universal integration point: tool output → PROVA graph.

    Routes by tool function:
      - information → working graph hypotheses
      - inference → core.Evidence + operating graph edge evidence
      - intent → annotate hypotheses with trial designs
      - report/null → no graph modification

    Creates InvestigationToolLink if investigation_id provided (backward compat).
    """
    tool_function = TOOL_FUNCTIONS.get(spec.tool_type)

    if tool_function is None:
        return IntegrationResult(
            success=True,
            tool_function=None,
            evidence_ids=[],
            evidence_weight=0.0,
            hypotheses_created=0,
            edges_updated=0,
            conflicts_detected=0,
            superseded_evidence_id=None,
            message=f"Tool '{spec.tool_type}' produces no evidence.",
        )

    # Backward compat: create InvestigationToolLink if investigation context exists
    if spec.investigation_id and tool_output:
        _link_investigation(spec.investigation_id, tool_output, spec.tool_type, user)

    # Route by function
    if tool_function == "information":
        return _handle_information(user, tenant, spec)

    elif tool_function == "inference":
        return _handle_inference(user, tenant, tool_output, spec)

    elif tool_function == "intent":
        return _handle_intent(user, tenant, spec)

    elif tool_function == "report":
        return IntegrationResult(
            success=True,
            tool_function="report",
            evidence_ids=[],
            evidence_weight=0.0,
            hypotheses_created=0,
            edges_updated=0,
            conflicts_detected=0,
            superseded_evidence_id=None,
            message="Report tools read the graph; no modification.",
        )

    return IntegrationResult(
        success=False,
        tool_function=tool_function,
        evidence_ids=[],
        evidence_weight=0.0,
        hypotheses_created=0,
        edges_updated=0,
        conflicts_detected=0,
        superseded_evidence_id=None,
        message=f"Unknown tool function: {tool_function}",
    )


# =============================================================================
# INFORMATION HANDLER — structured analysis → working graph hypotheses
# =============================================================================


def _handle_information(
    user,
    tenant,
    spec: FindingSpec,
) -> IntegrationResult:
    """RCA, Ishikawa, CE Matrix, FMEA, NCR, CAPA → working graph hypotheses.

    Information tools propose graph edits. They don't create evidence
    directly — they feed the working graph with hypotheses that must
    be trialed before promotion.
    """
    og = engine.get_or_create_operating_graph(tenant)

    # Get or create a working graph for this user
    wg, _ = WorkingGraph.objects.get_or_create(
        tenant=tenant,
        owner=user,
        operating_graph=og,
        project_id=spec.project_id,
        is_deleted=False,
        defaults={"state": {}},
    )

    created = 0
    for h_data in spec.hypotheses:
        hypothesis = ProvaHypothesis.objects.create(
            working_graph=wg,
            description=h_data.get("description", spec.summary),
            outcome_label=h_data.get("outcome_label", ""),
            prior=h_data.get("prior", 0.5),
            created_by=str(user.id),
        )

        # Create graph edits if provided
        for edit_data in h_data.get("edits", []):
            GraphEdit.objects.create(
                hypothesis=hypothesis,
                operation=edit_data.get("operation", "add_edge"),
                target_edge_id=edit_data.get("target_edge_id"),
                target_node_id=edit_data.get("target_node_id"),
                params=edit_data.get("params", {}),
            )

        created += 1

    # If no structured hypotheses, create one from the summary
    if not spec.hypotheses and spec.summary:
        ProvaHypothesis.objects.create(
            working_graph=wg,
            description=spec.summary,
            prior=0.5,
            created_by=str(user.id),
        )
        created += 1

    logger.info(
        "prova.bridge.information",
        extra={
            "tool_type": spec.tool_type,
            "hypotheses_created": created,
            "user": str(user.id),
        },
    )

    return IntegrationResult(
        success=True,
        tool_function="information",
        evidence_ids=[],
        evidence_weight=0.0,
        hypotheses_created=created,
        edges_updated=0,
        conflicts_detected=0,
        superseded_evidence_id=None,
        message=f"{created} hypothesis(es) added to working graph.",
    )


# =============================================================================
# INFERENCE HANDLER — analysis results → evidence on graph edges
# =============================================================================


def _handle_inference(
    user,
    tenant,
    tool_output,
    spec: FindingSpec,
) -> IntegrationResult:
    """SPC, DSW, DOE results, ML, Forecast → evidence attached to graph edges.

    1. Compute CANON-002 evidence weight (reliability)
    2. Create core.Evidence record
    3. Detect supersession (same tool re-run)
    4. Map to graph edges via field-level mapping
    5. Check for conflicts
    """
    og = engine.get_or_create_operating_graph(tenant)

    # 1. CANON-002 evidence weight (reliability)
    weight = canon_evidence_weight(
        source_tool=spec.tool_type,
        sample_size=spec.sample_size,
        measurement_system_id=spec.measurement_system_id,
        study_quality_factors=spec.study_quality_factors,
    )

    # 2. Create core.Evidence
    from core.models import Project

    project = None
    if spec.project_id:
        try:
            project = Project.objects.get(id=spec.project_id)
        except Project.DoesNotExist:
            pass

    source_desc = f"{spec.tool_type}:{spec.source_id}"
    if spec.source_field:
        source_desc += f":{spec.source_field}"

    evidence = Evidence.objects.create(
        project=project,
        summary=spec.summary,
        source_type=_tool_to_source_type(spec.tool_type),
        result_type=_tool_to_result_type(spec.tool_type),
        confidence=weight,
        p_value=spec.p_value,
        effect_size=spec.effect_size,
        sample_size=spec.sample_size,
        source_description=source_desc,
        raw_output=spec.raw_output,
        created_by=user,
    )

    evidence_ids = [str(evidence.id)]

    # 3. Supersession detection
    superseded_id = _detect_supersession(evidence, spec)

    # 4. Field-level mapping to graph edges
    edges_updated = 0
    conflicts_detected = 0

    for mapping in spec.edge_mappings:
        edge_id = mapping.get("edge_id")
        direction = mapping.get("direction", "supports")

        if not edge_id:
            continue

        try:
            edge = GraphEdge.objects.get(id=edge_id, operating_graph=og)
        except GraphEdge.DoesNotExist:
            continue

        # Attach evidence to edge
        edge.evidence.add(evidence)

        # Check for conflict: if direction is "weakens" and edge is confident
        if direction == "weakens" and edge.confidence > 0.7 and weight > 0.3:
            _create_conflict(og, edge, evidence, weight)
            conflicts_detected += 1
        elif direction == "supports":
            # Reinforce edge confidence based on evidence weight
            boost = weight * 0.1  # small incremental boost
            edge.confidence = min(0.99, edge.confidence + boost)
            edge.save(update_fields=["confidence", "updated_at"])

        edges_updated += 1

    # If no explicit edge mappings, try to auto-map via field matching
    if not spec.edge_mappings and spec.raw_output:
        auto_mapped = _auto_map_evidence(og, evidence, spec)
        edges_updated += auto_mapped

    logger.info(
        "prova.bridge.inference",
        extra={
            "tool_type": spec.tool_type,
            "evidence_id": str(evidence.id),
            "weight": round(weight, 4),
            "edges_updated": edges_updated,
            "conflicts": conflicts_detected,
        },
    )

    return IntegrationResult(
        success=True,
        tool_function="inference",
        evidence_ids=evidence_ids,
        evidence_weight=round(weight, 4),
        hypotheses_created=0,
        edges_updated=edges_updated,
        conflicts_detected=conflicts_detected,
        superseded_evidence_id=superseded_id,
        message=f"Evidence created (weight={weight:.3f}), {edges_updated} edge(s) updated.",
    )


# =============================================================================
# INTENT HANDLER — DOE design → annotate hypotheses
# =============================================================================


def _handle_intent(
    user,
    tenant,
    spec: FindingSpec,
) -> IntegrationResult:
    """DOE design → annotates target hypotheses with trial design metadata.

    Intent tools don't produce evidence. They prescribe which hypotheses
    to test and how. The PROVA trial configurator uses this as input.
    """
    annotated = 0
    for h_id in spec.target_hypothesis_ids:
        try:
            hypothesis = ProvaHypothesis.objects.get(
                id=h_id,
                working_graph__tenant=tenant,
            )
            # Store design linkage in hypothesis metadata
            meta = hypothesis.metadata or {}
            designs = meta.setdefault("linked_designs", [])
            designs.append(
                {
                    "tool_type": spec.tool_type,
                    "source_id": spec.source_id,
                    "design_metadata": spec.design_metadata,
                }
            )
            hypothesis.metadata = meta
            hypothesis.save(update_fields=["metadata", "updated_at"])
            annotated += 1
        except ProvaHypothesis.DoesNotExist:
            continue

    return IntegrationResult(
        success=True,
        tool_function="intent",
        evidence_ids=[],
        evidence_weight=0.0,
        hypotheses_created=0,
        edges_updated=0,
        conflicts_detected=0,
        superseded_evidence_id=None,
        message=f"{annotated} hypothesis(es) annotated with trial design.",
    )


# =============================================================================
# HELPERS
# =============================================================================


def _detect_supersession(evidence: Evidence, spec: FindingSpec) -> Optional[str]:
    """Detect re-runs from the same tool and create supersedes FK.

    CANON-002 §8.4: same (tool_type, source_id) → new supersedes old.
    """
    if not spec.source_id:
        return None

    source_desc = f"{spec.tool_type}:{spec.source_id}"

    prior = (
        Evidence.objects.filter(source_description__startswith=source_desc)
        .exclude(id=evidence.id)
        .order_by("-created_at")
        .first()
    )

    if prior:
        evidence.supersedes = prior
        evidence.save(update_fields=["supersedes"])
        logger.info(
            "prova.bridge.supersession",
            extra={
                "new_id": str(evidence.id),
                "superseded_id": str(prior.id),
                "source_tool": spec.tool_type,
            },
        )
        return str(prior.id)

    return None


def _create_conflict(
    og: OperatingGraph,
    edge: GraphEdge,
    evidence: Evidence,
    weight: float,
):
    """Create a conflict when inference evidence contradicts an edge."""
    from .models import Conflict

    conflict = Conflict.objects.create(
        edge=edge,
        operating_graph=og,
        magnitude=weight,
        evaluation_cost=engine._compute_edge_evaluation_cost(og, edge),
        proposed_resolutions=engine._propose_resolutions(
            edge,
            {
                "evidence_id": str(evidence.id),
                "weight": weight,
            },
        ),
        status=ConflictStatus.OPEN,
    )
    conflict.competing_evidence.add(evidence)

    # Break the edge
    edge.status = EdgeStatus.BROKEN
    edge.save(update_fields=["status", "updated_at"])

    logger.warning(
        "prova.bridge.conflict",
        extra={
            "edge_id": str(edge.id),
            "evidence_id": str(evidence.id),
            "magnitude": weight,
        },
    )


def _auto_map_evidence(
    og: OperatingGraph,
    evidence: Evidence,
    spec: FindingSpec,
) -> int:
    """Attempt to auto-map evidence to graph edges by matching field names.

    Looks at the raw_output keys and tries to match them to node labels
    in the operating graph. Simple heuristic — explicit edge_mappings
    in the spec are always preferred.
    """
    if not spec.raw_output:
        return 0

    output_keys = set()
    if isinstance(spec.raw_output, dict):
        output_keys = {k.lower() for k in spec.raw_output.keys()}

    if not output_keys:
        return 0

    # Find edges where source or target label matches output keys
    matched = 0
    edges = GraphEdge.objects.filter(
        operating_graph=og,
        status=EdgeStatus.ACTIVE,
    ).select_related("source", "target")

    for edge in edges:
        source_match = edge.source.label.lower() in output_keys
        target_match = edge.target.label.lower() in output_keys
        if source_match or target_match:
            edge.evidence.add(evidence)
            matched += 1

    return matched


def _link_investigation(investigation_id, tool_output, tool_type, user):
    """Backward compat: create InvestigationToolLink for existing investigations."""
    try:
        from core.models.investigation import Investigation

        investigation = Investigation.objects.get(id=investigation_id)
        ct = ContentType.objects.get_for_model(tool_output)
        InvestigationToolLink.objects.get_or_create(
            investigation=investigation,
            content_type=ct,
            object_id=tool_output.id,
            defaults={
                "tool_type": tool_type,
                "tool_function": TOOL_FUNCTIONS.get(tool_type, "information"),
                "linked_by": user,
            },
        )
    except Exception:
        pass  # Investigation system may not be active


def _tool_to_source_type(tool_type: str) -> str:
    """Map tool type to core.Evidence.source_type."""
    mapping = {
        "spc": "analysis",
        "dsw": "analysis",
        "doe_results": "experiment",
        "ml": "analysis",
        "forecast": "analysis",
        "rca": "analysis",
        "ishikawa": "analysis",
        "ce_matrix": "analysis",
        "fmea": "analysis",
        "ncr": "observation",
        "capa": "analysis",
        "user": "observation",
        "observation": "observation",
    }
    return mapping.get(tool_type, "analysis")


def _tool_to_result_type(tool_type: str) -> str:
    """Map tool type to core.Evidence.result_type."""
    mapping = {
        "spc": "statistical",
        "dsw": "statistical",
        "doe_results": "statistical",
        "ml": "statistical",
        "forecast": "statistical",
        "rca": "qualitative",
        "ishikawa": "qualitative",
        "ce_matrix": "qualitative",
        "fmea": "qualitative",
        "ncr": "qualitative",
        "capa": "qualitative",
        "user": "qualitative",
        "observation": "qualitative",
    }
    return mapping.get(tool_type, "qualitative")
