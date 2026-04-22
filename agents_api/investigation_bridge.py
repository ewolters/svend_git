"""
Investigation bridge — CANON-002 §12.3, §8.4, §10.1.

Universal integration point: tool output → investigation graph.
Tools call connect_tool() with a function-specific spec, and this module
handles Synara graph operations, evidence weighting, tool linking,
auto-transitions, supersession detection, and confirmation thresholds.

Reference: docs/standards/CANON-002.md §12.3, §8.4, §10.1
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from django.contrib.contenttypes.models import ContentType
from forgesia import Synara

from agents_api.evidence_weights import TOOL_FUNCTIONS, compute_evidence_weight
from core.models.investigation import Investigation, InvestigationToolLink

logger = logging.getLogger("svend.investigation")

# ---------------------------------------------------------------------------
# Confirmation thresholds — CANON-002 §10
# ---------------------------------------------------------------------------

CONFIRMED_THRESHOLD = 0.85
REJECTED_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Specs — what each tool function passes to connect_tool()
# ---------------------------------------------------------------------------


@dataclass
class HypothesisSpec:
    """What an information tool wants to add to the graph."""

    description: str
    behavior_class: str = ""
    domain_conditions: dict = field(default_factory=dict)
    prior: float = 0.5
    causes: str | None = None  # hypothesis_id this causes


@dataclass
class InferenceSpec:
    """What an inference tool wants to add as evidence."""

    event_description: str
    context: dict = field(default_factory=dict)
    supports: list[str] = field(default_factory=list)
    weakens: list[str] = field(default_factory=list)
    raw_output: dict = field(default_factory=dict)
    sample_size: int | None = None
    measurement_system_id: str | None = None
    study_quality_factors: dict | None = None


@dataclass
class IntentSpec:
    """What an intent tool wants to annotate on the graph."""

    target_hypothesis_ids: list[str] = field(default_factory=list)
    design_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Investigation access + Synara serialization
# ---------------------------------------------------------------------------


def get_investigation(investigation_id: str, user) -> Investigation:
    """Load investigation, verify membership."""
    inv = Investigation.objects.get(id=investigation_id)
    if inv.owner == user:
        return inv
    if inv.members.filter(id=user.id).exists():
        return inv
    raise PermissionError("User is not a member of this investigation")


def load_synara(investigation: Investigation) -> Synara:
    """Load Synara engine from investigation state.

    Deep-copies state before deserializing because Synara.from_dict()
    mutates nested dicts in-place (datetime parsing), which would corrupt
    the JSONField on the next full save.
    """
    if investigation.synara_state:
        return Synara.from_dict(copy.deepcopy(investigation.synara_state))
    return Synara()


def save_synara(investigation: Investigation, synara: Synara):
    """Persist Synara state back to investigation."""
    investigation.synara_state = synara.to_dict()
    investigation.save(update_fields=["synara_state", "updated_at"])


# ---------------------------------------------------------------------------
# connect_tool() — universal dispatcher
# ---------------------------------------------------------------------------


def connect_tool(
    investigation_id: str,
    tool_output,  # Any tool model instance
    tool_type: str,  # "spc", "rca", "ishikawa", etc.
    user,
    spec: HypothesisSpec | InferenceSpec | IntentSpec | list[HypothesisSpec],
) -> dict:
    """
    Universal integration point: tool output → investigation graph.

    Returns dict with:
      - "linked": bool
      - "graph_updated": bool
      - "posteriors": dict (if inference)
      - "expansion_signal": dict | None (if inference)
      - "hypotheses_added": int (if information)
      - "evidence_weight": float (if inference)
    """
    investigation = get_investigation(investigation_id, user)
    result = {"linked": True, "graph_updated": False}

    # Auto-transition open → active on first tool connection
    if investigation.status == Investigation.Status.OPEN:
        investigation.transition_to(Investigation.Status.ACTIVE, user)

    # Create tool link (idempotent via unique_together)
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

    # Load Synara
    synara = load_synara(investigation)
    tool_function = TOOL_FUNCTIONS.get(tool_type)

    if tool_function == "information":
        result = _handle_information(synara, spec, tool_type, result)

    elif tool_function == "inference":
        result = _handle_inference(synara, spec, tool_type, result, investigation, tool_output)

    elif tool_function == "intent":
        synara, result = _handle_intent(synara, spec, tool_output, result)

    elif tool_function == "report":
        # Reports read the graph, they don't modify it
        result["graph_updated"] = False

    # Persist
    save_synara(investigation, synara)

    logger.info(
        "investigation.tool_connected",
        extra={
            "investigation_id": str(investigation.id),
            "tool_type": tool_type,
            "tool_id": str(tool_output.id),
            "function": tool_function,
            "graph_updated": result["graph_updated"],
        },
    )

    return result


# ---------------------------------------------------------------------------
# Function handlers
# ---------------------------------------------------------------------------


def _handle_information(synara, spec, tool_type, result):
    """Information tools create hypotheses and causal links."""
    specs = spec if isinstance(spec, list) else [spec]
    added = 0
    for hs in specs:
        h = synara.create_hypothesis(
            description=hs.description,
            behavior_class=hs.behavior_class,
            domain_conditions=hs.domain_conditions,
            prior=hs.prior,
        )
        h_id = h.id if hasattr(h, "id") else str(h)
        if hs.causes:
            synara.create_link(
                from_id=h_id,
                to_id=hs.causes,
                strength=0.7,
                mechanism=f"Causal link from {tool_type}",
            )
        added += 1
    result["graph_updated"] = True
    result["hypotheses_added"] = added
    return result


def _handle_inference(synara, spec, tool_type, result, investigation=None, tool_output=None):
    """Inference tools compute evidence weight and create evidence."""
    assert isinstance(spec, InferenceSpec)
    weight = compute_evidence_weight(
        source_tool=tool_type,
        sample_size=spec.sample_size,
        measurement_system_id=spec.measurement_system_id,
        study_quality_factors=spec.study_quality_factors,
    )
    update_result = synara.create_evidence(
        event=spec.event_description,
        context=spec.context,
        supports=spec.supports,
        weakens=spec.weakens,
        strength=weight,
        source=tool_type,
        data=spec.raw_output,
    )
    result["graph_updated"] = True
    result["posteriors"] = {h_id: round(p, 4) for h_id, p in update_result.posteriors.items()}
    result["expansion_signal"] = update_result.expansion_signal.to_dict() if update_result.expansion_signal else None
    result["evidence_weight"] = round(weight, 4)

    # Supersession detection (§8.4)
    if investigation and tool_output:
        _detect_and_apply_supersession(
            investigation=investigation,
            source_tool=tool_type,
            source_id=str(tool_output.id),
            new_evidence_id=update_result.evidence_id,
        )

    # Confirmation thresholds (§10.1)
    if result["posteriors"]:
        confirmation_events = _apply_confirmation_thresholds(investigation, synara, result["posteriors"])
        result["confirmation_changes"] = confirmation_events

    return result


def _handle_intent(synara, spec, tool_output, result):
    """Intent tools annotate hypotheses with design linkage."""
    assert isinstance(spec, IntentSpec)
    full_data = synara.to_dict()
    hypotheses = full_data.get("graph", {}).get("hypotheses", {})
    for h_id in spec.target_hypothesis_ids:
        if h_id in hypotheses:
            annotations = hypotheses[h_id].setdefault("annotations", [])
            annotations.append(
                {
                    "design_id": str(tool_output.id),
                    "metadata": spec.design_metadata,
                }
            )
    # Rebuild synara from modified state
    synara = Synara.from_dict(full_data)
    result["graph_updated"] = True
    return synara, result


# ---------------------------------------------------------------------------
# Layer 3 export — CANON-002 §9.2
# ---------------------------------------------------------------------------


def export_investigation(
    investigation_id: str,
    target_project_id: str,
    user,
) -> dict:
    """
    Export a concluded investigation to a Layer 3 project container.

    Creates an Evidence record on the target project with the conclusion
    package frozen as raw_output. Transitions investigation to exported.

    Raises ValueError if investigation is not in 'concluded' state.
    """
    investigation = get_investigation(investigation_id, user)
    if investigation.status != Investigation.Status.CONCLUDED:
        raise ValueError(f"Cannot export investigation in '{investigation.status}' state — must be 'concluded'")

    from core.models import Evidence, Project

    target = Project.objects.get(id=target_project_id, user=user)
    synara = load_synara(investigation)

    package = _build_conclusion_package(investigation, synara, user)
    top_h = package["top_hypothesis"]

    Evidence.objects.create(
        project=target,
        source_type="analysis",
        result_type="qualitative",
        summary=f"{top_h['description']} (posterior: {top_h['posterior']:.2f})",
        confidence=top_h["posterior"],
        details=_build_export_details(package),
        raw_output=package,
        source_description=f"investigation:{investigation.id}",
    )

    investigation.export_package = package
    investigation.exported_to_project = target
    investigation.save(update_fields=["export_package", "exported_to_project", "updated_at"])

    investigation.transition_to(Investigation.Status.EXPORTED, user)

    logger.info(
        "investigation.exported",
        extra={
            "investigation_id": str(investigation.id),
            "target_project_id": str(target.id),
            "top_posterior": top_h["posterior"],
            "evidence_count": package["investigation_metadata"]["evidence_count"],
        },
    )

    return package


def _build_conclusion_package(investigation, synara, user) -> dict:
    """Build the §9.1 conclusion package from current Synara state."""
    hypotheses = synara.graph.hypotheses
    evidence_list = synara.graph.evidence
    links = synara.graph.links

    sorted_h = sorted(hypotheses.values(), key=lambda h: h.posterior, reverse=True)
    top = sorted_h[0] if sorted_h else None

    causal_chain = []
    if top:
        causal_chain = _trace_causal_chain(top.id, hypotheses, links)

    def h_status(posterior: float) -> str:
        if posterior >= CONFIRMED_THRESHOLD:
            return "confirmed"
        if posterior <= REJECTED_THRESHOLD:
            return "rejected"
        return "uncertain"

    expansion_signals = [
        {
            "signal_type": "expansion",
            "description": getattr(sig, "message", ""),
            "event": getattr(sig, "event", ""),
        }
        for sig in getattr(synara, "expansion_signals", [])
    ]

    tool_types = list(
        InvestigationToolLink.objects.filter(investigation=investigation).values_list("tool_type", flat=True).distinct()
    )

    duration_days = 0
    if investigation.concluded_at and investigation.created_at:
        duration_days = (investigation.concluded_at - investigation.created_at).days

    return {
        "investigation_id": str(investigation.id),
        "investigation_version": investigation.version,
        "status": "concluded",
        "concluded_at": (investigation.concluded_at.isoformat() if investigation.concluded_at else None),
        "concluded_by": str(user.id),
        "top_hypothesis": {
            "id": str(top.id) if top else None,
            "description": top.description if top else "No hypotheses",
            "posterior": round(top.posterior, 4) if top else 0.0,
            "status": h_status(top.posterior) if top else "uncertain",
            "causal_chain": causal_chain,
        },
        "competing_hypotheses": [
            {
                "id": str(h.id),
                "description": h.description,
                "posterior": round(h.posterior, 4),
                "status": h_status(h.posterior),
            }
            for h in sorted_h[1:]
            if h.posterior > REJECTED_THRESHOLD
        ],
        "evidence_summary": [
            {
                "id": str(e.id),
                "summary": e.event,
                "source_tool": getattr(e, "source", "unknown"),
                "evidence_weight": round(e.strength, 4),
                "supports": [str(s) for s in getattr(e, "supports", [])],
                "weakens": [str(w) for w in getattr(e, "weakens", [])],
            }
            for e in evidence_list
        ],
        "unresolved_signals": expansion_signals,
        "investigation_metadata": {
            "tools_used": tool_types,
            "evidence_count": len(evidence_list),
            "hypothesis_count": len(hypotheses),
            "duration_days": duration_days,
        },
    }


def _trace_causal_chain(hypothesis_id, hypotheses, links) -> list[dict]:
    """
    Walk causal links backward from hypothesis to build a chain.
    Max depth 20 to prevent cycles.
    """
    chain = []
    visited = set()
    current = hypothesis_id
    max_depth = 20

    while current and len(chain) < max_depth:
        if current in visited:
            break
        visited.add(current)

        incoming = [link for link in links if link.to_id == current]
        if not incoming:
            break

        strongest = max(incoming, key=lambda cl: cl.strength)
        source_h = hypotheses.get(strongest.from_id)
        if not source_h:
            break

        chain.append(
            {
                "hypothesis_id": str(source_h.id),
                "description": source_h.description,
                "posterior": round(source_h.posterior, 4),
                "link_strength": round(strongest.strength, 4),
                "mechanism": getattr(strongest, "mechanism", ""),
            }
        )
        current = strongest.from_id

    chain.reverse()
    return chain


def _build_export_details(package: dict) -> str:
    """Build human-readable details string for the Evidence record."""
    meta = package["investigation_metadata"]
    details = (
        f"Investigation conclusion — "
        f"{meta['evidence_count']} evidence nodes, "
        f"{meta['hypothesis_count']} hypotheses, "
        f"{meta['duration_days']} days"
    )
    n_signals = len(package.get("unresolved_signals", []))
    if n_signals > 0:
        details += (
            f". WARNING: Investigation has {n_signals} unresolved expansion signals — causal surface may be incomplete"
        )
    return details


# ---------------------------------------------------------------------------
# Supersession detection — CANON-002 §8.4
# ---------------------------------------------------------------------------


def _detect_and_apply_supersession(
    investigation,
    source_tool: str,
    source_id: str,
    new_evidence_id: str,
):
    """
    Detect re-runs from the same tool+source and create supersedes FK.

    Matches on (source_tool, source_id) — same tool output producing newer
    evidence supersedes prior evidence from the same source.
    """
    import uuid as _uuid

    from core.models import Evidence

    # Validate new_evidence_id is a valid UUID — Synara kernel evidence IDs
    # (e.g. "e_abc123") are not UUIDs and can't be used in ORM queries.
    try:
        _uuid.UUID(new_evidence_id)
    except (ValueError, AttributeError):
        return

    # Find prior evidence from the same source_tool with same source_description
    # that is not already superseded and is not the new evidence itself
    prior = (
        Evidence.objects.filter(
            source_description=f"{source_tool}:{source_id}",
        )
        .exclude(id=new_evidence_id)
        .exclude(id__in=Evidence.objects.filter(supersedes__isnull=False).values("supersedes_id"))
        .order_by("-created_at")
        .first()
    )

    if prior:
        try:
            new_evidence = Evidence.objects.get(id=new_evidence_id)
            new_evidence.supersedes = prior
            new_evidence.save(update_fields=["supersedes"])
            logger.info(
                "evidence.superseded",
                extra={
                    "new_id": str(new_evidence_id),
                    "superseded_id": str(prior.id),
                    "source_tool": source_tool,
                    "investigation_id": (str(investigation.id) if investigation else None),
                },
            )
        except Evidence.DoesNotExist:
            pass  # Evidence record not found


# ---------------------------------------------------------------------------
# Confirmation thresholds — CANON-002 §10.1
# ---------------------------------------------------------------------------


def _apply_confirmation_thresholds(
    investigation,
    synara: Synara,
    posteriors: dict[str, float],
) -> list[dict]:
    """
    Check all updated hypotheses against confirmation/rejection thresholds.
    Returns list of status change events.

    On confirmation (≥0.85):
      - Mark hypothesis as confirmed in graph metadata
      - Suppress expansion signals for that hypothesis

    On rejection (≤0.15):
      - Mark hypothesis as rejected
      - Deactivate outgoing causal links (strength → 0)

    On reversal (confirmed/rejected → uncertain):
      - Restore expansion signals and causal links
    """
    events = []

    for h_id, posterior in posteriors.items():
        h = synara.graph.hypotheses.get(h_id)
        if not h:
            continue

        prev_status = getattr(h, "confirmation_status", "uncertain")

        if posterior >= CONFIRMED_THRESHOLD and prev_status != "confirmed":
            h.confirmation_status = "confirmed"
            events.append(
                {
                    "hypothesis_id": h_id,
                    "transition": f"{prev_status} → confirmed",
                    "posterior": round(posterior, 4),
                }
            )

        elif posterior <= REJECTED_THRESHOLD and prev_status != "rejected":
            h.confirmation_status = "rejected"
            # Deactivate outgoing causal links
            for link in synara.graph.links:
                if link.from_id == h_id:
                    link.strength = 0.0
            events.append(
                {
                    "hypothesis_id": h_id,
                    "transition": f"{prev_status} → rejected",
                    "posterior": round(posterior, 4),
                }
            )

        elif REJECTED_THRESHOLD < posterior < CONFIRMED_THRESHOLD and prev_status != "uncertain":
            # Reversal — re-entered uncertain zone
            h.confirmation_status = "uncertain"
            # Re-enable deactivated links
            for link in synara.graph.links:
                if link.from_id == h_id and link.strength == 0.0:
                    link.strength = 0.7  # Default causal link strength
            events.append(
                {
                    "hypothesis_id": h_id,
                    "transition": f"{prev_status} → uncertain",
                    "posterior": round(posterior, 4),
                }
            )

    if events:
        logger.info(
            "investigation.confirmation_changes",
            extra={
                "investigation_id": str(investigation.id) if investigation else None,
                "changes": events,
            },
        )

    return events
