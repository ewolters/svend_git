"""
Investigation bridge — CANON-002 §12.3.

Universal integration point: tool output → investigation graph.
Tools call connect_tool() with a function-specific spec, and this module
handles Synara graph operations, evidence weighting, tool linking,
and auto-transitions.

Reference: docs/standards/CANON-002.md §12.3, §8.4, §10.1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.contrib.contenttypes.models import ContentType

from agents_api.evidence_weights import TOOL_FUNCTIONS, compute_evidence_weight
from agents_api.synara.synara import Synara
from core.models.investigation import Investigation, InvestigationToolLink

logger = logging.getLogger("svend.investigation")


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
    """Load Synara engine from investigation state."""
    if investigation.synara_state:
        return Synara.from_dict(investigation.synara_state)
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
        result = _handle_inference(synara, spec, tool_type, result)

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


def _handle_inference(synara, spec, tool_type, result):
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
