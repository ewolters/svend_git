"""
Synara API Views: Connect Workbench to Belief Engine

Maps workbench artifacts to Synara primitives:
- Artifact "hypothesis" → Synara.add_hypothesis()
- Artifact "evidence" → Synara.add_evidence() → belief update
- Guide observations ← expansion signals, validation results

DSL endpoints for formal hypothesis authoring:
- Parse hypothesis text → structured AST
- Validate hypothesis logic
- Evaluate hypothesis against data
"""

import json
import logging
from uuid import uuid4

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth

from .synara import (
    Synara,
    HypothesisRegion,
    Evidence,
    CausalLink,
)
from .synara.llm_interface import SynaraLLMInterface
from .synara.dsl import DSLParser, format_hypothesis
from .synara.logic_engine import LogicEngine, validate_hypothesis, parse_and_evaluate

logger = logging.getLogger(__name__)

# In-memory storage for Synara instances per workbench
# In production, serialize to database
_synara_instances: dict[str, Synara] = {}


def get_synara(workbench_id: str) -> Synara:
    """Get or create Synara instance for a workbench."""
    if workbench_id not in _synara_instances:
        _synara_instances[workbench_id] = Synara()
    return _synara_instances[workbench_id]


# =============================================================================
# Hypothesis Management
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def add_hypothesis(request, workbench_id: str):
    """
    Add a hypothesis region to the belief engine.

    Request body:
    {
        "description": "Temperature drift causes defects",
        "domain_conditions": {"shift": "night"},
        "behavior_class": "defect_increase",
        "latent_causes": ["coolant_viscosity"],
        "prior": 0.4
    }
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    synara = get_synara(workbench_id)

    h = synara.create_hypothesis(
        description=body.get("description", ""),
        domain_conditions=body.get("domain_conditions", {}),
        behavior_class=body.get("behavior_class", ""),
        latent_causes=body.get("latent_causes", []),
        prior=body.get("prior", 0.5),
        source="user",
    )

    return JsonResponse({
        "success": True,
        "hypothesis": h.to_dict(),
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_hypotheses(request, workbench_id: str):
    """Get all hypotheses, sorted by posterior."""

    synara = get_synara(workbench_id)
    hypotheses = synara.get_all_hypotheses()

    return JsonResponse({
        "hypotheses": [h.to_dict() for h in hypotheses],
        "count": len(hypotheses),
    })


@csrf_exempt
@require_http_methods(["DELETE"])
@require_auth
def delete_hypothesis(request, workbench_id: str, hypothesis_id: str):
    """Remove a hypothesis from the belief engine."""

    synara = get_synara(workbench_id)

    if hypothesis_id in synara.graph.hypotheses:
        del synara.graph.hypotheses[hypothesis_id]
        return JsonResponse({"success": True})
    else:
        return JsonResponse({"error": "Hypothesis not found"}, status=404)


# =============================================================================
# Causal Links
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def add_link(request, workbench_id: str):
    """
    Add a causal link between hypotheses.

    Request body:
    {
        "from_id": "h_123",
        "to_id": "h_456",
        "mechanism": "Temperature affects viscosity",
        "strength": 0.8
    }
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    synara = get_synara(workbench_id)

    # Validate hypotheses exist
    if body.get("from_id") not in synara.graph.hypotheses:
        return JsonResponse({"error": "Source hypothesis not found"}, status=400)
    if body.get("to_id") not in synara.graph.hypotheses:
        return JsonResponse({"error": "Target hypothesis not found"}, status=400)

    link = synara.create_link(
        from_id=body["from_id"],
        to_id=body["to_id"],
        mechanism=body.get("mechanism", ""),
        strength=body.get("strength", 0.7),
    )

    return JsonResponse({
        "success": True,
        "link": link.to_dict(),
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_links(request, workbench_id: str):
    """Get all causal links."""

    synara = get_synara(workbench_id)

    return JsonResponse({
        "links": [link.to_dict() for link in synara.graph.links],
        "count": len(synara.graph.links),
    })


# =============================================================================
# Evidence & Belief Update
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def add_evidence(request, workbench_id: str):
    """
    Add evidence and trigger belief update.

    Request body:
    {
        "event": "out_of_control_point",
        "context": {"shift": "night", "time": "03:00"},
        "supports": ["h_123"],
        "weakens": ["h_456"],
        "strength": 0.9,
        "source": "spc"
    }

    Returns belief update results including any expansion signals.
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    synara = get_synara(workbench_id)

    result = synara.create_evidence(
        event=body.get("event", ""),
        context=body.get("context", {}),
        supports=body.get("supports", []),
        weakens=body.get("weakens", []),
        strength=body.get("strength", 1.0),
        source=body.get("source", "user"),
        data=body.get("data"),
    )

    return JsonResponse({
        "success": True,
        "update": result.to_dict(),
        "expansion_signal": result.expansion_signal.to_dict() if result.expansion_signal else None,
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_evidence(request, workbench_id: str):
    """Get all evidence."""

    synara = get_synara(workbench_id)

    return JsonResponse({
        "evidence": [e.to_dict() for e in synara.graph.evidence],
        "count": len(synara.graph.evidence),
    })


# =============================================================================
# Expansion Signals
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_expansions(request, workbench_id: str):
    """Get pending expansion signals."""

    synara = get_synara(workbench_id)
    pending = synara.get_pending_expansions()

    return JsonResponse({
        "expansions": [s.to_dict() for s in pending],
        "count": len(pending),
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def resolve_expansion(request, workbench_id: str, signal_id: str):
    """
    Resolve an expansion signal.

    Request body:
    {
        "resolution": "new_hypothesis" | "expanded_hypothesis" | "dismissed",
        "new_hypothesis": {...}  // if resolution is new_hypothesis
    }
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    synara = get_synara(workbench_id)

    new_h = None
    if body.get("resolution") == "new_hypothesis" and body.get("new_hypothesis"):
        h_data = body["new_hypothesis"]
        new_h = HypothesisRegion(
            id=f"h_{uuid4().hex[:8]}",
            description=h_data.get("description", ""),
            domain_conditions=h_data.get("domain_conditions", {}),
            behavior_class=h_data.get("behavior_class", ""),
            latent_causes=h_data.get("latent_causes", []),
            prior=h_data.get("prior", 0.3),
            posterior=h_data.get("prior", 0.3),
            source="expansion",
        )

    success = synara.resolve_expansion(
        signal_id=signal_id,
        resolution=body.get("resolution", "dismissed"),
        new_hypothesis=new_h,
    )

    if success:
        return JsonResponse({
            "success": True,
            "new_hypothesis": new_h.to_dict() if new_h else None,
        })
    else:
        return JsonResponse({"error": "Signal not found"}, status=404)


# =============================================================================
# Analysis & Queries
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_belief_state(request, workbench_id: str):
    """Get current belief state summary."""

    synara = get_synara(workbench_id)
    interface = SynaraLLMInterface(synara)

    top = synara.get_most_likely_cause()
    competing = synara.get_competing_hypotheses()
    pending = synara.get_pending_expansions()

    return JsonResponse({
        "summary": interface.get_state_summary(),
        "top_hypothesis": top.to_dict() if top else None,
        "competing_hypotheses": [h.to_dict() for h in competing],
        "pending_expansions": len(pending),
        "total_evidence": len(synara.graph.evidence),
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def explain_hypothesis(request, workbench_id: str, hypothesis_id: str):
    """Get explanation for a hypothesis's probability."""

    synara = get_synara(workbench_id)
    explanation = synara.explain_belief(hypothesis_id)

    return JsonResponse(explanation)


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_causal_chains(request, workbench_id: str, hypothesis_id: str):
    """Get all causal chains leading to a hypothesis."""

    synara = get_synara(workbench_id)
    chains = synara.get_causal_chains_to(hypothesis_id)

    return JsonResponse({
        "hypothesis_id": hypothesis_id,
        "chains": chains,
        "count": len(chains),
    })


# =============================================================================
# LLM Integration Prompts
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_validation_prompt(request, workbench_id: str):
    """Get prompt for LLM to validate the causal graph."""

    synara = get_synara(workbench_id)
    interface = SynaraLLMInterface(synara)

    return JsonResponse({
        "prompt": interface.generate_validation_prompt(),
        "context": interface.format_for_context(),
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_hypothesis_prompt(request, workbench_id: str, signal_id: str):
    """Get prompt for LLM to generate hypotheses from expansion signal."""

    synara = get_synara(workbench_id)

    signal = next(
        (s for s in synara.expansion_signals if s.id == signal_id),
        None
    )
    if not signal:
        return JsonResponse({"error": "Signal not found"}, status=404)

    interface = SynaraLLMInterface(synara)

    return JsonResponse({
        "prompt": interface.generate_hypothesis_prompt(signal),
        "context": interface.format_for_context(),
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def apply_validation_result(request, workbench_id: str):
    """
    Apply LLM validation result to the graph.

    Request body: parsed JSON from LLM validation response
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    synara = get_synara(workbench_id)
    interface = SynaraLLMInterface(synara)

    analysis = interface.parse_validation_response(body)

    return JsonResponse({
        "success": True,
        "issues_count": len(analysis.issues),
        "issues": [
            {
                "type": i.issue_type,
                "severity": i.severity,
                "description": i.description,
            }
            for i in analysis.issues
        ],
        "summary": analysis.summary,
    })


# =============================================================================
# Serialization
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def export_synara(request, workbench_id: str):
    """Export Synara state as JSON."""

    synara = get_synara(workbench_id)

    return JsonResponse(synara.to_dict())


@csrf_exempt
@require_http_methods(["POST"])
@gated
def import_synara(request, workbench_id: str):
    """Import Synara state from JSON."""

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    synara = Synara.from_dict(body)
    _synara_instances[workbench_id] = synara

    return JsonResponse({
        "success": True,
        "hypothesis_count": len(synara.graph.hypotheses),
        "evidence_count": len(synara.graph.evidence),
    })


# =============================================================================
# DSL: Formal Hypothesis Language
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def parse_hypothesis_dsl(request, workbench_id: str):
    """
    Parse a formal hypothesis statement.

    Request body:
    {
        "text": "if [num_holidays] > 3 then [monthly_sales] < 100000"
    }

    Returns parsed AST and variable references.
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    text = body.get("text", "")
    if not text:
        return JsonResponse({"error": "No hypothesis text provided"}, status=400)

    parser = DSLParser()
    hypothesis = parser.parse(text)

    return JsonResponse({
        "success": True,
        "hypothesis": hypothesis.to_dict(),
        "formatted": {
            "natural": format_hypothesis(hypothesis, "natural"),
            "formal": format_hypothesis(hypothesis, "formal"),
            "code": format_hypothesis(hypothesis, "code"),
        },
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def validate_hypothesis_dsl(request, workbench_id: str):
    """
    Validate a hypothesis for logical issues.

    Request body:
    {
        "text": "ALWAYS [temperature] > 20 AND [temperature] < 30"
    }

    Returns validation results without evaluating against data.
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    text = body.get("text", "")
    if not text:
        return JsonResponse({"error": "No hypothesis text provided"}, status=400)

    result = validate_hypothesis(text)

    return JsonResponse({
        "success": True,
        **result,
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def evaluate_hypothesis_dsl(request, workbench_id: str):
    """
    Evaluate a hypothesis against data.

    Request body:
    {
        "text": "if [num_holidays] > 3 then [monthly_sales] < 100000",
        "data": [
            {"num_holidays": 4, "monthly_sales": 80000},
            {"num_holidays": 5, "monthly_sales": 120000}
        ],
        "variable_context": {
            "num_holidays": "holiday_count"  // optional column mapping
        }
    }

    Returns evaluation result with supporting/refuting evidence.
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    text = body.get("text", "")
    data = body.get("data", [])
    variable_context = body.get("variable_context", {})

    if not text:
        return JsonResponse({"error": "No hypothesis text provided"}, status=400)
    if not data:
        return JsonResponse({"error": "No data provided for evaluation"}, status=400)

    evaluation = parse_and_evaluate(text, data, variable_context)

    return JsonResponse({
        "success": True,
        "evaluation": evaluation.to_dict(),
        "hypothesis": evaluation.hypothesis.to_dict(),
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def add_formal_hypothesis(request, workbench_id: str):
    """
    Add a formal hypothesis to the belief engine.

    Combines DSL parsing with Synara integration:
    1. Parses the formal statement
    2. Validates logic
    3. Creates HypothesisRegion with formal structure
    4. Adds to Synara graph

    Request body:
    {
        "text": "if [num_holidays] > 3 then [monthly_sales] < 100000",
        "prior": 0.5,
        "domain_context": {"department": "sales"}
    }
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    text = body.get("text", "")
    if not text:
        return JsonResponse({"error": "No hypothesis text provided"}, status=400)

    # Parse and validate
    validation = validate_hypothesis(text)
    if not validation["valid"]:
        return JsonResponse({
            "success": False,
            "error": "Invalid hypothesis",
            "fallacies": validation["fallacies"],
        }, status=400)

    synara = get_synara(workbench_id)

    # Create hypothesis with formal structure
    h = synara.create_hypothesis(
        description=text,  # Raw formal statement as description
        domain_conditions=body.get("domain_context", {}),
        behavior_class=validation["hypothesis"]["structure"].get("type", "formal"),
        latent_causes=validation["variables"],  # Variables as "causes" to track
        prior=body.get("prior", 0.5),
        source="dsl",
    )

    # Store the parsed structure as metadata
    # (In a full implementation, we'd extend HypothesisRegion)

    return JsonResponse({
        "success": True,
        "hypothesis": h.to_dict(),
        "parsed": validation["hypothesis"],
        "formatted": {
            "natural": format_hypothesis(DSLParser().parse(text), "natural"),
            "formal": format_hypothesis(DSLParser().parse(text), "formal"),
        },
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def evaluate_workbench_hypothesis(request, workbench_id: str, hypothesis_id: str):
    """
    Evaluate a specific hypothesis against workbench data.

    Request body:
    {
        "data_source": "uploaded_file" | "manual",
        "data": [...],  // if manual
        "file_id": "...",  // if uploaded_file
        "variable_context": {}
    }
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    synara = get_synara(workbench_id)

    if hypothesis_id not in synara.graph.hypotheses:
        return JsonResponse({"error": "Hypothesis not found"}, status=404)

    hypothesis_region = synara.graph.hypotheses[hypothesis_id]

    # Get data
    data = body.get("data", [])
    if not data:
        return JsonResponse({"error": "No data provided"}, status=400)

    # Evaluate the formal statement
    evaluation = parse_and_evaluate(
        hypothesis_region.description,
        data,
        body.get("variable_context", {}),
    )

    # Create evidence based on evaluation
    if evaluation.result.value == "supported":
        result = synara.create_evidence(
            event=f"Evaluation: {evaluation.result.value}",
            context={"evaluation_confidence": evaluation.confidence},
            supports=[hypothesis_id],
            weakens=[],
            strength=evaluation.confidence,
            source="dsl_evaluation",
            data={"supporting_count": evaluation.supporting_evidence.matching_count},
        )
    elif evaluation.result.value == "refuted":
        result = synara.create_evidence(
            event=f"Evaluation: {evaluation.result.value}",
            context={"evaluation_confidence": evaluation.confidence},
            supports=[],
            weakens=[hypothesis_id],
            strength=evaluation.confidence,
            source="dsl_evaluation",
            data={"refuting_count": evaluation.refuting_evidence.matching_count},
        )
    else:
        result = None

    return JsonResponse({
        "success": True,
        "evaluation": evaluation.to_dict(),
        "belief_update": result.to_dict() if result else None,
        "hypothesis_posterior": synara.graph.hypotheses[hypothesis_id].posterior,
    })
