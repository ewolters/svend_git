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
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid

from .synara import (
    HypothesisRegion,
    Synara,
)
from .synara.dsl import DSLParser, format_hypothesis
from .synara.llm_interface import SynaraLLMInterface
from .synara.logic_engine import parse_and_evaluate, validate_hypothesis

logger = logging.getLogger(__name__)

# In-memory cache for Synara instances (backed by core.Project.synara_state)
# Bounded to prevent unbounded memory growth
_SYNARA_CACHE_MAX = 128
_synara_cache: dict[str, Synara] = {}


def _resolve_project(workbench_id: str, user=None):
    """Resolve a workbench_id to a core.Project.

    Tries core.Project first, then agents_api.Problem (via its core_project FK).
    Returns the Project or None. Filters by user when provided to prevent IDOR.
    """
    from core.models import Project

    from .models import Problem

    # Build user filter
    user_filter = {"user": user} if user else {}

    # Try as core.Project UUID directly
    try:
        return Project.objects.get(id=workbench_id, **user_filter)
    except (Project.DoesNotExist, ValueError):
        pass

    # Try as agents_api.Problem UUID → follow FK
    try:
        problem = Problem.objects.get(id=workbench_id, **user_filter)
        if problem.core_project:
            return problem.core_project
        # Auto-create core.Project if Problem exists but has no link
        return problem.ensure_core_project()
    except (Problem.DoesNotExist, ValueError):
        pass

    return None


def get_synara(workbench_id: str, user=None) -> Synara:
    """Get or create Synara instance for a workbench.

    Loads from core.Project.synara_state on cache miss.
    """
    if workbench_id in _synara_cache:
        return _synara_cache[workbench_id]

    project = _resolve_project(workbench_id, user=user)
    if project and project.synara_state:
        synara = Synara.from_dict(project.synara_state)
    else:
        synara = Synara()

    if len(_synara_cache) >= _SYNARA_CACHE_MAX:
        _synara_cache.pop(next(iter(_synara_cache)), None)
    _synara_cache[workbench_id] = synara
    return synara


def save_synara(workbench_id: str, synara: Synara = None, user=None) -> bool:
    """Persist Synara state to core.Project.synara_state.

    Returns True if saved successfully, False if no project found.
    """
    if synara is None:
        synara = _synara_cache.get(workbench_id)
    if synara is None:
        return False

    project = _resolve_project(workbench_id, user=user)
    if project:
        project.synara_state = synara.to_dict()
        project.save(update_fields=["synara_state", "updated_at"])
        return True

    logger.warning(f"No project found for workbench_id={workbench_id}, state not persisted")
    return False


def _require_project(workbench_id: str, user=None):
    """Validate that a project exists for the given workbench_id.

    Returns (project, None) on success, or (None, JsonResponse) on failure.
    Use before any Synara mutation that creates hypotheses, evidence, or links.
    """
    project = _resolve_project(workbench_id, user=user)
    if not project:
        return None, JsonResponse({"error": "No study loaded. Create or select a study first."}, status=400)
    return project, None


# =============================================================================
# Hypothesis Management
# =============================================================================


@require_http_methods(["POST"])
@gated_paid
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

    project, err = _require_project(workbench_id, user=request.user)
    if err:
        return err

    synara = get_synara(workbench_id, user=request.user)

    h = synara.create_hypothesis(
        description=body.get("description", ""),
        domain_conditions=body.get("domain_conditions", {}),
        behavior_class=body.get("behavior_class", ""),
        latent_causes=body.get("latent_causes", []),
        prior=body.get("prior", 0.5),
        source="user",
    )

    if not save_synara(workbench_id, synara, user=request.user):
        return JsonResponse({"error": "Failed to persist hypothesis."}, status=500)

    return JsonResponse(
        {
            "success": True,
            "hypothesis": h.to_dict(),
        }
    )


@require_http_methods(["GET"])
@gated_paid
def get_hypotheses(request, workbench_id: str):
    """Get all hypotheses, sorted by posterior."""

    synara = get_synara(workbench_id, user=request.user)
    hypotheses = synara.get_all_hypotheses()

    return JsonResponse(
        {
            "hypotheses": [h.to_dict() for h in hypotheses],
            "count": len(hypotheses),
        }
    )


@require_http_methods(["DELETE"])
@gated_paid
def delete_hypothesis(request, workbench_id: str, hypothesis_id: str):
    """Remove a hypothesis from the belief engine."""

    synara = get_synara(workbench_id, user=request.user)

    if hypothesis_id in synara.graph.hypotheses:
        del synara.graph.hypotheses[hypothesis_id]
        if not save_synara(workbench_id, synara, user=request.user):
            return JsonResponse({"error": "Failed to persist deletion."}, status=500)
        return JsonResponse({"success": True})
    else:
        return JsonResponse({"error": "Hypothesis not found"}, status=404)


# =============================================================================
# Causal Links
# =============================================================================


@require_http_methods(["POST"])
@gated_paid
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

    project, err = _require_project(workbench_id, user=request.user)
    if err:
        return err

    synara = get_synara(workbench_id, user=request.user)

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

    if not save_synara(workbench_id, synara, user=request.user):
        return JsonResponse({"error": "Failed to persist link."}, status=500)

    return JsonResponse(
        {
            "success": True,
            "link": link.to_dict(),
        }
    )


@require_http_methods(["GET"])
@gated_paid
def get_links(request, workbench_id: str):
    """Get all causal links."""

    synara = get_synara(workbench_id, user=request.user)

    return JsonResponse(
        {
            "links": [link.to_dict() for link in synara.graph.links],
            "count": len(synara.graph.links),
        }
    )


# =============================================================================
# Evidence & Belief Update
# =============================================================================


@require_http_methods(["POST"])
@gated_paid
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

    project, err = _require_project(workbench_id, user=request.user)
    if err:
        return err

    synara = get_synara(workbench_id, user=request.user)

    result = synara.create_evidence(
        event=body.get("event", ""),
        context=body.get("context", {}),
        supports=body.get("supports", []),
        weakens=body.get("weakens", []),
        strength=body.get("strength", 1.0),
        source=body.get("source", "user"),
        data=body.get("data"),
    )

    if not save_synara(workbench_id, synara, user=request.user):
        return JsonResponse({"error": "Failed to persist evidence."}, status=500)

    return JsonResponse(
        {
            "success": True,
            "update": result.to_dict(),
            "expansion_signal": result.expansion_signal.to_dict() if result.expansion_signal else None,
        }
    )


@require_http_methods(["GET"])
@gated_paid
def get_evidence(request, workbench_id: str):
    """Get all evidence."""

    synara = get_synara(workbench_id, user=request.user)

    return JsonResponse(
        {
            "evidence": [e.to_dict() for e in synara.graph.evidence],
            "count": len(synara.graph.evidence),
        }
    )


@require_http_methods(["DELETE"])
@gated_paid
def delete_evidence(request, workbench_id: str, evidence_id: str):
    """Remove an evidence item from the belief engine."""

    synara = get_synara(workbench_id, user=request.user)

    original_len = len(synara.graph.evidence)
    synara.graph.evidence = [e for e in synara.graph.evidence if e.id != evidence_id]

    if len(synara.graph.evidence) == original_len:
        return JsonResponse({"error": "Evidence not found"}, status=404)

    if not save_synara(workbench_id, synara, user=request.user):
        return JsonResponse({"error": "Failed to persist deletion."}, status=500)
    return JsonResponse({"success": True})


@require_http_methods(["DELETE"])
@gated_paid
def delete_link(request, workbench_id: str):
    """Remove a causal link between hypotheses.

    Query params: ?from_id=h_123&to_id=h_456
    """

    from_id = request.GET.get("from_id", "")
    to_id = request.GET.get("to_id", "")

    if not from_id or not to_id:
        return JsonResponse({"error": "from_id and to_id required"}, status=400)

    synara = get_synara(workbench_id, user=request.user)

    original_len = len(synara.graph.links)
    synara.graph.links = [lnk for lnk in synara.graph.links if not (lnk.from_id == from_id and lnk.to_id == to_id)]

    if len(synara.graph.links) == original_len:
        return JsonResponse({"error": "Link not found"}, status=404)

    if not save_synara(workbench_id, synara, user=request.user):
        return JsonResponse({"error": "Failed to persist deletion."}, status=500)
    return JsonResponse({"success": True})


# =============================================================================
# Expansion Signals
# =============================================================================


@require_http_methods(["GET"])
@gated_paid
def get_expansions(request, workbench_id: str):
    """Get pending expansion signals."""

    synara = get_synara(workbench_id, user=request.user)
    pending = synara.get_pending_expansions()

    return JsonResponse(
        {
            "expansions": [s.to_dict() for s in pending],
            "count": len(pending),
        }
    )


@require_http_methods(["POST"])
@gated_paid
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

    # Guard: creating a hypothesis requires a project
    if body.get("resolution") == "new_hypothesis":
        project, err = _require_project(workbench_id, user=request.user)
        if err:
            return err

    synara = get_synara(workbench_id, user=request.user)

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
        if not save_synara(workbench_id, synara, user=request.user):
            return JsonResponse({"error": "Failed to persist expansion resolution."}, status=500)
        return JsonResponse(
            {
                "success": True,
                "new_hypothesis": new_h.to_dict() if new_h else None,
            }
        )
    else:
        return JsonResponse({"error": "Signal not found"}, status=404)


# =============================================================================
# Analysis & Queries
# =============================================================================


@require_http_methods(["GET"])
@gated_paid
def get_belief_state(request, workbench_id: str):
    """Get current belief state summary."""

    synara = get_synara(workbench_id, user=request.user)
    interface = SynaraLLMInterface(synara)

    top = synara.get_most_likely_cause()
    competing = synara.get_competing_hypotheses()
    pending = synara.get_pending_expansions()

    return JsonResponse(
        {
            "summary": interface.get_state_summary(),
            "top_hypothesis": top.to_dict() if top else None,
            "competing_hypotheses": [h.to_dict() for h in competing],
            "pending_expansions": len(pending),
            "total_evidence": len(synara.graph.evidence),
        }
    )


@require_http_methods(["GET"])
@gated_paid
def explain_hypothesis(request, workbench_id: str, hypothesis_id: str):
    """Get explanation for a hypothesis's probability."""

    synara = get_synara(workbench_id, user=request.user)
    explanation = synara.explain_belief(hypothesis_id)

    return JsonResponse(explanation)


@require_http_methods(["GET"])
@gated_paid
def get_causal_chains(request, workbench_id: str, hypothesis_id: str):
    """Get all causal chains leading to a hypothesis."""

    synara = get_synara(workbench_id, user=request.user)
    chains = synara.get_causal_chains_to(hypothesis_id)

    return JsonResponse(
        {
            "hypothesis_id": hypothesis_id,
            "chains": chains,
            "count": len(chains),
        }
    )


# =============================================================================
# LLM Integration Prompts
# =============================================================================


@require_http_methods(["GET"])
@gated_paid
def get_validation_prompt(request, workbench_id: str):
    """Get prompt for LLM to validate the causal graph."""

    synara = get_synara(workbench_id, user=request.user)
    interface = SynaraLLMInterface(synara)

    return JsonResponse(
        {
            "prompt": interface.generate_validation_prompt(),
            "context": interface.format_for_context(),
        }
    )


@require_http_methods(["GET"])
@gated_paid
def get_hypothesis_prompt(request, workbench_id: str, signal_id: str):
    """Get prompt for LLM to generate hypotheses from expansion signal."""

    synara = get_synara(workbench_id, user=request.user)

    signal = next((s for s in synara.expansion_signals if s.id == signal_id), None)
    if not signal:
        return JsonResponse({"error": "Signal not found"}, status=404)

    interface = SynaraLLMInterface(synara)

    return JsonResponse(
        {
            "prompt": interface.generate_hypothesis_prompt(signal),
            "context": interface.format_for_context(),
        }
    )


@require_http_methods(["POST"])
@gated_paid
def apply_validation_result(request, workbench_id: str):
    """
    Apply LLM validation result to the graph.

    Request body: parsed JSON from LLM validation response
    """

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    synara = get_synara(workbench_id, user=request.user)
    interface = SynaraLLMInterface(synara)

    analysis = interface.parse_validation_response(body)

    return JsonResponse(
        {
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
        }
    )


# =============================================================================
# LLM-Powered Endpoints (Server-Side API Calls)
# =============================================================================


@require_http_methods(["POST"])
@gated_paid
def llm_validate(request, workbench_id: str):
    """
    Server-side LLM validation of the causal graph.

    Generates prompt, calls Claude, parses response, returns analysis.
    No request body needed.
    """

    synara = get_synara(workbench_id, user=request.user)
    interface = SynaraLLMInterface(synara)

    analysis = interface.validate_graph_llm(user=request.user)
    if analysis is None:
        return JsonResponse(
            {
                "error": "LLM unavailable — check ANTHROPIC_API_KEY",
                "fallback_prompt": interface.generate_validation_prompt(),
            },
            status=503,
        )

    return JsonResponse(
        {
            "success": True,
            "issues": [
                {
                    "type": i.issue_type,
                    "severity": i.severity,
                    "description": i.description,
                    "involved_hypotheses": i.involved_hypotheses,
                    "suggested_fix": i.suggested_fix,
                }
                for i in analysis.issues
            ],
            "strengths": analysis.strengths,
            "gaps": analysis.gaps,
            "suggested_hypotheses": analysis.suggested_hypotheses,
            "suggested_evidence": analysis.suggested_evidence,
            "summary": analysis.summary,
        }
    )


@require_http_methods(["POST"])
@gated_paid
def llm_generate_hypotheses(request, workbench_id: str, signal_id: str):
    """
    Server-side LLM hypothesis generation from expansion signal.

    Generates prompt, calls Claude, parses response, adds hypotheses to graph.
    """

    project, err = _require_project(workbench_id, user=request.user)
    if err:
        return err

    synara = get_synara(workbench_id, user=request.user)

    signal = next((s for s in synara.expansion_signals if s.id == signal_id), None)
    if not signal:
        return JsonResponse({"error": "Signal not found"}, status=404)

    interface = SynaraLLMInterface(synara)
    hypotheses = interface.generate_hypotheses_llm(
        user=request.user,
        expansion_signal=signal,
    )

    if not hypotheses:
        return JsonResponse(
            {
                "error": "LLM unavailable or returned no hypotheses",
                "fallback_prompt": interface.generate_hypothesis_prompt(signal),
            },
            status=503,
        )

    if not save_synara(workbench_id, synara, user=request.user):
        return JsonResponse({"error": "Failed to persist generated hypotheses."}, status=500)

    return JsonResponse(
        {
            "success": True,
            "hypotheses": [h.to_dict() for h in hypotheses],
            "count": len(hypotheses),
        }
    )


@require_http_methods(["POST"])
@gated_paid
def llm_interpret_evidence(request, workbench_id: str):
    """
    Server-side LLM interpretation of the most recent evidence update.

    Request body (optional):
    {
        "evidence_id": "e_abc123"  // specific evidence, defaults to most recent
    }
    """

    try:
        body = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        body = {}

    synara = get_synara(workbench_id, user=request.user)

    if not synara.graph.evidence:
        return JsonResponse({"error": "No evidence to interpret"}, status=400)

    evidence_id = body.get("evidence_id")
    if evidence_id:
        evidence = next((e for e in synara.graph.evidence if e.id == evidence_id), None)
        if not evidence:
            return JsonResponse({"error": "Evidence not found"}, status=404)
    else:
        evidence = synara.graph.evidence[-1]

    # Find the corresponding update result
    update_result = next(
        (u for u in synara.update_history if u.evidence_id == evidence.id),
        None,
    )
    if not update_result:
        return JsonResponse({"error": "No update result for this evidence"}, status=400)

    interface = SynaraLLMInterface(synara)
    interpretation = interface.interpret_evidence_llm(
        user=request.user,
        evidence=evidence,
        update_result=update_result,
    )

    if interpretation is None:
        return JsonResponse(
            {
                "error": "LLM unavailable",
                "fallback_prompt": interface.generate_evidence_interpretation_prompt(evidence, update_result),
            },
            status=503,
        )

    return JsonResponse(
        {
            "success": True,
            "evidence_id": evidence.id,
            "interpretation": interpretation,
        }
    )


@require_http_methods(["POST"])
@gated_paid
def llm_document(request, workbench_id: str):
    """
    Server-side LLM documentation of findings.

    Request body (optional):
    {
        "format": "summary" | "a3" | "8d" | "technical"
    }
    """

    try:
        body = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        body = {}

    format_type = body.get("format", "summary")

    synara = get_synara(workbench_id, user=request.user)
    interface = SynaraLLMInterface(synara)

    document = interface.document_findings_llm(
        user=request.user,
        format_type=format_type,
    )

    if document is None:
        return JsonResponse(
            {
                "error": "LLM unavailable",
                "fallback_prompt": interface.generate_documentation_prompt(format_type),
            },
            status=503,
        )

    return JsonResponse(
        {
            "success": True,
            "format": format_type,
            "document": document,
        }
    )


# =============================================================================
# Serialization
# =============================================================================


@require_http_methods(["GET"])
@gated_paid
def export_synara(request, workbench_id: str):
    """Export Synara state as JSON."""

    synara = get_synara(workbench_id, user=request.user)

    return JsonResponse(synara.to_dict())


@require_http_methods(["POST"])
@gated_paid
def import_synara(request, workbench_id: str):
    """Import Synara state from JSON."""

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    project, err = _require_project(workbench_id, user=request.user)
    if err:
        return err

    synara = Synara.from_dict(body)
    if len(_synara_cache) >= _SYNARA_CACHE_MAX:
        _synara_cache.pop(next(iter(_synara_cache)), None)
    _synara_cache[workbench_id] = synara

    if not save_synara(workbench_id, synara, user=request.user):
        return JsonResponse({"error": "Failed to persist imported state."}, status=500)

    return JsonResponse(
        {
            "success": True,
            "hypothesis_count": len(synara.graph.hypotheses),
            "evidence_count": len(synara.graph.evidence),
        }
    )


# =============================================================================
# DSL: Formal Hypothesis Language
# =============================================================================


@require_http_methods(["POST"])
@gated_paid
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

    return JsonResponse(
        {
            "success": True,
            "hypothesis": hypothesis.to_dict(),
            "formatted": {
                "natural": format_hypothesis(hypothesis, "natural"),
                "formal": format_hypothesis(hypothesis, "formal"),
                "code": format_hypothesis(hypothesis, "code"),
            },
        }
    )


@require_http_methods(["POST"])
@gated_paid
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

    return JsonResponse(
        {
            "success": True,
            **result,
        }
    )


@require_http_methods(["POST"])
@gated_paid
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

    return JsonResponse(
        {
            "success": True,
            "evaluation": evaluation.to_dict(),
            "hypothesis": evaluation.hypothesis.to_dict(),
        }
    )


@require_http_methods(["POST"])
@gated_paid
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

    project, err = _require_project(workbench_id, user=request.user)
    if err:
        return err

    # Parse and validate
    validation = validate_hypothesis(text)
    if not validation["valid"]:
        return JsonResponse(
            {
                "success": False,
                "error": "Invalid hypothesis",
                "fallacies": validation["fallacies"],
            },
            status=400,
        )

    synara = get_synara(workbench_id, user=request.user)

    # Create hypothesis with formal structure
    h = synara.create_hypothesis(
        description=text,  # Raw formal statement as description
        domain_conditions=body.get("domain_context", {}),
        behavior_class=validation["hypothesis"]["structure"].get("type", "formal"),
        latent_causes=validation["variables"],  # Variables as "causes" to track
        prior=body.get("prior", 0.5),
        source="dsl",
    )

    if not save_synara(workbench_id, synara, user=request.user):
        return JsonResponse({"error": "Failed to persist formal hypothesis."}, status=500)

    return JsonResponse(
        {
            "success": True,
            "hypothesis": h.to_dict(),
            "parsed": validation["hypothesis"],
            "formatted": {
                "natural": format_hypothesis(DSLParser().parse(text), "natural"),
                "formal": format_hypothesis(DSLParser().parse(text), "formal"),
            },
        }
    )


@require_http_methods(["POST"])
@gated_paid
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

    synara = get_synara(workbench_id, user=request.user)

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

    if result:
        if not save_synara(workbench_id, synara, user=request.user):
            return JsonResponse({"error": "Failed to persist evaluation results."}, status=500)

    return JsonResponse(
        {
            "success": True,
            "evaluation": evaluation.to_dict(),
            "belief_update": result.to_dict() if result else None,
            "hypothesis_posterior": synara.graph.hypotheses[hypothesis_id].posterior,
        }
    )
