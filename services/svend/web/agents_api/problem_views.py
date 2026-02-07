"""Problem Session API views.

CRUD operations for problem sessions - the core of the Decision Science Workbench.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth
from .models import Problem

logger = logging.getLogger(__name__)

# Shared context directory for cross-agent access
CONTEXT_DIR = Path(settings.BASE_DIR).parent / "shared_context" / "problems"


# =============================================================================
# Context File Management
# =============================================================================

def get_context_path(problem_id: str, user_id: int) -> Path:
    """Get the path to a problem's context file."""
    user_dir = CONTEXT_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / f"{problem_id}.json"


def write_context_file(problem: Problem) -> Path:
    """
    Write/update the shared context file for a problem.

    This file is readable by all agents and tracks the full problem state.
    """
    context_path = get_context_path(str(problem.id), problem.user_id)

    context = {
        "problem_id": str(problem.id),
        "user_id": problem.user_id,
        "title": problem.title,
        "status": problem.status,
        "last_updated": datetime.now().isoformat(),

        # Methodology tracking
        "methodology": {
            "type": problem.methodology,
            "dmaic_phase": problem.dmaic_phase,
            "phase_history": problem.phase_history,
            "guidance": problem.get_phase_guidance() if problem.dmaic_phase else None,
        },

        # The Effect (what we're trying to understand)
        "effect": {
            "description": problem.effect_description,
            "magnitude": problem.effect_magnitude,
            "first_observed": problem.effect_first_observed,
            "confidence": problem.effect_confidence,
        },

        # Decision context
        "context": {
            "domain": problem.domain,
            "stakeholders": problem.stakeholders,
            "constraints": problem.constraints,
            "prior_beliefs": problem.prior_beliefs,
            "can_experiment": problem.can_experiment,
            "available_data": problem.available_data,
        },

        # Current hypotheses with probabilities
        "hypotheses": problem.hypotheses,

        # Evidence gathered
        "evidence": problem.evidence,

        # What we've ruled out
        "dead_ends": problem.dead_ends,

        # Current understanding
        "understanding": {
            "probable_causes": problem.probable_causes,
            "key_uncertainties": problem.key_uncertainties,
            "recommended_next_steps": problem.recommended_next_steps,
        },

        # Cognitive bias warnings
        "bias_warnings": problem.bias_warnings,

        # Interview state (if any)
        "interview": problem.interview_state if hasattr(problem, 'interview_state') else None,

        # Resolution (if resolved)
        "resolution": {
            "summary": problem.resolution_summary,
            "confidence": problem.resolution_confidence,
        } if problem.status == "resolved" else None,

        # For agent handoff
        "agent_context": {
            "source": "problem_session",
            "can_update": True,
            "endpoints": {
                "add_evidence": f"/api/problems/{problem.id}/evidence/",
                "add_hypothesis": f"/api/problems/{problem.id}/hypotheses/",
            },
        },
    }

    with open(context_path, 'w') as f:
        json.dump(context, f, indent=2, default=str)

    logger.info(f"Updated context file for problem {problem.id}")
    return context_path


def read_context_file(problem_id: str, user_id: int) -> dict | None:
    """Read a problem's context file."""
    context_path = get_context_path(problem_id, user_id)
    if context_path.exists():
        with open(context_path) as f:
            return json.load(f)
    return None


# =============================================================================
# Interview Session Management
# =============================================================================

# In-memory interview sessions (could be moved to cache/Redis for scale)
_interview_sessions = {}


def get_interview_session(problem_id: str):
    """Get or create an interview session for a problem."""
    if problem_id not in _interview_sessions:
        import sys
        sys.path.insert(0, "/home/eric/kjerne/services/svend/agents/agents")
        from guide.decision import DecisionGuide

        guide = DecisionGuide()
        guide.start()
        _interview_sessions[problem_id] = guide

    return _interview_sessions[problem_id]


def clear_interview_session(problem_id: str):
    """Clear an interview session."""
    if problem_id in _interview_sessions:
        del _interview_sessions[problem_id]


def problem_to_dict(problem: Problem) -> dict:
    """Serialize a Problem to dict."""
    result = {
        "id": str(problem.id),
        "title": problem.title,
        "status": problem.status,

        # Methodology
        "methodology": problem.methodology,
        "dmaic_phase": problem.dmaic_phase,
        "phase_history": problem.phase_history,

        # Effect
        "effect": {
            "description": problem.effect_description,
            "magnitude": problem.effect_magnitude,
            "first_observed": problem.effect_first_observed,
            "confidence": problem.effect_confidence,
        },

        # Context
        "context": {
            "domain": problem.domain,
            "stakeholders": problem.stakeholders,
            "constraints": problem.constraints,
            "prior_beliefs": problem.prior_beliefs,
            "can_experiment": problem.can_experiment,
            "available_data": problem.available_data,
        },

        # Living state
        "hypotheses": problem.hypotheses,
        "evidence": problem.evidence,
        "dead_ends": problem.dead_ends,

        # Understanding
        "probable_causes": problem.probable_causes,
        "key_uncertainties": problem.key_uncertainties,
        "recommended_next_steps": problem.recommended_next_steps,

        # Bias warnings
        "bias_warnings": problem.bias_warnings,

        # Resolution
        "resolution": {
            "summary": problem.resolution_summary,
            "confidence": problem.resolution_confidence,
        } if problem.status == "resolved" else None,

        # Meta
        "created_at": problem.created_at.isoformat(),
        "updated_at": problem.updated_at.isoformat(),
    }

    # Add phase guidance if using DMAIC
    if problem.methodology == "dmaic" and problem.dmaic_phase:
        result["phase_guidance"] = problem.get_phase_guidance()

    return result


@csrf_exempt
@require_http_methods(["GET", "POST"])
def problems_list(request):
    """List or create problems."""

    if request.method == "GET":
        # List user's problems
        status_filter = request.GET.get("status")
        problems = Problem.objects.filter(user=request.user)

        if status_filter:
            problems = problems.filter(status=status_filter)

        return JsonResponse({
            "problems": [
                {
                    "id": str(p.id),
                    "title": p.title,
                    "status": p.status,
                    "effect_summary": p.effect_description[:100] + "..." if len(p.effect_description) > 100 else p.effect_description,
                    "hypothesis_count": len(p.hypotheses),
                    "evidence_count": len(p.evidence),
                    "top_cause": p.probable_causes[0] if p.probable_causes else None,
                    "updated_at": p.updated_at.isoformat(),
                }
                for p in problems
            ]
        })

    # POST - create new problem
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Validate required fields
    if not data.get("title"):
        return JsonResponse({"error": "Title is required"}, status=400)
    if not data.get("effect_description"):
        return JsonResponse({"error": "Effect description is required"}, status=400)

    # Create problem
    problem = Problem.objects.create(
        user=request.user,
        title=data["title"],
        effect_description=data["effect_description"],
        effect_magnitude=data.get("effect_magnitude", ""),
        effect_first_observed=data.get("effect_first_observed", ""),
        effect_confidence=data.get("effect_confidence", "medium"),
        domain=data.get("domain", ""),
        stakeholders=data.get("stakeholders", []),
        constraints=data.get("constraints", []),
        prior_beliefs=data.get("prior_beliefs", []),
        can_experiment=data.get("can_experiment", True),
        available_data=data.get("available_data", ""),
    )

    # Check for initial biases in the description
    try:
        import sys
        sys.path.insert(0, "/home/eric/kjerne/services/svend/agents/agents")
        from guide.decision import detect_biases

        all_text = f"{data['title']} {data['effect_description']} {data.get('available_data', '')}"
        biases = detect_biases(all_text)

        if biases:
            problem.bias_warnings = [
                {
                    "type": b.bias_type,
                    "description": b.description,
                    "evidence": b.evidence,
                    "suggestion": b.suggestion,
                    "timestamp": datetime.now().isoformat(),
                }
                for b in biases
            ]
            problem.save(update_fields=["bias_warnings"])
    except Exception as e:
        logger.warning(f"Bias detection failed: {e}")

    # Phase 1 dual-write: create linked core.Project
    try:
        core_project = problem.ensure_core_project()
        logger.info(f"Created core.Project {core_project.id} for Problem {problem.id}")
    except Exception as e:
        logger.warning(f"Dual-write failed for Problem {problem.id}: {e}")

    # Write initial context file
    write_context_file(problem)

    return JsonResponse({
        "id": str(problem.id),
        "success": True,
        "bias_warnings": problem.bias_warnings,
    })


@csrf_exempt
@require_http_methods(["GET", "PATCH", "DELETE"])
def problem_detail(request, problem_id):
    """Get, update, or delete a problem."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(problem_to_dict(problem))

    if request.method == "DELETE":
        problem.delete()
        return JsonResponse({"success": True})

    # PATCH - update
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Update allowed fields
    updatable = [
        "title", "status", "effect_description", "effect_magnitude",
        "effect_first_observed", "effect_confidence", "domain",
        "stakeholders", "constraints", "prior_beliefs", "can_experiment",
        "available_data", "key_uncertainties", "recommended_next_steps",
    ]

    for field in updatable:
        if field in data:
            setattr(problem, field, data[field])

    problem.save()
    write_context_file(problem)
    return JsonResponse({"success": True})


@csrf_exempt
@require_http_methods(["POST"])
@gated
def add_hypothesis(request, problem_id):
    """Add a hypothesis to a problem."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not data.get("cause"):
        return JsonResponse({"error": "Cause is required"}, status=400)

    hypothesis = problem.add_hypothesis(
        cause=data["cause"],
        mechanism=data.get("mechanism", ""),
        probability=data.get("probability", 0.5),
    )

    # Phase 1 dual-write: sync hypothesis to core.Hypothesis
    try:
        problem.sync_hypothesis_to_core(hypothesis)
    except Exception as e:
        logger.warning(f"Hypothesis dual-write failed for Problem {problem.id}: {e}")

    write_context_file(problem)

    return JsonResponse({
        "success": True,
        "hypothesis": hypothesis,
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def add_evidence(request, problem_id):
    """Add evidence to a problem."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not data.get("summary"):
        return JsonResponse({"error": "Summary is required"}, status=400)

    evidence = problem.add_evidence(
        summary=data["summary"],
        evidence_type=data.get("type", "observation"),
        source=data.get("source", ""),
        supports=data.get("supports", []),
        weakens=data.get("weakens", []),
    )

    # Phase 1 dual-write: sync evidence to core.Evidence + EvidenceLinks
    try:
        problem.sync_evidence_to_core(evidence)
    except Exception as e:
        logger.warning(f"Evidence dual-write failed for Problem {problem.id}: {e}")

    # Update probable causes after new evidence
    problem.update_understanding()
    write_context_file(problem)

    return JsonResponse({
        "success": True,
        "evidence": evidence,
        "updated_hypotheses": problem.hypotheses,
        "probable_causes": problem.probable_causes,
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def reject_hypothesis(request, problem_id, hypothesis_id):
    """Reject a hypothesis and move to dead ends."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    reason = data.get("reason", "")
    if not reason:
        return JsonResponse({"error": "Reason for rejection is required"}, status=400)

    problem.reject_hypothesis(hypothesis_id, reason)

    # Phase 1 dual-write: mark core.Hypothesis as rejected
    try:
        core_hyp = problem._find_core_hypothesis(hypothesis_id)
        if core_hyp:
            core_hyp.status = "rejected"
            core_hyp.save(update_fields=["status"])
    except Exception as e:
        logger.warning(f"Reject dual-write failed for Problem {problem.id}: {e}")

    write_context_file(problem)

    return JsonResponse({
        "success": True,
        "dead_ends": problem.dead_ends,
        "probable_causes": problem.probable_causes,
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def resolve_problem(request, problem_id):
    """Mark a problem as resolved."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    summary = data.get("summary", "")
    if not summary:
        return JsonResponse({"error": "Resolution summary is required"}, status=400)

    problem.resolve(
        summary=summary,
        confidence=data.get("confidence", "medium"),
    )

    # Phase 1 dual-write: resolve the core.Project
    try:
        if problem.core_project:
            problem.core_project.resolve(
                summary=summary,
                confidence=data.get("confidence", "medium"),
            )
    except Exception as e:
        logger.warning(f"Resolve dual-write failed for Problem {problem.id}: {e}")

    write_context_file(problem)

    return JsonResponse({
        "success": True,
        "status": problem.status,
        "resolution": {
            "summary": problem.resolution_summary,
            "confidence": problem.resolution_confidence,
        },
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def generate_hypotheses(request, problem_id):
    """Use LLM to generate hypotheses based on problem context."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    # Track query usage
    if not request.user.can_query():
        return JsonResponse({
            "error": "Daily query limit reached",
            "limit": request.user.daily_limit,
        }, status=429)

    request.user.increment_queries()

    try:
        # Use shared LLM to generate hypotheses
        from .views import get_shared_llm

        llm = get_shared_llm()
        if not llm:
            return JsonResponse({"error": "LLM not available"}, status=503)

        prompt = f"""Based on this problem, suggest 3-5 possible causes (hypotheses).

PROBLEM: {problem.title}

EFFECT OBSERVED:
{problem.effect_description}
Magnitude: {problem.effect_magnitude or 'Not specified'}
First observed: {problem.effect_first_observed or 'Not specified'}

CONTEXT:
Domain: {problem.domain or 'Not specified'}
Available data: {problem.available_data or 'None specified'}
Can experiment: {'Yes' if problem.can_experiment else 'No'}

EXISTING HYPOTHESES:
{json.dumps(problem.hypotheses, indent=2) if problem.hypotheses else 'None yet'}

Generate hypotheses as a JSON array:
[
  {{"cause": "Brief cause description", "mechanism": "How this would cause the effect", "testable": true, "initial_probability": 0.3}},
  ...
]

Focus on testable, specific causes. Vary the probabilities based on how likely each seems given the context."""

        response = llm.complete(prompt, max_tokens=1000)

        # Parse response
        import re
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            hypotheses_data = json.loads(match.group())

            added = []
            for h in hypotheses_data:
                hyp = problem.add_hypothesis(
                    cause=h.get("cause", ""),
                    mechanism=h.get("mechanism", ""),
                    probability=h.get("initial_probability", 0.5),
                )
                # Phase 1 dual-write
                try:
                    problem.sync_hypothesis_to_core(hyp)
                except Exception as e:
                    logger.warning(f"LLM hypothesis dual-write failed: {e}")
                added.append(hyp)

            write_context_file(problem)

            return JsonResponse({
                "success": True,
                "generated": added,
                "all_hypotheses": problem.hypotheses,
            })

        return JsonResponse({"error": "Failed to parse LLM response"}, status=500)

    except Exception as e:
        logger.exception("Hypothesis generation failed")
        return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# Interview Endpoints
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def start_interview(request, problem_id):
    """Start a guided interview for a problem."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    resume = data.get("resume", False)

    # Check if there's saved progress to resume
    if resume and problem.interview_state:
        # Resume from saved state
        clear_interview_session(str(problem_id))
        guide = get_interview_session(str(problem_id))
        guide.load_state(problem.interview_state)
    elif problem.interview_state and not data.get("fresh", False):
        # Saved state exists - ask user if they want to resume
        saved = problem.interview_state
        return JsonResponse({
            "success": True,
            "has_saved_progress": True,
            "saved_progress": {
                "answered": len(saved.get("answers", {})),
                "current_section": saved.get("current_section", 0) + 1,
            },
            "message": "You have saved interview progress. Resume or start fresh?",
        })
    else:
        # Start fresh
        clear_interview_session(str(problem_id))
        guide = get_interview_session(str(problem_id))
        # Clear any saved state
        problem.interview_state = None
        problem.save(update_fields=["interview_state"])

    question = guide.get_current_question()
    section = guide.get_current_section()
    progress = guide.get_progress()

    return JsonResponse({
        "success": True,
        "interview_started": True,
        "resumed": resume,
        "section": {
            "id": section.id,
            "title": section.title,
            "description": section.description,
        } if section else None,
        "question": _format_question(question) if question else None,
        "progress": progress,
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def interview_answer(request, problem_id):
    """Submit an answer to the current interview question."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    answer = data.get("answer")
    if answer is None:
        return JsonResponse({"error": "Answer is required"}, status=400)

    guide = get_interview_session(str(problem_id))

    # Submit the answer
    success, message, next_question = guide.submit_answer(answer)

    if not success:
        return JsonResponse({
            "success": False,
            "error": message,
            "question": _format_question(guide.get_current_question()),
        }, status=400)

    # Save progress after each answer
    problem.interview_state = guide.save_state()
    problem.save(update_fields=["interview_state", "updated_at"])

    # Check if interview is complete
    if guide.is_complete():
        # Synthesize results and update problem
        result = guide.synthesize()
        brief = guide.get_brief()

        # Update problem with interview results
        _apply_interview_to_problem(problem, guide, brief)

        # Clear the session and saved state
        clear_interview_session(str(problem_id))
        problem.interview_state = None
        problem.save(update_fields=["interview_state"])

        return JsonResponse({
            "success": True,
            "complete": True,
            "message": "Interview complete!",
            "brief": brief.to_dict(),
            "bias_warnings": [
                {
                    "type": b["type"],
                    "description": b["description"],
                    "suggestion": b["suggestion"],
                }
                for b in brief.detected_biases
            ],
            "recommended_agent": brief.recommended_agent,
            "recommended_action": brief.recommended_action,
            "routing_reasoning": brief.routing_reasoning,
        })

    # Return next question
    section = guide.get_current_section()
    progress = guide.get_progress()

    return JsonResponse({
        "success": True,
        "complete": False,
        "section": {
            "id": section.id,
            "title": section.title,
            "description": section.description,
        } if section else None,
        "question": _format_question(next_question) if next_question else None,
        "progress": progress,
        "bias_count": len(guide.bias_warnings),
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def interview_skip(request, problem_id):
    """Skip the current interview question (if optional)."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    guide = get_interview_session(str(problem_id))

    success, message, next_question = guide.skip_question()

    if not success:
        return JsonResponse({
            "success": False,
            "error": message,
        }, status=400)

    # Save progress
    problem.interview_state = guide.save_state()
    problem.save(update_fields=["interview_state", "updated_at"])

    section = guide.get_current_section()
    progress = guide.get_progress()

    return JsonResponse({
        "success": True,
        "section": {
            "id": section.id,
            "title": section.title,
            "description": section.description,
        } if section else None,
        "question": _format_question(next_question) if next_question else None,
        "progress": progress,
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def interview_save(request, problem_id):
    """Save interview progress and exit."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    if str(problem_id) not in _interview_sessions:
        return JsonResponse({
            "success": False,
            "error": "No active interview session to save",
        }, status=400)

    guide = _interview_sessions[str(problem_id)]

    # Save state to database
    problem.interview_state = guide.save_state()
    problem.save(update_fields=["interview_state", "updated_at"])

    # Clear the in-memory session
    clear_interview_session(str(problem_id))

    progress = guide.get_progress()

    return JsonResponse({
        "success": True,
        "message": "Interview progress saved. You can resume later.",
        "progress": progress,
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def interview_status(request, problem_id):
    """Get current interview status."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    # Check for active in-memory session
    if str(problem_id) in _interview_sessions:
        guide = _interview_sessions[str(problem_id)]
        question = guide.get_current_question()
        section = guide.get_current_section()
        progress = guide.get_progress()

        return JsonResponse({
            "active": True,
            "complete": guide.is_complete(),
            "has_saved_progress": False,
            "section": {
                "id": section.id,
                "title": section.title,
                "description": section.description,
            } if section else None,
            "question": _format_question(question) if question else None,
            "progress": progress,
            "bias_count": len(guide.bias_warnings),
        })

    # Check for saved progress
    if problem.interview_state:
        saved = problem.interview_state
        return JsonResponse({
            "active": False,
            "has_saved_progress": True,
            "saved_progress": {
                "answered": len(saved.get("answers", {})),
                "current_section": saved.get("current_section", 0) + 1,
                "bias_count": len(saved.get("bias_warnings", [])),
            },
            "message": "You have saved interview progress that can be resumed.",
        })

    return JsonResponse({
        "active": False,
        "has_saved_progress": False,
        "message": "No active interview session",
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_context_file(request, problem_id):
    """Get the shared context file for a problem."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    context = read_context_file(str(problem_id), request.user.id)

    if context is None:
        # Generate it now
        write_context_file(problem)
        context = read_context_file(str(problem_id), request.user.id)

    return JsonResponse({
        "success": True,
        "context": context,
        "path": str(get_context_path(str(problem_id), request.user.id)),
    })


# =============================================================================
# Helper Functions
# =============================================================================

def _format_question(question) -> dict | None:
    """Format a question object for JSON response."""
    if question is None:
        return None

    return {
        "id": question.id,
        "text": question.text,
        "type": question.question_type.value,
        "options": question.options if question.options else None,
        "required": question.required,
        "help_text": question.help_text,
        "scale_min": question.scale_min if question.question_type.value == "scale" else None,
        "scale_max": question.scale_max if question.question_type.value == "scale" else None,
    }


def _apply_interview_to_problem(problem: Problem, guide, brief):
    """Apply interview results to a problem."""
    answers = guide.state.answers

    # Update effect description if we got more detail
    if answers.get("situation"):
        if problem.effect_description:
            problem.effect_description += f"\n\nContext: {answers['situation']}"
        else:
            problem.effect_description = answers["situation"]

    # Update available data
    if answers.get("have_data"):
        problem.available_data = answers["have_data"]

    # Add constraints from time horizon
    if answers.get("time_horizon"):
        if not problem.constraints:
            problem.constraints = []
        problem.constraints.append(f"Timeline: {answers['time_horizon']}")

    # Add stakeholders
    if brief.stakeholders:
        problem.stakeholders = brief.stakeholders

    # Add prior beliefs from initial lean
    if answers.get("initial_lean"):
        if not problem.prior_beliefs:
            problem.prior_beliefs = []
        problem.prior_beliefs.append({
            "belief": answers["initial_lean"],
            "confidence": answers.get("confidence", 5),
        })

    # Add key uncertainties from data gaps
    if brief.data_gaps:
        problem.key_uncertainties = brief.data_gaps

    # Add recommended next steps
    if brief.recommended_action:
        problem.recommended_next_steps = [{
            "action": brief.recommended_action,
            "agent": brief.recommended_agent,
            "reasoning": brief.routing_reasoning,
        }]

    # Add bias warnings from interview
    if brief.detected_biases:
        existing_types = {b.get("type") for b in problem.bias_warnings}
        for bias in brief.detected_biases:
            if bias["type"] not in existing_types:
                problem.bias_warnings.append({
                    "type": bias["type"],
                    "description": bias["description"],
                    "evidence": bias.get("evidence", ""),
                    "suggestion": bias["suggestion"],
                    "timestamp": datetime.now().isoformat(),
                    "source": "interview",
                })

    problem.save()
    write_context_file(problem)


# =============================================================================
# Methodology & Phase Management
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def set_methodology(request, problem_id):
    """Set the problem-solving methodology for a problem."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    methodology = data.get("methodology")
    valid_methodologies = ["none", "dmaic", "doe", "pdca", "a3"]

    if methodology not in valid_methodologies:
        return JsonResponse({
            "error": f"Invalid methodology. Choose from: {valid_methodologies}"
        }, status=400)

    problem.set_methodology(methodology)
    write_context_file(problem)

    return JsonResponse({
        "success": True,
        "methodology": problem.methodology,
        "dmaic_phase": problem.dmaic_phase,
        "phase_guidance": problem.get_phase_guidance() if problem.dmaic_phase else None,
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated
def advance_phase(request, problem_id):
    """Advance to the next DMAIC phase."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    if problem.methodology != "dmaic":
        return JsonResponse({
            "error": "Problem is not using DMAIC methodology"
        }, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = {}

    notes = data.get("notes", "")

    # Get current phase guidance to find next phase
    guidance = problem.get_phase_guidance()
    next_phase = guidance.get("next_phase")

    if not next_phase:
        return JsonResponse({
            "error": "Already at final phase (Control)",
            "current_phase": problem.dmaic_phase,
        }, status=400)

    problem.advance_phase(next_phase, notes)
    write_context_file(problem)

    return JsonResponse({
        "success": True,
        "previous_phase": guidance.get("focus", ""),
        "current_phase": problem.dmaic_phase,
        "phase_guidance": problem.get_phase_guidance(),
        "phase_history": problem.phase_history,
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def get_phase_guidance(request, problem_id):
    """Get guidance for the current phase."""

    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
    except Problem.DoesNotExist:
        return JsonResponse({"error": "Problem not found"}, status=404)

    return JsonResponse({
        "methodology": problem.methodology,
        "current_phase": problem.dmaic_phase,
        "guidance": problem.get_phase_guidance() if problem.dmaic_phase else None,
        "phase_history": problem.phase_history,
        "methodologies_available": [
            {"id": "none", "name": "None/General", "description": "No specific methodology"},
            {"id": "dmaic", "name": "Six Sigma DMAIC", "description": "Define, Measure, Analyze, Improve, Control"},
            {"id": "doe", "name": "Design of Experiments", "description": "Structured experimentation approach"},
            {"id": "pdca", "name": "Plan-Do-Check-Act", "description": "Continuous improvement cycle"},
            {"id": "a3", "name": "A3 Problem Solving", "description": "Toyota-style structured problem solving"},
        ],
    })
