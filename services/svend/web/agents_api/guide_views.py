"""Guide API - AI assistant for analysis, summarization, and report generation.

This module provides rate-limited LLM access across the application:
- DSW analysis guidance
- Whiteboard summarization
- Project/CAPA report generation

Rate limits are enforced per user tier via LLMManager.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid, require_auth, require_enterprise
from .llm_manager import LLMManager

logger = logging.getLogger(__name__)


# System prompts for different contexts
SYSTEM_PROMPTS = {
    "dsw": """You are a data analysis assistant in the SVEND Decision Science Workbench.
You help users:
- Understand their data and suggest appropriate analyses
- Interpret statistical results (SPC, capability, DOE, regression, hypothesis tests)
- Connect findings to project hypotheses and update probability assessments
- Recommend next steps based on findings and investigation goals
- Explain statistical concepts in plain language

When a project is linked, help the user:
- Evaluate evidence for/against their hypotheses
- Suggest likelihood ratios based on analysis results
- Identify what additional analyses would be most informative
- Connect findings back to the problem statement

Be concise but thorough. Use markdown formatting. Focus on actionable insights.""",

    "whiteboard": """You are a facilitation assistant for kaizen and problem-solving sessions.
You help users:
- Summarize whiteboard content (brainstorming, fishbone diagrams, process maps)
- Identify patterns and themes in ideas
- Suggest hypotheses based on if-then relationships
- Structure findings for reports

Be concise. Focus on synthesis and actionable takeaways.""",

    "project": """You are a project summarization assistant for quality and operational excellence.
You help users:
- Compile project findings into structured reports (CAPA, 8D, A3)
- Synthesize data from DSW analyses and whiteboard sessions
- Write clear problem statements and root cause summaries
- Generate action plans and control measures

Follow the user's template structure. Be professional and precise.""",

    "general": """You are an AI assistant for SVEND, a decision science platform.
You help with data analysis, problem-solving, and quality improvement.
Be helpful, concise, and professional.""",
}


@csrf_exempt
@require_enterprise
@require_http_methods(["POST"])
def guide_chat(request):
    """General-purpose guide chat endpoint.

    Request body:
    {
        "message": "User's question or request",
        "context": "dsw" | "whiteboard" | "project" | "general",
        "data": {
            // Optional context data (analysis results, elements, etc.)
        },
        "history": [
            // Optional conversation history
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

    Response:
    {
        "response": "Assistant's response",
        "model": "model used",
        "rate_limit": {"remaining": N, "limit": N}
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message = body.get("message", "").strip()
    if not message:
        return JsonResponse({"error": "Message required"}, status=400)

    context_type = body.get("context", "general")
    context_data = body.get("data", {})
    history = body.get("history", [])

    # Build system prompt with context
    system = SYSTEM_PROMPTS.get(context_type, SYSTEM_PROMPTS["general"])

    # Add context data to system prompt if provided
    if context_data:
        if context_type == "dsw":
            context_parts = []
            # Add project context if linked
            if "project" in context_data and context_data["project"]:
                proj = context_data["project"]
                context_parts.append(f"=== LINKED PROJECT ===")
                context_parts.append(f"Project: {proj.get('title', 'Untitled')}")
                if proj.get("problem_statement"):
                    context_parts.append(f"Problem: {proj['problem_statement']}")
                hypotheses = proj.get("hypotheses", [])
                if hypotheses:
                    context_parts.append("\nHypotheses under investigation:")
                    for i, h in enumerate(hypotheses[:5], 1):
                        prob = int((h.get("probability") or 0.5) * 100)
                        context_parts.append(f"  {i}. \"{h.get('statement', '')}\" - {prob}% probability ({h.get('status', 'investigating')})")
                    context_parts.append("\nHelp the user evaluate evidence for/against these hypotheses.")
            # Add session context
            if "summary" in context_data:
                context_parts.append(f"\n=== CURRENT SESSION ===\n{context_data['summary']}")
            if context_parts:
                system += "\n\n" + "\n".join(context_parts)
        elif context_type == "whiteboard" and "elements" in context_data:
            elements_summary = summarize_whiteboard_elements(context_data["elements"])
            system += f"\n\nWhiteboard content:\n{elements_summary}"
        elif context_type == "project" and "project" in context_data:
            system += f"\n\nProject: {context_data['project'].get('title', 'Untitled')}"
            if context_data.get("template"):
                system += f"\nTemplate: {context_data['template']}"

    # Build messages list
    messages = []
    for msg in history[-10:]:  # Keep last 10 messages for context
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    # Call LLM with rate limiting
    result = LLMManager.chat(
        user=request.user,
        messages=messages,
        system=system,
        max_tokens=2048,
        temperature=0.7,
    )

    if result is None:
        return JsonResponse({
            "error": "LLM unavailable. Please check API configuration.",
        }, status=503)

    if result.get("rate_limited"):
        return JsonResponse({
            "error": result["error"],
            "rate_limited": True,
            "rate_limit": result.get("rate_limit", {}),
        }, status=429)

    return JsonResponse({
        "response": result["content"],
        "model": result["model"],
        "rate_limit": result.get("rate_limit", {}),
    })


@csrf_exempt
@require_enterprise
@require_http_methods(["POST"])
def summarize_project(request):
    """Summarize an entire project for report generation.

    Request body:
    {
        "project_id": "uuid",
        "template": "capa" | "8d" | "a3" | "custom",
        "custom_template": "..." (if template == "custom"),
        "include": {
            "hypotheses": true,
            "evidence": true,
            "dsw_results": true,
            "whiteboard": true
        }
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    project_id = body.get("project_id")
    if not project_id:
        return JsonResponse({"error": "project_id required"}, status=400)

    template = body.get("template", "capa")
    include = body.get("include", {})

    # Load project data
    try:
        from core.models import Project, Hypothesis
        from .models import Board

        project = Project.objects.get(id=project_id, user=request.user)
    except Project.DoesNotExist:
        return JsonResponse({"error": "Project not found"}, status=404)

    # Gather context
    context_parts = [f"Project: {project.title}"]
    if project.description:
        context_parts.append(f"Description: {project.description}")

    # Hypotheses
    if include.get("hypotheses", True):
        hypotheses = Hypothesis.objects.filter(project=project)
        if hypotheses.exists():
            context_parts.append("\nHypotheses:")
            for h in hypotheses[:20]:
                status = h.status
                prob = f"{h.current_probability:.0%}"
                context_parts.append(f"- [{status}] {h.statement} (P={prob})")

    # Whiteboard content
    if include.get("whiteboard", True):
        boards = Board.objects.filter(project=project)
        for board in boards[:5]:
            if board.elements:
                context_parts.append(f"\nWhiteboard '{board.name}':")
                elements_summary = summarize_whiteboard_elements(board.elements)
                context_parts.append(elements_summary)

    # Build template-specific prompt
    template_prompts = {
        "capa": """Generate a CAPA (Corrective and Preventive Action) report with:
1. Problem Description
2. Root Cause Analysis
3. Corrective Actions (immediate)
4. Preventive Actions (systemic)
5. Verification Plan
6. Effectiveness Check Criteria""",

        "8d": """Generate an 8D report with:
D0: Preparation/Emergency Response
D1: Team Formation
D2: Problem Description
D3: Containment Actions
D4: Root Cause Analysis
D5: Permanent Corrective Actions
D6: Implementation & Validation
D7: Preventive Actions
D8: Team Recognition""",

        "a3": """Generate an A3 report (single page summary) with:
- Background (why this matters)
- Current Condition
- Goal/Target Condition
- Root Cause Analysis
- Countermeasures
- Implementation Plan
- Follow-up""",
    }

    template_instruction = body.get("custom_template") or template_prompts.get(template, template_prompts["capa"])

    system = SYSTEM_PROMPTS["project"]
    user_message = f"""Based on this project data, generate a report.

{chr(10).join(context_parts)}

---
{template_instruction}

Generate the report now, filling in based on available data. If data is missing for a section, note what's needed."""

    # Call LLM
    result = LLMManager.chat(
        user=request.user,
        messages=[{"role": "user", "content": user_message}],
        system=system,
        max_tokens=4096,
        temperature=0.5,  # Lower temperature for more focused output
    )

    if result is None:
        return JsonResponse({"error": "LLM unavailable"}, status=503)

    if result.get("rate_limited"):
        return JsonResponse({
            "error": result["error"],
            "rate_limited": True,
        }, status=429)

    return JsonResponse({
        "report": result["content"],
        "template": template,
        "project_id": str(project.id),
        "project_title": project.title,
        "model": result["model"],
        "rate_limit": result.get("rate_limit", {}),
    })


@csrf_exempt
@require_auth
@require_http_methods(["GET"])
def rate_limit_status(request):
    """Get current rate limit status for the user."""
    from .models import check_rate_limit, LLMUsage
    from django.utils import timezone

    allowed, remaining, limit = check_rate_limit(request.user)
    usage = LLMUsage.get_daily_usage(request.user, timezone.now().date())

    tier = getattr(request.user, 'subscription_tier', 'FREE') or 'FREE'

    return JsonResponse({
        "tier": tier,
        "limit": limit,
        "remaining": remaining,
        "used": usage.get("total_requests") or 0,
        "input_tokens_today": usage.get("total_input_tokens") or 0,
        "output_tokens_today": usage.get("total_output_tokens") or 0,
    })


def summarize_whiteboard_elements(elements):
    """Create a text summary of whiteboard elements."""
    if not elements:
        return "No elements"

    summary_parts = []

    # Group by type
    by_type = {}
    for el in elements:
        el_type = el.get("type", "unknown")
        if el_type not in by_type:
            by_type[el_type] = []
        by_type[el_type].append(el)

    for el_type, items in by_type.items():
        if el_type == "postit":
            texts = [el.get("text", "") for el in items if el.get("text")]
            if texts:
                summary_parts.append(f"Post-its ({len(texts)}): " + "; ".join(texts[:10]))
        elif el_type in ("rectangle", "oval", "diamond"):
            texts = [el.get("text", "") for el in items if el.get("text")]
            if texts:
                summary_parts.append(f"Shapes ({len(texts)}): " + "; ".join(texts[:10]))
        elif el_type == "fishbone":
            for el in items:
                effect = el.get("effect", "Unknown")
                categories = el.get("categories", [])
                causes = []
                for cat in categories:
                    causes.extend([c.get("text", "") for c in cat.get("causes", [])])
                summary_parts.append(f"Fishbone - Effect: {effect}, Causes: {', '.join(causes[:10])}")
        elif el_type in ("gate-and", "gate-or"):
            summary_parts.append(f"Logic gates: {len(items)} {el_type.replace('gate-', '').upper()} gate(s)")

    return "\n".join(summary_parts) if summary_parts else "Various elements"
