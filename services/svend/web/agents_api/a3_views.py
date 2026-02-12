"""A3 Report API views.

A3 is a Toyota-style single-page problem-solving format.
This module provides CRUD operations and import functionality.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from accounts.permissions import gated_paid
from .models import A3Report, Board, DSWResult
from core.models import Project, Hypothesis

logger = logging.getLogger(__name__)


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def list_a3_reports(request):
    """List user's A3 reports.

    Query params:
    - project_id: filter by project
    - status: filter by status
    """
    reports = A3Report.objects.filter(owner=request.user).select_related('project')

    project_id = request.GET.get("project_id")
    if project_id:
        reports = reports.filter(project_id=project_id)

    status = request.GET.get("status")
    if status:
        reports = reports.filter(status=status)

    return JsonResponse({
        "reports": [r.to_dict() for r in reports[:50]],
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def create_a3_report(request):
    """Create a new A3 report."""
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    project_id = data.get("project_id")
    if not project_id:
        return JsonResponse({"error": "project_id required"}, status=400)

    try:
        project = Project.objects.get(id=project_id, user=request.user)
    except Project.DoesNotExist:
        return JsonResponse({"error": "Project not found"}, status=404)

    title = data.get("title", "Untitled A3")

    report = A3Report.objects.create(
        owner=request.user,
        project=project,
        title=title,
        background=data.get("background", ""),
        current_condition=data.get("current_condition", ""),
        goal=data.get("goal", ""),
        root_cause=data.get("root_cause", ""),
        countermeasures=data.get("countermeasures", ""),
        implementation_plan=data.get("implementation_plan", ""),
        follow_up=data.get("follow_up", ""),
    )

    return JsonResponse({
        "id": str(report.id),
        "report": report.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def get_a3_report(request, report_id):
    """Get a single A3 report with full details."""
    report = get_object_or_404(A3Report, id=report_id, owner=request.user)

    # Also get related data for import suggestions
    project = report.project
    hypotheses = list(Hypothesis.objects.filter(project=project).values(
        'id', 'statement', 'current_probability', 'status'
    )[:20])
    boards = list(Board.objects.filter(project=project).values(
        'id', 'name', 'room_code'
    )[:10])

    # Get DSW results linked to this project
    dsw_results = DSWResult.objects.filter(
        project=project
    ).order_by('-created_at')[:20]

    return JsonResponse({
        "report": report.to_dict(),
        "project": {
            "id": str(project.id),
            "title": project.title,
            "description": project.description,
        },
        "available_imports": {
            "hypotheses": [
                {"id": str(h["id"]), "statement": h["statement"],
                 "probability": h["current_probability"], "status": h["status"]}
                for h in hypotheses
            ],
            "boards": [
                {"id": str(b["id"]), "name": b["name"], "room_code": b["room_code"]}
                for b in boards
            ],
            "dsw_results": [
                {"id": r.id, "title": r.title, "type": r.result_type,
                 "summary": r.get_summary(), "created": r.created_at.isoformat()}
                for r in dsw_results
            ],
        },
    })


@csrf_exempt
@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_a3_report(request, report_id):
    """Update an A3 report."""
    report = get_object_or_404(A3Report, id=report_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Update fields
    for field in ['title', 'status', 'background', 'current_condition', 'goal',
                  'root_cause', 'countermeasures', 'implementation_plan', 'follow_up']:
        if field in data:
            setattr(report, field, data[field])

    if 'imported_from' in data:
        report.imported_from = data['imported_from']

    report.save()

    return JsonResponse({
        "success": True,
        "report": report.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def delete_a3_report(request, report_id):
    """Delete an A3 report."""
    report = get_object_or_404(A3Report, id=report_id, owner=request.user)
    report.delete()

    return JsonResponse({"success": True})


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def import_to_a3(request, report_id):
    """Import content from other tools into an A3 section.

    Request body:
    {
        "section": "root_cause" | "current_condition" | "countermeasures" | ...,
        "source_type": "hypothesis" | "whiteboard" | "dsw",
        "source_id": "uuid",
        "append": true  // false to replace
    }
    """
    report = get_object_or_404(A3Report, id=report_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    section = data.get("section")
    source_type = data.get("source_type")
    source_id = data.get("source_id")
    append = data.get("append", True)

    valid_sections = ['background', 'current_condition', 'goal', 'root_cause',
                      'countermeasures', 'implementation_plan', 'follow_up']
    if section not in valid_sections:
        return JsonResponse({"error": f"Invalid section. Must be one of: {valid_sections}"}, status=400)

    # Get content based on source type
    content = ""
    import_ref = {"source": source_type, "id": source_id}

    if source_type == "hypothesis":
        try:
            hypothesis = Hypothesis.objects.get(id=source_id, project=report.project)
            content = f"**Hypothesis:** {hypothesis.statement}\n"
            if hypothesis.mechanism:
                content += f"**Mechanism:** {hypothesis.mechanism}\n"
            content += f"**Probability:** {hypothesis.current_probability:.0%} ({hypothesis.status})\n"
            import_ref["summary"] = hypothesis.statement[:100]
        except Hypothesis.DoesNotExist:
            return JsonResponse({"error": "Hypothesis not found"}, status=404)

    elif source_type == "whiteboard":
        try:
            board = Board.objects.get(id=source_id, project=report.project)
            # Summarize whiteboard content
            elements = board.elements or []
            content = f"**From Whiteboard:** {board.name}\n\n"

            # Extract text from elements
            for el in elements[:20]:
                text = el.get("text") or el.get("title") or el.get("effect")
                if text:
                    el_type = el.get("type", "item")
                    content += f"- [{el_type}] {text}\n"

            # Extract causal connections
            causal_conns = [c for c in (board.connections or []) if c.get("type") == "causal"]
            if causal_conns:
                content += "\n**Causal Relationships:**\n"
                for conn in causal_conns[:10]:
                    from_el = next((e for e in elements if e.get("id") == conn["from"]["elementId"]), None)
                    to_el = next((e for e in elements if e.get("id") == conn["to"]["elementId"]), None)
                    if from_el and to_el:
                        from_text = from_el.get("text") or from_el.get("title") or "?"
                        to_text = to_el.get("text") or to_el.get("title") or "?"
                        content += f"- If {from_text} → Then {to_text}\n"

            import_ref["summary"] = f"{board.name} ({len(elements)} elements)"
        except Board.DoesNotExist:
            return JsonResponse({"error": "Whiteboard not found"}, status=404)

    elif source_type == "project":
        # Import project description
        content = f"**Project:** {report.project.title}\n\n{report.project.description or ''}"
        import_ref["summary"] = report.project.title

    elif source_type == "dsw":
        # Import DSW analysis result
        try:
            dsw_result = DSWResult.objects.get(id=source_id, user=request.user)
            import json as json_module
            data = json_module.loads(dsw_result.data)

            content = f"**DSW Analysis:** {dsw_result.title}\n\n"

            # Add analysis type/ID
            if data.get("analysis_id"):
                content += f"**Type:** {data['analysis_id'].replace('_', ' ').title()}\n"

            # Add summary
            summary = data.get("summary", "") or data.get("guide_observation", "")
            if summary:
                # Strip color tags
                import re
                clean_summary = re.sub(r"<<COLOR:\w+>>|<</COLOR>>", "", summary)
                content += f"\n{clean_summary}\n"

            # Note about plots
            plots_count = data.get("plots_count", 0)
            if plots_count:
                content += f"\n*{plots_count} visualization(s) available in DSW*\n"

            import_ref["summary"] = dsw_result.title or f"DSW: {data.get('analysis_id', 'Analysis')}"
        except DSWResult.DoesNotExist:
            return JsonResponse({"error": "DSW result not found"}, status=404)

    else:
        return JsonResponse({"error": f"Unknown source_type: {source_type}"}, status=400)

    # Update section content
    current = getattr(report, section)
    if append and current:
        new_content = current + "\n\n---\n\n" + content
    else:
        new_content = content

    setattr(report, section, new_content)

    # Track import reference
    imports = report.imported_from or {}
    if section not in imports:
        imports[section] = []
    imports[section].append(import_ref)
    report.imported_from = imports

    report.save()

    return JsonResponse({
        "success": True,
        "section": section,
        "content": new_content,
        "report": report.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def auto_populate_a3(request, report_id):
    """Auto-populate A3 sections using LLM based on project data.

    Request body:
    {
        "sections": ["background", "root_cause", ...]  // sections to populate
    }
    """
    report = get_object_or_404(A3Report, id=report_id, owner=request.user)

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    sections = data.get("sections", ["background", "current_condition", "root_cause"])

    # Gather project context
    project = report.project
    hypotheses = list(Hypothesis.objects.filter(project=project)[:10])
    boards = list(Board.objects.filter(project=project)[:5])

    context_parts = [f"Project: {project.title}"]
    if project.description:
        context_parts.append(f"Description: {project.description}")

    if hypotheses:
        context_parts.append("\nHypotheses:")
        for h in hypotheses:
            context_parts.append(f"- [{h.status}] {h.statement} (P={h.current_probability:.0%})")

    for board in boards:
        if board.elements:
            context_parts.append(f"\nWhiteboard '{board.name}':")
            for el in board.elements[:10]:
                text = el.get("text") or el.get("title")
                if text:
                    context_parts.append(f"- {text}")

    context = "\n".join(context_parts)

    # Use LLM to generate content
    from .llm_manager import LLMManager

    section_prompts = {
        "background": "Write a brief Background section explaining why this problem matters. Include business impact.",
        "current_condition": "Describe the Current Condition based on the data. What is happening now?",
        "goal": "Define a clear Goal/Target Condition. What does success look like?",
        "root_cause": "Summarize the Root Cause Analysis. What are the main causes based on the hypotheses?",
        "countermeasures": "Suggest Countermeasures to address the root causes.",
        "implementation_plan": "Outline an Implementation Plan with key actions.",
        "follow_up": "Describe Follow-up steps to verify the countermeasures worked.",
    }

    results = {}
    for section in sections:
        if section not in section_prompts:
            continue

        prompt = f"""Based on this project context, {section_prompts[section]}

Project Context:
{context}

A3 Title: {report.title}

Write a concise response (2-4 sentences) suitable for an A3 report section."""

        response = LLMManager.chat(
            user=request.user,
            messages=[{"role": "user", "content": prompt}],
            system="You are helping create an A3 problem-solving report. Be concise and actionable.",
            max_tokens=500,
            temperature=0.7,
        )

        if response and not response.get("rate_limited"):
            content = response.get("content", "")
            setattr(report, section, content)
            results[section] = content
        elif response and response.get("rate_limited"):
            return JsonResponse({
                "error": response["error"],
                "rate_limited": True,
                "partial_results": results,
            }, status=429)

    report.save()

    return JsonResponse({
        "success": True,
        "populated_sections": list(results.keys()),
        "report": report.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def embed_diagram(request, report_id):
    """Embed a whiteboard diagram (SVG) into an A3 section.

    Request body:
    {
        "section": "root_cause" | "current_condition" | ...,
        "room_code": "ABC123"
    }
    """
    report = get_object_or_404(A3Report, id=report_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    section = data.get("section")
    room_code = data.get("room_code")

    valid_sections = ['background', 'current_condition', 'goal', 'root_cause',
                      'countermeasures', 'implementation_plan', 'follow_up']
    if section not in valid_sections:
        return JsonResponse({"error": f"Invalid section"}, status=400)

    if not room_code:
        return JsonResponse({"error": "room_code required"}, status=400)

    # Get the whiteboard
    try:
        board = Board.objects.get(room_code=room_code.upper())
    except Board.DoesNotExist:
        return JsonResponse({"error": "Whiteboard not found"}, status=404)

    # Import the SVG renderer from whiteboard_views
    from .whiteboard_views import _render_element_svg, _render_connection_svg

    elements = board.elements or []
    connections = board.connections or []

    if not elements:
        return JsonResponse({"error": "Whiteboard is empty"}, status=400)

    # Calculate bounding box
    min_x = min(el.get('x', 0) for el in elements)
    min_y = min(el.get('y', 0) for el in elements)
    max_x = max(el.get('x', 0) + el.get('width', 120) for el in elements)
    max_y = max(el.get('y', 0) + el.get('height', 60) for el in elements)

    padding = 20
    width = max_x - min_x + padding * 2
    height = max_y - min_y + padding * 2
    offset_x = min_x - padding
    offset_y = min_y - padding

    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#1a1a1a">',
        '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#4a9f6e"/></marker></defs>',
    ]

    for conn in connections:
        svg_parts.append(_render_connection_svg(conn, elements, offset_x, offset_y))

    for el in elements:
        svg_parts.append(_render_element_svg(el, offset_x, offset_y))

    svg_parts.append('</svg>')
    svg_content = '\n'.join(svg_parts)

    # Store in embedded_diagrams
    diagrams = report.embedded_diagrams or {}
    if section not in diagrams:
        diagrams[section] = []

    # Generate a unique ID for this diagram
    import uuid
    diagram_id = str(uuid.uuid4())[:8]

    diagrams[section].append({
        "id": diagram_id,
        "svg": svg_content,
        "board_name": board.name,
        "room_code": board.room_code,
        "width": width,
        "height": height,
    })

    report.embedded_diagrams = diagrams
    report.save()

    return JsonResponse({
        "success": True,
        "diagram_id": diagram_id,
        "section": section,
        "board_name": board.name,
        "svg": svg_content,
    })


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def remove_diagram(request, report_id, diagram_id):
    """Remove an embedded diagram from an A3 report."""
    report = get_object_or_404(A3Report, id=report_id, owner=request.user)

    diagrams = report.embedded_diagrams or {}
    removed = False

    for section, section_diagrams in diagrams.items():
        diagrams[section] = [d for d in section_diagrams if d.get("id") != diagram_id]
        if len(diagrams[section]) < len(section_diagrams):
            removed = True

    if not removed:
        return JsonResponse({"error": "Diagram not found"}, status=404)

    report.embedded_diagrams = diagrams
    report.save()

    return JsonResponse({"success": True})
