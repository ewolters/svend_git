"""Report API views for CAPA, 8D, and future report types.

Mirrors A3 patterns but with flexible sections driven by REPORT_TYPES registry.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from accounts.permissions import gated_paid
from .models import Report, Board, DSWResult, RCASession
from .report_types import REPORT_TYPES
from core.models import Project, Hypothesis

logger = logging.getLogger(__name__)


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def list_report_types(request):
    """List available report types and their section definitions."""
    return JsonResponse({
        "report_types": {
            key: {
                "name": rt["name"],
                "description": rt["description"],
                "sections": rt["sections"],
                "layout": rt.get("layout", "single_column"),
            }
            for key, rt in REPORT_TYPES.items()
        },
    })


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def list_reports(request):
    """List user's reports.

    Query params:
    - project_id: filter by project
    - report_type: filter by type (capa, 8d)
    - status: filter by status
    """
    reports = Report.objects.filter(owner=request.user).select_related('project')

    project_id = request.GET.get("project_id")
    if project_id:
        reports = reports.filter(project_id=project_id)

    report_type = request.GET.get("report_type")
    if report_type:
        reports = reports.filter(report_type=report_type)

    status = request.GET.get("status")
    if status:
        reports = reports.filter(status=status)

    return JsonResponse({
        "reports": [r.to_dict() for r in reports[:50]],
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def create_report(request):
    """Create a new report.

    Request body:
    {
        "project_id": "uuid",
        "report_type": "capa" | "8d",
        "title": "Optional title"
    }
    """
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    project_id = data.get("project_id")
    report_type = data.get("report_type")

    if not project_id:
        return JsonResponse({"error": "project_id required"}, status=400)
    if not report_type or report_type not in REPORT_TYPES:
        return JsonResponse({
            "error": f"report_type must be one of: {', '.join(REPORT_TYPES.keys())}",
        }, status=400)

    try:
        project = Project.objects.get(id=project_id, user=request.user)
    except Project.DoesNotExist:
        return JsonResponse({"error": "Study not found"}, status=404)

    type_def = REPORT_TYPES[report_type]
    title = data.get("title", f"{type_def['name']} — {project.title}")

    # Initialize empty sections from the type definition
    sections = {s["key"]: "" for s in type_def["sections"]}

    report = Report.objects.create(
        owner=request.user,
        project=project,
        report_type=report_type,
        title=title,
        sections=sections,
    )

    return JsonResponse({
        "id": str(report.id),
        "report": report.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def get_report(request, report_id):
    """Get a single report with full details and import suggestions."""
    report = get_object_or_404(Report, id=report_id, owner=request.user)

    project = report.project
    hypotheses = list(Hypothesis.objects.filter(project=project).values(
        'id', 'statement', 'current_probability', 'status'
    )[:20])
    boards = list(Board.objects.filter(project=project).values(
        'id', 'name', 'room_code'
    )[:10])
    dsw_results = DSWResult.objects.filter(
        project=project
    ).order_by('-created_at')[:20]
    rca_sessions = RCASession.objects.filter(
        project=project
    ).order_by('-updated_at')[:10]

    # Get the type definition for the frontend
    type_def = REPORT_TYPES.get(report.report_type, {})

    return JsonResponse({
        "report": report.to_dict(),
        "type_definition": type_def,
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
            "rca_sessions": [
                {"id": str(r.id), "title": r.title, "event": r.event[:200],
                 "root_cause": r.root_cause[:200] if r.root_cause else "",
                 "status": r.status}
                for r in rca_sessions
            ],
        },
    })


@csrf_exempt
@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_report(request, report_id):
    """Update a report's title, status, or section content."""
    report = get_object_or_404(Report, id=report_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "title" in data:
        report.title = data["title"]
    if "status" in data:
        report.status = data["status"]

    # Update individual sections
    if "sections" in data and isinstance(data["sections"], dict):
        sections = report.sections or {}
        for key, content in data["sections"].items():
            sections[key] = content
        report.sections = sections

    if "imported_from" in data:
        report.imported_from = data["imported_from"]

    report.save()

    return JsonResponse({
        "success": True,
        "report": report.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def delete_report(request, report_id):
    """Delete a report."""
    report = get_object_or_404(Report, id=report_id, owner=request.user)
    report.delete()
    return JsonResponse({"success": True})


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def import_to_report(request, report_id):
    """Import content from other tools into a report section.

    Request body:
    {
        "section": "root_cause_analysis",
        "source_type": "hypothesis" | "whiteboard" | "dsw" | "project" | "rca",
        "source_id": "uuid",
        "append": true
    }
    """
    report = get_object_or_404(Report, id=report_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    section = data.get("section")
    source_type = data.get("source_type")
    source_id = data.get("source_id")
    append = data.get("append", True)

    # Validate section exists in this report type
    type_def = REPORT_TYPES.get(report.report_type, {})
    valid_keys = [s["key"] for s in type_def.get("sections", [])]
    if section not in valid_keys:
        return JsonResponse({"error": f"Invalid section for {report.report_type}"}, status=400)

    content = ""
    import_ref = {"source": source_type, "id": str(source_id) if source_id else ""}

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
            elements = board.elements or []
            content = f"**From Whiteboard:** {board.name}\n\n"
            for el in elements[:20]:
                text = el.get("text") or el.get("title") or el.get("effect")
                if text:
                    el_type = el.get("type", "item")
                    content += f"- [{el_type}] {text}\n"
            causal_conns = [c for c in (board.connections or []) if c.get("type") == "causal"]
            if causal_conns:
                content += "\n**Causal Relationships:**\n"
                for conn in causal_conns[:10]:
                    from_el = next((e for e in elements if e.get("id") == conn["from"]["elementId"]), None)
                    to_el = next((e for e in elements if e.get("id") == conn["to"]["elementId"]), None)
                    if from_el and to_el:
                        from_text = from_el.get("text") or from_el.get("title") or "?"
                        to_text = to_el.get("text") or to_el.get("title") or "?"
                        content += f"- If {from_text} -> Then {to_text}\n"
            import_ref["summary"] = f"{board.name} ({len(elements)} elements)"
        except Board.DoesNotExist:
            return JsonResponse({"error": "Whiteboard not found"}, status=404)

    elif source_type == "project":
        content = f"**Study:** {report.project.title}\n\n{report.project.description or ''}"
        import_ref["summary"] = report.project.title

    elif source_type == "dsw":
        try:
            dsw_result = DSWResult.objects.get(id=source_id, user=request.user)
            import json as json_module
            result_data = json_module.loads(dsw_result.data)
            content = f"**DSW Analysis:** {dsw_result.title}\n\n"
            if result_data.get("analysis_id"):
                content += f"**Type:** {result_data['analysis_id'].replace('_', ' ').title()}\n"
            summary = result_data.get("summary", "") or result_data.get("guide_observation", "")
            if summary:
                import re
                clean_summary = re.sub(r"<<COLOR:\w+>>|<</COLOR>>", "", summary)
                content += f"\n{clean_summary}\n"
            import_ref["summary"] = dsw_result.title or "DSW Analysis"
        except DSWResult.DoesNotExist:
            return JsonResponse({"error": "DSW result not found"}, status=404)

    elif source_type == "rca":
        try:
            rca = RCASession.objects.get(id=source_id, owner=request.user)
            content = f"**Root Cause Analysis:** {rca.title or 'RCA Session'}\n\n"
            content += f"**Event:** {rca.event}\n\n"
            if rca.chain:
                content += "**Causal Chain:**\n"
                for i, step in enumerate(rca.chain):
                    claim = step.get("claim", "")
                    content += f"{i+1}. {claim}\n"
            if rca.root_cause:
                content += f"\n**Root Cause:** {rca.root_cause}\n"
            if rca.countermeasure:
                content += f"**Countermeasure:** {rca.countermeasure}\n"
            import_ref["summary"] = rca.title or rca.event[:100]
        except RCASession.DoesNotExist:
            return JsonResponse({"error": "RCA session not found"}, status=404)

    else:
        return JsonResponse({"error": f"Unknown source_type: {source_type}"}, status=400)

    # Update section content
    sections = report.sections or {}
    current = sections.get(section, "")
    if append and current:
        sections[section] = current + "\n\n---\n\n" + content
    else:
        sections[section] = content
    report.sections = sections

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
        "content": sections[section],
        "report": report.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def auto_populate_report(request, report_id):
    """Auto-populate empty report sections using LLM.

    Request body:
    {
        "sections": ["problem_description", "root_cause_analysis", ...]
    }
    """
    report = get_object_or_404(Report, id=report_id, owner=request.user)

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    type_def = REPORT_TYPES.get(report.report_type, {})
    section_defs = {s["key"]: s for s in type_def.get("sections", [])}

    # Default: populate all empty sections
    requested = data.get("sections")
    if not requested:
        sections = report.sections or {}
        requested = [key for key in section_defs if not sections.get(key, "").strip()]

    # Gather project context
    project = report.project
    hypotheses = list(Hypothesis.objects.filter(project=project)[:10])
    boards = list(Board.objects.filter(project=project)[:5])

    context_parts = [f"Study: {project.title}"]
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

    from .llm_manager import LLMManager

    results = {}
    sections = report.sections or {}

    for section_key in requested:
        sec_def = section_defs.get(section_key)
        if not sec_def:
            continue

        prompt = f"""Based on the project context below, write the "{sec_def['label']}" section of a {type_def['name']}.

{sec_def.get('help', '')}

Project Context:
{context}

Report Title: {report.title}

Write a concise but thorough response (3-5 sentences) suitable for this report section."""

        response = LLMManager.chat(
            user=request.user,
            messages=[{"role": "user", "content": prompt}],
            system=f"You are helping create a {type_def['name']}. Be concise, professional, and actionable.",
            max_tokens=600,
            temperature=0.7,
        )

        if response and not response.get("rate_limited"):
            content = response.get("content", "")
            sections[section_key] = content
            results[section_key] = content
        elif response and response.get("rate_limited"):
            report.sections = sections
            report.save()
            return JsonResponse({
                "error": response["error"],
                "rate_limited": True,
                "partial_results": results,
            }, status=429)

    report.sections = sections
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
    """Embed a whiteboard diagram (SVG) into a report section.

    Request body:
    {
        "section": "root_cause_analysis",
        "room_code": "ABC123"
    }
    """
    report = get_object_or_404(Report, id=report_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    section = data.get("section")
    room_code = data.get("room_code")

    type_def = REPORT_TYPES.get(report.report_type, {})
    valid_keys = [s["key"] for s in type_def.get("sections", [])]
    if section not in valid_keys:
        return JsonResponse({"error": "Invalid section"}, status=400)

    if not room_code:
        return JsonResponse({"error": "room_code required"}, status=400)

    try:
        board = Board.objects.get(room_code=room_code.upper())
    except Board.DoesNotExist:
        return JsonResponse({"error": "Whiteboard not found"}, status=404)

    from .whiteboard_views import _render_element_svg, _render_connection_svg

    elements = board.elements or []
    connections = board.connections or []

    if not elements:
        return JsonResponse({"error": "Whiteboard is empty"}, status=400)

    min_x = min(el.get('x', 0) for el in elements)
    min_y = min(el.get('y', 0) for el in elements)
    max_x = max(el.get('x', 0) + el.get('width', 120) for el in elements)
    max_y = max(el.get('y', 0) + el.get('height', 60) for el in elements)

    padding = 20
    width = max_x - min_x + padding * 2
    height = max_y - min_y + padding * 2
    offset_x = min_x - padding
    offset_y = min_y - padding

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

    diagrams = report.embedded_diagrams or {}
    if section not in diagrams:
        diagrams[section] = []

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
    """Remove an embedded diagram from a report."""
    report = get_object_or_404(Report, id=report_id, owner=request.user)

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
