"""Ishikawa (Fishbone) diagram API views.

CRUD for common cause analysis diagrams with 6M categories.
Follows rca_views.py pattern: session CRUD, evidence hooks, auto-project.
"""

import copy
import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from core.models import Project

from .evidence_bridge import create_tool_evidence
from .models import DEFAULT_6M_BRANCHES, IshikawaDiagram

logger = logging.getLogger(__name__)


def _ensure_ishikawa_project(diagram, user):
    """Auto-create a Study for a standalone Ishikawa diagram.

    Same pattern as _ensure_rca_project() — invisible, tagged, no notification.
    """
    if diagram.project_id:
        return diagram.project

    project = Project.objects.create(
        user=user,
        title=diagram.title or "Ishikawa Analysis",
        project_class="investigation",
        methodology="none",
        tags=["auto-created", "ishikawa"],
    )
    diagram.project = project
    diagram.save(update_fields=["project"])
    logger.info("Auto-created project %s for Ishikawa %s", project.id, diagram.id)
    return project


def _ishikawa_evidence_hooks(diagram, user):
    """Create evidence when an Ishikawa diagram is completed.

    Pushes top-level causes per category as analysis evidence.
    Only fires when status is 'complete'. Idempotent via evidence_bridge.
    """
    if not diagram.project:
        return
    if diagram.status != "complete":
        return

    for branch in diagram.branches:
        category = branch.get("category", "")
        for cause in branch.get("causes", []):
            text = cause.get("text", "").strip()
            if not text:
                continue
            create_tool_evidence(
                project=diagram.project,
                user=user,
                summary=f"Ishikawa [{category}]: {text[:200]}",
                source_tool="ishikawa",
                source_id=str(diagram.id),
                source_field=f"branch_{category}_{text[:50]}",
                source_type="analysis",
            )


def _ishikawa_connect_investigation(request, investigation_id, diagram):
    """CANON-002 §12 — connect Ishikawa causes to investigation graph."""
    from core.models import MeasurementSystem

    from .investigation_bridge import HypothesisSpec, connect_tool

    try:
        tool_output, _ = MeasurementSystem.objects.get_or_create(
            name="Ishikawa Diagram",
            owner=request.user,
            defaults={"system_type": "variable"},
        )
        specs = []
        for branch in diagram.branches:
            category = branch.get("category", "")
            for cause in branch.get("causes", []):
                text = cause.get("text", "").strip()
                if text:
                    specs.append(
                        HypothesisSpec(
                            description=f"Ishikawa [{category}]: {text[:300]}",
                            prior=0.5,
                        )
                    )
        if specs:
            connect_tool(
                investigation_id=investigation_id,
                tool_output=tool_output,
                tool_type="ishikawa",
                user=request.user,
                spec=specs,
            )
    except Exception:
        logger.exception("Ishikawa investigation bridge error for diagram %s", diagram.id)


# --- CRUD Endpoints ---


@gated_paid
@require_http_methods(["GET"])
def list_diagrams(request):
    """List user's Ishikawa diagrams."""
    diagrams = IshikawaDiagram.objects.filter(owner=request.user).order_by("-updated_at")[:50]
    return JsonResponse({"diagrams": [d.to_dict() for d in diagrams]})


@gated_paid
@require_http_methods(["POST"])
def create_diagram(request):
    """Create a new Ishikawa diagram with 6M branches initialized."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    effect = data.get("effect", "").strip()
    if not effect:
        return JsonResponse({"error": "Effect description required"}, status=400)

    # Deep copy default branches to avoid sharing reference
    branches = copy.deepcopy(DEFAULT_6M_BRANCHES)

    diagram = IshikawaDiagram.objects.create(
        owner=request.user,
        title=data.get("title", "").strip(),
        effect=effect,
        branches=branches,
        status="draft",
    )

    # Link to project if provided, otherwise auto-create
    project_id = data.get("project_id")
    if project_id:
        try:
            project = Project.objects.get(id=project_id, user=request.user)
            diagram.project = project
            diagram.save(update_fields=["project"])
        except Project.DoesNotExist:
            pass

    _ensure_ishikawa_project(diagram, request.user)

    return JsonResponse({"diagram": diagram.to_dict()}, status=201)


@gated_paid
@require_http_methods(["GET"])
def get_diagram(request, diagram_id):
    """Get a single Ishikawa diagram."""
    try:
        diagram = IshikawaDiagram.objects.get(id=diagram_id, owner=request.user)
        return JsonResponse({"diagram": diagram.to_dict()})
    except IshikawaDiagram.DoesNotExist:
        return JsonResponse({"error": "Diagram not found"}, status=404)


@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_diagram(request, diagram_id):
    """Update an Ishikawa diagram."""
    try:
        diagram = IshikawaDiagram.objects.get(id=diagram_id, owner=request.user)
    except IshikawaDiagram.DoesNotExist:
        return JsonResponse({"error": "Diagram not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "title" in data:
        diagram.title = data["title"]
    if "effect" in data:
        diagram.effect = data["effect"]
    if "branches" in data:
        diagram.branches = data["branches"]
    if "status" in data and data["status"] in dict(IshikawaDiagram.Status.choices):
        diagram.status = data["status"]

    diagram.save()

    # Evidence hooks — after save
    _ensure_ishikawa_project(diagram, request.user)
    _ishikawa_evidence_hooks(diagram, request.user)

    # CANON-002 §12 — investigation bridge (dual-write)
    investigation_id = data.get("investigation_id")
    if investigation_id and diagram.status == "complete":
        _ishikawa_connect_investigation(request, investigation_id, diagram)

    return JsonResponse({"diagram": diagram.to_dict()})


@gated_paid
@require_http_methods(["DELETE"])
def delete_diagram(request, diagram_id):
    """Delete an Ishikawa diagram."""
    try:
        diagram = IshikawaDiagram.objects.get(id=diagram_id, owner=request.user)
        diagram.delete()
        return JsonResponse({"success": True})
    except IshikawaDiagram.DoesNotExist:
        return JsonResponse({"error": "Diagram not found"}, status=404)
