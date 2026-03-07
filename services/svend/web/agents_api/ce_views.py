"""Cause & Effect (C&E) Matrix API views.

CRUD for C&E scoring matrices with weighted inputs/outputs.
Follows rca_views.py pattern: session CRUD, evidence hooks, auto-project.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from core.models import Project

from .evidence_bridge import create_tool_evidence
from .models import CEMatrix

logger = logging.getLogger(__name__)


def _ensure_ce_project(matrix, user):
    """Auto-create a Study for a standalone C&E matrix.

    Same pattern as _ensure_rca_project() — invisible, tagged, no notification.
    """
    if matrix.project_id:
        return matrix.project

    project = Project.objects.create(
        user=user,
        title=matrix.title or "C&E Matrix Analysis",
        project_class="investigation",
        methodology="none",
        tags=["auto-created", "ce-matrix"],
    )
    matrix.project = project
    matrix.save(update_fields=["project"])
    logger.info("Auto-created project %s for C&E Matrix %s", project.id, matrix.id)
    return project


def _ce_evidence_hooks(matrix, user):
    """Create evidence when a C&E matrix is completed.

    Pushes top 3 highest-scored inputs as analysis evidence.
    Only fires when status is 'complete'. Idempotent via evidence_bridge.
    """
    if not matrix.project:
        return
    if matrix.status != "complete":
        return

    totals = matrix.compute_totals()
    for entry in totals[:3]:
        name = entry.get("input_name", "").strip()
        total = entry.get("total", 0)
        if not name or total == 0:
            continue
        create_tool_evidence(
            project=matrix.project,
            user=user,
            summary=f"C&E Matrix top input: {name} (score: {total:.0f})",
            source_tool="ce_matrix",
            source_id=str(matrix.id),
            source_field=f"input_{entry.get('input_index', 0)}",
            source_type="analysis",
        )


# --- CRUD Endpoints ---


@gated_paid
@require_http_methods(["GET"])
def list_matrices(request):
    """List user's C&E matrices."""
    matrices = CEMatrix.objects.filter(owner=request.user).order_by("-updated_at")[:50]
    return JsonResponse({"matrices": [m.to_dict() for m in matrices]})


@gated_paid
@require_http_methods(["POST"])
def create_matrix(request):
    """Create a new C&E matrix."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = data.get("title", "").strip()

    matrix = CEMatrix.objects.create(
        owner=request.user,
        title=title,
        outputs=data.get("outputs", []),
        inputs=data.get("inputs", []),
        scores=data.get("scores", {}),
        status="draft",
    )

    # Link to project if provided, otherwise auto-create
    project_id = data.get("project_id")
    if project_id:
        try:
            project = Project.objects.get(id=project_id, user=request.user)
            matrix.project = project
            matrix.save(update_fields=["project"])
        except Project.DoesNotExist:
            pass

    _ensure_ce_project(matrix, request.user)

    return JsonResponse({"matrix": matrix.to_dict()}, status=201)


@gated_paid
@require_http_methods(["GET"])
def get_matrix(request, matrix_id):
    """Get a single C&E matrix with computed totals."""
    try:
        matrix = CEMatrix.objects.get(id=matrix_id, owner=request.user)
        return JsonResponse({"matrix": matrix.to_dict()})
    except CEMatrix.DoesNotExist:
        return JsonResponse({"error": "Matrix not found"}, status=404)


@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_matrix(request, matrix_id):
    """Update a C&E matrix."""
    try:
        matrix = CEMatrix.objects.get(id=matrix_id, owner=request.user)
    except CEMatrix.DoesNotExist:
        return JsonResponse({"error": "Matrix not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "title" in data:
        matrix.title = data["title"]
    if "outputs" in data:
        matrix.outputs = data["outputs"]
    if "inputs" in data:
        matrix.inputs = data["inputs"]
    if "scores" in data:
        matrix.scores = data["scores"]
    if "status" in data and data["status"] in dict(CEMatrix.Status.choices):
        matrix.status = data["status"]

    matrix.save()

    # Evidence hooks — after save
    _ensure_ce_project(matrix, request.user)
    _ce_evidence_hooks(matrix, request.user)

    return JsonResponse({"matrix": matrix.to_dict()})


@gated_paid
@require_http_methods(["DELETE"])
def delete_matrix(request, matrix_id):
    """Delete a C&E matrix."""
    try:
        matrix = CEMatrix.objects.get(id=matrix_id, owner=request.user)
        matrix.delete()
        return JsonResponse({"success": True})
    except CEMatrix.DoesNotExist:
        return JsonResponse({"error": "Matrix not found"}, status=404)
