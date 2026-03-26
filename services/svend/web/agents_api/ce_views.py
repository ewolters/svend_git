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

from .models import CEMatrix

logger = logging.getLogger(__name__)


def _ce_connect_investigation(request, investigation_id, matrix):
    """CANON-002 §12 — connect C&E top inputs to investigation graph."""
    from core.models import MeasurementSystem

    from .investigation_bridge import HypothesisSpec, connect_tool

    try:
        tool_output, _ = MeasurementSystem.objects.get_or_create(
            name="C&E Matrix",
            owner=request.user,
            defaults={"system_type": "variable"},
        )
        specs = []
        totals = matrix.compute_totals()
        for entry in totals[:3]:
            name = entry.get("input_name", "").strip()
            total = entry.get("total", 0)
            if name and total > 0:
                specs.append(
                    HypothesisSpec(
                        description=f"C&E Matrix top input: {name} (score: {total:.0f})",
                        prior=0.5,
                    )
                )
        if specs:
            connect_tool(
                investigation_id=investigation_id,
                tool_output=tool_output,
                tool_type="ce_matrix",
                user=request.user,
                spec=specs,
            )
    except Exception:
        logger.exception(
            "C&E Matrix investigation bridge error for matrix %s", matrix.id
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

    # CANON-002 §12 — investigation bridge
    investigation_id = data.get("investigation_id")
    if investigation_id and matrix.status == "complete":
        _ce_connect_investigation(request, investigation_id, matrix)

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
