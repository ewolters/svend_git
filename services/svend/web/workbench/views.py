"""API views for Workbench."""

import json
import re

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from .models import Artifact, Workbench


@login_required
@require_http_methods(["GET"])
def list_workbenches(request):
    """List all workbenches for the current user."""
    workbenches = Workbench.objects.filter(user=request.user)

    # Filter by status if provided
    status = request.GET.get("status")
    if status:
        workbenches = workbenches.filter(status=status)

    # Filter by template if provided
    template = request.GET.get("template")
    if template:
        workbenches = workbenches.filter(template=template)

    return JsonResponse(
        {
            "workbenches": [
                {
                    "id": str(w.id),
                    "title": w.title,
                    "template": w.template,
                    "status": w.status,
                    "artifact_count": w.artifacts.count(),
                    "updated_at": w.updated_at.isoformat(),
                    "created_at": w.created_at.isoformat(),
                }
                for w in workbenches
            ]
        }
    )


@login_required
@require_http_methods(["POST"])
def create_workbench(request):
    """Create a new workbench."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "Title is required"}, status=400)

    template = data.get("template", Workbench.Template.BLANK)
    if template not in [t.value for t in Workbench.Template]:
        return JsonResponse({"error": "Invalid template"}, status=400)

    workbench = Workbench.objects.create(
        user=request.user,
        title=title,
        description=data.get("description", ""),
        template=template,
    )

    # Initialize template-specific state
    if template != Workbench.Template.BLANK:
        workbench.init_template()

    return JsonResponse(
        {
            "success": True,
            "workbench": {
                "id": str(workbench.id),
                "title": workbench.title,
                "template": workbench.template,
                "template_state": workbench.template_state,
            },
        },
        status=201,
    )


@login_required
@require_http_methods(["GET"])
def get_workbench(request, workbench_id):
    """Get a workbench with all its artifacts (full JSON for agent context)."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)
    return JsonResponse(workbench.to_json())


@login_required
@require_http_methods(["PATCH"])
def update_workbench(request, workbench_id):
    """Update workbench metadata."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Update allowed fields
    if "title" in data:
        workbench.title = data["title"]
    if "description" in data:
        workbench.description = data["description"]
    if "status" in data:
        workbench.status = data["status"]
    if "conclusion" in data:
        workbench.conclusion = data["conclusion"]
    if "conclusion_confidence" in data:
        workbench.conclusion_confidence = data["conclusion_confidence"]
    if "layout" in data:
        workbench.layout = data["layout"]
    if "datasets" in data:
        workbench.datasets = data["datasets"]
    if "guide_observations" in data:
        workbench.guide_observations = data["guide_observations"]

    workbench.save()

    return JsonResponse({"success": True})


@login_required
@require_http_methods(["DELETE"])
def delete_workbench(request, workbench_id):
    """Delete a workbench (moves to archived status by default)."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    # Check if permanent delete requested
    permanent = request.GET.get("permanent", "false").lower() == "true"

    if permanent:
        workbench.delete()
    else:
        workbench.status = Workbench.Status.ARCHIVED
        workbench.save(update_fields=["status"])

    return JsonResponse({"success": True})


@login_required
@require_http_methods(["POST"])
def export_workbench(request, workbench_id):
    """Export workbench as JSON file."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    response = JsonResponse(workbench.to_json())
    safe_title = re.sub(r'[\x00-\x1f\x7f"\\/:*?<>|]', "_", workbench.title) or "workbench"
    response["Content-Disposition"] = f'attachment; filename="{safe_title}.json"'
    return response


@login_required
@require_http_methods(["POST"])
def import_workbench(request):
    """Import a workbench from JSON."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    workbench = Workbench.from_json(data, request.user)

    return JsonResponse(
        {
            "success": True,
            "workbench_id": str(workbench.id),
        },
        status=201,
    )


# =============================================================================
# Artifact endpoints
# =============================================================================


@login_required
@require_http_methods(["POST"])
def create_artifact(request, workbench_id):
    """Create an artifact in a workbench."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    artifact_type = data.get("type")
    if not artifact_type:
        return JsonResponse({"error": "Artifact type is required"}, status=400)

    artifact = workbench.add_artifact(
        artifact_type=artifact_type,
        content=data.get("content", {}),
        title=data.get("title", ""),
        source=data.get("source", "user"),
        probability=data.get("probability"),
        tags=data.get("tags", []),
    )

    # Handle layout position if provided
    if "position" in data:
        workbench.layout[str(artifact.id)] = data["position"]
        workbench.save(update_fields=["layout"])

    return JsonResponse(
        {
            "success": True,
            "artifact": artifact.to_json(),
        },
        status=201,
    )


@login_required
@require_http_methods(["GET"])
def get_artifact(request, workbench_id, artifact_id):
    """Get a single artifact."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)
    artifact = get_object_or_404(Artifact, id=artifact_id, workbench=workbench)

    return JsonResponse(artifact.to_json())


@login_required
@require_http_methods(["PATCH"])
def update_artifact(request, workbench_id, artifact_id):
    """Update an artifact."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)
    artifact = get_object_or_404(Artifact, id=artifact_id, workbench=workbench)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Update allowed fields
    if "title" in data:
        artifact.title = data["title"]
    if "content" in data:
        artifact.content = data["content"]
    if "probability" in data:
        artifact.probability = data["probability"]
    if "tags" in data:
        artifact.tags = data["tags"]
    if "supports" in data:
        artifact.supports_hypotheses = data["supports"]
    if "weakens" in data:
        artifact.weakens_hypotheses = data["weakens"]

    artifact.save()

    # Update layout if position provided
    if "position" in data:
        workbench.layout[str(artifact.id)] = data["position"]
        workbench.save(update_fields=["layout"])

    return JsonResponse({"success": True, "artifact": artifact.to_json()})


@login_required
@require_http_methods(["DELETE"])
def delete_artifact(request, workbench_id, artifact_id):
    """Delete an artifact."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)
    artifact = get_object_or_404(Artifact, id=artifact_id, workbench=workbench)

    # Remove from layout
    if str(artifact.id) in workbench.layout:
        del workbench.layout[str(artifact.id)]
        workbench.save(update_fields=["layout"])

    # Remove from connections
    workbench.connections = [
        c for c in workbench.connections if c["from"] != str(artifact.id) and c["to"] != str(artifact.id)
    ]
    workbench.save(update_fields=["connections"])

    artifact.delete()

    return JsonResponse({"success": True})


# =============================================================================
# Connection endpoints
# =============================================================================


@login_required
@require_http_methods(["POST"])
def connect_artifacts(request, workbench_id):
    """Connect two artifacts."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    from_id = data.get("from")
    to_id = data.get("to")
    label = data.get("label", "")

    if not from_id or not to_id:
        return JsonResponse({"error": "from and to are required"}, status=400)

    workbench.connect_artifacts(from_id, to_id, label)

    return JsonResponse({"success": True})


@login_required
@require_http_methods(["DELETE"])
def disconnect_artifacts(request, workbench_id):
    """Remove a connection between artifacts."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    from_id = data.get("from")
    to_id = data.get("to")

    workbench.connections = [c for c in workbench.connections if not (c["from"] == from_id and c["to"] == to_id)]
    workbench.save(update_fields=["connections"])

    return JsonResponse({"success": True})


# =============================================================================
# Template-specific endpoints
# =============================================================================


@login_required
@require_http_methods(["POST"])
def advance_phase(request, workbench_id):
    """Advance DMAIC phase."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    if workbench.template != Workbench.Template.DMAIC:
        return JsonResponse({"error": "Not a DMAIC workbench"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = {}

    notes = data.get("notes", "")
    workbench.advance_dmaic_phase(notes)

    return JsonResponse(
        {
            "success": True,
            "current_phase": workbench.template_state.get("current_phase"),
        }
    )


# =============================================================================
# Guide endpoints
# =============================================================================


@login_required
@require_http_methods(["POST"])
def add_guide_observation(request, workbench_id):
    """Add an AI Guide observation (called by agents)."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    observation = data.get("observation", "")
    suggestion = data.get("suggestion", "")

    if not observation:
        return JsonResponse({"error": "Observation is required"}, status=400)

    workbench.add_guide_observation(observation, suggestion)

    return JsonResponse({"success": True})


@login_required
@require_http_methods(["POST"])
def acknowledge_observation(request, workbench_id, observation_index):
    """Mark a guide observation as acknowledged."""
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)

    try:
        idx = int(observation_index)
        if 0 <= idx < len(workbench.guide_observations):
            workbench.guide_observations[idx]["acknowledged"] = True
            workbench.save(update_fields=["guide_observations"])
            return JsonResponse({"success": True})
    except (ValueError, IndexError):
        pass

    return JsonResponse({"error": "Invalid observation index"}, status=400)
