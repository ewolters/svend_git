"""API views for Workbench."""

import json
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404

from .models import Project, Workbench, Artifact, Hypothesis, Evidence, Conversation


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

    return JsonResponse({
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
    })


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

    return JsonResponse({
        "success": True,
        "workbench": {
            "id": str(workbench.id),
            "title": workbench.title,
            "template": workbench.template,
            "template_state": workbench.template_state,
        }
    }, status=201)


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
    response["Content-Disposition"] = f'attachment; filename="{workbench.title}.json"'
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

    return JsonResponse({
        "success": True,
        "workbench_id": str(workbench.id),
    }, status=201)


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

    return JsonResponse({
        "success": True,
        "artifact": artifact.to_json(),
    }, status=201)


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
        c for c in workbench.connections
        if c["from"] != str(artifact.id) and c["to"] != str(artifact.id)
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

    workbench.connections = [
        c for c in workbench.connections
        if not (c["from"] == from_id and c["to"] == to_id)
    ]
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

    return JsonResponse({
        "success": True,
        "current_phase": workbench.template_state.get("current_phase"),
    })


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


# =============================================================================
# Project CRUD
# =============================================================================

@login_required
@require_http_methods(["GET"])
def list_projects(request):
    """List all projects for the current user."""
    projects = Project.objects.filter(user=request.user)

    # Filter by status if provided
    status = request.GET.get("status")
    if status:
        projects = projects.filter(status=status)

    return JsonResponse({
        "projects": [p.to_dict() for p in projects]
    })


@login_required
@require_http_methods(["POST"])
def create_project(request):
    """Create a new project."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = data.get("title", "").strip()
    hypothesis = data.get("hypothesis", "").strip()

    if not title:
        return JsonResponse({"error": "Title is required"}, status=400)
    if not hypothesis:
        return JsonResponse({"error": "Hypothesis is required"}, status=400)

    project = Project.objects.create(
        user=request.user,
        title=title,
        hypothesis=hypothesis,
        description=data.get("description", ""),
        domain=data.get("domain", ""),
    )

    return JsonResponse({
        "success": True,
        "project": project.to_dict()
    }, status=201)


@login_required
@require_http_methods(["GET"])
def get_project(request, project_id):
    """Get a project with its workbenches."""
    project = get_object_or_404(Project, id=project_id, user=request.user)

    data = project.to_dict()
    data["workbenches"] = [
        {
            "id": str(w.id),
            "title": w.title,
            "template": w.template,
            "status": w.status,
            "artifact_count": w.artifacts.count(),
            "updated_at": w.updated_at.isoformat(),
        }
        for w in project.workbenches.all()
    ]

    # Include knowledge graph if exists
    if hasattr(project, 'knowledge_graph') and project.knowledge_graph:
        data["knowledge_graph"] = project.knowledge_graph.to_dict()

    return JsonResponse(data)


@login_required
@require_http_methods(["PATCH"])
def update_project(request, project_id):
    """Update project metadata."""
    project = get_object_or_404(Project, id=project_id, user=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Update allowed fields
    if "title" in data:
        project.title = data["title"]
    if "hypothesis" in data:
        project.hypothesis = data["hypothesis"]
    if "description" in data:
        project.description = data["description"]
    if "domain" in data:
        project.domain = data["domain"]
    if "status" in data:
        project.status = data["status"]
    if "conclusion" in data:
        project.conclusion = data["conclusion"]
    if "conclusion_status" in data:
        project.conclusion_status = data["conclusion_status"]

    project.save()

    return JsonResponse({"success": True, "project": project.to_dict()})


@login_required
@require_http_methods(["DELETE"])
def delete_project(request, project_id):
    """Delete a project (archives by default)."""
    project = get_object_or_404(Project, id=project_id, user=request.user)

    permanent = request.GET.get("permanent", "false").lower() == "true"

    if permanent:
        project.delete()
    else:
        project.status = Project.Status.ARCHIVED
        project.save(update_fields=["status"])

    return JsonResponse({"success": True})


@login_required
@require_http_methods(["POST"])
def add_workbench_to_project(request, project_id):
    """Add an existing workbench to a project, or create a new one."""
    project = get_object_or_404(Project, id=project_id, user=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    workbench_id = data.get("workbench_id")

    if workbench_id:
        # Link existing workbench
        workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user)
        workbench.project = project
        workbench.save(update_fields=["project"])
    else:
        # Create new workbench in project
        title = data.get("title", "").strip()
        if not title:
            return JsonResponse({"error": "Title is required for new workbench"}, status=400)

        template = data.get("template", Workbench.Template.BLANK)
        workbench = Workbench.objects.create(
            user=request.user,
            project=project,
            title=title,
            description=data.get("description", ""),
            template=template,
        )

        if template != Workbench.Template.BLANK:
            workbench.init_template()

    return JsonResponse({
        "success": True,
        "workbench": {
            "id": str(workbench.id),
            "title": workbench.title,
            "project_id": str(project.id),
        }
    })


@login_required
@require_http_methods(["POST"])
def remove_workbench_from_project(request, project_id, workbench_id):
    """Remove a workbench from a project (doesn't delete it)."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    workbench = get_object_or_404(Workbench, id=workbench_id, user=request.user, project=project)

    workbench.project = None
    workbench.save(update_fields=["project"])

    return JsonResponse({"success": True})


# =============================================================================
# Hypothesis CRUD
# =============================================================================

@login_required
@require_http_methods(["GET"])
def list_hypotheses(request, project_id):
    """List all hypotheses for a project."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypotheses = project.hypotheses.all()

    # Filter by status if provided
    status = request.GET.get("status")
    if status:
        hypotheses = hypotheses.filter(status=status)

    return JsonResponse({
        "hypotheses": [h.to_dict() for h in hypotheses]
    })


@login_required
@require_http_methods(["POST"])
def create_hypothesis(request, project_id):
    """Create a new hypothesis in a project."""
    project = get_object_or_404(Project, id=project_id, user=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    statement = data.get("statement", "").strip()
    if not statement:
        return JsonResponse({"error": "Statement is required"}, status=400)

    hypothesis = Hypothesis.objects.create(
        project=project,
        statement=statement,
        mechanism=data.get("mechanism", ""),
        prior_probability=data.get("prior_probability", 0.5),
        current_probability=data.get("prior_probability", 0.5),
    )

    return JsonResponse({
        "success": True,
        "hypothesis": hypothesis.to_dict()
    }, status=201)


@login_required
@require_http_methods(["GET"])
def get_hypothesis(request, project_id, hypothesis_id):
    """Get a hypothesis with its evidence and conversations."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    data = hypothesis.to_dict()
    data["evidence"] = [e.to_dict() for e in hypothesis.evidence.all()]
    data["conversations"] = [
        {
            "id": str(c.id),
            "title": c.title,
            "message_count": len(c.messages),
            "updated_at": c.updated_at.isoformat(),
        }
        for c in hypothesis.conversations.all()
    ]

    return JsonResponse(data)


@login_required
@require_http_methods(["PATCH"])
def update_hypothesis(request, project_id, hypothesis_id):
    """Update a hypothesis."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "statement" in data:
        hypothesis.statement = data["statement"]
    if "mechanism" in data:
        hypothesis.mechanism = data["mechanism"]
    if "prior_probability" in data:
        hypothesis.prior_probability = data["prior_probability"]
    if "current_probability" in data:
        hypothesis.current_probability = data["current_probability"]
    if "status" in data:
        hypothesis.status = data["status"]
    if "conclusion_notes" in data:
        hypothesis.conclusion_notes = data["conclusion_notes"]

    hypothesis.save()

    return JsonResponse({"success": True, "hypothesis": hypothesis.to_dict()})


@login_required
@require_http_methods(["DELETE"])
def delete_hypothesis(request, project_id, hypothesis_id):
    """Delete a hypothesis."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    hypothesis.delete()

    return JsonResponse({"success": True})


@login_required
@require_http_methods(["POST"])
def update_hypothesis_probability(request, project_id, hypothesis_id):
    """Update hypothesis probability using Bayesian update."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    likelihood_ratio = data.get("likelihood_ratio")
    if likelihood_ratio is None:
        return JsonResponse({"error": "likelihood_ratio is required"}, status=400)

    old_prob = hypothesis.current_probability
    hypothesis.update_probability(likelihood_ratio)

    return JsonResponse({
        "success": True,
        "old_probability": old_prob,
        "new_probability": hypothesis.current_probability,
    })


# =============================================================================
# Evidence CRUD
# =============================================================================

@login_required
@require_http_methods(["GET"])
def list_evidence(request, project_id, hypothesis_id):
    """List all evidence for a hypothesis."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    evidence = hypothesis.evidence.all()

    # Filter by type if provided
    evidence_type = request.GET.get("type")
    if evidence_type:
        evidence = evidence.filter(evidence_type=evidence_type)

    # Filter by direction if provided
    direction = request.GET.get("direction")
    if direction:
        evidence = evidence.filter(direction=direction)

    return JsonResponse({
        "evidence": [e.to_dict() for e in evidence]
    })


@login_required
@require_http_methods(["POST"])
def create_evidence(request, project_id, hypothesis_id):
    """Create new evidence for a hypothesis."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    summary = data.get("summary", "").strip()
    if not summary:
        return JsonResponse({"error": "Summary is required"}, status=400)

    evidence = Evidence.objects.create(
        hypothesis=hypothesis,
        summary=summary,
        evidence_type=data.get("evidence_type", Evidence.EvidenceType.OBSERVATION),
        direction=data.get("direction", Evidence.Direction.SUPPORTS),
        strength=data.get("strength", 0.5),
        source=data.get("source", ""),
        source_artifact_id=data.get("source_artifact_id"),
        details=data.get("details", {}),
    )

    # Optionally auto-update hypothesis probability based on evidence
    if data.get("auto_update_probability", False):
        strength = evidence.strength
        if evidence.direction == Evidence.Direction.SUPPORTS:
            likelihood_ratio = 1 + strength * 2  # 1.0 to 3.0
        elif evidence.direction == Evidence.Direction.WEAKENS:
            likelihood_ratio = 1 / (1 + strength * 2)  # 0.33 to 1.0
        else:
            likelihood_ratio = 1.0
        hypothesis.update_probability(likelihood_ratio)

    return JsonResponse({
        "success": True,
        "evidence": evidence.to_dict()
    }, status=201)


@login_required
@require_http_methods(["GET"])
def get_evidence(request, project_id, hypothesis_id, evidence_id):
    """Get a single piece of evidence."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)
    evidence = get_object_or_404(Evidence, id=evidence_id, hypothesis=hypothesis)

    return JsonResponse(evidence.to_dict())


@login_required
@require_http_methods(["DELETE"])
def delete_evidence(request, project_id, hypothesis_id, evidence_id):
    """Delete evidence."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)
    evidence = get_object_or_404(Evidence, id=evidence_id, hypothesis=hypothesis)

    evidence.delete()

    return JsonResponse({"success": True})


# =============================================================================
# Conversation CRUD
# =============================================================================

@login_required
@require_http_methods(["GET"])
def list_conversations(request, project_id, hypothesis_id):
    """List all conversations for a hypothesis."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    conversations = hypothesis.conversations.all()

    return JsonResponse({
        "conversations": [
            {
                "id": str(c.id),
                "title": c.title,
                "message_count": len(c.messages),
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in conversations
        ]
    })


@login_required
@require_http_methods(["POST"])
def create_conversation(request, project_id, hypothesis_id):
    """Create a new conversation for a hypothesis."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = {}

    title = data.get("title", "New Conversation").strip()

    conversation = Conversation.objects.create(
        hypothesis=hypothesis,
        title=title,
    )

    # Build initial context from hypothesis
    conversation.context = conversation.build_context()
    conversation.save(update_fields=["context"])

    return JsonResponse({
        "success": True,
        "conversation": conversation.to_dict()
    }, status=201)


@login_required
@require_http_methods(["GET"])
def get_conversation(request, project_id, hypothesis_id, conversation_id):
    """Get a conversation with all messages."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)
    conversation = get_object_or_404(Conversation, id=conversation_id, hypothesis=hypothesis)

    return JsonResponse(conversation.to_dict())


@login_required
@require_http_methods(["PATCH"])
def update_conversation(request, project_id, hypothesis_id, conversation_id):
    """Update conversation metadata."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)
    conversation = get_object_or_404(Conversation, id=conversation_id, hypothesis=hypothesis)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "title" in data:
        conversation.title = data["title"]
        conversation.save(update_fields=["title", "updated_at"])

    return JsonResponse({"success": True})


@login_required
@require_http_methods(["POST"])
def add_message(request, project_id, hypothesis_id, conversation_id):
    """Add a message to a conversation."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)
    conversation = get_object_or_404(Conversation, id=conversation_id, hypothesis=hypothesis)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    role = data.get("role", "user")
    content = data.get("content", "").strip()

    if not content:
        return JsonResponse({"error": "Content is required"}, status=400)

    conversation.add_message(role, content)

    return JsonResponse({
        "success": True,
        "message_count": len(conversation.messages)
    })


@login_required
@require_http_methods(["DELETE"])
def delete_conversation(request, project_id, hypothesis_id, conversation_id):
    """Delete a conversation."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)
    conversation = get_object_or_404(Conversation, id=conversation_id, hypothesis=hypothesis)

    conversation.delete()

    return JsonResponse({"success": True})


@login_required
@require_http_methods(["POST"])
def refresh_conversation_context(request, project_id, hypothesis_id, conversation_id):
    """Refresh the context snapshot in a conversation."""
    project = get_object_or_404(Project, id=project_id, user=request.user)
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id, project=project)
    conversation = get_object_or_404(Conversation, id=conversation_id, hypothesis=hypothesis)

    conversation.context = conversation.build_context()
    conversation.save(update_fields=["context", "updated_at"])

    return JsonResponse({
        "success": True,
        "context": conversation.context
    })
