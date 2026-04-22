"""Composable QMS views — CRUD for templates and artifacts."""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth
from qms_core.permissions import get_tenant, qms_queryset, resolve_site

from .models import Artifact, ArtifactSection, ToolTemplate
from .workflow_models import SignalTypeRegistry, WorkflowTemplate

logger = logging.getLogger(__name__)


# =============================================================================
# ToolTemplate CRUD
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def template_list(request):
    """List available templates (system + tenant-specific)."""
    tenant = get_tenant(request.user)
    # System templates + tenant templates
    if tenant:
        qs = ToolTemplate.objects.filter(tenant__isnull=True) | ToolTemplate.objects.filter(tenant=tenant)
    else:
        qs = ToolTemplate.objects.filter(tenant__isnull=True)

    templates = [
        {
            "id": str(t.id),
            "name": t.name,
            "slug": t.slug,
            "description": t.description,
            "icon": t.icon,
            "is_system": t.is_system,
            "version": t.version,
            "status_flow": t.status_flow,
            "section_count": len(t.get_section_defs()),
        }
        for t in qs.order_by("name")
    ]
    return JsonResponse({"templates": templates})


@require_http_methods(["GET"])
@require_auth
def template_detail(request, template_id):
    """Get full template detail including schema."""
    tenant = get_tenant(request.user)
    try:
        t = ToolTemplate.objects.get(id=template_id)
    except ToolTemplate.DoesNotExist:
        return JsonResponse({"error": "Template not found"}, status=404)

    # Access check: system templates are public, tenant templates require membership
    if t.tenant and (not tenant or t.tenant_id != tenant.id):
        return JsonResponse({"error": "Template not found"}, status=404)

    return JsonResponse(
        {
            "template": {
                "id": str(t.id),
                "name": t.name,
                "slug": t.slug,
                "description": t.description,
                "icon": t.icon,
                "is_system": t.is_system,
                "version": t.version,
                "schema": t.schema,
                "status_flow": t.status_flow,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
            }
        }
    )


# =============================================================================
# Artifact CRUD
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def artifact_list(request):
    """List artifacts for the current user/tenant."""
    qs, tenant, _ = qms_queryset(Artifact, request.user)

    # Optional filters
    template_slug = request.GET.get("template")
    status = request.GET.get("status")
    if template_slug:
        qs = qs.filter(template__slug=template_slug)
    if status:
        qs = qs.filter(status=status)

    artifacts = [
        {
            "id": str(a.id),
            "template_slug": a.template.slug,
            "template_name": a.template.name,
            "title": a.title,
            "status": a.status,
            "updated_at": a.updated_at.isoformat(),
        }
        for a in qs.select_related("template")[:100]
    ]
    return JsonResponse({"artifacts": artifacts})


@require_http_methods(["POST"])
@require_auth
def artifact_create(request):
    """Create a new artifact from a template."""
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    template_id = data.get("template_id")
    title = data.get("title", "").strip()

    if not template_id:
        return JsonResponse({"error": "template_id is required"}, status=400)
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    try:
        template = ToolTemplate.objects.get(id=template_id)
    except ToolTemplate.DoesNotExist:
        return JsonResponse({"error": "Template not found"}, status=404)

    tenant = get_tenant(request.user)

    # Access check for tenant templates
    if template.tenant and (not tenant or template.tenant_id != tenant.id):
        return JsonResponse({"error": "Template not found"}, status=404)

    artifact = Artifact.objects.create(
        template=template,
        tenant=tenant,
        owner=request.user,
        title=title,
        status=template.status_flow[0] if template.status_flow else "draft",
    )

    # Set site if provided
    site_id = data.get("site_id")
    if site_id:
        site = resolve_site(request.user, site_id)
        if site:
            artifact.site = site
            artifact.save(update_fields=["site"])

    # Set project if provided
    project_id = data.get("project_id")
    if project_id:
        from core.models import Project

        try:
            project = Project.objects.get(id=project_id)
            artifact.project = project
            artifact.save(update_fields=["project"])
        except Project.DoesNotExist:
            pass

    # Pre-create empty sections from template schema
    for i, sdef in enumerate(template.get_section_defs()):
        ArtifactSection.objects.create(
            artifact=artifact,
            section_key=sdef["key"],
            primitive_type=sdef["type"],
            data=_default_data(sdef["type"]),
            sort_order=i,
        )

    return JsonResponse(
        {"artifact": artifact.to_dict()},
        status=201,
    )


@require_http_methods(["GET"])
@require_auth
def artifact_detail(request, artifact_id):
    """Get full artifact with all sections."""
    qs, _, _ = qms_queryset(Artifact, request.user)
    try:
        artifact = qs.prefetch_related("sections").select_related("template").get(id=artifact_id)
    except Artifact.DoesNotExist:
        return JsonResponse({"error": "Artifact not found"}, status=404)

    return JsonResponse({"artifact": artifact.to_dict()})


@require_http_methods(["PATCH"])
@require_auth
def artifact_update(request, artifact_id):
    """Update artifact metadata or section data."""
    qs, _, _ = qms_queryset(Artifact, request.user)
    try:
        artifact = qs.select_related("template").get(id=artifact_id)
    except Artifact.DoesNotExist:
        return JsonResponse({"error": "Artifact not found"}, status=404)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Update metadata
    if "title" in data:
        artifact.title = data["title"]
    if "status" in data:
        new_status = data["status"]
        flow = artifact.template.status_flow
        if flow and new_status not in flow:
            return JsonResponse(
                {"error": f"Invalid status '{new_status}'. Valid: {flow}"},
                status=400,
            )
        artifact.status = new_status
    artifact.save()

    # Update sections
    sections = data.get("sections", {})
    for section_key, section_data in sections.items():
        try:
            section = artifact.sections.get(section_key=section_key)
            section.data = section_data
            section.save(update_fields=["data"])
        except ArtifactSection.DoesNotExist:
            logger.warning("Section %s not found on artifact %s", section_key, artifact_id)

    return JsonResponse({"artifact": artifact.to_dict()})


@require_http_methods(["DELETE"])
@require_auth
def artifact_delete(request, artifact_id):
    """Delete an artifact and its sections."""
    from qms_core.pull_views import check_delete_friction

    qs, _, _ = qms_queryset(Artifact, request.user)
    try:
        artifact = qs.get(id=artifact_id)
    except Artifact.DoesNotExist:
        return JsonResponse({"error": "Artifact not found"}, status=404)

    force = request.GET.get("force") == "true"
    ok, err_resp, _ = check_delete_friction("qms", "Artifact", artifact_id, force=force)
    if not ok:
        return err_resp

    artifact.delete()
    return JsonResponse({"deleted": True})


def _default_data(primitive_type):
    """Return default empty data structure for a primitive type."""
    defaults = {
        "text": {"content": ""},
        "grid": {"rows": []},
        "tree": {"root": "", "branches": []},
        "checklist": {"items": []},
        "action_list": {"items": []},
    }
    return defaults.get(primitive_type, {})


# =============================================================================
# Workflow views
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def workflow_list(request):
    """List available workflow templates (system + tenant-specific)."""
    tenant = get_tenant(request.user)
    if tenant:
        qs = WorkflowTemplate.objects.filter(tenant__isnull=True) | WorkflowTemplate.objects.filter(tenant=tenant)
    else:
        qs = WorkflowTemplate.objects.filter(tenant__isnull=True)

    workflows = []
    for wf in qs.filter(is_active=True).prefetch_related("phases"):
        workflows.append(
            {
                "id": str(wf.id),
                "name": wf.name,
                "is_system": wf.is_system,
                "phase_count": wf.phases.count(),
                "phases": [{"key": p.key, "label": p.label, "color": p.color} for p in wf.phases.all()],
            }
        )
    return JsonResponse({"workflows": workflows})


@require_http_methods(["GET"])
@require_auth
def workflow_detail(request, workflow_id):
    """Full workflow detail with phases, transitions, and available templates per phase."""
    tenant = get_tenant(request.user)
    try:
        wf = WorkflowTemplate.objects.prefetch_related(
            "phases__available_templates", "transitions__from_phase", "transitions__to_phase"
        ).get(id=workflow_id)
    except WorkflowTemplate.DoesNotExist:
        return JsonResponse({"error": "Workflow not found"}, status=404)

    if wf.tenant and (not tenant or wf.tenant_id != tenant.id):
        return JsonResponse({"error": "Workflow not found"}, status=404)

    phases = []
    for p in wf.phases.all():
        phases.append(
            {
                "id": str(p.id),
                "key": p.key,
                "label": p.label,
                "sort_order": p.sort_order,
                "color": p.color,
                "available_templates": [
                    {"id": str(t.id), "name": t.name, "slug": t.slug} for t in p.available_templates.all()
                ],
            }
        )

    transitions = [
        {
            "id": str(tr.id),
            "from_phase": tr.from_phase.key,
            "to_phase": tr.to_phase.key,
            "label": tr.label,
            "gate_conditions": tr.gate_conditions,
        }
        for tr in wf.transitions.all()
    ]

    return JsonResponse(
        {
            "workflow": {
                "id": str(wf.id),
                "name": wf.name,
                "is_system": wf.is_system,
                "is_active": wf.is_active,
                "phases": phases,
                "transitions": transitions,
            }
        }
    )


@require_http_methods(["GET"])
@require_auth
def signal_type_list(request):
    """List available signal types (system + tenant-specific)."""
    tenant = get_tenant(request.user)
    if tenant:
        qs = SignalTypeRegistry.objects.filter(tenant__isnull=True) | SignalTypeRegistry.objects.filter(tenant=tenant)
    else:
        qs = SignalTypeRegistry.objects.filter(tenant__isnull=True)

    types = [
        {
            "id": str(st.id),
            "key": st.key,
            "label": st.label,
            "default_severity": st.default_severity,
            "is_system": st.is_system,
            "icon": st.icon,
            "auto_phase_id": str(st.auto_phase_id) if st.auto_phase_id else None,
        }
        for st in qs
    ]
    return JsonResponse({"signal_types": types})
