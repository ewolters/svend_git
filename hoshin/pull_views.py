"""Hoshin Kanri pull contract views — expose projects/actions for cross-tool integration.

Container: HoshinProject (wraps core.Project with Hoshin extensions)
Artifacts: HoshinProject (JSON sub-artifacts), ActionItem, ResourceCommitment
"""

from qms_core import pull_views
from qms_core.permissions import get_accessible_sites, require_tenant

from .models import ActionItem, HoshinProject, ResourceCommitment

SOURCE_APP = "hoshin"
CONTAINER_TYPE = "HoshinProject"


def _queryset(request):
    """Return HoshinProjects accessible via tenant/site."""
    tenant, err = require_tenant(request.user)
    if err:
        return HoshinProject.objects.none()
    sites, _ = get_accessible_sites(request.user, tenant)
    return HoshinProject.objects.filter(site__in=sites).select_related("project", "site").order_by("-updated_at")


def _serialize(obj):
    return {
        "id": str(obj.id),
        "title": obj.project.title,
        "hoshin_status": obj.hoshin_status,
        "fiscal_year": obj.fiscal_year,
        "site_name": obj.site.name if obj.site else None,
        "updated_at": obj.updated_at.isoformat(),
    }


def _get_container(request, pk):
    tenant, err = require_tenant(request.user)
    if err:
        return None
    sites, _ = get_accessible_sites(request.user, tenant)
    try:
        return HoshinProject.objects.select_related("project", "site").get(id=pk, site__in=sites)
    except HoshinProject.DoesNotExist:
        return None


def _manifest(obj):
    return obj.to_manifest()


def _get_artifact_dict(request, artifact_id):
    """Try HoshinProject first, then ActionItem, then ResourceCommitment."""
    # Try HoshinProject
    obj = _get_container(request, artifact_id)
    if obj is not None:
        return obj.to_dict()

    # Try ActionItem
    try:
        item = ActionItem.objects.select_related("project").get(id=artifact_id)
        # Verify access via hoshin project
        tenant, err = require_tenant(request.user)
        if err:
            return None
        sites, _ = get_accessible_sites(request.user, tenant)
        if HoshinProject.objects.filter(project=item.project, site__in=sites).exists():
            return item.to_dict()
    except ActionItem.DoesNotExist:
        pass

    # Try ResourceCommitment
    try:
        rc = ResourceCommitment.objects.select_related("employee", "project__project").get(id=artifact_id)
        tenant, err = require_tenant(request.user)
        if err:
            return None
        sites, _ = get_accessible_sites(request.user, tenant)
        if rc.project.site_id and sites.filter(id=rc.project.site_id).exists():
            return rc.to_dict()
    except ResourceCommitment.DoesNotExist:
        pass

    return None


def _get_artifact(request, artifact_id):
    """Get raw artifact object for sub-artifact traversal."""
    obj = _get_container(request, artifact_id)
    if obj is not None:
        return obj

    try:
        item = ActionItem.objects.select_related("project").get(id=artifact_id)
        tenant, err = require_tenant(request.user)
        if not err:
            sites, _ = get_accessible_sites(request.user, tenant)
            if HoshinProject.objects.filter(project=item.project, site__in=sites).exists():
                return item
    except ActionItem.DoesNotExist:
        pass

    try:
        rc = ResourceCommitment.objects.select_related("employee", "project__project").get(id=artifact_id)
        tenant, err = require_tenant(request.user)
        if not err:
            sites, _ = get_accessible_sites(request.user, tenant)
            if rc.project.site_id and sites.filter(id=rc.project.site_id).exists():
                return rc
    except ResourceCommitment.DoesNotExist:
        pass

    return None


def _sub_artifact(obj, key_path):
    parts = key_path.strip("/").split("/")
    data = obj.to_dict()
    for part in parts:
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list):
            try:
                data = data[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return data


# View functions


def container_list(request):
    return pull_views.pull_container_list(request, queryset_fn=_queryset, serialize_fn=_serialize)


def container_detail(request, pk):
    return pull_views.pull_container_detail(request, pk, get_obj_fn=_get_container, manifest_fn=_manifest)


def artifact_detail(request, artifact_id):
    return pull_views.pull_artifact_detail(request, artifact_id, get_artifact_fn=_get_artifact_dict)


def artifact_sub(request, artifact_id, key_path):
    return pull_views.pull_sub_artifact(
        request,
        artifact_id,
        key_path,
        get_artifact_fn=_get_artifact,
        sub_artifact_fn=_sub_artifact,
    )


def register_reference(request, artifact_id):
    return pull_views.pull_register_reference(
        request,
        artifact_id,
        source_app=SOURCE_APP,
        source_type=CONTAINER_TYPE,
    )


def list_references(request, container_id):
    return pull_views.pull_list_references(
        request,
        container_id,
        source_app=SOURCE_APP,
        source_type=CONTAINER_TYPE,
    )
