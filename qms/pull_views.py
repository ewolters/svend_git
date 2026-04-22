"""Composable QMS pull contract views — expose artifacts for cross-tool integration.

Follows the same pattern as fmea/pull_views.py, a3/pull_views.py etc.
"""

from qms_core import pull_views
from qms_core.permissions import qms_queryset

from .models import Artifact

SOURCE_APP = "qms"
CONTAINER_TYPE = "Artifact"


def _queryset(request):
    qs, _, _ = qms_queryset(Artifact, request.user)
    return qs.select_related("template").order_by("-updated_at")


def _serialize(obj):
    return {
        "id": str(obj.id),
        "title": obj.title,
        "template_slug": obj.template.slug,
        "template_name": obj.template.name,
        "status": obj.status,
        "updated_at": obj.updated_at.isoformat(),
    }


def _get_container(request, pk):
    qs, _, _ = qms_queryset(Artifact, request.user)
    try:
        return qs.select_related("template").prefetch_related("sections").get(id=pk)
    except Artifact.DoesNotExist:
        return None


def _manifest(obj):
    return obj.to_manifest()


def _get_artifact_dict(request, artifact_id):
    obj = _get_container(request, artifact_id)
    if obj is None:
        return None
    return obj.to_dict()


def _get_artifact(request, artifact_id):
    return _get_container(request, artifact_id)


def _sub_artifact(obj, key_path):
    """Retrieve a sub-artifact field from an Artifact."""
    return obj.get_sub_artifact(key_path)


# View functions


def container_list(request):
    return pull_views.pull_container_list(request, queryset_fn=_queryset, serialize_fn=_serialize)


def container_detail(request, pk):
    return pull_views.pull_container_detail(request, pk, get_obj_fn=_get_container, manifest_fn=_manifest)


def artifact_detail_view(request, artifact_id):
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
