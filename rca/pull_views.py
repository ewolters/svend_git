"""RCA pull contract views — expose RCA artifacts for cross-tool integration."""

from qms_core import pull_views
from qms_core.permissions import qms_queryset

from .models import RCASession

SOURCE_APP = "rca"
CONTAINER_TYPE = "RCASession"


def _queryset(request):
    qs, _, _ = qms_queryset(RCASession, request.user)
    return qs.order_by("-updated_at")


def _serialize(obj):
    return {
        "id": str(obj.id),
        "title": obj.title or obj.event[:80],
        "status": obj.status,
        "updated_at": obj.updated_at.isoformat(),
    }


def _get_container(request, pk):
    qs, _, _ = qms_queryset(RCASession, request.user)
    try:
        return qs.get(id=pk)
    except RCASession.DoesNotExist:
        return None


def _manifest(obj):
    return obj.to_manifest()


def _get_artifact_dict(request, artifact_id):
    """RCA is a single-artifact container — the session itself is the artifact."""
    obj = _get_container(request, artifact_id)
    if obj is None:
        return None
    return obj.to_dict()


def _get_artifact(request, artifact_id):
    return _get_container(request, artifact_id)


def _sub_artifact(obj, key_path):
    """Retrieve a sub-artifact field from an RCASession."""
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
