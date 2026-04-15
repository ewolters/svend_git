"""FMEA pull contract views — expose FMEA artifacts for cross-tool integration."""

from qms_core import pull_views
from qms_core.permissions import qms_queryset

from .models import FMEA, FMEARow

SOURCE_APP = "fmea"
CONTAINER_TYPE = "FMEA"
ARTIFACT_TYPE = "FMEARow"


def _queryset(request):
    qs, _, _ = qms_queryset(FMEA, request.user)
    return qs.order_by("-updated_at")


def _serialize(obj):
    return {
        "id": str(obj.id),
        "title": obj.title,
        "status": obj.status,
        "fmea_type": obj.fmea_type,
        "updated_at": obj.updated_at.isoformat(),
    }


def _get_container(request, pk):
    qs, _, _ = qms_queryset(FMEA, request.user)
    try:
        return qs.get(id=pk)
    except FMEA.DoesNotExist:
        return None


def _manifest(obj):
    return obj.to_manifest()


def _get_row_dict(request, artifact_id):
    try:
        row = FMEARow.objects.select_related("fmea").get(id=artifact_id)
    except FMEARow.DoesNotExist:
        return None
    # Verify access
    qs, _, _ = qms_queryset(FMEA, request.user)
    if not qs.filter(id=row.fmea_id).exists():
        return None
    return row.to_dict()


def _get_row(request, artifact_id):
    try:
        row = FMEARow.objects.select_related("fmea").get(id=artifact_id)
    except FMEARow.DoesNotExist:
        return None
    qs, _, _ = qms_queryset(FMEA, request.user)
    if not qs.filter(id=row.fmea_id).exists():
        return None
    return row


def _sub_artifact(row, key_path):
    """Retrieve a sub-artifact field from an FMEARow."""
    parts = key_path.strip("/").split("/")
    obj = row.to_dict()
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return obj


def _delete_fmea(obj):
    obj.delete()


# View functions


def container_list(request):
    return pull_views.pull_container_list(request, queryset_fn=_queryset, serialize_fn=_serialize)


def container_detail(request, pk):
    return pull_views.pull_container_detail(request, pk, get_obj_fn=_get_container, manifest_fn=_manifest)


def artifact_detail(request, artifact_id):
    return pull_views.pull_artifact_detail(request, artifact_id, get_artifact_fn=_get_row_dict)


def artifact_sub(request, artifact_id, key_path):
    return pull_views.pull_sub_artifact(
        request,
        artifact_id,
        key_path,
        get_artifact_fn=_get_row,
        sub_artifact_fn=_sub_artifact,
    )


def register_reference(request, artifact_id):
    return pull_views.pull_register_reference(
        request,
        artifact_id,
        source_app=SOURCE_APP,
        source_type=ARTIFACT_TYPE,
    )


def list_references(request, container_id):
    return pull_views.pull_list_references(
        request,
        container_id,
        source_app=SOURCE_APP,
        source_type=CONTAINER_TYPE,
    )


def delete_with_friction(request, container_id):
    return pull_views.pull_delete_with_friction(
        request,
        container_id,
        source_app=SOURCE_APP,
        source_type=CONTAINER_TYPE,
        get_obj_fn=_get_container,
        delete_fn=_delete_fmea,
    )
