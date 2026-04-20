"""Safety pull contract views — expose Frontier Card observations for cross-tool integration."""

from qms_core import pull_views
from qms_core.permissions import get_accessible_sites, get_tenant

from .models import FrontierCard

SOURCE_APP = "safety"
CONTAINER_TYPE = "FrontierCard"


def _queryset(request):
    """Return FrontierCards accessible to this user via tenant/site access."""
    tenant = get_tenant(request.user)
    if not tenant:
        return FrontierCard.objects.none()
    sites, _ = get_accessible_sites(request.user, tenant)
    return FrontierCard.objects.filter(site__in=sites).select_related("zone").order_by("-audit_date")


def _serialize(obj):
    return {
        "id": str(obj.id),
        "title": f"{obj.zone.name} — {obj.audit_date}",
        "classification": obj.classification,
        "at_risk_count": obj.at_risk_count,
        "audit_date": obj.audit_date.isoformat(),
    }


def _get_container(request, pk):
    tenant = get_tenant(request.user)
    if not tenant:
        return None
    sites, _ = get_accessible_sites(request.user, tenant)
    try:
        return FrontierCard.objects.select_related("zone").get(id=pk, site__in=sites)
    except FrontierCard.DoesNotExist:
        return None


def _manifest(obj):
    return obj.to_manifest()


def _get_card_dict(request, artifact_id):
    obj = _get_container(request, artifact_id)
    if obj is None:
        return None
    return obj.to_dict()


def _get_card(request, artifact_id):
    return _get_container(request, artifact_id)


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
    return pull_views.pull_artifact_detail(request, artifact_id, get_artifact_fn=_get_card_dict)


def artifact_sub(request, artifact_id, key_path):
    return pull_views.pull_sub_artifact(
        request,
        artifact_id,
        key_path,
        get_artifact_fn=_get_card,
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
