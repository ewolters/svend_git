"""Generic pull contract view helpers.

Each app wires these into its own urls.py by passing app-specific callables
for queryset generation, manifest creation, and sub-artifact retrieval.

Architecture: docs/planning/object_271/qms_architecture.md §2.3
"""

import json
import logging

from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth
from qms_core.models import ArtifactReference
from qms_core.permissions import get_tenant

logger = logging.getLogger(__name__)


# =============================================================================
# Container list + detail
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def pull_container_list(request, *, queryset_fn, serialize_fn):
    """List containers (sessions, FMEAs, RCA sessions, etc.) for pull browsing.

    Args:
        queryset_fn: (request) -> QuerySet — filtered for user/tenant
        serialize_fn: (obj) -> dict — minimal serialization for list view
    """
    qs = queryset_fn(request)
    items = [serialize_fn(obj) for obj in qs[:100]]
    return JsonResponse({"containers": items})


@require_http_methods(["GET"])
@require_auth
def pull_container_detail(request, pk, *, get_obj_fn, manifest_fn):
    """Return full manifest for a single container.

    Args:
        get_obj_fn: (request, pk) -> obj or None
        manifest_fn: (obj) -> dict — full manifest with available sub-artifacts
    """
    obj = get_obj_fn(request, pk)
    if obj is None:
        return JsonResponse({"error": "Not found"}, status=404)
    return JsonResponse({"manifest": manifest_fn(obj)})


# =============================================================================
# Artifact + sub-artifact access
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def pull_artifact_detail(request, artifact_id, *, get_artifact_fn):
    """Return full artifact data (e.g. an FMEARow, a SessionAnalysis).

    Args:
        get_artifact_fn: (request, artifact_id) -> dict or None
    """
    data = get_artifact_fn(request, artifact_id)
    if data is None:
        return JsonResponse({"error": "Artifact not found"}, status=404)
    return JsonResponse({"artifact": data})


@require_http_methods(["GET"])
@require_auth
def pull_sub_artifact(request, artifact_id, key_path, *, get_artifact_fn, sub_artifact_fn):
    """Access a specific sub-artifact field by key path.

    Args:
        get_artifact_fn: (request, artifact_id) -> model instance or None
        sub_artifact_fn: (obj, key_path) -> value or None
    """
    obj = get_artifact_fn(request, artifact_id)
    if obj is None:
        return JsonResponse({"error": "Artifact not found"}, status=404)

    value = sub_artifact_fn(obj, key_path)
    if value is None:
        return JsonResponse({"error": f"Sub-artifact not found: {key_path}"}, status=404)

    return JsonResponse({"key": key_path, "value": value})


# =============================================================================
# Reference registration + listing
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def pull_register_reference(request, artifact_id, *, source_app, source_type):
    """Register that a consumer is pulling from this artifact.

    POST body: {"consumer_app": "a3", "consumer_type": "A3Report", "consumer_id": "<uuid>",
                "artifact_key": "statistics/p_value"}
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    consumer_app = data.get("consumer_app", "")
    consumer_type = data.get("consumer_type", "")
    consumer_id = data.get("consumer_id", "")
    artifact_key = data.get("artifact_key", "")

    if not consumer_app or not consumer_type or not consumer_id:
        return JsonResponse({"error": "consumer_app, consumer_type, consumer_id are required"}, status=400)

    tenant = get_tenant(request.user)

    ref, created = ArtifactReference.objects.get_or_create(
        source_app=source_app,
        source_id=artifact_id,
        artifact_key=artifact_key,
        consumer_app=consumer_app,
        consumer_type=consumer_type,
        consumer_id=consumer_id,
        defaults={
            "source_type": source_type,
            "tenant": tenant,
            "created_by": request.user,
        },
    )

    return JsonResponse(
        {
            "reference_id": str(ref.id),
            "created": created,
        },
        status=201 if created else 200,
    )


@require_http_methods(["GET"])
@require_auth
def pull_list_references(request, container_id, *, source_app, source_type):
    """List all references pointing to artifacts in this container.

    Returns consumers that have registered references to any artifact
    within the container.
    """
    refs = ArtifactReference.objects.filter(
        source_app=source_app,
        source_type=source_type,
        source_id=container_id,
    ).order_by("-created_at")

    return JsonResponse(
        {
            "references": [
                {
                    "id": str(r.id),
                    "artifact_key": r.artifact_key,
                    "consumer_app": r.consumer_app,
                    "consumer_type": r.consumer_type,
                    "consumer_id": str(r.consumer_id),
                    "created_at": r.created_at.isoformat(),
                    "source_deleted_at": r.source_deleted_at.isoformat() if r.source_deleted_at else None,
                }
                for r in refs
            ]
        }
    )


# =============================================================================
# Delete with friction — utility + view
# =============================================================================


def check_delete_friction(source_app, source_type, source_id, force=False):
    """Check for active references and optionally tombstone them.

    Call this from any delete view before actually deleting the object.

    Returns:
        (ok_to_delete, error_response_or_none, tombstoned_count)

    If ok_to_delete is False, return the error_response (409 with consumer list).
    If ok_to_delete is True, proceed with deletion — references are already tombstoned.
    """
    active_refs = ArtifactReference.objects.filter(
        source_app=source_app,
        source_type=source_type,
        source_id=source_id,
        source_deleted_at__isnull=True,
    )

    ref_count = active_refs.count()

    if ref_count > 0 and not force:
        consumers = [
            {
                "consumer_app": r.consumer_app,
                "consumer_type": r.consumer_type,
                "consumer_id": str(r.consumer_id),
                "artifact_key": r.artifact_key,
            }
            for r in active_refs[:20]
        ]
        return (
            False,
            JsonResponse(
                {
                    "error": "Cannot delete — active references exist",
                    "reference_count": ref_count,
                    "consumers": consumers,
                    "hint": "Add ?force=true to delete anyway (consumers will see tombstone)",
                },
                status=409,
            ),
            0,
        )

    # Tombstone all references
    if ref_count > 0:
        now = timezone.now()
        active_refs.update(source_deleted_at=now)
        logger.info(
            "Tombstoned %d references for %s:%s/%s",
            ref_count,
            source_app,
            source_type,
            source_id,
        )

    return True, None, ref_count
