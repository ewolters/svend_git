"""
Investigation API views — CANON-002 §13.

Endpoints for investigation lifecycle: create, detail, list,
transition, reopen, export, member management, and graph retrieval.

<!-- impl: investigation_views:list_create_investigations -->
<!-- impl: investigation_views:get_investigation -->
<!-- impl: investigation_views:transition_investigation -->
<!-- impl: investigation_views:reopen_investigation -->
<!-- impl: investigation_views:export_investigation_view -->
<!-- impl: investigation_views:manage_members -->
<!-- impl: investigation_views:get_graph -->
<!-- impl: investigation_views:list_tools -->
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from agents_api.investigation_bridge import (
    export_investigation,
    get_investigation,
    load_synara,
)
from core.models import (
    Investigation,
    InvestigationMembership,
    InvestigationToolLink,
)

logger = logging.getLogger("svend.investigation")


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _serialize_investigation(inv, include_graph=False):
    """Serialize an Investigation for API responses."""
    data = {
        "id": str(inv.id),
        "title": inv.title,
        "description": inv.description,
        "status": inv.status,
        "version": inv.version,
        "owner_id": str(inv.owner_id),
        "parent_version_id": (str(inv.parent_version_id) if inv.parent_version_id else None),
        "exported_to_project_id": (str(inv.exported_to_project_id) if inv.exported_to_project_id else None),
        "created_at": inv.created_at.isoformat() if inv.created_at else None,
        "updated_at": inv.updated_at.isoformat() if inv.updated_at else None,
        "concluded_at": inv.concluded_at.isoformat() if inv.concluded_at else None,
        "exported_at": inv.exported_at.isoformat() if inv.exported_at else None,
    }
    if include_graph:
        synara = load_synara(inv)
        graph = synara.to_dict().get("graph", {})
        data["graph"] = {
            "hypotheses": graph.get("hypotheses", {}),
            "links": [
                {
                    "from_id": link.get("from_id", ""),
                    "to_id": link.get("to_id", ""),
                    "strength": link.get("strength", 0),
                    "mechanism": link.get("mechanism", ""),
                }
                for link in graph.get("links", [])
            ],
            "evidence": [
                {
                    "id": e.get("id", ""),
                    "event": e.get("event", ""),
                    "strength": e.get("strength", 0),
                    "source": e.get("source", ""),
                }
                for e in graph.get("evidence", [])
            ],
        }
    return data


def _serialize_member(membership):
    return {
        "id": str(membership.id),
        "user_id": str(membership.user_id),
        "username": membership.user.username,
        "role": membership.role,
        "joined_at": membership.joined_at.isoformat(),
    }


def _serialize_tool_link(link):
    return {
        "id": str(link.id),
        "tool_type": link.tool_type,
        "tool_function": link.tool_function,
        "object_id": str(link.object_id),
        "content_type": link.content_type.model,
        "linked_at": link.linked_at.isoformat(),
        "linked_by_id": str(link.linked_by_id),
    }


# ---------------------------------------------------------------------------
# List / Create
# ---------------------------------------------------------------------------


@gated_paid
@require_http_methods(["GET", "POST"])
def list_create_investigations(request):
    """
    GET  — List user's investigations (owned + member).
    POST — Create a new investigation.

    POST body: {"title": str, "description": str (optional)}
    """
    if request.method == "GET":
        owned = Investigation.objects.filter(owner=request.user)
        member_of = Investigation.objects.filter(members=request.user).exclude(owner=request.user)
        investigations = list(owned) + list(member_of)
        return JsonResponse({"investigations": [_serialize_investigation(inv) for inv in investigations]})

    # POST — create
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    inv = Investigation.objects.create(
        title=title,
        description=data.get("description", ""),
        owner=request.user,
    )
    # Auto-create owner membership
    InvestigationMembership.objects.create(
        investigation=inv,
        user=request.user,
        role=Investigation.MemberRole.OWNER,
    )

    logger.info("investigation.created", extra={"investigation_id": str(inv.id)})
    return JsonResponse({"investigation": _serialize_investigation(inv)}, status=201)


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


@gated_paid
@require_http_methods(["GET", "DELETE"])
def investigation_detail(request, investigation_id):
    """
    GET    — Investigation detail with graph summary.
    DELETE — Delete investigation (only if status=open).
    """
    try:
        inv = get_investigation(str(investigation_id), request.user)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)
    except PermissionError:
        return JsonResponse({"error": "Not a member of this investigation"}, status=403)

    if request.method == "GET":
        return JsonResponse({"investigation": _serialize_investigation(inv, include_graph=True)})

    # DELETE
    if inv.status != Investigation.Status.OPEN:
        return JsonResponse(
            {"error": f"Cannot delete investigation in '{inv.status}' state — must be 'open'"},
            status=400,
        )
    if inv.owner != request.user:
        return JsonResponse({"error": "Only the owner can delete an investigation"}, status=403)

    inv_id = str(inv.id)
    inv.delete()
    logger.info("investigation.deleted", extra={"investigation_id": inv_id})
    return JsonResponse({"deleted": True, "id": inv_id})


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


@gated_paid
@require_http_methods(["POST"])
def transition_investigation(request, investigation_id):
    """
    Transition investigation to a new state.

    POST body: {"target_status": "active"|"concluded"|"exported"}
    """
    try:
        inv = get_investigation(str(investigation_id), request.user)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)
    except PermissionError:
        return JsonResponse({"error": "Not a member of this investigation"}, status=403)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    target = data.get("target_status")
    if not target:
        return JsonResponse({"error": "target_status is required"}, status=400)

    try:
        inv.transition_to(target, request.user)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    logger.info(
        "investigation.transitioned",
        extra={"investigation_id": str(inv.id), "status": inv.status},
    )
    return JsonResponse({"investigation": _serialize_investigation(inv)})


# ---------------------------------------------------------------------------
# Reopen
# ---------------------------------------------------------------------------


@gated_paid
@require_http_methods(["POST"])
def reopen_investigation(request, investigation_id):
    """
    Reopen a concluded/exported investigation, creating a new version.
    """
    try:
        inv = get_investigation(str(investigation_id), request.user)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)
    except PermissionError:
        return JsonResponse({"error": "Not a member of this investigation"}, status=403)

    try:
        new_inv = inv.reopen(request.user)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    logger.info(
        "investigation.reopened",
        extra={
            "old_id": str(inv.id),
            "new_id": str(new_inv.id),
            "version": new_inv.version,
        },
    )
    return JsonResponse({"investigation": _serialize_investigation(new_inv)}, status=201)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@gated_paid
@require_http_methods(["POST"])
def export_investigation_view(request, investigation_id):
    """
    Export a concluded investigation to a target project.

    POST body: {"target_project_id": uuid}
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    target_project_id = data.get("target_project_id")
    if not target_project_id:
        return JsonResponse({"error": "target_project_id is required"}, status=400)

    from qms_core.permissions import resolve_project

    _proj, err = resolve_project(request.user, target_project_id)
    if err:
        return err
    if not _proj:
        return JsonResponse({"error": "Target project not found"}, status=404)

    try:
        package = export_investigation(
            investigation_id=str(investigation_id),
            target_project_id=target_project_id,
            user=request.user,
        )
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)
    except PermissionError:
        return JsonResponse({"error": "Not a member of this investigation"}, status=403)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"package": package})


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


@gated_paid
@require_http_methods(["GET", "POST", "DELETE"])
def manage_members(request, investigation_id):
    """
    GET    — List investigation members.
    POST   — Add a member (owner only). Body: {"user_id": uuid, "role": str}
    DELETE — Remove a member (owner only). Body: {"user_id": uuid}
    """
    try:
        inv = get_investigation(str(investigation_id), request.user)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)
    except PermissionError:
        return JsonResponse({"error": "Not a member of this investigation"}, status=403)

    if request.method == "GET":
        members = InvestigationMembership.objects.filter(investigation=inv).select_related("user")
        return JsonResponse({"members": [_serialize_member(m) for m in members]})

    # POST and DELETE require owner
    if inv.owner != request.user:
        return JsonResponse({"error": "Only the owner can manage members"}, status=403)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user_id = data.get("user_id")
    if not user_id:
        return JsonResponse({"error": "user_id is required"}, status=400)

    from django.contrib.auth import get_user_model

    User = get_user_model()
    try:
        target_user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return JsonResponse({"error": "User not found"}, status=404)

    if request.method == "POST":
        role = data.get("role", Investigation.MemberRole.CONTRIBUTOR)
        membership, created = InvestigationMembership.objects.get_or_create(
            investigation=inv,
            user=target_user,
            defaults={"role": role},
        )
        if not created:
            membership.role = role
            membership.save(update_fields=["role"])
        return JsonResponse({"member": _serialize_member(membership)}, status=201 if created else 200)

    # DELETE
    deleted, _ = InvestigationMembership.objects.filter(investigation=inv, user=target_user).delete()
    if not deleted:
        return JsonResponse({"error": "User is not a member"}, status=404)
    return JsonResponse({"removed": True, "user_id": str(user_id)})


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@gated_paid
@require_http_methods(["GET"])
def get_graph(request, investigation_id):
    """
    Return the full Synara causal graph for an investigation.
    """
    try:
        inv = get_investigation(str(investigation_id), request.user)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)
    except PermissionError:
        return JsonResponse({"error": "Not a member of this investigation"}, status=403)

    synara = load_synara(inv)
    graph = synara.to_dict().get("graph", {})

    return JsonResponse(
        {
            "investigation_id": str(inv.id),
            "hypotheses": graph.get("hypotheses", {}),
            "links": graph.get("links", []),
            "evidence": graph.get("evidence", []),
        }
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@gated_paid
@require_http_methods(["GET"])
def list_tools(request, investigation_id):
    """
    List tool outputs linked to this investigation.
    """
    try:
        inv = get_investigation(str(investigation_id), request.user)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)
    except PermissionError:
        return JsonResponse({"error": "Not a member of this investigation"}, status=403)

    links = InvestigationToolLink.objects.filter(investigation=inv).select_related("content_type")
    return JsonResponse(
        {
            "investigation_id": str(inv.id),
            "tools": [_serialize_tool_link(link) for link in links],
        }
    )
