"""ForgeRack session CRUD — save/load rack configurations."""

import json

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth
from agents_api.models import RackSession

FREE_SESSION_LIMIT = 3


def _rack_queryset(user):
    """Get rack sessions accessible to the user."""
    from django.db.models import Q

    qs = RackSession.objects.filter(Q(user=user))
    if hasattr(user, "tenant") and user.tenant:
        qs = RackSession.objects.filter(Q(user=user) | Q(tenant=user.tenant))
    return qs


def _can_edit(user, session):
    if session.user_id == user.id:
        return True
    if session.tenant and hasattr(user, "tenant") and session.tenant == user.tenant:
        return True
    return False


def _check_limit(user):
    """Free users limited to FREE_SESSION_LIMIT sessions."""
    sub = getattr(user, "subscription", None)
    if sub and getattr(sub, "tier", "free") != "free":
        return True  # Paid — no limit
    count = RackSession.objects.filter(user=user).count()
    return count < FREE_SESSION_LIMIT


@require_auth
@require_http_methods(["GET"])
def list_rack_sessions(request):
    sessions = _rack_queryset(request.user).values("id", "title", "session_type", "status", "updated_at")
    return JsonResponse(
        [
            {
                "id": str(s["id"]),
                "title": s["title"],
                "session_type": s["session_type"],
                "status": s["status"],
                "updated_at": s["updated_at"].isoformat() if s["updated_at"] else None,
            }
            for s in sessions
        ],
        safe=False,
    )


@require_auth
@require_http_methods(["POST"])
def create_rack_session(request):
    if not _check_limit(request.user):
        return JsonResponse(
            {"error": f"Free tier limited to {FREE_SESSION_LIMIT} saved sessions"},
            status=403,
        )

    data = json.loads(request.body)
    session = RackSession.objects.create(
        title=data.get("title", "Untitled Rack"),
        description=data.get("description", ""),
        session_type=data.get("session_type", RackSession.SessionType.SANDBOX),
        state=data.get("state", {}),
        user=request.user,
        tenant=getattr(request.user, "tenant", None),
    )
    return JsonResponse(session.to_dict(), status=201)


@require_auth
@require_http_methods(["GET"])
def get_rack_session(request, session_id):
    session = get_object_or_404(RackSession, id=session_id)
    if not _can_edit(request.user, session):
        return JsonResponse({"error": "Not authorized"}, status=403)
    return JsonResponse(session.to_dict())


@require_auth
@require_http_methods(["POST", "PUT"])
def update_rack_session(request, session_id):
    session = get_object_or_404(RackSession, id=session_id)
    if not _can_edit(request.user, session):
        return JsonResponse({"error": "Not authorized"}, status=403)

    data = json.loads(request.body)
    if "title" in data:
        session.title = data["title"]
    if "description" in data:
        session.description = data["description"]
    if "state" in data:
        session.state = data["state"]
    if "status" in data:
        session.status = data["status"]
    if "session_type" in data:
        session.session_type = data["session_type"]
    session.save()
    return JsonResponse(session.to_dict())


@require_auth
@require_http_methods(["DELETE"])
def delete_rack_session(request, session_id):
    session = get_object_or_404(RackSession, id=session_id)
    if not _can_edit(request.user, session):
        return JsonResponse({"error": "Not authorized"}, status=403)
    session.delete()
    return JsonResponse({"status": "deleted"})
