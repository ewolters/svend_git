"""Notification views — NTF-001 §6.

Endpoints:
    GET  /api/notifications/                   — list (filterable, paginated)
    GET  /api/notifications/stream/            — SSE real-time stream
    GET  /api/notifications/unread-count/       — lightweight unread count
    POST /api/notifications/<uuid>/read/        — mark single as read
    POST /api/notifications/read-all/           — mark all as read
    GET/PUT /api/notifications/preferences/     — notification preferences
"""

import json
import time

from django.core.signing import BadSignature, Signer
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth

from .models import Notification, NotificationType

# Valid type values for preference validation
_VALID_TYPES = frozenset(t.value for t in NotificationType)


# ── SSE connection tracking (NTF-001 §9.3) ──────────────────────────────
_active_streams = {}  # user_id → count
_MAX_STREAMS_PER_USER = 2


# ── List ─────────────────────────────────────────────────────────────────


@csrf_exempt
@require_auth
@require_http_methods(["GET"])
def notification_list(request):
    """NTF-001 §6.2 — list notifications with filtering."""
    qs = Notification.objects.filter(recipient=request.user)

    # Filters
    unread = request.GET.get("unread")
    if unread and unread.lower() == "true":
        qs = qs.filter(is_read=False)

    ntype = request.GET.get("type")
    if ntype:
        qs = qs.filter(notification_type=ntype)

    entity_type = request.GET.get("entity_type")
    if entity_type:
        qs = qs.filter(entity_type=entity_type)

    entity_id = request.GET.get("entity_id")
    if entity_id:
        qs = qs.filter(entity_id=entity_id)

    # Pagination
    try:
        limit = min(int(request.GET.get("limit", 50)), 200)
    except (ValueError, TypeError):
        limit = 50
    try:
        offset = max(int(request.GET.get("offset", 0)), 0)
    except (ValueError, TypeError):
        offset = 0

    notifications = qs[offset : offset + limit]
    return JsonResponse([n.to_dict() for n in notifications], safe=False)


# ── SSE Stream ───────────────────────────────────────────────────────────


@csrf_exempt
@require_auth
@require_http_methods(["GET"])
def notification_stream(request):
    """NTF-001 §5.1 — SSE endpoint with DB polling."""
    user = request.user
    uid = user.id

    # Rate limit SSE connections per user (NTF-001 §9.3)
    current = _active_streams.get(uid, 0)
    if current >= _MAX_STREAMS_PER_USER:
        return JsonResponse(
            {"error": "Too many active streams. Close existing tabs."},
            status=429,
        )

    def event_stream():
        _active_streams[uid] = _active_streams.get(uid, 0) + 1
        try:
            from django.utils import timezone

            last_check = timezone.now()
            # Check for Last-Event-ID header for reconnection
            last_event_id = request.META.get("HTTP_LAST_EVENT_ID")
            if last_event_id:
                try:
                    from django.utils.dateparse import parse_datetime

                    parsed = parse_datetime(last_event_id)
                    if parsed:
                        last_check = parsed
                except (ValueError, TypeError):
                    pass

            start_time = time.monotonic()
            max_duration = 110  # seconds — under gunicorn's 120s timeout
            ping_interval = 15  # seconds
            poll_interval = 3  # seconds
            last_ping = time.monotonic()

            while time.monotonic() - start_time < max_duration:
                # Check for new notifications
                new_notifs = list(
                    Notification.objects.filter(
                        recipient_id=uid,
                        created_at__gt=last_check,
                    ).order_by("created_at")[:20]
                )

                for notif in new_notifs:
                    data = json.dumps(notif.to_dict())
                    ts = notif.created_at.isoformat()
                    yield f"event: notification\nid: {ts}\ndata: {data}\n\n"
                    last_check = notif.created_at

                if not new_notifs:
                    last_check = timezone.now()

                # Keepalive ping
                now = time.monotonic()
                if now - last_ping >= ping_interval:
                    yield "event: ping\ndata: {}\n\n"
                    last_ping = now

                time.sleep(poll_interval)

        finally:
            _active_streams[uid] = max(_active_streams.get(uid, 1) - 1, 0)

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


# ── Unread Count ─────────────────────────────────────────────────────────


@csrf_exempt
@require_auth
@require_http_methods(["GET"])
def notification_unread_count(request):
    """NTF-001 §6.3 — lightweight unread count."""
    count = Notification.objects.filter(recipient=request.user, is_read=False).count()
    return JsonResponse({"count": count})


# ── Mark Read ────────────────────────────────────────────────────────────


@csrf_exempt
@require_auth
@require_http_methods(["POST"])
def notification_mark_read(request, notification_id):
    """NTF-001 §6.4 — mark single notification as read."""
    try:
        notif = Notification.objects.get(id=notification_id, recipient=request.user)
    except Notification.DoesNotExist:
        return JsonResponse({"error": "Notification not found"}, status=404)

    if not notif.is_read:
        notif.is_read = True
        notif.save(update_fields=["is_read"])

    return JsonResponse({"ok": True})


@csrf_exempt
@require_auth
@require_http_methods(["POST"])
def notification_mark_all_read(request):
    """NTF-001 §6.4 — mark all notifications as read."""
    updated = Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
    return JsonResponse({"updated": updated})


# ── Preferences ──────────────────────────────────────────────────────────


@csrf_exempt
@require_auth
@require_http_methods(["GET", "PUT"])
def notification_preferences(request):
    """NTF-001 §6.5 — get or update notification preferences."""
    user = request.user

    if request.method == "GET":
        prefs = (getattr(user, "preferences", None) or {}).get("notifications", {})
        return JsonResponse(
            {
                "muted_types": prefs.get("muted_types", []),
                "email_enabled": prefs.get("email_enabled", False),
                "email_mode": prefs.get("email_mode", "immediate"),
            }
        )

    # PUT
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    muted = data.get("muted_types", [])
    if not isinstance(muted, list):
        return JsonResponse({"error": "muted_types must be a list"}, status=400)

    # Validate type keys
    invalid = [t for t in muted if t not in _VALID_TYPES]
    if invalid:
        return JsonResponse(
            {"error": f"Invalid notification types: {', '.join(invalid)}"},
            status=400,
        )

    # Update preferences
    if not user.preferences:
        user.preferences = {}
    user.preferences.setdefault("notifications", {})
    user.preferences["notifications"]["muted_types"] = muted

    if "email_enabled" in data:
        user.preferences["notifications"]["email_enabled"] = bool(data["email_enabled"])

    if "email_mode" in data:
        from .email import VALID_EMAIL_MODES

        mode = data["email_mode"]
        if mode not in VALID_EMAIL_MODES:
            return JsonResponse(
                {"error": f"Invalid email_mode. Must be one of: {', '.join(sorted(VALID_EMAIL_MODES))}"},
                status=400,
            )
        user.preferences["notifications"]["email_mode"] = mode

    user.save(update_fields=["preferences"])

    return JsonResponse(
        {
            "muted_types": user.preferences["notifications"].get("muted_types", []),
            "email_enabled": user.preferences["notifications"].get("email_enabled", False),
            "email_mode": user.preferences["notifications"].get("email_mode", "immediate"),
        }
    )


# ── Per-Type Unsubscribe ────────────────────────────────────────────────


@csrf_exempt
@require_http_methods(["GET"])
def notification_type_unsubscribe(request):
    """NTF-001 §5.2.7 — per-type unsubscribe via signed URL.

    No auth required — the signed token IS the credential.
    """
    from django.contrib.auth import get_user_model

    token = request.GET.get("token", "")
    if not token:
        return JsonResponse({"error": "Missing token"}, status=400)

    signer = Signer(salt="ntf-type-unsub")
    try:
        payload = signer.unsign(token)
    except BadSignature:
        return JsonResponse({"error": "Invalid or expired token"}, status=400)

    try:
        user_id, notification_type = payload.split(":", 1)
    except ValueError:
        return JsonResponse({"error": "Malformed token"}, status=400)

    if notification_type not in _VALID_TYPES:
        return JsonResponse({"error": "Invalid notification type"}, status=400)

    User = get_user_model()
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return JsonResponse({"error": "User not found"}, status=404)

    if not user.preferences:
        user.preferences = {}
    user.preferences.setdefault("notifications", {})
    muted = user.preferences["notifications"].setdefault("muted_types", [])
    if notification_type not in muted:
        muted.append(notification_type)
    user.save(update_fields=["preferences"])

    return JsonResponse({"ok": True, "muted_type": notification_type})
