"""Notification token views — NTF-001 §5.2.4.

Unauthenticated one-click action endpoint for email notifications.
Token IS the credential — mirrors agents_api/token_views.py pattern.
"""

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .tokens import NotificationToken


@csrf_exempt
@require_http_methods(["GET", "POST"])
def notification_token_view(request, token):
    """Render (GET) or execute (POST) a notification action token.

    No auth required — the token IS the credential (NTF-001 §5.2.3).
    Tokens are single-use, time-limited, and action-scoped.
    """
    tok = get_object_or_404(NotificationToken, token=token)

    if not tok.is_valid:
        return JsonResponse(
            {"error": "Token expired or already used"},
            status=410,
        )

    if request.method == "GET":
        notif = tok.notification
        return JsonResponse(
            {
                "token": tok.to_dict(),
                "notification": notif.to_dict(),
            }
        )

    # POST — execute the scoped action and mark used
    tok.use()

    # If action is "acknowledge", also mark notification as read
    if tok.action_type == "acknowledge":
        notif = tok.notification
        if not notif.is_read:
            notif.is_read = True
            notif.save(update_fields=["is_read"])

    return JsonResponse({"ok": True, "action": tok.action_type})
