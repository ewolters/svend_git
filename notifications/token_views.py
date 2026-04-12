"""Notification token views — NTF-001 §5.2.4.

Unauthenticated one-click action endpoint for email notifications.
Token IS the credential — mirrors agents_api/token_views.py pattern.

Renders HTML pages (not JSON) because recipients arrive from email links.
"""

from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .tokens import NotificationToken

# ── Shared card styling (matches unsubscribe page in views.py) ────────────

_CARD_HEAD = (
    '<!DOCTYPE html><html><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
    "<title>{title} — Svend</title>"
    "<style>"
    "body{{margin:0;padding:60px 20px;background:#f4f7f4;"
    "font-family:'Inter',-apple-system,sans-serif;text-align:center;}}"
    ".card{{max-width:500px;margin:0 auto;background:#fff;border-radius:8px;"
    "padding:40px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}"
    "h2{{color:#1a2a1a;margin:0 0 12px;}}"
    "p{{color:#5a6a5a;line-height:1.6;}}"
    ".msg{{background:#f0f5f0;border-radius:6px;padding:16px;margin:16px 0;"
    "text-align:left;color:#333;}}"
    ".btn{{display:inline-block;padding:12px 24px;background:#4a9f6e;color:#fff;"
    "text-decoration:none;border-radius:6px;font-weight:600;border:none;"
    "cursor:pointer;font-size:14px;}}"
    ".btn:hover{{background:#3d8a5c;}}"
    ".meta{{font-size:12px;color:#7a8f7a;margin-top:8px;}}"
    "a{{color:#4a9f6e;}}"
    "</style></head><body><div class='card'>"
)
_CARD_TAIL = "</div></body></html>"


def _card(title, body, status=200):
    html = _CARD_HEAD.format(title=title) + body + _CARD_TAIL
    return HttpResponse(html, content_type="text/html", status=status)


# ── Entity URL routing (NTF-001 §4.3) ────────────────────────────────────

_ENTITY_URLS = {
    "ncr": "/app/qms/#ncr-{id}",
    "capa": "/app/qms/#capa-{id}",
    "document": "/app/qms/#doc-{id}",
    "review": "/app/qms/#review-{id}",
    "audit": "/app/qms/#audit-{id}",
    "training": "/app/qms/#training-{id}",
    "fmea": "/app/fmea/{id}/",
    "action": "/app/qms/#action-{id}",
    "signature": "/app/qms/#sig-{id}",
    "hoshin_project": "/app/hoshin/",
}


def _entity_link(notif):
    """Build a 'View in Svend' link if entity routing exists."""
    if notif.entity_type and notif.entity_id:
        pattern = _ENTITY_URLS.get(notif.entity_type)
        if pattern:
            url = "https://svend.ai" + pattern.format(id=notif.entity_id)
            return f'<p><a href="{url}" class="btn" style="background:#2a5a3e;">View in Svend</a></p>'
    return ""


# ── View ──────────────────────────────────────────────────────────────────


@csrf_exempt
@require_http_methods(["GET", "POST"])
def notification_token_view(request, token):
    """Render (GET) or execute (POST) a notification action token.

    No auth required — the token IS the credential (NTF-001 §5.2.4).
    Tokens are single-use, time-limited, and action-scoped.
    """
    tok = get_object_or_404(NotificationToken, token=token)

    if not tok.is_valid:
        reason = "already been used" if tok.used_at else "expired"
        return _card(
            "Token Expired",
            (
                f"<h2>This link has {reason}</h2>"
                "<p>Notification action links are single-use and expire after 72 hours.</p>"
                '<p><a href="https://svend.ai/app/">Go to Svend</a></p>'
            ),
            status=410,
        )

    notif = tok.notification

    if request.method == "GET":
        type_label = notif.get_notification_type_display()
        message_block = ""
        if notif.message:
            message_block = f'<div class="msg">{notif.message}</div>'

        return _card(
            notif.title,
            (
                f"<h2>{notif.title}</h2>"
                f'<p class="meta">{type_label}</p>'
                f"{message_block}"
                f"{_entity_link(notif)}"
                '<form method="post" style="margin-top:24px;">'
                '<button type="submit" class="btn">Acknowledge</button>'
                "</form>"
            ),
        )

    # POST — execute the scoped action and mark used
    tok.use()

    if tok.action_type == "acknowledge":
        if not notif.is_read:
            notif.is_read = True
            notif.save(update_fields=["is_read"])

    return _card(
        "Done",
        (
            "<h2>Acknowledged</h2>"
            f"<p>Notification &ldquo;{notif.title}&rdquo; has been marked as read.</p>"
            f"{_entity_link(notif)}"
            '<p><a href="https://svend.ai/app/">Back to Svend</a></p>'
        ),
    )
