"""ActionToken views — unauthenticated, token-is-credential.

Renders HTML pages for email click-through (commitment confirm/decline).
Falls back to JSON for Accept: application/json requests.

Standard: QMS-002 §2.3, SEC-001
"""

import logging

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import ActionToken

logger = logging.getLogger(__name__)

# ── Shared card styling (matches notifications/token_views.py) ────────────

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
    ".details{{background:#f0f5f0;border-radius:6px;padding:16px;margin:16px 0;"
    "text-align:left;color:#333;}}"
    ".details p{{margin:4px 0;}}"
    ".btn{{display:inline-block;padding:12px 24px;background:#4a9f6e;color:#fff;"
    "text-decoration:none;border-radius:6px;font-weight:600;border:none;"
    "cursor:pointer;font-size:14px;margin:4px;}}"
    ".btn:hover{{background:#3d8a5c;}}"
    ".btn-decline{{background:#6b7280;}}"
    ".btn-decline:hover{{background:#4b5563;}}"
    "a{{color:#4a9f6e;}}"
    "</style></head><body><div class='card'>"
)
_CARD_TAIL = "</div></body></html>"

_COMMITMENT_ACTIONS = {"confirm_availability", "decline"}


def _card(title, body, status=200):
    html = _CARD_HEAD.format(title=title) + body + _CARD_TAIL
    return HttpResponse(html, content_type="text/html", status=status)


def _wants_json(request):
    accept = request.META.get("HTTP_ACCEPT", "")
    return "application/json" in accept and "text/html" not in accept


def _load_commitment(tok):
    """Load the ResourceCommitment from token's scoped_to, or None."""
    commitment_id = (tok.scoped_to or {}).get("commitment_id")
    if not commitment_id:
        return None
    try:
        from .models import ResourceCommitment

        return ResourceCommitment.objects.select_related(
            "employee", "project__project", "requested_by"
        ).get(id=commitment_id)
    except ResourceCommitment.DoesNotExist:
        return None


def _commitment_details_html(commitment):
    """Render commitment details as an HTML block."""
    return (
        '<div class="details">'
        f"<p><strong>Project:</strong> {commitment.project.project.title}</p>"
        f"<p><strong>Role:</strong> {commitment.get_role_display()}</p>"
        f"<p><strong>Dates:</strong> {commitment.start_date} to {commitment.end_date}</p>"
        f"<p><strong>Hours/day:</strong> {commitment.hours_per_day}</p>"
        "</div>"
    )


@csrf_exempt
@require_http_methods(["GET", "POST"])
def action_token_view(request, token):
    """Render (GET) or execute (POST) an action token.

    No auth required — the token IS the credential.
    Tokens are single-use, time-limited, and action-scoped.

    Commitment tokens (confirm_availability, decline) render HTML cards.
    Other tokens return JSON for backward compatibility.
    """
    tok = get_object_or_404(ActionToken, token=token)

    is_commitment = tok.action_type in _COMMITMENT_ACTIONS

    if not tok.is_valid:
        if is_commitment and not _wants_json(request):
            reason = "already been used" if tok.used_at else "expired"
            return _card(
                "Link Expired",
                (
                    f"<h2>This link has {reason}</h2>"
                    "<p>Action links are single-use and expire after 72 hours.</p>"
                    '<p><a href="https://svend.ai">Go to Svend</a></p>'
                ),
                status=410,
            )
        return JsonResponse({"error": "Token expired or already used"}, status=410)

    if request.method == "GET":
        if is_commitment and not _wants_json(request):
            return _get_commitment_card(tok)
        return JsonResponse(tok.to_dict())

    # POST
    if is_commitment and not _wants_json(request):
        return _post_commitment_action(tok)

    # Legacy JSON path
    tok.use()
    return JsonResponse({"ok": True, "action": tok.action_type})


def _get_commitment_card(tok):
    """Render HTML card for commitment confirm/decline."""
    commitment = _load_commitment(tok)
    if not commitment:
        return _card("Not Found", "<h2>Commitment not found</h2>", status=404)

    if tok.action_type == "confirm_availability":
        heading = "Confirm Resource Commitment"
        btn_class = "btn"
        btn_label = "Confirm Participation"
    else:
        heading = "Decline Resource Commitment"
        btn_class = "btn btn-decline"
        btn_label = "Decline"

    return _card(
        heading,
        (
            f"<h2>{heading}</h2>"
            f"<p>Hi {commitment.employee.name},</p>"
            f"{_commitment_details_html(commitment)}"
            '<form method="post" style="margin-top:24px;">'
            f'<button type="submit" class="{btn_class}">{btn_label}</button>'
            "</form>"
        ),
    )


def _post_commitment_action(tok):
    """Execute commitment confirm/decline and render success card."""
    commitment = _load_commitment(tok)
    if not commitment:
        tok.use()
        return _card("Not Found", "<h2>Commitment not found</h2>", status=404)

    # Validate transition
    if commitment.status != "requested":
        tok.use()
        return _card(
            "Already Actioned",
            (
                "<h2>This commitment has already been actioned</h2>"
                f"<p>Current status: <strong>{commitment.get_status_display()}</strong></p>"
                '<p><a href="https://svend.ai">Go to Svend</a></p>'
            ),
            status=409,
        )

    # Transition
    new_status = (
        "confirmed" if tok.action_type == "confirm_availability" else "declined"
    )
    commitment.status = new_status
    commitment.save(update_fields=["status", "updated_at"])
    tok.use()

    # Notify the requester
    try:
        from agents_api.commitment_notifications import notify_commitment_response

        notify_commitment_response(commitment, old_status="requested")
    except Exception:
        logger.exception("Failed to notify requester for commitment %s", commitment.id)

    if new_status == "confirmed":
        return _card(
            "Confirmed",
            (
                "<h2>Commitment Confirmed</h2>"
                f"<p>Thank you, {commitment.employee.name}! "
                f"You're confirmed for this project.</p>"
                f"{_commitment_details_html(commitment)}"
                '<p><a href="https://svend.ai">Go to Svend</a></p>'
            ),
        )
    else:
        return _card(
            "Declined",
            (
                "<h2>Commitment Declined</h2>"
                "<p>The project coordinator has been notified.</p>"
                f"{_commitment_details_html(commitment)}"
                '<p><a href="https://svend.ai">Go to Svend</a></p>'
            ),
        )
