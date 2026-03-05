"""Notification email builders — NTF-001 §5.2.

Builds single and digest notification emails using the shared EMAIL_TEMPLATE.
"""

import logging

from django.core.mail import send_mail as django_send_mail

from api.internal_views import EMAIL_TEMPLATE
from api.views import make_unsubscribe_url

logger = logging.getLogger(__name__)

# NTF-001 §5.2.2 — valid email delivery modes
VALID_EMAIL_MODES = frozenset({"immediate", "daily", "weekly"})


def _notification_action_url(token_value):
    """Build the full URL for a notification action token."""
    return f"https://svend.ai/ntf/{token_value}/"


def _notification_type_unsub_url(user, notification_type):
    """Build a signed URL for per-type unsubscribe."""
    from django.core.signing import Signer
    signer = Signer(salt="ntf-type-unsub")
    signed = signer.sign(f"{user.id}:{notification_type}")
    return f"https://svend.ai/api/notifications/unsubscribe/?token={signed}"


def build_notification_email(notification, token):
    """Build a single notification email body (HTML).

    Returns (subject, body_html) tuple.
    """
    subject = f"[Svend] {notification.title}"

    action_url = _notification_action_url(token.token)
    type_unsub_url = _notification_type_unsub_url(
        token.user, notification.notification_type
    )

    body_parts = [
        f"<h2 style='margin:0 0 16px;font-size:18px;color:#1a2a1a;'>{notification.title}</h2>",
    ]

    if notification.message:
        body_parts.append(
            f"<p style='margin:8px 0;color:#333;'>{notification.message}</p>"
        )

    # Action button
    body_parts.append(
        f"<p style='margin:24px 0;'>"
        f"<a href=\"{action_url}\" style=\"display:inline-block;padding:12px 24px;"
        f"background:#4a9f6e;color:#fff;text-decoration:none;border-radius:6px;"
        f"font-weight:600;\">Acknowledge</a></p>"
    )

    # Per-type unsubscribe
    body_parts.append(
        f"<p style='margin:16px 0 0;font-size:12px;color:#7a8f7a;'>"
        f"<a href=\"{type_unsub_url}\" style=\"color:#7a8f7a;\">"
        f"Stop receiving {notification.get_notification_type_display()} emails</a></p>"
    )

    body_html = "\n".join(body_parts)
    return subject, body_html


def build_digest_email(user, notifications_with_tokens, period):
    """Build a digest email for multiple notifications.

    Args:
        user: User instance.
        notifications_with_tokens: List of (notification, token) tuples.
        period: "daily" or "weekly".

    Returns:
        (subject, body_html) tuple.
    """
    count = len(notifications_with_tokens)
    subject = f"[Svend] {period.title()} digest — {count} notification{'s' if count != 1 else ''}"

    body_parts = [
        f"<h2 style='margin:0 0 16px;font-size:18px;color:#1a2a1a;'>"
        f"Your {period} notification summary</h2>",
        f"<p style='margin:8px 0;color:#555;'>"
        f"You have {count} unread notification{'s' if count != 1 else ''}.</p>",
    ]

    for notif, token in notifications_with_tokens[:20]:  # Cap at 20 in digest
        action_url = _notification_action_url(token.token)
        body_parts.append(
            f"<div style='padding:12px 0;border-bottom:1px solid #e8efe8;'>"
            f"<strong style='color:#1a2a1a;'>{notif.title}</strong>"
        )
        if notif.message:
            body_parts.append(
                f"<br><span style='color:#555;font-size:13px;'>{notif.message}</span>"
            )
        body_parts.append(
            f"<br><a href=\"{action_url}\" style=\"color:#4a9f6e;font-size:13px;\">Acknowledge</a>"
            f"</div>"
        )

    body_parts.append(
        "<p style='margin:16px 0;'>"
        "<a href=\"https://svend.ai/app/\" style=\"color:#4a9f6e;\">View all in Svend</a></p>"
    )

    body_html = "\n".join(body_parts)
    return subject, body_html


def send_notification_email(user, subject, body_html, notification_type=None):
    """Wrap body in EMAIL_TEMPLATE, add unsubscribe link, and send.

    Args:
        user: Recipient user instance.
        subject: Email subject line.
        body_html: Inner HTML content (will be wrapped in template).
        notification_type: Optional — for per-type unsubscribe link.

    Returns:
        True if sent, False on failure.
    """
    unsub_url = make_unsubscribe_url(user)
    full_html = EMAIL_TEMPLATE.format(body=body_html, unsub_url=unsub_url)

    try:
        django_send_mail(
            subject=subject,
            message="",
            from_email=None,  # Uses DEFAULT_FROM_EMAIL
            recipient_list=[user.email],
            html_message=full_html,
        )
        return True
    except Exception:
        logger.exception("Failed to send notification email to %s", user.email)
        return False
