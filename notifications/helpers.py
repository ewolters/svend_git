"""Notification helper — NTF-001 §7.1.

All notification creation MUST go through notify().
"""

import logging

from .models import Notification, NotificationType

logger = logging.getLogger(__name__)

# Valid type values for fast lookup
_VALID_TYPES = frozenset(t.value for t in NotificationType)


def notify(recipient, notification_type, title, message="", entity_type="", entity_id=None):
    """Create a notification, respecting user preferences.

    Args:
        recipient: User instance.
        notification_type: NotificationType value (string or enum member).
        title: Short headline (max 300 chars).
        message: Optional extended body.
        entity_type: Source record type key (e.g., "ncr", "capa").
        entity_id: Source record UUID.

    Returns:
        Notification instance, or None if the type is muted by user preferences.
    """
    # Resolve enum member to string value
    type_val = notification_type.value if hasattr(notification_type, "value") else notification_type

    if type_val not in _VALID_TYPES:
        raise ValueError(f"Invalid notification type: {type_val}")

    # Check user preferences for muted types (NTF-001 §7.1)
    prefs = getattr(recipient, "preferences", None) or {}
    notif_prefs = prefs.get("notifications", {})
    muted = notif_prefs.get("muted_types", [])
    if type_val in muted:
        return None

    notification = Notification.objects.create(
        recipient=recipient,
        notification_type=type_val,
        title=title[:300],
        message=message,
        entity_type=entity_type,
        entity_id=entity_id,
    )

    # NTF-001 §5.2 — maybe schedule email delivery
    _maybe_schedule_email(recipient, notification)

    return notification


def _maybe_schedule_email(recipient, notification):
    """Schedule an email notification if the user has opted in.

    NTF-001 §5.2.2 — checks email_enabled, is_email_opted_out, and email_mode.
    Never blocks notification creation — all errors are caught and logged.
    """
    try:
        # Global opt-out check (CAN-SPAM)
        if getattr(recipient, "is_email_opted_out", False):
            return

        prefs = (getattr(recipient, "preferences", None) or {}).get("notifications", {})
        if not prefs.get("email_enabled", False):
            return

        email_mode = prefs.get("email_mode", "immediate")
        if email_mode in ("daily", "weekly"):
            # Cron handles digest modes
            return

        # Immediate mode — create token and schedule task
        from syn.sched.scheduler import schedule_task

        from .tokens import NotificationToken

        tok = NotificationToken.objects.create(
            user=recipient,
            notification=notification,
            action_type="acknowledge",
        )

        schedule_task(
            name=f"ntf-email-{notification.id}",
            func="notifications.tasks.send_notification_email_task",
            args={
                "notification_id": str(notification.id),
                "token_id": str(tok.id),
            },
            delay_seconds=5,
            queue="core",
        )
    except Exception:
        logger.exception("Failed to schedule notification email for %s", notification.id)
