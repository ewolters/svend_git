"""Notification email task handlers — NTF-001 §5.2.5.

Registered with syn.sched in svend_tasks.py.
"""

import logging
from datetime import timedelta

from django.utils import timezone

logger = logging.getLogger(__name__)

# NTF-001 §5.2.5 — rate limit: max 20 emails per user per hour
_RATE_LIMIT_PER_HOUR = 20
# Max users per digest batch run
_DIGEST_BATCH_CAP = 100


def send_notification_email_task(payload, context=None):
    """Send a single notification email (immediate mode).

    Expected payload args: notification_id, token_id
    """
    from notifications.email import build_notification_email, send_notification_email
    from notifications.tokens import NotificationToken

    args = (
        payload.get("args", {})
        if isinstance(payload, dict)
        else getattr(payload, "args", {}) or {}
    )
    notification_id = args.get("notification_id")
    token_id = args.get("token_id")

    if not notification_id or not token_id:
        return {"error": "Missing notification_id or token_id"}

    try:
        tok = NotificationToken.objects.select_related("notification", "user").get(
            id=token_id
        )
    except NotificationToken.DoesNotExist:
        return {"error": f"Token {token_id} not found"}

    # Rate limit check
    one_hour_ago = timezone.now() - timedelta(hours=1)
    recent_count = NotificationToken.objects.filter(
        user=tok.user,
        email_sent_at__gte=one_hour_ago,
    ).count()
    if recent_count >= _RATE_LIMIT_PER_HOUR:
        logger.warning("Rate limit hit for user %s — skipping email", tok.user.email)
        return {"skipped": True, "reason": "rate_limit"}

    subject, body_html = build_notification_email(tok.notification, tok)
    sent = send_notification_email(
        tok.user, subject, body_html, tok.notification.notification_type
    )

    if sent:
        tok.email_sent_at = timezone.now()
        tok.save(update_fields=["email_sent_at"])
        return {"sent": True, "notification_id": str(notification_id)}
    else:
        tok.email_failed_at = timezone.now()
        tok.save(update_fields=["email_failed_at"])
        return {"sent": False, "notification_id": str(notification_id)}


def send_daily_digest(payload, context=None):
    """Send daily digest emails to users with email_mode='daily'.

    Runs daily at 08:00 UTC via cron.
    """
    return _send_digest("daily", timedelta(hours=24))


def send_weekly_digest(payload, context=None):
    """Send weekly digest emails to users with email_mode='weekly'.

    Runs weekly Monday 08:00 UTC via cron.
    """
    return _send_digest("weekly", timedelta(days=7))


def _send_digest(period, lookback):
    """Shared digest logic for daily/weekly."""
    from django.contrib.auth import get_user_model

    from notifications.email import build_digest_email, send_notification_email
    from notifications.models import Notification
    from notifications.tokens import NotificationToken

    User = get_user_model()
    cutoff = timezone.now() - lookback

    # Find users with email enabled and matching mode
    users = User.objects.filter(
        is_active=True,
        is_email_opted_out=False,
    ).exclude(
        email=""
    )[:_DIGEST_BATCH_CAP]

    sent_count = 0
    for user in users:
        prefs = (getattr(user, "preferences", None) or {}).get("notifications", {})
        if not prefs.get("email_enabled", False):
            continue
        if prefs.get("email_mode", "immediate") != period:
            continue

        muted = set(prefs.get("muted_types", []))

        # Fetch unread notifications since cutoff
        unread = list(
            Notification.objects.filter(
                recipient=user,
                is_read=False,
                created_at__gte=cutoff,
            )
            .exclude(notification_type__in=muted)
            .order_by("-created_at")[:20]
        )

        if not unread:
            continue

        # Create tokens for each notification
        pairs = []
        for notif in unread:
            tok = NotificationToken.objects.create(
                user=user,
                notification=notif,
                action_type="acknowledge",
            )
            pairs.append((notif, tok))

        subject, body_html = build_digest_email(user, pairs, period)
        sent = send_notification_email(user, subject, body_html)

        if sent:
            now = timezone.now()
            for _, tok in pairs:
                tok.email_sent_at = now
                tok.save(update_fields=["email_sent_at"])
            sent_count += 1

    return {"period": period, "users_emailed": sent_count}


def cleanup_expired_tokens(payload=None, context=None):
    """Delete expired, unused notification tokens older than 7 days.

    Runs weekly Sunday 04:00 UTC.
    """
    from notifications.tokens import NotificationToken

    cutoff = timezone.now() - timedelta(days=7)
    deleted, _ = NotificationToken.objects.filter(
        expires_at__lt=cutoff,
        used_at__isnull=True,
    ).delete()
    return {"deleted": deleted}
