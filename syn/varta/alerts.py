"""
Security alert delivery.

Sends email alerts for high-severity security events using
Django's existing email configuration (Gmail/Resend).
Throttled to prevent alert storms.
"""

import logging
import time
from collections import defaultdict

from notifications.email_service import email_service

logger = logging.getLogger("syn.varta")

# ── Alert throttling ─────────────────────────────────────────────────
# Prevent alert storms: max 1 email per alert type per cooldown period
_alert_timestamps: dict[str, float] = defaultdict(float)
ALERT_COOLDOWN = 900  # 15 minutes between same-type alerts
DAILY_ALERT_LIMIT = 20
_daily_count = 0
_daily_reset = time.time()


def _check_throttle(alert_type: str) -> bool:
    """Return True if alert should be sent (not throttled)."""
    global _daily_count, _daily_reset

    now = time.time()

    # Reset daily counter
    if now - _daily_reset > 86400:
        _daily_count = 0
        _daily_reset = now

    if _daily_count >= DAILY_ALERT_LIMIT:
        return False

    last_sent = _alert_timestamps.get(alert_type, 0)
    if now - last_sent < ALERT_COOLDOWN:
        return False

    return True


def send_security_alert(
    alert_type: str,
    subject: str,
    body: str,
    severity: str = "HIGH",
):
    """
    Send a security alert email.

    Args:
        alert_type: Dedup key (e.g., "honeypot_hit", "brute_force")
        subject: Email subject
        body: Alert details
        severity: CRITICAL, HIGH, MEDIUM, LOW
    """
    global _daily_count

    if not _check_throttle(alert_type):
        logger.debug("Alert throttled: %s", alert_type)
        return

    recipient = "eric.wolters@svend.ai"
    full_subject = f"[ВАРТА {severity}] {subject}"

    full_body = (
        f"Severity: {severity}\n"
        f"Alert: {alert_type}\n"
        f"Server: kjerne\n"
        f"─────────────────────────\n\n"
        f"{body}\n\n"
        f"─────────────────────────\n"
        f"Варта Active Defense • svend.ai\n"
    )

    result = email_service.send(
        to=recipient,
        subject=full_subject,
        body_text=full_body,
        wrap_template=False,
    )
    if result.sent:
        _alert_timestamps[alert_type] = time.time()
        _daily_count += 1
        logger.info("Security alert sent: %s [%s]", alert_type, severity)
    else:
        logger.warning("Failed to send security alert: %s — %s", alert_type, result.error)
