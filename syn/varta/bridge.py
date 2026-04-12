"""
fail2ban bridge — writes structured security events to a log file
that fail2ban watches for automatic IP banning.

Log format (one line per event):
  [VARTA] ip=1.2.3.4 action=honeypot score=10 detail=wp_admin

fail2ban filter matches the [VARTA] prefix and extracts the IP.
"""

import logging

# Dedicated logger that writes to /var/log/svend/varta.log
# Configured in Django settings.LOGGING
varta_logger = logging.getLogger("syn.varta.actions")


def log_security_action(
    ip: str,
    action: str,
    detail: str = "",
    score: int = 0,
):
    """
    Write a security event for fail2ban to consume.

    Args:
        ip: Client IP address
        action: Event type (honeypot, injection, traversal, brute_force, anomaly)
        detail: Human-readable description
        score: Threat score (fail2ban filter can use threshold)
    """
    # Single-line format for fail2ban regex parsing
    msg = f"[VARTA] ip={ip} action={action} score={score} detail={detail}"
    varta_logger.warning(msg)
