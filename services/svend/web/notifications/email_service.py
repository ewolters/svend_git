"""Centralized email service — the ONLY path for sending email from Svend.

⚠ MANDATORY: All email sending MUST go through this service. Direct use of
django.core.mail.send_mail() is prohibited outside this file. A pre-commit
hook enforces this.

This follows the same singleton pattern as agents_api/llm_service.py —
one entry point, one place to add rate limiting, deliverability tracking,
provider switching, or audit logging.

Usage:
    from notifications.email_service import email_service

    # HTML email with branded template
    result = email_service.send(
        to="user@example.com",
        subject="Your report is ready",
        body_html="<p>Click here to view...</p>",
    )

    # Plain text email (verification, transactional)
    result = email_service.send(
        to="user@example.com",
        subject="Verify your account",
        body_text="Click this link: ...",
    )

    # With unsubscribe URL (marketing, notifications)
    result = email_service.send(
        to=user.email,
        subject="Weekly digest",
        body_html=digest_html,
        unsubscribe_url=email_service.make_unsubscribe_url(user),
    )

    if result.sent:
        ...
    else:
        logger.error(result.error)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Email template (branded wrapper)
# ---------------------------------------------------------------------------

EMAIL_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f7f4;font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f7f4;padding:40px 20px;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;">
<tr><td style="background:#1a2a1a;padding:24px 32px;">
  <span style="color:#4a9f6e;font-size:20px;font-weight:600;letter-spacing:1px;">SVEND</span>
</td></tr>
<tr><td style="padding:32px;color:#1a1a1a;font-size:15px;line-height:1.7;">
  {body}
</td></tr>
<tr><td style="padding:16px 32px 24px;border-top:1px solid #e8efe8;color:#7a8f7a;font-size:12px;">
  SVEND &middot; Decision Science Workbench &middot; <a href="https://svend.ai" style="color:#4a9f6e;">svend.ai</a>
  {unsub_line}
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class EmailResult:
    """Result of an email send attempt."""

    sent: bool
    error: str = ""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class EmailService:
    """Centralized email sending service.

    All email in Svend routes through this class. Provides:
    - Branded HTML template wrapping
    - Plain text fallback
    - Consistent error handling (never raises, returns EmailResult)
    - Single point for future: rate limiting, provider switching, tracking
    """

    def send(
        self,
        to,
        subject,
        body_html=None,
        body_text=None,
        unsubscribe_url=None,
        wrap_template=True,
        from_email=None,
    ):
        """Send an email.

        Args:
            to: Recipient email address (string).
            subject: Email subject line.
            body_html: HTML body content. If wrap_template=True, wrapped in branded template.
            body_text: Plain text body. Used as fallback or as primary if no body_html.
            unsubscribe_url: Optional unsubscribe link for the template footer.
            wrap_template: If True (default), wrap body_html in EMAIL_TEMPLATE.
            from_email: Sender address. None uses DEFAULT_FROM_EMAIL.

        Returns:
            EmailResult with sent=True on success, sent=False with error on failure.
        """
        from django.core.mail import send_mail as _django_send_mail

        if not to:
            return EmailResult(sent=False, error="No recipient email")

        if not body_html and not body_text:
            return EmailResult(sent=False, error="No email body provided")

        # Build HTML message
        html_message = None
        if body_html:
            if wrap_template:
                unsub_line = ""
                if unsubscribe_url:
                    unsub_line = f'<br><a href="{unsubscribe_url}" style="color:#7a8f7a;">Unsubscribe</a>'
                html_message = EMAIL_TEMPLATE.format(body=body_html, unsub_line=unsub_line)
            else:
                html_message = body_html

        # Plain text fallback
        message = body_text or ""

        try:
            _django_send_mail(
                subject=subject,
                message=message,
                from_email=from_email,
                recipient_list=[to],
                html_message=html_message,
            )
            return EmailResult(sent=True)
        except Exception as e:
            logger.exception("Email send failed: to=%s subject=%s", to, subject[:50])
            return EmailResult(sent=False, error=str(e))

    @staticmethod
    def make_unsubscribe_url(user):
        """Generate a signed unsubscribe URL for a user."""
        from django.core.signing import Signer

        signer = Signer(salt="email-unsubscribe")
        token = signer.sign(str(user.id))
        return f"https://svend.ai/api/email/unsubscribe/?token={token}"


# Module-level singleton — import this
email_service = EmailService()
