"""Commitment email task handlers for non-user employees.

Sends branded HTML emails with one-click Confirm/Decline ActionToken links.
Registered with syn.sched in svend_tasks.py.
"""

import logging

from django.core.mail import send_mail as django_send_mail

logger = logging.getLogger(__name__)


def build_commitment_request_email(commitment, confirm_token, decline_token):
    """Build HTML email body for a commitment request to a non-user employee.

    Returns (subject, body_html) tuple. body_html is inner content —
    caller wraps in EMAIL_TEMPLATE.
    """
    employee = commitment.employee
    project_title = commitment.project.project.title
    requester = commitment.requested_by
    requester_name = (
        requester.get_full_name() or requester.email
        if requester
        else "A project coordinator"
    )
    role_display = commitment.get_role_display()

    subject = f"[Svend] Resource commitment request: {project_title}"

    confirm_url = f"https://svend.ai/action/{confirm_token.token}/"
    decline_url = f"https://svend.ai/action/{decline_token.token}/"

    body_parts = [
        "<h2 style='margin:0 0 16px;font-size:18px;color:#1a2a1a;'>Resource Commitment Request</h2>",
        f"<p style='margin:8px 0;color:#333;'>Hi {employee.name},</p>",
        f"<p style='margin:8px 0;color:#333;'>"
        f"{requester_name} has requested your participation on a continuous "
        f"improvement project.</p>",
        # Details block
        "<div style='background:#f0f5f0;border-radius:6px;padding:16px;margin:16px 0;'>",
        f"<p style='margin:4px 0;'><strong>Project:</strong> {project_title}</p>",
        f"<p style='margin:4px 0;'><strong>Role:</strong> {role_display}</p>",
        f"<p style='margin:4px 0;'><strong>Dates:</strong> {commitment.start_date} to {commitment.end_date}</p>",
        f"<p style='margin:4px 0;'><strong>Hours/day:</strong> {commitment.hours_per_day}</p>",
        "</div>",
        # CTA buttons
        "<p style='margin:24px 0;'>",
        f'<a href="{confirm_url}" style="display:inline-block;padding:12px 24px;'
        f"background:#4a9f6e;color:#fff;text-decoration:none;border-radius:6px;"
        f'font-weight:600;margin-right:12px;">Confirm</a>',
        f'<a href="{decline_url}" style="display:inline-block;padding:12px 24px;'
        f"background:#6b7280;color:#fff;text-decoration:none;border-radius:6px;"
        f'font-weight:600;">Decline</a>',
        "</p>",
        "<p style='margin:16px 0 0;font-size:12px;color:#7a8f7a;'>"
        "These links expire in 72 hours and can only be used once.</p>",
    ]

    return subject, "\n".join(body_parts)


def send_commitment_request_email_task(payload, context=None):
    """syn.sched task: send commitment request email to non-user employee.

    Expected payload args: commitment_id, confirm_token_id, decline_token_id
    """
    from agents_api.models import ActionToken, ResourceCommitment
    from api.internal_views import EMAIL_TEMPLATE

    args = (
        payload.get("args", {})
        if isinstance(payload, dict)
        else getattr(payload, "args", {}) or {}
    )
    commitment_id = args.get("commitment_id")
    confirm_token_id = args.get("confirm_token_id")
    decline_token_id = args.get("decline_token_id")

    if not all([commitment_id, confirm_token_id, decline_token_id]):
        return {"error": "Missing required args"}

    try:
        commitment = ResourceCommitment.objects.select_related(
            "employee", "project__project", "requested_by"
        ).get(id=commitment_id)
        confirm_tok = ActionToken.objects.get(id=confirm_token_id)
        decline_tok = ActionToken.objects.get(id=decline_token_id)
    except (ResourceCommitment.DoesNotExist, ActionToken.DoesNotExist) as e:
        return {"error": str(e)}

    subject, body_html = build_commitment_request_email(
        commitment, confirm_tok, decline_tok
    )

    # Wrap in branded template — no unsubscribe for transactional non-user email
    full_html = EMAIL_TEMPLATE.format(body=body_html, unsub_url="https://svend.ai")

    try:
        django_send_mail(
            subject=subject,
            message="",
            from_email=None,
            recipient_list=[commitment.employee.email],
            html_message=full_html,
        )
        return {"sent": True, "email": commitment.employee.email}
    except Exception:
        logger.exception(
            "Failed to send commitment email to %s", commitment.employee.email
        )
        return {"sent": False, "error": "SMTP failure"}
