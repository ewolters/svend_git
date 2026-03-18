"""Commitment notification dispatcher — dual-path for users and non-users.

When a ResourceCommitment is created:
- Path A (Employee.user_link set): in-app notification via notify()
- Path B (no user_link): email with ActionToken confirm/decline links

On confirm/decline: notify the requested_by user.
"""

import logging

logger = logging.getLogger(__name__)


def notify_commitment_requested(commitment):
    """Send notification to the assigned employee about a new commitment.

    Never raises — all errors are caught and logged.
    """
    try:
        _dispatch_commitment_requested(commitment)
    except Exception:
        logger.exception("Failed to send commitment notification for %s", commitment.id)


def _dispatch_commitment_requested(commitment):
    employee = commitment.employee
    project_title = commitment.project.project.title
    requester = commitment.requested_by
    requester_name = requester.get_full_name() or requester.email if requester else "Someone"

    title = f"Resource commitment requested: {project_title}"
    message = (
        f"{requester_name} has requested your participation as "
        f'{commitment.get_role_display()} on "{project_title}" '
        f"from {commitment.start_date} to {commitment.end_date} "
        f"({commitment.hours_per_day}h/day)."
    )

    if employee.user_link_id:
        # Path A — Svend user: in-app notification + email via notify()
        from notifications.helpers import notify

        notify(
            recipient=employee.user_link,
            notification_type="commitment_requested",
            title=title,
            message=message,
            entity_type="hoshin_project",
            entity_id=commitment.project.project_id,
        )
    elif employee.email:
        # Path B — non-user: ActionToken email with confirm/decline
        _send_non_user_commitment_email(commitment, title, message)


def _send_non_user_commitment_email(commitment, title, message):
    """Create ActionTokens and schedule email for non-user employee."""
    from agents_api.models import ActionToken
    from syn.sched.scheduler import schedule_task

    scope = {"commitment_id": str(commitment.id)}

    confirm_tok = ActionToken.objects.create(
        employee=commitment.employee,
        action_type="confirm_availability",
        scoped_to=scope,
    )
    decline_tok = ActionToken.objects.create(
        employee=commitment.employee,
        action_type="decline",
        scoped_to=scope,
    )

    schedule_task(
        name=f"commitment-email-{commitment.id}",
        func="agents_api.send_commitment_request_email",
        args={
            "commitment_id": str(commitment.id),
            "confirm_token_id": str(confirm_tok.id),
            "decline_token_id": str(decline_tok.id),
        },
        delay_seconds=5,
        queue="core",
    )


def notify_commitment_response(commitment, old_status):
    """Notify the requester when a commitment is confirmed or declined.

    Called from both the in-app PUT handler and the ActionToken POST handler.
    Never raises — all errors are caught and logged.
    """
    try:
        _dispatch_commitment_response(commitment, old_status)
    except Exception:
        logger.exception("Failed to send commitment response notification for %s", commitment.id)


def _dispatch_commitment_response(commitment, old_status):
    if old_status != "requested":
        return
    if commitment.status not in ("confirmed", "declined"):
        return
    if not commitment.requested_by:
        return

    from notifications.helpers import notify

    employee_name = commitment.employee.name
    project_title = commitment.project.project.title

    if commitment.status == "confirmed":
        ntype = "commitment_confirmed"
        title = f"{employee_name} confirmed commitment to {project_title}"
    else:
        ntype = "commitment_declined"
        title = f"{employee_name} declined commitment to {project_title}"

    notify(
        recipient=commitment.requested_by,
        notification_type=ntype,
        title=title,
        message=f"Role: {commitment.get_role_display()}, {commitment.start_date} to {commitment.end_date}.",
        entity_type="hoshin_project",
        entity_id=commitment.project.project_id,
    )
