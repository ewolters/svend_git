"""
Svend task handler registry for syn.sched.

Maps existing Tempora task handlers to syn.sched's TaskRegistry.
This module is imported during app startup to register all handlers.

Phase 5 Migration:
- Step 1: Register handlers here (this file)
- Step 2: Shadow mode — both schedulers run, compare results
- Step 3: Cutover — register schedules with syn.sched.Schedule
- Step 4: Remove tempora from INSTALLED_APPS
"""

import logging

from syn.sched.core import TaskRegistry
from syn.sched.types import QueueType, TaskPriority, RetryStrategy

logger = logging.getLogger(__name__)


def register_svend_tasks():
    """Register all Svend task handlers with syn.sched's TaskRegistry."""

    # ---- api/tasks.py handlers ----

    from api.tasks import (
        publish_scheduled_posts,
        send_onboarding_email,
        process_onboarding_drip,
        process_automations,
        evaluate_experiments,
        claude_growth_review,
        crm_send_one_email,
    )

    TaskRegistry.register(
        task_name="api.publish_scheduled_posts",
        handler=publish_scheduled_posts,
        queue=QueueType.CORE,
        priority=TaskPriority.LOW,
        timeout_seconds=30,
        max_attempts=1,
    )

    TaskRegistry.register(
        task_name="api.send_onboarding_email",
        handler=send_onboarding_email,
        queue=QueueType.CORE,
        priority=TaskPriority.NORMAL,
        timeout_seconds=30,
        max_attempts=3,
    )

    TaskRegistry.register(
        task_name="api.process_onboarding_drip",
        handler=process_onboarding_drip,
        queue=QueueType.CORE,
        priority=TaskPriority.LOW,
        timeout_seconds=60,
        max_attempts=1,
    )

    TaskRegistry.register(
        task_name="api.process_automations",
        handler=process_automations,
        queue=QueueType.CORE,
        priority=TaskPriority.NORMAL,
        timeout_seconds=120,
        max_attempts=1,
    )

    TaskRegistry.register(
        task_name="api.evaluate_experiments",
        handler=evaluate_experiments,
        queue=QueueType.CORE,
        priority=TaskPriority.LOW,
        timeout_seconds=60,
        max_attempts=1,
    )

    TaskRegistry.register(
        task_name="api.claude_growth_review",
        handler=claude_growth_review,
        queue=QueueType.CORE,
        priority=TaskPriority.LOW,
        timeout_seconds=300,
        max_attempts=1,
    )

    TaskRegistry.register(
        task_name="api.crm_send_one_email",
        handler=crm_send_one_email,
        queue=QueueType.CORE,
        priority=TaskPriority.NORMAL,
        timeout_seconds=30,
        max_attempts=2,
    )

    # ---- forge/tasks.py handler ----

    from forge.tasks import generate_data_task

    TaskRegistry.register(
        task_name="forge.tasks.generate_data_task",
        handler=generate_data_task,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=300,
        max_attempts=2,
    )

    # ---- Compliance checks (syn.audit) ----

    def compliance_daily_handler(task):
        from syn.audit.compliance import run_daily_checks
        results = run_daily_checks()
        return {
            "checks_run": len(results),
            "passed": sum(1 for r in results if r.status == "pass"),
        }

    def compliance_monthly_report_handler(task):
        from syn.audit.compliance import generate_monthly_report
        report = generate_monthly_report()
        return {"report_id": str(report.id), "pass_rate": report.pass_rate}

    def audit_cleanup_violations_handler(task):
        """Clean up resolved violations older than 90 days + expired health pings."""
        from datetime import timedelta
        from django.utils import timezone
        from syn.audit.models import IntegrityViolation, HealthPing
        cutoff = timezone.now() - timedelta(days=90)
        deleted_violations, _ = IntegrityViolation.objects.filter(
            is_resolved=True, resolved_at__lt=cutoff
        ).delete()
        deleted_pings, _ = HealthPing.objects.filter(timestamp__lt=cutoff).delete()
        return {"deleted_violations": deleted_violations, "deleted_pings": deleted_pings}

    def health_ping_handler(task):
        """Ping /api/health/ and record result. Fires andon on 3+ consecutive failures."""
        import time
        import requests
        from syn.audit.models import HealthPing

        url = "http://127.0.0.1:8000/api/health/"
        # X-Forwarded-Proto bypasses SECURE_SSL_REDIRECT for localhost pings
        headers = {"X-Forwarded-Proto": "https"}
        start = time.time()
        is_healthy = False
        try:
            resp = requests.get(url, timeout=10, headers=headers)
            elapsed_ms = (time.time() - start) * 1000
            is_healthy = resp.status_code == 200
            if is_healthy:
                try:
                    data = resp.json()
                    is_healthy = data.get("status") == "ok"
                except Exception:
                    is_healthy = False
            HealthPing.objects.create(
                is_healthy=is_healthy,
                status_code=resp.status_code,
                response_time_ms=round(elapsed_ms, 1),
            )
        except requests.RequestException as e:
            elapsed_ms = (time.time() - start) * 1000
            HealthPing.objects.create(
                is_healthy=False,
                status_code=None,
                response_time_ms=round(elapsed_ms, 1),
                error=str(e)[:500],
            )

        # Andon alert: 3 consecutive failures → notify staff
        recent = list(HealthPing.objects.order_by("-timestamp")[:3])
        if len(recent) == 3 and all(not p.is_healthy for p in recent):
            _fire_availability_andon(recent)

        return {"is_healthy": is_healthy}

    TaskRegistry.register(
        task_name="audit.compliance_daily",
        handler=compliance_daily_handler,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=300,
        max_attempts=2,
    )

    TaskRegistry.register(
        task_name="audit.compliance_monthly_report",
        handler=compliance_monthly_report_handler,
        queue=QueueType.BATCH,
        priority=TaskPriority.LOW,
        timeout_seconds=600,
        max_attempts=2,
    )

    TaskRegistry.register(
        task_name="audit.cleanup_violations",
        handler=audit_cleanup_violations_handler,
        queue=QueueType.BATCH,
        priority=TaskPriority.LOW,
        timeout_seconds=120,
        max_attempts=1,
    )

    TaskRegistry.register(
        task_name="audit.health_ping",
        handler=health_ping_handler,
        queue=QueueType.CORE,
        priority=TaskPriority.HIGH,
        timeout_seconds=30,
        max_attempts=1,
    )

    # ---- notifications/tasks.py handlers ----

    from notifications.tasks import (
        send_notification_email_task,
        send_daily_digest,
        send_weekly_digest,
        cleanup_expired_tokens,
    )

    TaskRegistry.register(
        task_name="notifications.send_email",
        handler=send_notification_email_task,
        queue=QueueType.CORE,
        priority=TaskPriority.NORMAL,
        timeout_seconds=30,
        max_attempts=3,
    )

    TaskRegistry.register(
        task_name="notifications.daily_digest",
        handler=send_daily_digest,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=120,
        max_attempts=2,
    )

    TaskRegistry.register(
        task_name="notifications.weekly_digest",
        handler=send_weekly_digest,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=120,
        max_attempts=2,
    )

    TaskRegistry.register(
        task_name="notifications.cleanup_tokens",
        handler=cleanup_expired_tokens,
        queue=QueueType.BATCH,
        priority=TaskPriority.LOW,
        timeout_seconds=60,
        max_attempts=1,
    )

    logger.info("[syn.sched] Registered %d Svend task handlers", len(TaskRegistry._handlers))


def _fire_availability_andon(recent_pings):
    """Send notification to staff on consecutive health ping failures (SLA-001 §12)."""
    from django.contrib.auth import get_user_model
    from notifications.helpers import notify

    User = get_user_model()
    staff = User.objects.filter(is_staff=True)
    errors = [p.error or f"HTTP {p.status_code}" for p in recent_pings]

    for user in staff:
        notify(
            recipient=user,
            notification_type="system",
            title="AVAILABILITY ALERT: 3 consecutive health ping failures",
            message=f"Health endpoint /api/health/ failed 3 times in a row. Errors: {'; '.join(errors)}",
            entity_type="health_ping",
        )
    logger.warning("[syn.sched] Availability andon fired: 3 consecutive health ping failures")


# Schedule definitions matching current Tempora configuration.
# Used during cutover (Phase 5.4) to create syn.sched.Schedule records.
SVEND_SCHEDULES = [
    {
        "schedule_id": "publish_scheduled_posts",
        "task_name": "api.publish_scheduled_posts",
        "cron": "*/15 * * * *",
        "priority": TaskPriority.LOW,
        "queue": "core",
    },
    {
        "schedule_id": "process_onboarding_drip",
        "task_name": "api.process_onboarding_drip",
        "cron": "*/10 * * * *",
        "priority": TaskPriority.NORMAL,
        "queue": "core",
    },
    {
        "schedule_id": "process_automations",
        "task_name": "api.process_automations",
        "cron": "*/30 * * * *",
        "priority": TaskPriority.NORMAL,
        "queue": "core",
    },
    {
        "schedule_id": "evaluate_experiments",
        "task_name": "api.evaluate_experiments",
        "cron": "0 6 * * *",
        "priority": TaskPriority.LOW,
        "queue": "core",
    },
    {
        "schedule_id": "claude_growth_review",
        "task_name": "api.claude_growth_review",
        "cron": "0 20 * * 0",
        "priority": TaskPriority.LOW,
        "queue": "core",
    },
    {
        "schedule_id": "health-ping",
        "task_name": "audit.health_ping",
        "cron": "* * * * *",  # Every minute
        "priority": TaskPriority.HIGH,
        "queue": "core",
    },
    {
        "schedule_id": "compliance-daily",
        "task_name": "audit.compliance_daily",
        "cron": "0 2 * * *",
        "priority": TaskPriority.NORMAL,
        "queue": "batch",
    },
    {
        "schedule_id": "compliance-monthly-report",
        "task_name": "audit.compliance_monthly_report",
        "cron": "0 4 1 * *",
        "priority": TaskPriority.LOW,
        "queue": "batch",
    },
    {
        "schedule_id": "audit-cleanup-violations",
        "task_name": "audit.cleanup_violations",
        "cron": "0 3 * * 0",  # Weekly Sunday 03:00 UTC
        "priority": TaskPriority.LOW,
        "queue": "batch",
    },
    # ---- Notification email schedules ----
    {
        "schedule_id": "notifications-daily-digest",
        "task_name": "notifications.daily_digest",
        "cron": "0 8 * * *",  # Daily 08:00 UTC
        "priority": TaskPriority.NORMAL,
        "queue": "batch",
    },
    {
        "schedule_id": "notifications-weekly-digest",
        "task_name": "notifications.weekly_digest",
        "cron": "0 8 * * 1",  # Monday 08:00 UTC
        "priority": TaskPriority.NORMAL,
        "queue": "batch",
    },
    {
        "schedule_id": "notifications-cleanup-tokens",
        "task_name": "notifications.cleanup_tokens",
        "cron": "0 4 * * 0",  # Sunday 04:00 UTC
        "priority": TaskPriority.LOW,
        "queue": "batch",
    },
]
