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
from syn.sched.types import QueueType, TaskPriority

logger = logging.getLogger(__name__)


def register_svend_tasks():
    """Register all Svend task handlers with syn.sched's TaskRegistry."""

    # ---- api/tasks.py handlers ----

    from api.tasks import (
        claude_growth_review,
        crm_send_one_email,
        evaluate_experiments,
        process_automations,
        process_onboarding_drip,
        publish_scheduled_posts,
        send_onboarding_email,
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

    # ---- agents_api/commitment_tasks.py handler ----

    from agents_api.commitment_tasks import send_commitment_request_email_task

    TaskRegistry.register(
        task_name="agents_api.send_commitment_request_email",
        handler=send_commitment_request_email_task,
        queue=QueueType.CORE,
        priority=TaskPriority.NORMAL,
        timeout_seconds=30,
        max_attempts=3,
    )

    # ---- forge/tasks.py handler ----

    from forge.tasks import generate_data_task_async

    TaskRegistry.register(
        task_name="forge.tasks.generate_data_task",
        handler=generate_data_task_async,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=300,
        max_attempts=2,
    )

    # ---- Compliance checks (syn.audit) ----

    def compliance_daily_handler(payload, context):
        from syn.audit.compliance import run_daily_checks

        results = run_daily_checks()
        return {
            "checks_run": len(results),
            "passed": sum(1 for r in results if r.status == "pass"),
        }

    def compliance_monthly_report_handler(payload, context):
        from syn.audit.compliance import generate_monthly_report

        report = generate_monthly_report()
        return {"report_id": str(report.id), "pass_rate": report.pass_rate}

    def audit_cleanup_violations_handler(payload, context):
        """Clean up resolved violations older than 90 days + expired health pings."""
        from datetime import timedelta

        from django.utils import timezone

        from syn.audit.models import HealthPing, IntegrityViolation

        cutoff = timezone.now() - timedelta(days=90)
        deleted_violations, _ = IntegrityViolation.objects.filter(is_resolved=True, resolved_at__lt=cutoff).delete()
        deleted_pings, _ = HealthPing.objects.filter(timestamp__lt=cutoff).delete()
        return {
            "deleted_violations": deleted_violations,
            "deleted_pings": deleted_pings,
        }

    def health_ping_handler(payload, context):
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

    def test_execution_handler(payload, context):
        """Run full test suite via compliance check at 2:00 AM EST (07:00 UTC).

        ⚠ COMPLIANCE-CRITICAL: This is the automated nightly test run per
        TST-001 §10.4. It runs the full pytest suite in a subprocess against
        a disposable test database. The production database is never touched.

        DO NOT disable this task without a ChangeRequest (CHG-001).
        """
        from syn.audit.compliance import run_check

        test_result = run_check("test_execution")
        coverage_result = run_check("test_coverage")
        return {
            "test_status": test_result.status,
            "test_details": test_result.details,
            "coverage_status": coverage_result.status,
            "coverage_details": coverage_result.details,
        }

    TaskRegistry.register(
        task_name="audit.test_execution",
        handler=test_execution_handler,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=720,  # 12 min — test suite timeout (600s) + coverage overhead
        max_attempts=1,  # Do not retry — flaky retries mask real failures
    )

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
        cleanup_expired_tokens,
        send_daily_digest,
        send_notification_email_task,
        send_weekly_digest,
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

    # ---- Webhook retry processing (NTF-001 §5.4) ----

    def webhook_retry_handler(payload, context):
        """Process pending webhook delivery retries."""
        from notifications.webhook_delivery import process_retries

        count = process_retries()
        return {"retries_processed": count}

    TaskRegistry.register(
        task_name="notifications.webhook_retries",
        handler=webhook_retry_handler,
        queue=QueueType.CORE,
        priority=TaskPriority.NORMAL,
        timeout_seconds=120,
        max_attempts=1,
    )

    # ---- Harada daily reminders ----

    from agents_api.harada_tasks import harada_daily_reminders

    TaskRegistry.register(
        task_name="api.harada_daily_reminders",
        handler=harada_daily_reminders,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=120,
        max_attempts=1,
    )

    # ---- CI Readiness clustering (k-prototypes) ----

    from agents_api.clustering import run_clustering

    TaskRegistry.register(
        task_name="api.run_ci_clustering",
        handler=run_clustering,
        queue=QueueType.BATCH,
        priority=TaskPriority.LOW,
        timeout_seconds=300,
        max_attempts=2,
    )

    # ---- Front page digest (Qwen 14B theme extraction) ----

    from agents_api.front_page_tasks import generate_front_page_digest

    TaskRegistry.register(
        task_name="api.generate_front_page_digest",
        handler=generate_front_page_digest,
        queue=QueueType.BATCH,
        priority=TaskPriority.LOW,
        timeout_seconds=600,
        max_attempts=2,
    )

    # ---- Privacy export tasks (PRIV-001) ----

    from accounts.privacy_tasks import cleanup_expired_exports, generate_export

    TaskRegistry.register(
        task_name="privacy.generate_export",
        handler=generate_export,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=300,
        max_attempts=2,
    )

    TaskRegistry.register(
        task_name="privacy.cleanup_expired_exports",
        handler=cleanup_expired_exports,
        queue=QueueType.BATCH,
        priority=TaskPriority.LOW,
        timeout_seconds=120,
        max_attempts=1,
    )

    # ---- Loop policy sweep (LOOP-001 §4.6.3) ----

    from loop.evaluator import run_policy_sweep

    TaskRegistry.register(
        task_name="loop.policy_sweep",
        handler=run_policy_sweep,
        queue=QueueType.BATCH,
        priority=TaskPriority.NORMAL,
        timeout_seconds=300,
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
    # ⚠ COMPLIANCE-CRITICAL: Nightly test execution — TST-001 §10.4
    # Runs full pytest suite + coverage check at 2:00 AM EST (07:00 UTC).
    # DO NOT disable without a ChangeRequest (CHG-001).
    {
        "schedule_id": "test-execution-nightly",
        "task_name": "audit.test_execution",
        "cron": "0 7 * * *",  # 07:00 UTC = 2:00 AM EST
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
    # ---- Webhook retry processing (NTF-001 §5.4) ----
    {
        "schedule_id": "webhook-retries",
        "task_name": "notifications.webhook_retries",
        "cron": "* * * * *",  # Every minute — retries have their own delay logic
        "priority": TaskPriority.NORMAL,
        "queue": "core",
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
    # ---- Privacy export schedules (PRIV-001) ----
    {
        "schedule_id": "privacy-cleanup-exports",
        "task_name": "privacy.cleanup_expired_exports",
        "cron": "0 3 * * 0",  # Sunday 03:00 UTC
        "priority": TaskPriority.LOW,
        "queue": "batch",
    },
    # ---- Harada daily reminders ----
    {
        "schedule_id": "harada-daily-reminders",
        "task_name": "api.harada_daily_reminders",
        "cron": "0 8 * * *",  # Daily 08:00 UTC
        "priority": TaskPriority.NORMAL,
        "queue": "batch",
    },
    # ---- Loop policy sweep (LOOP-001 §4.6.3) ----
    {
        "schedule_id": "loop-policy-sweep",
        "task_name": "loop.policy_sweep",
        "cron": "0 6 * * *",  # Daily 06:00 UTC
        "priority": TaskPriority.NORMAL,
        "queue": "batch",
    },
]
