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

    logger.info("[syn.sched] Registered %d Svend task handlers", len(TaskRegistry._handlers))


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
]
