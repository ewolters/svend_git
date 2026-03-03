"""
syn.sched Scheduler API
========================

Simple API for scheduling tasks with the cognitive scheduler.
Drop-in replacement for tempora.scheduler with syn.sched models.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from django.utils import timezone

logger = logging.getLogger(__name__)


def schedule_task(
    name: str,
    func: str,
    args: Optional[Dict[str, Any]] = None,
    run_at: Optional[datetime] = None,
    delay_seconds: int = 0,
    cron: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    priority: int = 2,
    queue: str = "core",
    tenant_id: Optional[uuid.UUID] = None,
) -> str:
    """
    Schedule a task for execution.

    Args:
        name: Human-readable task name
        func: Dotted path to function (e.g., "myapp.tasks.send_email")
        args: Keyword arguments to pass to the function
        run_at: Specific datetime to run (one-time)
        delay_seconds: Delay before execution
        cron: Cron expression for recurring (e.g., "0 2 * * *")
        interval_seconds: Interval for recurring
        priority: Task priority (0-4, default 2=NORMAL)
        queue: Target queue (core, batch, realtime, governance)
        tenant_id: Tenant for isolation

    Returns:
        Task ID for tracking
    """
    from syn.sched.models import CognitiveTask, Schedule
    from syn.sched.types import TaskState, ScheduleType

    args = args or {}
    tenant_id = tenant_id or uuid.UUID("00000000-0000-0000-0000-000000000000")

    # Handle recurring schedules
    if cron or interval_seconds:
        cron_fields = {}
        if cron:
            parts = cron.strip().split()
            if len(parts) == 5:
                cron_fields = {
                    "cron_minute": parts[0],
                    "cron_hour": parts[1],
                    "cron_day_of_month": parts[2],
                    "cron_month": parts[3],
                    "cron_day_of_week": parts[4],
                }
            else:
                logger.warning(f"Invalid cron expression '{cron}', using defaults")

        next_run = timezone.now()
        if cron:
            try:
                from croniter import croniter
                next_run = croniter(cron, timezone.now()).get_next(datetime)
                if timezone.is_naive(next_run):
                    from django.utils.timezone import make_aware
                    next_run = make_aware(next_run)
            except ImportError:
                pass

        schedule = Schedule.objects.create(
            schedule_id=name,
            name=name,
            tenant_id=tenant_id,
            task_name=func,
            payload_template=args,
            schedule_type=ScheduleType.CRON.value if cron else ScheduleType.INTERVAL.value,
            interval_seconds=interval_seconds or 0,
            priority=priority,
            queue=queue,
            enabled=True,
            next_run_at=next_run,
            **cron_fields,
        )
        logger.info(f"Created schedule: {name} ({schedule.id})")
        return str(schedule.id)

    # Calculate scheduled time
    scheduled_at = None
    if run_at:
        scheduled_at = run_at
    elif delay_seconds > 0:
        scheduled_at = timezone.now() + timedelta(seconds=delay_seconds)

    # Create one-time task
    task = CognitiveTask.objects.create(
        tenant_id=tenant_id,
        task_name=func,
        payload=args,
        priority=priority,
        queue=queue,
        state=TaskState.PENDING.value,
        scheduled_at=scheduled_at,
    )

    logger.info(f"Scheduled task: {name} ({task.id})")
    return str(task.id)


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get the status of a scheduled task."""
    from syn.sched.models import CognitiveTask

    try:
        task_uuid = uuid.UUID(task_id)
        task = CognitiveTask.objects.get(id=task_uuid)
        return {
            "id": str(task.id),
            "name": task.task_name,
            "state": task.state,
            "created_at": task.created_at.isoformat() if hasattr(task, 'created_at') else None,
            "scheduled_at": task.scheduled_at.isoformat() if task.scheduled_at else None,
            "attempts": task.attempts,
            "result": task.result,
            "error_message": task.error_message,
        }
    except CognitiveTask.DoesNotExist:
        return None
    except ValueError:
        return None


def cancel_task(task_id: str) -> bool:
    """Cancel a pending task."""
    from syn.sched.models import CognitiveTask
    from syn.sched.types import TaskState

    try:
        task_uuid = uuid.UUID(task_id)
        task = CognitiveTask.objects.get(id=task_uuid)

        if task.state in [TaskState.PENDING.value, "scheduled"]:
            task.state = TaskState.CANCELLED.value
            task.save()
            logger.info(f"Cancelled task: {task_id}")
            return True

        return False
    except CognitiveTask.DoesNotExist:
        return False


__all__ = [
    "schedule_task",
    "get_task_status",
    "cancel_task",
]
