"""
Tempora Scheduler API
=====================

Simple API for scheduling tasks with the Tempora distributed scheduler.
"""

import importlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Union

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

    Example:
        # One-time task
        schedule_task(
            name="send_welcome_email",
            func="myapp.tasks.send_email",
            args={"user_id": 123},
            run_at=timezone.now() + timedelta(hours=1)
        )

        # Recurring task
        schedule_task(
            name="daily_cleanup",
            func="myapp.tasks.cleanup",
            cron="0 2 * * *"
        )
    """
    from tempora.models import CognitiveTask, Schedule
    from tempora.types import TaskPriority, TaskState, QueueType, ScheduleType

    args = args or {}
    tenant_id = tenant_id or uuid.UUID("00000000-0000-0000-0000-000000000000")

    # Handle recurring schedules
    if cron or interval_seconds:
        schedule = Schedule.objects.create(
            schedule_id=name,
            tenant_id=tenant_id,
            task_name=func,
            payload_template=args,
            schedule_type=ScheduleType.CRON.value if cron else ScheduleType.INTERVAL.value,
            cron_expression=cron or "",
            interval_seconds=interval_seconds or 0,
            priority=priority,
            queue=queue,
            enabled=True,
            next_run_at=timezone.now(),
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
    """
    Get the status of a scheduled task.

    Args:
        task_id: Task ID from schedule_task

    Returns:
        Task status dict or None if not found
    """
    from tempora.models import CognitiveTask

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
    """
    Cancel a pending task.

    Args:
        task_id: Task ID to cancel

    Returns:
        True if cancelled, False if not found or not cancellable
    """
    from tempora.models import CognitiveTask
    from tempora.types import TaskState

    try:
        task_uuid = uuid.UUID(task_id)
        task = CognitiveTask.objects.get(id=task_uuid)

        if task.state in [TaskState.PENDING.value, TaskState.SCHEDULED.value]:
            task.state = TaskState.CANCELLED.value
            task.save()
            logger.info(f"Cancelled task: {task_id}")
            return True

        return False
    except CognitiveTask.DoesNotExist:
        return False


def get_cluster_health() -> Dict[str, Any]:
    """
    Get health status of the Tempora cluster.

    Returns:
        Cluster health information
    """
    from tempora.models import ClusterMember

    try:
        members = ClusterMember.objects.all()
        leader = members.filter(is_leader=True).first()

        return {
            "status": "healthy" if leader else "degraded",
            "leader": leader.node_id if leader else None,
            "term": leader.term if leader else 0,
            "nodes": {
                m.node_id: {
                    "status": "leader" if m.is_leader else "follower",
                    "last_heartbeat": m.last_heartbeat.isoformat() if m.last_heartbeat else None,
                }
                for m in members
            },
        }
    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e),
        }


__all__ = [
    "schedule_task",
    "get_task_status",
    "cancel_task",
    "get_cluster_health",
]
