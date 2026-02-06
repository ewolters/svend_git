"""
Tempora Events
==============

Event definitions and helpers for scheduler observability.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from django.utils import timezone

logger = logging.getLogger(__name__)

# Event handlers registry
_event_handlers: Dict[str, List[Callable]] = {}


# =============================================================================
# EVENT DEFINITIONS
# =============================================================================

SCHEDULER_EVENTS = {
    # Task lifecycle
    "scheduler.task.created": "Task created and queued",
    "scheduler.task.started": "Task execution started",
    "scheduler.task.completed": "Task completed successfully",
    "scheduler.task.failed": "Task execution failed",
    "scheduler.task.retrying": "Task scheduled for retry",
    # Schedule events
    "scheduler.schedule.triggered": "Schedule triggered task creation",
    "scheduler.schedule.paused": "Schedule paused",
    "scheduler.schedule.resumed": "Schedule resumed",
    # Cascade events
    "scheduler.cascade.child_created": "Child task created in cascade",
    "scheduler.cascade.depth_exceeded": "Cascade depth limit exceeded",
    # DLQ events
    "scheduler.dlq.enqueued": "Task moved to dead letter queue",
    "scheduler.dlq.replayed": "Task replayed from DLQ",
    # Quota events
    "scheduler.quota.exceeded": "Tenant quota exceeded",
    "scheduler.quota.warning": "Tenant quota approaching limit",
    # Retry events
    "scheduler.retry.scheduled": "Retry scheduled",
    "scheduler.retry.exhausted": "Retry attempts exhausted",
    # Circuit breaker
    "scheduler.circuit.opened": "Circuit breaker opened",
    "scheduler.circuit.closed": "Circuit breaker closed",
    "scheduler.circuit.rejected": "Request rejected by circuit breaker",
    # Scheduler lifecycle
    "scheduler.started": "Scheduler started",
    "scheduler.stopped": "Scheduler stopped",
    "scheduler.worker.started": "Worker started",
    "scheduler.worker.stopped": "Worker stopped",
    # Backpressure
    "scheduler.backpressure.level_changed": "Throttle level changed",
    "scheduler.backpressure.emergency": "Emergency backpressure triggered",
}


# =============================================================================
# EVENT EMISSION
# =============================================================================


def emit_scheduler_event(
    event_type: str,
    payload: Dict[str, Any],
    correlation_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Emit a scheduler event.

    Args:
        event_type: Event type key from SCHEDULER_EVENTS
        payload: Event data
        correlation_id: Optional correlation ID for tracing
        tenant_id: Optional tenant ID for multi-tenancy
    """
    event = {
        "event_type": event_type,
        "timestamp": timezone.now().isoformat(),
        "payload": payload,
    }

    if correlation_id:
        event["correlation_id"] = correlation_id
    if tenant_id:
        event["tenant_id"] = tenant_id

    # Log the event
    logger.info(
        f"[TEMPORA] {event_type}",
        extra={"event": event},
    )

    # Call registered handlers
    handlers = _event_handlers.get(event_type, []) + _event_handlers.get("*", [])
    for handler in handlers:
        try:
            handler(event)
        except Exception as e:
            logger.error(f"Event handler error: {e}")


def register_event_handler(event_type: str, handler: Callable) -> None:
    """Register a handler for an event type. Use '*' for all events."""
    if event_type not in _event_handlers:
        _event_handlers[event_type] = []
    _event_handlers[event_type].append(handler)


def unregister_event_handler(event_type: str, handler: Callable) -> None:
    """Unregister a handler."""
    if event_type in _event_handlers:
        _event_handlers[event_type] = [
            h for h in _event_handlers[event_type] if h != handler
        ]


# =============================================================================
# PAYLOAD BUILDERS
# =============================================================================


def build_task_created_payload(task) -> Dict[str, Any]:
    """Build payload for task.created event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "tenant_id": str(task.tenant_id),
        "correlation_id": str(task.correlation_id),
        "priority": task.priority,
        "queue": task.queue,
        "cascade_depth": getattr(task, "cascade_depth", 0),
    }


def build_task_started_payload(task, worker_id: str, attempt: int) -> Dict[str, Any]:
    """Build payload for task.started event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "worker_id": worker_id,
        "attempt_number": attempt,
        "correlation_id": str(task.correlation_id),
        "tenant_id": str(task.tenant_id),
    }


def build_task_completed_payload(task, duration_ms: int, attempts: int) -> Dict[str, Any]:
    """Build payload for task.completed event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "duration_ms": duration_ms,
        "attempt_number": attempts,
        "correlation_id": str(task.correlation_id),
        "tenant_id": str(task.tenant_id),
    }


def build_task_failed_payload(
    task,
    error_type: str,
    error_message: str,
    attempt: int,
    will_retry: bool,
) -> Dict[str, Any]:
    """Build payload for task.failed event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "error_type": error_type,
        "error_message": error_message,
        "attempt_number": attempt,
        "will_retry": will_retry,
        "correlation_id": str(task.correlation_id),
        "tenant_id": str(task.tenant_id),
    }


def build_schedule_triggered_payload(schedule, task) -> Dict[str, Any]:
    """Build payload for schedule.triggered event."""
    return {
        "schedule_id": schedule.schedule_id,
        "task_id": str(task.id),
        "task_name": schedule.task_name,
        "run_count": schedule.run_count,
        "tenant_id": str(schedule.tenant_id),
    }


def build_cascade_child_payload(parent_task, child_task) -> Dict[str, Any]:
    """Build payload for cascade.child_created event."""
    return {
        "parent_task_id": str(parent_task.id),
        "child_task_id": str(child_task.id),
        "child_task_name": child_task.task_name,
        "cascade_depth": child_task.cascade_depth,
        "root_correlation_id": str(parent_task.root_correlation_id),
    }


def build_dlq_enqueued_payload(dlq_entry) -> Dict[str, Any]:
    """Build payload for dlq.enqueued event."""
    return {
        "dlq_entry_id": str(dlq_entry.id),
        "original_task_id": str(dlq_entry.original_task_id),
        "task_name": dlq_entry.task_name,
        "failure_reason": dlq_entry.failure_reason,
        "tenant_id": str(dlq_entry.tenant_id),
    }


def build_quota_exceeded_payload(
    tenant_id: str,
    quota_type: str,
    current: int,
    limit: int,
    action: str,
) -> Dict[str, Any]:
    """Build payload for quota.exceeded event."""
    return {
        "tenant_id": tenant_id,
        "quota_type": quota_type,
        "current_value": current,
        "limit": limit,
        "action_taken": action,
    }


def build_retry_scheduled_payload(task, delay_seconds: int) -> Dict[str, Any]:
    """Build payload for retry.scheduled event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "attempt_number": task.attempts,
        "delay_seconds": delay_seconds,
        "correlation_id": str(task.correlation_id),
    }


__all__ = [
    "SCHEDULER_EVENTS",
    "emit_scheduler_event",
    "register_event_handler",
    "unregister_event_handler",
    "build_task_created_payload",
    "build_task_started_payload",
    "build_task_completed_payload",
    "build_task_failed_payload",
    "build_schedule_triggered_payload",
    "build_cascade_child_payload",
    "build_dlq_enqueued_payload",
    "build_quota_exceeded_payload",
    "build_retry_scheduled_payload",
]
