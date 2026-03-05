"""
Synara Cognitive Scheduler Events (SCH-001/002, EVT-001)
========================================================

Event catalog and emitters for the cognitive scheduler.

Standard:     SCH-001 §scheduler_events, EVT-001/002
Compliance:   ISO 9001:2015 §9.5, SOC 2 CC7.2
Version:      1.0.0
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEDULER EVENT CATALOG (SCH-001 §scheduler_events)
# =============================================================================


SCHEDULER_EVENTS = {
    # ==========================================================================
    # Task Lifecycle Events (SCH-001 §task_states)
    # ==========================================================================
    "scheduler.task.created": {
        "description": "New task created in scheduler",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "queue": "string",
            "priority": "integer",
            "priority_score": "float",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.task.scheduled": {
        "description": "Task scheduled for execution",
        "category": "scheduler",
        "audit": True,
        "priority": 4,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "scheduled_at": "datetime",
            "worker_id": "string",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.task.started": {
        "description": "Task execution started",
        "category": "scheduler",
        "audit": True,
        "priority": 4,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "worker_id": "string",
            "attempt_number": "integer",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.task.completed": {
        "description": "Task execution completed successfully",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "duration_ms": "integer",
            "attempt_number": "integer",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.task.failed": {
        "description": "Task execution failed",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "error_type": "string",
            "error_message": "string",
            "attempt_number": "integer",
            "will_retry": "boolean",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.task.cancelled": {
        "description": "Task cancelled before completion",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "cancelled_by": "string",
            "reason": "string",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.task.timeout": {
        "description": "Task exceeded execution timeout",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "timeout_seconds": "integer",
            "elapsed_seconds": "integer",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.task.deadline_exceeded": {
        "description": "Task exceeded hard deadline",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "deadline": "datetime",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    # ==========================================================================
    # Retry Events (SCH-002 §retry_strategies)
    # ==========================================================================
    "scheduler.retry.scheduled": {
        "description": "Task retry scheduled",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "attempt_number": "integer",
            "max_attempts": "integer",
            "retry_strategy": "string",
            "retry_delay_seconds": "integer",
            "next_retry_at": "datetime",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.retry.exhausted": {
        "description": "Task exhausted all retry attempts",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "task_id": "uuid",
            "task_name": "string",
            "total_attempts": "integer",
            "final_error": "string",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    # ==========================================================================
    # Dead Letter Queue Events (SCH-002 §dlq_handling)
    # ==========================================================================
    "scheduler.dlq.enqueued": {
        "description": "Task moved to Dead Letter Queue",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "dlq_id": "uuid",
            "task_id": "uuid",
            "task_name": "string",
            "failure_reason": "string",
            "failure_count": "integer",
            "correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.dlq.reprocessed": {
        "description": "DLQ entry reprocessed",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "dlq_id": "uuid",
            "original_task_id": "uuid",
            "new_task_id": "uuid",
            "reprocess_count": "integer",
            "reprocessed_by": "string",
            "tenant_id": "uuid",
        },
    },
    "scheduler.dlq.resolved": {
        "description": "DLQ entry resolved",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "dlq_id": "uuid",
            "task_id": "uuid",
            "resolution_status": "string",
            "resolved_by": "string",
            "resolution_notes": "string",
            "tenant_id": "uuid",
        },
    },
    "scheduler.dlq.discarded": {
        "description": "DLQ entry discarded",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "dlq_id": "uuid",
            "task_id": "uuid",
            "discarded_by": "string",
            "reason": "string",
            "tenant_id": "uuid",
        },
    },
    # ==========================================================================
    # Circuit Breaker Events (SCH-002 §circuit_breaker)
    # ==========================================================================
    "scheduler.circuit.opened": {
        "description": "Circuit breaker opened (blocking calls)",
        "category": "scheduler",
        "audit": True,
        "priority": 1,
        "payload_schema": {
            "service_name": "string",
            "failure_count": "integer",
            "failure_threshold": "integer",
            "last_error": "string",
        },
    },
    "scheduler.circuit.half_opened": {
        "description": "Circuit breaker transitioned to half-open (testing)",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "service_name": "string",
            "recovery_timeout_seconds": "integer",
            "opened_at": "datetime",
        },
    },
    "scheduler.circuit.closed": {
        "description": "Circuit breaker closed (normal operation)",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "service_name": "string",
            "success_count": "integer",
            "recovery_duration_seconds": "integer",
        },
    },
    "scheduler.circuit.rejected": {
        "description": "Request rejected by open circuit",
        "category": "scheduler",
        "audit": False,
        "priority": 3,
        "payload_schema": {
            "service_name": "string",
            "circuit_state": "string",
            "task_id": "uuid",
            "correlation_id": "uuid",
        },
    },
    # ==========================================================================
    # Schedule Events (SCH-001 §temporal_patterns)
    # ==========================================================================
    "scheduler.schedule.created": {
        "description": "New schedule created",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "schedule_id": "string",
            "name": "string",
            "schedule_type": "string",
            "task_name": "string",
            "next_run_at": "datetime",
            "created_by": "string",
            "tenant_id": "uuid",
        },
    },
    "scheduler.schedule.updated": {
        "description": "Schedule configuration updated",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "schedule_id": "string",
            "changes": "object",
            "updated_by": "string",
            "tenant_id": "uuid",
        },
    },
    "scheduler.schedule.enabled": {
        "description": "Schedule enabled",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "schedule_id": "string",
            "name": "string",
            "next_run_at": "datetime",
            "enabled_by": "string",
            "tenant_id": "uuid",
        },
    },
    "scheduler.schedule.disabled": {
        "description": "Schedule disabled",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "schedule_id": "string",
            "name": "string",
            "disabled_by": "string",
            "reason": "string",
            "tenant_id": "uuid",
        },
    },
    "scheduler.schedule.triggered": {
        "description": "Schedule triggered task creation",
        "category": "scheduler",
        "audit": True,
        "priority": 4,
        "payload_schema": {
            "schedule_id": "string",
            "task_id": "uuid",
            "run_number": "integer",
            "next_run_at": "datetime",
            "tenant_id": "uuid",
        },
    },
    "scheduler.schedule.expired": {
        "description": "Schedule expired and disabled",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "schedule_id": "string",
            "name": "string",
            "expires_at": "datetime",
            "total_runs": "integer",
            "tenant_id": "uuid",
        },
    },
    "scheduler.schedule.max_runs_reached": {
        "description": "Schedule reached maximum run count",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "schedule_id": "string",
            "name": "string",
            "max_runs": "integer",
            "tenant_id": "uuid",
        },
    },
    # ==========================================================================
    # Cascade Events (SCH-002 §reflex_throttling)
    # ==========================================================================
    "scheduler.cascade.started": {
        "description": "Cascade chain started",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "root_task_id": "uuid",
            "root_correlation_id": "uuid",
            "task_name": "string",
            "tenant_id": "uuid",
        },
    },
    "scheduler.cascade.child_created": {
        "description": "Child task created in cascade",
        "category": "scheduler",
        "audit": False,
        "priority": 4,
        "payload_schema": {
            "parent_task_id": "uuid",
            "child_task_id": "uuid",
            "cascade_depth": "integer",
            "root_correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.cascade.depth_exceeded": {
        "description": "Cascade depth limit exceeded",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "parent_task_id": "uuid",
            "attempted_depth": "integer",
            "max_depth": "integer",
            "root_correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    "scheduler.cascade.throttled": {
        "description": "Cascade throttled due to budget",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "parent_task_id": "uuid",
            "cascade_depth": "integer",
            "tasks_at_depth": "integer",
            "budget_limit": "integer",
            "root_correlation_id": "uuid",
            "tenant_id": "uuid",
        },
    },
    # ==========================================================================
    # Worker Events (SCH-001 §worker_pool)
    # ==========================================================================
    "scheduler.worker.started": {
        "description": "Worker instance started",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "worker_id": "string",
            "queues": "array",
            "concurrency": "integer",
        },
    },
    "scheduler.worker.stopped": {
        "description": "Worker instance stopped",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "worker_id": "string",
            "reason": "string",
            "tasks_completed": "integer",
            "uptime_seconds": "integer",
        },
    },
    "scheduler.worker.heartbeat": {
        "description": "Worker heartbeat",
        "category": "scheduler",
        "audit": False,
        "priority": 5,
        "payload_schema": {
            "worker_id": "string",
            "active_tasks": "integer",
            "completed_tasks": "integer",
            "failed_tasks": "integer",
            "memory_mb": "float",
            "cpu_percent": "float",
        },
    },
    "scheduler.worker.overloaded": {
        "description": "Worker at capacity",
        "category": "scheduler",
        "audit": False,
        "priority": 3,
        "payload_schema": {
            "worker_id": "string",
            "active_tasks": "integer",
            "max_concurrency": "integer",
            "queue_depth": "integer",
        },
    },
    # ==========================================================================
    # Quota Events (SCH-002 §resource_quotas)
    # ==========================================================================
    "scheduler.quota.exceeded": {
        "description": "Tenant quota exceeded",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "tenant_id": "uuid",
            "quota_type": "string",
            "current_value": "integer",
            "limit": "integer",
            "action_taken": "string",
        },
    },
    "scheduler.quota.warning": {
        "description": "Tenant approaching quota limit",
        "category": "scheduler",
        "audit": False,
        "priority": 3,
        "payload_schema": {
            "tenant_id": "uuid",
            "quota_type": "string",
            "current_value": "integer",
            "limit": "integer",
            "percent_used": "float",
        },
    },
    "scheduler.rate_limited": {
        "description": "Request rate limited",
        "category": "scheduler",
        "audit": True,
        "priority": 3,
        "payload_schema": {
            "tenant_id": "uuid",
            "rate_limit_per_minute": "integer",
            "current_rate": "integer",
            "correlation_id": "uuid",
        },
    },
    # ==========================================================================
    # Scheduler System Events
    # ==========================================================================
    "scheduler.started": {
        "description": "Scheduler system started",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "version": "string",
            "worker_count": "integer",
            "queues": "array",
        },
    },
    "scheduler.stopped": {
        "description": "Scheduler system stopped",
        "category": "scheduler",
        "audit": True,
        "priority": 2,
        "payload_schema": {
            "reason": "string",
            "graceful": "boolean",
            "pending_tasks": "integer",
        },
    },
    "scheduler.error": {
        "description": "Scheduler system error",
        "category": "scheduler",
        "audit": True,
        "priority": 1,
        "payload_schema": {
            "error_type": "string",
            "error_message": "string",
            "component": "string",
            "correlation_id": "uuid",
        },
    },
}


# =============================================================================
# EVENT EMITTER
# =============================================================================


def emit_scheduler_event(
    event_name: str,
    payload: dict[str, Any],
    correlation_id: str | None = None,
    tenant_id: str | None = None,
    parent_correlation_id: str | None = None,
    validate: bool = True,
) -> str | None:
    """
    Emit a scheduler event through the kernel event system.

    Args:
        event_name: Event name (e.g., "scheduler.task.created")
        payload: Event payload
        correlation_id: Correlation ID for tracing
        tenant_id: Tenant ID for isolation
        parent_correlation_id: Parent correlation for lineage
        validate: Whether to validate payload against schema

    Returns:
        New correlation ID if emitted, None on error

    Standard: EVT-002 §10, SCH-001 §scheduler_events
    """
    if event_name not in SCHEDULER_EVENTS:
        logger.warning(f"Unknown scheduler event: {event_name}")

    try:
        from syn.kernel.event_primitives import emit_event

        result = emit_event(
            event_name=event_name,
            payload=payload,
            correlation_id=correlation_id,
            parent_correlation_id=parent_correlation_id,
        )

        return result

    except ImportError:
        # Kernel not available, log locally
        logger.info(
            f"Scheduler event (no kernel): {event_name}",
            extra={
                "event_name": event_name,
                "payload": payload,
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
            },
        )
        return correlation_id

    except Exception as e:
        logger.error(f"Failed to emit scheduler event {event_name}: {e}")
        return None


# =============================================================================
# PAYLOAD BUILDERS
# =============================================================================


def build_task_created_payload(task) -> dict[str, Any]:
    """Build payload for scheduler.task.created event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "queue": task.queue,
        "priority": task.priority,
        "priority_score": task.priority_score,
        "correlation_id": str(task.correlation_id),
        "tenant_id": str(task.tenant_id),
    }


def build_task_started_payload(task, worker_id: str, attempt_number: int) -> dict[str, Any]:
    """Build payload for scheduler.task.started event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "worker_id": worker_id,
        "attempt_number": attempt_number,
        "correlation_id": str(task.correlation_id),
        "tenant_id": str(task.tenant_id),
    }


def build_task_completed_payload(
    task,
    duration_ms: int,
    attempt_number: int,
) -> dict[str, Any]:
    """Build payload for scheduler.task.completed event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "duration_ms": duration_ms,
        "attempt_number": attempt_number,
        "correlation_id": str(task.correlation_id),
        "tenant_id": str(task.tenant_id),
    }


def build_task_failed_payload(
    task,
    error_type: str,
    error_message: str,
    attempt_number: int,
    will_retry: bool,
) -> dict[str, Any]:
    """Build payload for scheduler.task.failed event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "error_type": error_type,
        "error_message": error_message,
        "attempt_number": attempt_number,
        "will_retry": will_retry,
        "correlation_id": str(task.correlation_id),
        "tenant_id": str(task.tenant_id),
    }


def build_retry_scheduled_payload(
    task,
    retry_delay_seconds: int,
) -> dict[str, Any]:
    """Build payload for scheduler.retry.scheduled event."""
    return {
        "task_id": str(task.id),
        "task_name": task.task_name,
        "attempt_number": task.attempts,
        "max_attempts": task.max_attempts,
        "retry_strategy": task.retry_strategy,
        "retry_delay_seconds": retry_delay_seconds,
        "next_retry_at": task.next_retry_at.isoformat() if task.next_retry_at else None,
        "correlation_id": str(task.correlation_id),
        "tenant_id": str(task.tenant_id),
    }


def build_dlq_enqueued_payload(dlq_entry) -> dict[str, Any]:
    """Build payload for scheduler.dlq.enqueued event."""
    return {
        "dlq_id": str(dlq_entry.id),
        "task_id": str(dlq_entry.original_task.id),
        "task_name": dlq_entry.original_task.task_name,
        "failure_reason": dlq_entry.failure_reason,
        "failure_count": dlq_entry.failure_count,
        "correlation_id": str(dlq_entry.original_task.correlation_id),
        "tenant_id": str(dlq_entry.tenant_id),
    }


def build_circuit_opened_payload(circuit, last_error: str = "") -> dict[str, Any]:
    """Build payload for scheduler.circuit.opened event."""
    return {
        "service_name": circuit.service_name,
        "failure_count": circuit.failure_count,
        "failure_threshold": circuit.failure_threshold,
        "last_error": last_error,
    }


def build_circuit_closed_payload(
    circuit,
    recovery_duration_seconds: int,
) -> dict[str, Any]:
    """Build payload for scheduler.circuit.closed event."""
    return {
        "service_name": circuit.service_name,
        "success_count": circuit.success_count,
        "recovery_duration_seconds": recovery_duration_seconds,
    }


def build_schedule_triggered_payload(schedule, task) -> dict[str, Any]:
    """Build payload for scheduler.schedule.triggered event."""
    return {
        "schedule_id": schedule.schedule_id,
        "task_id": str(task.id),
        "run_number": schedule.run_count,
        "next_run_at": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
        "tenant_id": str(schedule.tenant_id),
    }


def build_cascade_child_payload(
    parent_task,
    child_task,
) -> dict[str, Any]:
    """Build payload for scheduler.cascade.child_created event."""
    return {
        "parent_task_id": str(parent_task.id),
        "child_task_id": str(child_task.id),
        "cascade_depth": child_task.cascade_depth,
        "root_correlation_id": str(child_task.root_correlation_id),
        "tenant_id": str(child_task.tenant_id),
    }


def build_quota_exceeded_payload(
    tenant_id: str,
    quota_type: str,
    current_value: int,
    limit: int,
    action_taken: str,
) -> dict[str, Any]:
    """Build payload for scheduler.quota.exceeded event."""
    return {
        "tenant_id": tenant_id,
        "quota_type": quota_type,
        "current_value": current_value,
        "limit": limit,
        "action_taken": action_taken,
    }


def build_worker_heartbeat_payload(
    worker_id: str,
    active_tasks: int,
    completed_tasks: int,
    failed_tasks: int,
    memory_mb: float,
    cpu_percent: float,
) -> dict[str, Any]:
    """Build payload for scheduler.worker.heartbeat event."""
    return {
        "worker_id": worker_id,
        "active_tasks": active_tasks,
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "memory_mb": memory_mb,
        "cpu_percent": cpu_percent,
    }
