"""
Scheduler Primitives (PCONF-001 §16)
====================================

Primitive operations for the Cognitive Scheduler.

Standard:     PCONF-001 §16 (Scheduler Primitives)
Compliance:   SCH-001 §5, SCH-002 §4-10, SCH-003 §4-8
Location:     syn/sched/primitives.py
Version:      1.0.0

Primitives:
Task Operations:
- sched.task.enqueue: Enqueue task (50ms SLO)
- sched.task.cancel: Cancel task (30ms SLO)
- sched.task.status: Get task status (10ms SLO)
- sched.task.retry: Retry failed task (50ms SLO)

Queue Operations:
- sched.queue.depth: Get queue depth (10ms SLO)
- sched.queue.purge: Purge queue (100ms SLO)

Circuit Breaker:
- sched.circuit.status: Get circuit status (10ms SLO)
- sched.circuit.reset: Reset circuit (20ms SLO)

DLQ Operations:
- sched.dlq.list: List DLQ entries (30ms SLO)
- sched.dlq.retry: Retry DLQ entry (50ms SLO)

Priority:
- sched.priority.calculate: Calculate task priority (20ms SLO)
"""

import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS (PCONF-001 §16)
# =============================================================================


class TaskStatus(str, Enum):
    """Task status per PCONF-001 §16 PCONF-SCH-001."""

    PENDING = "pending"
    PRE_FLIGHT = "pre_flight"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    THROTTLED = "throttled"
    CANCELLED = "cancelled"


class WorkerState(str, Enum):
    """Worker state per PCONF-001 §16 PCONF-SCH-103."""

    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    OFFLINE = "offline"


class CircuitState(str, Enum):
    """Circuit breaker state per PCONF-001 §16 PCONF-SCH-201."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Priority calculation formula: P = (U×0.4) + (C×0.3) + ((1-G)×0.2) + ((1-R)×0.1)
PRIORITY_WEIGHTS = {
    "urgency": 0.4,
    "confidence": 0.3,
    "governance": 0.2,
    "resource": 0.1,
}

# Backoff formula: min(base × 2^attempt, max_delay)
BACKOFF_BASE_MS = 1000
BACKOFF_MAX_MS = 60000


# =============================================================================
# TASK PRIMITIVES (PCONF-001 §16.task_operations)
# =============================================================================


def enqueue_task(
    *,
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Enqueue a task for execution.

    Primitive: sched.task.enqueue
    SLO: 50ms
    Standard: PCONF-001 §16 PCONF-SCH-001

    Args:
        params: {task_type: str, payload: dict, priority?: str, governance_required?: bool}
        context: {tenant_id, correlation_id, timestamp}

    Returns:
        {task_id: str, status: str, queued_at: datetime, queue_depth: int}

    Events:
        sched.task.enqueued
    """
    from syn.sched.events import emit_scheduler_event

    task_type = params.get("task_type")
    payload = params.get("payload", {})
    priority = params.get("priority", "NORMAL")
    governance_required = params.get("governance_required", False)
    cascade_budget = params.get("cascade_budget", 10)
    correlation_id = context.get("correlation_id")
    tenant_id = context.get("tenant_id")

    logger.info(f"[SCHED PRIMITIVE] Enqueuing task: {task_type}")

    try:
        from syn.sched.models import ScheduledTask

        task = ScheduledTask.objects.create(
            tenant_id=tenant_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            status=TaskStatus.QUEUED.value,
            governance_required=governance_required,
            cascade_budget=cascade_budget,
            correlation_id=correlation_id,
        )

        # Get queue depth
        queue_depth = ScheduledTask.objects.filter(
            tenant_id=tenant_id,
            status=TaskStatus.QUEUED.value,
        ).count()

        emit_scheduler_event(
            "sched.task.enqueued",
            {
                "task_id": str(task.id),
                "task_type": task_type,
                "priority": priority,
                "queue_depth": queue_depth,
            },
            correlation_id=correlation_id,
            tenant_id=tenant_id,
        )

        return {
            "task_id": str(task.id),
            "status": task.status,
            "queued_at": task.created_at.isoformat(),
            "queue_depth": queue_depth,
        }

    except Exception as e:
        logger.error(f"[SCHED PRIMITIVE] Enqueue failed: {e}")
        return {"error": str(e), "task_id": None}


def cancel_task(
    *,
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cancel a pending task.

    Primitive: sched.task.cancel
    SLO: 30ms
    Standard: PCONF-001 §16 PCONF-SCH-002

    Args:
        params: {task_id: str, reason?: str}
        context: {tenant_id, correlation_id}

    Returns:
        {cancelled: bool, previous_status: str}

    Events:
        sched.task.cancelled
    """
    from syn.sched.events import emit_scheduler_event

    task_id = params.get("task_id")
    reason = params.get("reason", "User requested")
    correlation_id = context.get("correlation_id")
    tenant_id = context.get("tenant_id")

    try:
        from syn.sched.models import ScheduledTask

        task = ScheduledTask.objects.get(id=task_id, tenant_id=tenant_id)
        previous_status = task.status

        # Can only cancel pending/queued tasks
        if task.status not in [TaskStatus.PENDING.value, TaskStatus.QUEUED.value]:
            return {
                "cancelled": False,
                "previous_status": previous_status,
                "error": f"Cannot cancel task in status: {previous_status}",
            }

        task.status = TaskStatus.CANCELLED.value
        task.cancelled_reason = reason
        task.save()

        emit_scheduler_event(
            "sched.task.cancelled",
            {
                "task_id": str(task_id),
                "previous_status": previous_status,
                "reason": reason,
            },
            correlation_id=correlation_id,
            tenant_id=tenant_id,
        )

        return {
            "cancelled": True,
            "previous_status": previous_status,
        }

    except Exception as e:
        logger.error(f"[SCHED PRIMITIVE] Cancel failed: {e}")
        return {"error": str(e), "cancelled": False}


def get_task_status(
    *,
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get task status.

    Primitive: sched.task.status
    SLO: 10ms
    Pure: Yes
    Standard: PCONF-001 §16 PCONF-SCH-003

    Args:
        params: {task_id: str}
        context: {tenant_id, correlation_id}

    Returns:
        {status: str, progress_pct: int, worker_id: str, started_at: datetime}
    """
    task_id = params.get("task_id")
    tenant_id = context.get("tenant_id")

    try:
        from syn.sched.models import ScheduledTask

        task = ScheduledTask.objects.get(id=task_id, tenant_id=tenant_id)

        return {
            "status": task.status,
            "progress_pct": getattr(task, "progress_pct", 0),
            "worker_id": getattr(task, "worker_id", None),
            "started_at": task.started_at.isoformat() if task.started_at else None,
        }

    except Exception as e:
        logger.error(f"[SCHED PRIMITIVE] Get status failed: {e}")
        return {"error": str(e)}


def retry_task(
    *,
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Retry a failed task with exponential backoff.

    Primitive: sched.task.retry
    SLO: 50ms
    Formula: min(base × 2^attempt, max_delay)
    Standard: PCONF-001 §16 PCONF-SCH-004

    Args:
        params: {task_id: str, delay_ms?: int, max_retries?: int}
        context: {tenant_id, correlation_id}

    Returns:
        {retry_count: int, next_attempt_at: datetime, backoff_ms: int}

    Events:
        sched.task.enqueued
    """
    from syn.sched.events import emit_scheduler_event

    task_id = params.get("task_id")
    delay_ms = params.get("delay_ms")
    max_retries = params.get("max_retries", 5)
    correlation_id = context.get("correlation_id")
    tenant_id = context.get("tenant_id")

    try:
        from syn.sched.models import ScheduledTask

        task = ScheduledTask.objects.get(id=task_id, tenant_id=tenant_id)

        if task.retry_count >= max_retries:
            return {
                "error": f"Max retries ({max_retries}) exceeded",
                "retry_count": task.retry_count,
            }

        # Calculate backoff
        attempt = task.retry_count + 1
        if delay_ms is None:
            delay_ms = min(BACKOFF_BASE_MS * (2 ** attempt), BACKOFF_MAX_MS)

        next_attempt_at = datetime.utcnow() + timedelta(milliseconds=delay_ms)

        task.status = TaskStatus.QUEUED.value
        task.retry_count = attempt
        task.next_attempt_at = next_attempt_at
        task.save()

        emit_scheduler_event(
            "sched.task.enqueued",
            {
                "task_id": str(task_id),
                "retry_count": attempt,
                "backoff_ms": delay_ms,
            },
            correlation_id=correlation_id,
            tenant_id=tenant_id,
        )

        return {
            "retry_count": attempt,
            "next_attempt_at": next_attempt_at.isoformat(),
            "backoff_ms": delay_ms,
        }

    except Exception as e:
        logger.error(f"[SCHED PRIMITIVE] Retry failed: {e}")
        return {"error": str(e)}


# =============================================================================
# QUEUE PRIMITIVES (PCONF-001 §16.queue_operations)
# =============================================================================


def get_queue_depth(
    *,
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get queue depth and metrics.

    Primitive: sched.queue.depth
    SLO: 10ms
    Pure: Yes
    Standard: PCONF-001 §16 PCONF-SCH-101

    Args:
        params: {queue_name: str}
        context: {tenant_id, correlation_id}

    Returns:
        {depth: int, oldest_task_age_ms: int, processing_rate: float}
    """
    queue_name = params.get("queue_name", "default")
    tenant_id = context.get("tenant_id")

    try:
        from syn.sched.models import CognitiveTask, TaskExecution

        queued = CognitiveTask.objects.filter(
            tenant_id=tenant_id,
            state=TaskStatus.QUEUED.value,
        )
        if queue_name and queue_name != "default":
            queued = queued.filter(queue_type=queue_name)

        depth = queued.count()

        # Get oldest task age
        oldest = queued.order_by("created_at").first()
        oldest_age_ms = 0
        if oldest:
            oldest_age_ms = int((datetime.utcnow() - oldest.created_at).total_seconds() * 1000)

        # Calculate processing rate (tasks completed per minute over last 5 minutes)
        processing_rate = 0.0
        try:
            five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
            completed_count = TaskExecution.objects.filter(
                task__tenant_id=tenant_id,
                completed_at__gte=five_minutes_ago,
                success=True,
            ).count()
            # Convert to tasks per minute
            processing_rate = round(completed_count / 5.0, 2)
        except Exception:
            # If TaskExecution query fails, default to 0.0
            pass

        return {
            "depth": depth,
            "oldest_task_age_ms": oldest_age_ms,
            "processing_rate": processing_rate,
        }

    except Exception as e:
        logger.error(f"[SCHED PRIMITIVE] Queue depth failed: {e}")
        return {"error": str(e)}


# =============================================================================
# CIRCUIT BREAKER PRIMITIVES (PCONF-001 §16.circuit_breaker)
# =============================================================================


def get_circuit_status(
    *,
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get circuit breaker status.

    Primitive: sched.circuit.status
    SLO: 10ms
    Pure: Yes
    State machine: CLOSED → OPEN → HALF_OPEN → CLOSED
    Standard: PCONF-001 §16 PCONF-SCH-201

    Args:
        params: {circuit_id: str}
        context: {tenant_id, correlation_id}

    Returns:
        {state: str, failure_count: int, last_failure_at: datetime, next_probe_at: datetime}
    """
    circuit_id = params.get("circuit_id")
    tenant_id = context.get("tenant_id")

    try:
        from syn.sched.models import CircuitBreaker

        circuit = CircuitBreaker.objects.get(id=circuit_id, tenant_id=tenant_id)

        return {
            "state": circuit.state,
            "failure_count": circuit.failure_count,
            "last_failure_at": circuit.last_failure_at.isoformat() if circuit.last_failure_at else None,
            "next_probe_at": circuit.next_probe_at.isoformat() if circuit.next_probe_at else None,
        }

    except Exception as e:
        logger.error(f"[SCHED PRIMITIVE] Circuit status failed: {e}")
        return {"error": str(e)}


def reset_circuit(
    *,
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reset circuit breaker to closed state.

    Primitive: sched.circuit.reset
    SLO: 20ms
    Standard: PCONF-001 §16 PCONF-SCH-202

    Args:
        params: {circuit_id: str, force?: bool}
        context: {tenant_id, correlation_id}

    Returns:
        {previous_state: str, new_state: str, reset_at: datetime}

    Events:
        sched.circuit.closed
    """
    from syn.sched.events import emit_scheduler_event

    circuit_id = params.get("circuit_id")
    force = params.get("force", False)
    correlation_id = context.get("correlation_id")
    tenant_id = context.get("tenant_id")

    try:
        from syn.sched.models import CircuitBreaker

        circuit = CircuitBreaker.objects.get(id=circuit_id, tenant_id=tenant_id)
        previous_state = circuit.state

        if not force and circuit.state == CircuitState.OPEN.value:
            return {
                "error": "Cannot reset OPEN circuit without force=true",
                "previous_state": previous_state,
            }

        circuit.state = CircuitState.CLOSED.value
        circuit.failure_count = 0
        circuit.last_failure_at = None
        circuit.save()

        emit_scheduler_event(
            "sched.circuit.closed",
            {
                "circuit_id": str(circuit_id),
                "previous_state": previous_state,
            },
            correlation_id=correlation_id,
            tenant_id=tenant_id,
        )

        return {
            "previous_state": previous_state,
            "new_state": CircuitState.CLOSED.value,
            "reset_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[SCHED PRIMITIVE] Circuit reset failed: {e}")
        return {"error": str(e)}


# =============================================================================
# PRIORITY PRIMITIVES (PCONF-001 §16.priority_operations)
# =============================================================================


def calculate_priority(
    *,
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate task priority score.

    Primitive: sched.priority.calculate
    SLO: 20ms
    Pure: Yes
    Formula: P = (U×0.4) + (C×0.3) + ((1-G)×0.2) + ((1-R)×0.1)
    Standard: PCONF-001 §16 PCONF-SCH-501

    Args:
        params: {task_context: dict} with urgency, confidence, governance, resource scores
        context: {tenant_id, correlation_id}

    Returns:
        {priority_score: float, factors: dict}
    """
    task_context = params.get("task_context", {})

    # Extract factors (0.0 to 1.0)
    urgency = task_context.get("urgency", 0.5)
    confidence = task_context.get("confidence", 0.5)
    governance = task_context.get("governance", 0.0)  # 0 = no governance needed
    resource = task_context.get("resource", 0.0)  # 0 = low resource usage

    # Calculate priority: P = (U×0.4) + (C×0.3) + ((1-G)×0.2) + ((1-R)×0.1)
    priority_score = (
        (urgency * PRIORITY_WEIGHTS["urgency"])
        + (confidence * PRIORITY_WEIGHTS["confidence"])
        + ((1 - governance) * PRIORITY_WEIGHTS["governance"])
        + ((1 - resource) * PRIORITY_WEIGHTS["resource"])
    )

    return {
        "priority_score": round(priority_score, 4),
        "factors": {
            "urgency": urgency,
            "confidence": confidence,
            "governance": governance,
            "resource": resource,
        },
    }


# =============================================================================
# PRIMITIVE REGISTRY
# =============================================================================


SCHED_PRIMITIVES = {
    # Task operations
    "sched.task.enqueue": enqueue_task,
    "sched.task.cancel": cancel_task,
    "sched.task.status": get_task_status,
    "sched.task.retry": retry_task,
    # Queue operations
    "sched.queue.depth": get_queue_depth,
    # Circuit breaker
    "sched.circuit.status": get_circuit_status,
    "sched.circuit.reset": reset_circuit,
    # Priority
    "sched.priority.calculate": calculate_priority,
}


def get_primitive(name: str):
    """Get scheduler primitive by name."""
    return SCHED_PRIMITIVES.get(name)


def list_primitives() -> List[str]:
    """List all scheduler primitive names."""
    return list(SCHED_PRIMITIVES.keys())
