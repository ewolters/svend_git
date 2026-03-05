"""
Scheduler Exceptions

Standard: SCH-001 §4.1 (Error Classification per ERR-001)
Author: Systems Architect
Version: 1.0
Date: 2025-12-10

This module defines all exceptions raised by the CognitiveScheduler.
These are classified per ERR-001 error handling standard.

Exception Hierarchy:
    SchedulerError (base)
    ├── SubmissionError (task cannot be submitted)
    │   ├── QuotaExceededError
    │   ├── ThrottledError
    │   └── ValidationError
    ├── ExecutionError (task execution failed)
    │   ├── CircuitOpenError
    │   ├── CascadeLimitError
    │   └── TimeoutError
    └── ConfigurationError (invalid configuration)

Usage:
    from syn.sched.exceptions import QuotaExceededError, ThrottledError

    try:
        scheduler.submit(task)
    except QuotaExceededError as e:
        logger.warning(f"Quota exceeded: {e}")
    except ThrottledError as e:
        logger.info(f"Throttled, retry after {e.retry_after_seconds}s")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from syn.sched.backpressure.throttle import ThrottleLevel


# =============================================================================
# BASE EXCEPTION
# =============================================================================


class SchedulerError(Exception):
    """
    Base exception for all scheduler errors.

    Standard: ERR-001 §4 (Error Classification)

    Attributes:
        code: Machine-readable error code
        details: Additional context for debugging
    """

    code: str = "SCHED_ERROR"

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        if code:
            self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception for logging/API responses."""
        return {
            "error": self.code,
            "message": str(self),
            "details": self.details,
        }


# =============================================================================
# SUBMISSION ERRORS
# =============================================================================


class SubmissionError(SchedulerError):
    """Task could not be submitted to the scheduler."""

    code = "SCHED_SUBMISSION_ERROR"


class QuotaExceededError(SubmissionError):
    """
    Raised when tenant quota is exceeded.

    Standard: SCH-001 §5.2 (Tenant Isolation)

    This error indicates the tenant has reached their task submission limit.
    The caller should either:
    - Wait for existing tasks to complete
    - Request a quota increase
    - Reduce submission rate

    Attributes:
        tenant_id: The tenant that exceeded quota
        current_count: Current task count
        quota_limit: Maximum allowed
    """

    code = "SCHED_QUOTA_EXCEEDED"

    def __init__(
        self,
        message: str = "Tenant quota exceeded",
        tenant_id: Optional[int] = None,
        current_count: int = 0,
        quota_limit: int = 0,
    ):
        super().__init__(
            message,
            details={
                "tenant_id": tenant_id,
                "current_count": current_count,
                "quota_limit": quota_limit,
            },
        )
        self.tenant_id = tenant_id
        self.current_count = current_count
        self.quota_limit = quota_limit


class ThrottledError(SubmissionError):
    """
    Raised when backpressure throttling prevents task submission.

    Standard: SCH-004 §4.3 (Backpressure Controller)

    This error indicates the system is under load and cannot accept
    new tasks at the current time. The caller should:
    - Wait for retry_after_seconds before retrying
    - Reduce submission rate
    - Check system health via dashboard

    Attributes:
        throttle_level: Current throttle level (NONE, LIGHT, MODERATE, etc.)
        retry_after_seconds: Suggested delay before retry
        reasons: List of reasons for throttling
    """

    code = "SCHED_THROTTLED"

    def __init__(
        self,
        message: str = "Task throttled due to backpressure",
        throttle_level: Optional["ThrottleLevel"] = None,
        retry_after_seconds: float = 0.0,
        reasons: Optional[List[str]] = None,
    ):
        # Import here to avoid circular imports
        from syn.sched.backpressure.throttle import ThrottleLevel

        if throttle_level is None:
            throttle_level = ThrottleLevel.NONE

        super().__init__(
            message,
            details={
                "throttle_level": throttle_level.value if throttle_level else "NONE",
                "retry_after_seconds": retry_after_seconds,
                "reasons": reasons or [],
            },
        )
        self.throttle_level = throttle_level
        self.retry_after_seconds = retry_after_seconds
        self.reasons = reasons or []


class TaskValidationError(SubmissionError):
    """
    Raised when task validation fails.

    Standard: SCH-002 §3 (Task Model)

    This error indicates the task definition is invalid.
    Check the field_errors for specific issues.

    Attributes:
        field_errors: Dict mapping field names to error messages
    """

    code = "SCHED_VALIDATION_ERROR"

    def __init__(
        self,
        message: str = "Task validation failed",
        field_errors: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message, details={"field_errors": field_errors or {}})
        self.field_errors = field_errors or {}


class HandlerNotFoundError(SubmissionError):
    """
    Raised when no handler is registered for a task name.

    Standard: SCH-001 §4.3 (Task Registry)

    Attributes:
        task_name: The task name that was not found
        available_handlers: List of registered handler names
    """

    code = "SCHED_HANDLER_NOT_FOUND"

    def __init__(
        self,
        task_name: str,
        available_handlers: Optional[List[str]] = None,
    ):
        super().__init__(
            f"No handler registered for task: {task_name}",
            details={
                "task_name": task_name,
                "available_handlers": available_handlers or [],
            },
        )
        self.task_name = task_name
        self.available_handlers = available_handlers or []


# =============================================================================
# EXECUTION ERRORS
# =============================================================================


class ExecutionError(SchedulerError):
    """Task execution failed."""

    code = "SCHED_EXECUTION_ERROR"


class CircuitOpenError(ExecutionError):
    """
    Raised when circuit breaker is open.

    Standard: SCH-001 §6 (Circuit Breaker per ERR-001)

    This error indicates the target service is experiencing failures
    and the circuit breaker has tripped. The caller should:
    - Wait for the circuit to close (half-open check)
    - Check service health
    - Consider alternative fallbacks

    Attributes:
        service_name: The protected service name
        failure_count: Number of consecutive failures
        opened_at: When the circuit opened
    """

    code = "SCHED_CIRCUIT_OPEN"

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        service_name: Optional[str] = None,
        failure_count: int = 0,
        opened_at: Optional[str] = None,
    ):
        super().__init__(
            message,
            details={
                "service_name": service_name,
                "failure_count": failure_count,
                "opened_at": opened_at,
            },
        )
        self.service_name = service_name
        self.failure_count = failure_count
        self.opened_at = opened_at


class CascadeLimitError(ExecutionError):
    """
    Raised when cascade depth limit is exceeded.

    Standard: SCH-001 §5 (Cascade Control)

    This error indicates a task has triggered too many child tasks,
    potentially causing cascade overload. The caller should:
    - Review task design for unbounded recursion
    - Implement explicit depth limits
    - Consider batch processing instead

    Attributes:
        cascade_depth: Current cascade depth
        max_depth: Maximum allowed depth
        correlation_id: Root correlation ID for tracing
    """

    code = "SCHED_CASCADE_LIMIT"

    def __init__(
        self,
        message: str = "Cascade depth limit exceeded",
        cascade_depth: int = 0,
        max_depth: int = 5,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(
            message,
            details={
                "cascade_depth": cascade_depth,
                "max_depth": max_depth,
                "correlation_id": correlation_id,
            },
        )
        self.cascade_depth = cascade_depth
        self.max_depth = max_depth
        self.correlation_id = correlation_id


class TaskTimeoutError(ExecutionError):
    """
    Raised when task execution times out.

    Standard: SCH-003 §5 (Task Executor)

    Attributes:
        task_id: The task that timed out
        timeout_seconds: Configured timeout
        elapsed_seconds: Actual execution time
    """

    code = "SCHED_TIMEOUT"

    def __init__(
        self,
        message: str = "Task execution timed out",
        task_id: Optional[str] = None,
        timeout_seconds: float = 0.0,
        elapsed_seconds: float = 0.0,
    ):
        super().__init__(
            message,
            details={
                "task_id": task_id,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
        )
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class ResourceExceededError(ExecutionError):
    """
    Raised when task exceeds resource limits.

    Standard: SCH-003 §5 (Execution Context)

    Attributes:
        resource_type: Type of resource exceeded (memory, cpu)
        limit: Configured limit
        actual: Actual usage
    """

    code = "SCHED_RESOURCE_EXCEEDED"

    def __init__(
        self,
        message: str = "Task exceeded resource limits",
        resource_type: str = "unknown",
        limit: float = 0.0,
        actual: float = 0.0,
    ):
        super().__init__(
            message,
            details={
                "resource_type": resource_type,
                "limit": limit,
                "actual": actual,
            },
        )
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class ConfigurationError(SchedulerError):
    """Invalid scheduler configuration."""

    code = "SCHED_CONFIG_ERROR"


class ScheduleConfigError(ConfigurationError):
    """Invalid schedule configuration."""

    code = "SCHED_SCHEDULE_CONFIG_ERROR"


# =============================================================================
# STATE ERRORS
# =============================================================================


class StateError(SchedulerError):
    """Scheduler state error."""

    code = "SCHED_STATE_ERROR"


class TaskNotFoundError(StateError):
    """Task not found in scheduler."""

    code = "SCHED_TASK_NOT_FOUND"

    def __init__(self, task_id: str):
        super().__init__(
            f"Task not found: {task_id}",
            details={"task_id": task_id},
        )
        self.task_id = task_id


class InvalidStateTransitionError(StateError):
    """Invalid task state transition."""

    code = "SCHED_INVALID_STATE_TRANSITION"

    def __init__(
        self,
        task_id: str,
        current_state: str,
        target_state: str,
    ):
        super().__init__(
            f"Cannot transition task {task_id} from {current_state} to {target_state}",
            details={
                "task_id": task_id,
                "current_state": current_state,
                "target_state": target_state,
            },
        )
        self.task_id = task_id
        self.current_state = current_state
        self.target_state = target_state


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base
    "SchedulerError",
    # Submission
    "SubmissionError",
    "QuotaExceededError",
    "ThrottledError",
    "TaskValidationError",
    "HandlerNotFoundError",
    # Execution
    "ExecutionError",
    "CircuitOpenError",
    "CascadeLimitError",
    "TaskTimeoutError",
    "ResourceExceededError",
    # Configuration
    "ConfigurationError",
    "ScheduleConfigError",
    # State
    "StateError",
    "TaskNotFoundError",
    "InvalidStateTransitionError",
]
