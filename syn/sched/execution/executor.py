"""
Task Executor - Isolated Task Execution with Resource Monitoring

Standard: SCH-003 §5 (Task Executor)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The TaskExecutor provides:
- Isolated task execution (process or thread)
- Resource monitoring (memory, CPU time)
- Timeout enforcement
- Graceful cancellation
- Execution context propagation
- Result/error capture
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from django.utils import timezone

from .queue import QueuedTask
from .resource_class import ResourceClass, WorkerConfig, get_worker_config

logger = logging.getLogger(__name__)

# Cross-platform resource module support (Windows doesn't have resource module)
# On Windows, we fall back to time-based approximations for CPU measurement
_HAS_RESOURCE = False
try:
    import resource

    _HAS_RESOURCE = True
except ImportError:
    # Windows fallback - resource module not available
    resource = None  # type: ignore


class ExecutionStatus(Enum):
    """Status of task execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    MEMORY_EXCEEDED = "memory_exceeded"


@dataclass
class ExecutionContext:
    """
    Context passed to task handler during execution.

    Standard: SCH-003 §5.1

    Provides:
    - Task identity (id, name, correlation)
    - Execution metadata (attempt, worker, deadline)
    - Resource limits
    - Cancellation support
    """

    task_id: uuid.UUID
    task_name: str
    correlation_id: uuid.UUID | None = None
    root_correlation_id: uuid.UUID | None = None
    tenant_id: uuid.UUID | None = None
    attempt: int = 1
    max_attempts: int = 3
    worker_id: str = ""
    deadline: datetime | None = None
    timeout_seconds: int = 60
    memory_limit_mb: int = 512
    resource_class: ResourceClass = ResourceClass.MIXED

    # Internal state
    _cancelled: threading.Event = field(default_factory=threading.Event)
    _started_at: datetime | None = None

    def is_cancelled(self) -> bool:
        """Check if execution has been cancelled."""
        return self._cancelled.is_set()

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled.set()

    def check_deadline(self) -> bool:
        """Check if deadline has passed."""
        if self.deadline and timezone.now() > self.deadline:
            return False
        return True

    def elapsed_seconds(self) -> float:
        """Get elapsed execution time."""
        if self._started_at:
            return (timezone.now() - self._started_at).total_seconds()
        return 0.0

    def remaining_seconds(self) -> float:
        """Get remaining time before timeout."""
        elapsed = self.elapsed_seconds()
        return max(0, self.timeout_seconds - elapsed)

    @classmethod
    def from_queued_task(
        cls,
        task: QueuedTask,
        worker_id: str,
        attempt: int = 1,
        timeout_seconds: int | None = None,
    ) -> ExecutionContext:
        """Create context from QueuedTask."""
        config = get_worker_config(task.resource_class)
        base_timeout = timeout_seconds or task.metadata.get("timeout_seconds", 60)
        effective_timeout = int(base_timeout * config.timeout_multiplier)

        return cls(
            task_id=task.task_id,
            task_name=task.task_name,
            correlation_id=task.metadata.get("correlation_id"),
            root_correlation_id=task.metadata.get("root_correlation_id"),
            tenant_id=task.tenant_id,
            attempt=attempt,
            max_attempts=task.metadata.get("max_attempts", 3),
            worker_id=worker_id,
            deadline=timezone.now() + timedelta(seconds=effective_timeout),
            timeout_seconds=effective_timeout,
            memory_limit_mb=config.memory_limit_mb,
            resource_class=task.resource_class,
        )


@dataclass
class ExecutionResult:
    """
    Result of task execution.

    Standard: SCH-003 §5.2
    """

    task_id: uuid.UUID
    status: ExecutionStatus
    result: Any | None = None
    error_message: str | None = None
    error_type: str | None = None
    error_traceback: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_time_ms: float = 0.0
    worker_id: str = ""
    attempt: int = 1

    @property
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    @property
    def is_retryable(self) -> bool:
        """Check if failure is retryable."""
        # Not retryable: permanent errors, cancellation
        non_retryable = {
            ExecutionStatus.CANCELLED,
            ExecutionStatus.SUCCESS,
        }
        if self.status in non_retryable:
            return False

        # Check error type
        permanent_errors = ["ValueError", "TypeError", "KeyError", "AttributeError"]
        if self.error_type in permanent_errors:
            return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": str(self.task_id),
            "status": self.status.value,
            "result": self.result,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "error_traceback": self.error_traceback,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "duration_ms": self.duration_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_time_ms": self.cpu_time_ms,
            "worker_id": self.worker_id,
            "attempt": self.attempt,
        }


def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB (cross-platform)."""
    try:
        if _HAS_RESOURCE and resource is not None:
            # Unix: Use resource module
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # KB to MB on Linux
        else:
            # Windows: Use psutil if available, else return 0
            try:
                import psutil

                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024 * 1024)  # bytes to MB
            except ImportError:
                return 0.0
    except Exception:
        return 0.0


def _get_cpu_time_ms() -> float:
    """Get current process CPU time in milliseconds (cross-platform)."""
    try:
        if _HAS_RESOURCE and resource is not None:
            # Unix: Use resource module for precise CPU measurement
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return (usage.ru_utime + usage.ru_stime) * 1000
        else:
            # Windows: Use time.process_time() as fallback
            return time.process_time() * 1000
    except Exception:
        return 0.0


def _execute_in_subprocess(
    handler: Callable,
    payload: dict[str, Any],
    context_dict: dict[str, Any],
    memory_limit_mb: int,
) -> tuple[Any, str, str, str, float, float]:
    """
    Execute handler in subprocess with resource limits.

    Cross-platform: Memory limits only work on Unix systems with the resource module.

    Returns: (result, error_message, error_type, error_traceback, memory_mb, cpu_ms)
    """
    # Set memory limit (Unix only - resource module required)
    if memory_limit_mb > 0 and sys.platform != "win32":
        try:
            import resource as res

            soft_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
            hard_limit = int(soft_limit * 1.1)  # 10% buffer
            res.setrlimit(res.RLIMIT_AS, (soft_limit, hard_limit))
        except (ImportError, AttributeError, OSError):
            pass  # Limit setting may not be available on all systems

    # Track resources
    start_cpu = _get_cpu_time_ms()
    start_memory = _get_memory_usage_mb()

    try:
        # Reconstruct context (simplified - no threading.Event in subprocess)
        from .executor import ExecutionContext

        context = ExecutionContext(
            task_id=uuid.UUID(context_dict["task_id"]),
            task_name=context_dict["task_name"],
            tenant_id=(uuid.UUID(context_dict["tenant_id"]) if context_dict.get("tenant_id") else None),
            attempt=context_dict.get("attempt", 1),
            worker_id=context_dict.get("worker_id", ""),
            timeout_seconds=context_dict.get("timeout_seconds", 60),
        )
        context._started_at = timezone.now()

        # Execute handler
        result = handler(payload, context)

        # Calculate resource usage
        end_cpu = _get_cpu_time_ms()
        end_memory = _get_memory_usage_mb()

        return (
            result,
            None,  # error_message
            None,  # error_type
            None,  # error_traceback
            max(end_memory - start_memory, 0),
            max(end_cpu - start_cpu, 0),
        )

    except MemoryError as e:
        return (
            None,
            str(e),
            "MemoryError",
            traceback.format_exc(),
            _get_memory_usage_mb(),
            _get_cpu_time_ms() - start_cpu,
        )

    except Exception as e:
        return (
            None,
            str(e),
            type(e).__name__,
            traceback.format_exc(),
            _get_memory_usage_mb() - start_memory,
            _get_cpu_time_ms() - start_cpu,
        )


class TaskExecutor:
    """
    Executes tasks with isolation, monitoring, and timeout enforcement.

    Standard: SCH-003 §5

    Features:
    - Process isolation for memory leak containment
    - Thread execution for lightweight tasks
    - Resource monitoring (memory, CPU)
    - Timeout enforcement with cleanup
    - Graceful cancellation support

    Usage:
        executor = TaskExecutor(worker_id="worker-1")
        result = executor.execute(queued_task, handler)
    """

    def __init__(
        self,
        worker_id: str,
        use_process_isolation: bool = True,
        default_timeout: int = 60,
    ):
        """
        Initialize task executor.

        Args:
            worker_id: Unique identifier for this executor
            use_process_isolation: Use subprocess for isolation
            default_timeout: Default timeout in seconds
        """
        self.worker_id = worker_id
        self.use_process_isolation = use_process_isolation
        self.default_timeout = default_timeout

        # Execution state
        self._current_context: ExecutionContext | None = None
        self._current_future: Future | None = None

        # Thread pool for lightweight tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{worker_id}-")

        # Process pool for isolated tasks
        self._process_pool: ProcessPoolExecutor | None = None
        if use_process_isolation:
            self._process_pool = ProcessPoolExecutor(max_workers=1)

    def execute(
        self,
        task: QueuedTask,
        handler: Callable,
        timeout_override: int | None = None,
    ) -> ExecutionResult:
        """
        Execute a task with the given handler.

        Args:
            task: QueuedTask to execute
            handler: Callable to invoke
            timeout_override: Override timeout in seconds

        Returns:
            ExecutionResult with status and metrics
        """
        # Create context
        context = ExecutionContext.from_queued_task(
            task,
            self.worker_id,
            attempt=task.metadata.get("attempts", 0) + 1,
            timeout_seconds=timeout_override,
        )
        context._started_at = timezone.now()
        self._current_context = context

        # Get worker config
        config = get_worker_config(task.resource_class)

        # Determine isolation mode
        use_process = (
            self.use_process_isolation and config.process_isolation and task.resource_class != ResourceClass.LIGHTWEIGHT
        )

        started_at = timezone.now()
        result = ExecutionResult(
            task_id=task.task_id,
            status=ExecutionStatus.RUNNING,
            started_at=started_at,
            worker_id=self.worker_id,
            attempt=context.attempt,
        )

        try:
            if use_process:
                exec_result = self._execute_in_process(task, handler, context, config)
            else:
                exec_result = self._execute_in_thread(task, handler, context, config)

            result = exec_result

        except FutureTimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error_message = f"Task exceeded timeout of {context.timeout_seconds}s"
            result.error_type = "TimeoutError"
            logger.warning(f"[EXECUTOR] Task {task.task_id} timed out after {context.timeout_seconds}s")

        except Exception as e:
            result.status = ExecutionStatus.FAILURE
            result.error_message = str(e)
            result.error_type = type(e).__name__
            result.error_traceback = traceback.format_exc()
            logger.error(f"[EXECUTOR] Task {task.task_id} failed: {e}")

        finally:
            result.completed_at = timezone.now()
            result.duration_ms = (result.completed_at - started_at).total_seconds() * 1000
            self._current_context = None
            self._current_future = None

        return result

    def _execute_in_process(
        self,
        task: QueuedTask,
        handler: Callable,
        context: ExecutionContext,
        config: WorkerConfig,
    ) -> ExecutionResult:
        """Execute task in isolated subprocess."""
        if not self._process_pool:
            return self._execute_in_thread(task, handler, context, config)

        # Prepare context dict for serialization
        context_dict = {
            "task_id": str(context.task_id),
            "task_name": context.task_name,
            "tenant_id": str(context.tenant_id) if context.tenant_id else None,
            "attempt": context.attempt,
            "worker_id": context.worker_id,
            "timeout_seconds": context.timeout_seconds,
        }

        # Submit to process pool
        future = self._process_pool.submit(
            _execute_in_subprocess,
            handler,
            task.payload,
            context_dict,
            config.memory_limit_mb,
        )
        self._current_future = future

        # Wait with timeout
        try:
            result_tuple = future.result(timeout=context.timeout_seconds)
            result_value, error_msg, error_type, error_tb, memory_mb, cpu_ms = result_tuple

            if error_msg:
                status = ExecutionStatus.MEMORY_EXCEEDED if error_type == "MemoryError" else ExecutionStatus.FAILURE
                return ExecutionResult(
                    task_id=task.task_id,
                    status=status,
                    error_message=error_msg,
                    error_type=error_type,
                    error_traceback=error_tb,
                    memory_peak_mb=memory_mb,
                    cpu_time_ms=cpu_ms,
                    worker_id=self.worker_id,
                    attempt=context.attempt,
                )
            else:
                return ExecutionResult(
                    task_id=task.task_id,
                    status=ExecutionStatus.SUCCESS,
                    result=result_value,
                    memory_peak_mb=memory_mb,
                    cpu_time_ms=cpu_ms,
                    worker_id=self.worker_id,
                    attempt=context.attempt,
                )

        except FutureTimeoutError:
            future.cancel()
            raise

    def _execute_in_thread(
        self,
        task: QueuedTask,
        handler: Callable,
        context: ExecutionContext,
        config: WorkerConfig,
    ) -> ExecutionResult:
        """Execute task in thread (lightweight tasks)."""
        start_memory = _get_memory_usage_mb()
        start_cpu = _get_cpu_time_ms()

        def _run():
            return handler(task.payload, context)

        future = self._thread_pool.submit(_run)
        self._current_future = future

        try:
            result_value = future.result(timeout=context.timeout_seconds)

            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.SUCCESS,
                result=result_value,
                memory_peak_mb=_get_memory_usage_mb() - start_memory,
                cpu_time_ms=_get_cpu_time_ms() - start_cpu,
                worker_id=self.worker_id,
                attempt=context.attempt,
            )

        except FutureTimeoutError:
            future.cancel()
            raise

        except Exception as e:
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILURE,
                error_message=str(e),
                error_type=type(e).__name__,
                error_traceback=traceback.format_exc(),
                memory_peak_mb=_get_memory_usage_mb() - start_memory,
                cpu_time_ms=_get_cpu_time_ms() - start_cpu,
                worker_id=self.worker_id,
                attempt=context.attempt,
            )

    def cancel(self) -> bool:
        """Cancel current execution."""
        if self._current_context:
            self._current_context.cancel()
        if self._current_future:
            return self._current_future.cancel()
        return False

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor pools."""
        self._thread_pool.shutdown(wait=wait)
        if self._process_pool:
            self._process_pool.shutdown(wait=wait)

    def is_busy(self) -> bool:
        """Check if executor is currently running a task."""
        return self._current_context is not None
