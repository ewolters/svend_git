"""
Worker Pool - Process Pool Management with Auto-Scaling

Standard: SCH-003 §6 (Worker Pool)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The WorkerPool provides:
- Resource-class-aware worker groups
- Auto-scaling based on queue depth and utilization
- Circuit breaker integration at pool level
- Health monitoring and worker recycling
- Graceful shutdown with task draining
- Metrics collection for observability

Architecture:
    Scheduler
        ↓
    ExecutionQueue (in-memory priority queue)
        ↓
    WorkerPool
        ├── WorkerGroup[CPU_BOUND] → TaskExecutor[] → Handler
        ├── WorkerGroup[IO_BOUND] → TaskExecutor[] → Handler
        ├── WorkerGroup[MIXED] → TaskExecutor[] → Handler
        └── WorkerGroup[LIGHTWEIGHT] → TaskExecutor[] → Handler

Replaces: Celery Worker (full autonomy)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from django.utils import timezone

from .executor import ExecutionResult, ExecutionStatus, TaskExecutor
from .queue import ExecutionQueue, QueuedTask
from .resource_class import (
    WORKER_CONFIGS,
    ResourceClass,
    WorkerConfig,
)

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """State of an individual worker."""

    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    RECYCLING = "recycling"


@dataclass
class WorkerInfo:
    """Information about a single worker."""

    worker_id: str
    resource_class: ResourceClass
    state: WorkerState = WorkerState.STARTING
    started_at: datetime | None = None
    last_heartbeat: datetime | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task_id: uuid.UUID | None = None
    current_task_started: datetime | None = None


@dataclass
class WorkerPoolMetrics:
    """
    Metrics for the worker pool.

    Standard: SCH-003 §6.3
    """

    total_workers: int = 0
    active_workers: int = 0
    idle_workers: int = 0
    workers_by_class: dict[str, int] = field(default_factory=dict)
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_timed_out: int = 0
    avg_task_duration_ms: float = 0.0
    queue_depth: int = 0
    utilization: float = 0.0
    last_scale_time: datetime | None = None
    scale_up_count: int = 0
    scale_down_count: int = 0
    workers_recycled: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_workers": self.total_workers,
            "active_workers": self.active_workers,
            "idle_workers": self.idle_workers,
            "workers_by_class": self.workers_by_class,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_timed_out": self.tasks_timed_out,
            "avg_task_duration_ms": self.avg_task_duration_ms,
            "queue_depth": self.queue_depth,
            "utilization": self.utilization,
            "last_scale_time": (
                self.last_scale_time.isoformat() if self.last_scale_time else None
            ),
            "scale_up_count": self.scale_up_count,
            "scale_down_count": self.scale_down_count,
            "workers_recycled": self.workers_recycled,
        }


class WorkerGroup:
    """
    Group of workers for a specific resource class.

    Standard: SCH-003 §6.1

    Features:
    - Maintains worker count within min/max bounds
    - Auto-scales based on utilization
    - Recycles workers after task threshold
    - Health monitoring via heartbeats
    """

    def __init__(
        self,
        resource_class: ResourceClass,
        config: WorkerConfig,
        task_handler_getter: Callable[[str], Callable | None],
        result_callback: Callable[[ExecutionResult], None],
    ):
        """
        Initialize worker group.

        Args:
            resource_class: Resource class for this group
            config: Worker configuration
            task_handler_getter: Function to get handler by task name
            result_callback: Callback for execution results
        """
        self.resource_class = resource_class
        self.config = config
        self._get_handler = task_handler_getter
        self._on_result = result_callback

        # Workers
        self._workers: dict[str, WorkerInfo] = {}
        self._executors: dict[str, TaskExecutor] = {}
        self._worker_threads: dict[str, threading.Thread] = {}

        # State
        self._running = False
        self._lock = threading.RLock()
        self._last_scale_time: datetime | None = None

        # Metrics
        self._task_durations: list[float] = []

    def start(self) -> None:
        """Start the worker group."""
        with self._lock:
            if self._running:
                return

            self._running = True

            # Create initial workers
            for i in range(self.config.default_workers):
                self._create_worker()

            logger.info(
                f"[WORKER_GROUP] Started {self.resource_class.value} with {len(self._workers)} workers"
            )

    def stop(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """Stop all workers in the group."""
        with self._lock:
            self._running = False

            # Signal all workers to stop
            for worker_id, info in self._workers.items():
                info.state = WorkerState.STOPPING

            # Shutdown executors
            for executor in self._executors.values():
                executor.shutdown(wait=graceful)

            # Wait for threads if graceful
            if graceful:
                deadline = time.time() + timeout
                for thread in self._worker_threads.values():
                    remaining = max(0, deadline - time.time())
                    thread.join(timeout=remaining)

            self._workers.clear()
            self._executors.clear()
            self._worker_threads.clear()

            logger.info(f"[WORKER_GROUP] Stopped {self.resource_class.value}")

    def _create_worker(self) -> str:
        """Create a new worker."""
        worker_id = f"{self.resource_class.value}-{uuid.uuid4().hex[:8]}"

        # Create executor
        executor = TaskExecutor(
            worker_id=worker_id,
            use_process_isolation=self.config.process_isolation,
            default_timeout=int(60 * self.config.timeout_multiplier),
        )

        # Create worker info
        info = WorkerInfo(
            worker_id=worker_id,
            resource_class=self.resource_class,
            state=WorkerState.IDLE,
            started_at=timezone.now(),
            last_heartbeat=timezone.now(),
        )

        self._workers[worker_id] = info
        self._executors[worker_id] = executor

        logger.debug(f"[WORKER_GROUP] Created worker {worker_id}")
        return worker_id

    def _remove_worker(self, worker_id: str) -> None:
        """Remove a worker."""
        with self._lock:
            if worker_id in self._executors:
                self._executors[worker_id].shutdown(wait=False)
                del self._executors[worker_id]

            if worker_id in self._workers:
                del self._workers[worker_id]

            if worker_id in self._worker_threads:
                del self._worker_threads[worker_id]

            logger.debug(f"[WORKER_GROUP] Removed worker {worker_id}")

    def execute_task(self, task: QueuedTask) -> ExecutionResult | None:
        """
        Execute a task using an available worker.

        Args:
            task: Task to execute

        Returns:
            ExecutionResult or None if no worker available
        """
        with self._lock:
            # Find idle worker
            worker_id = None
            for wid, info in self._workers.items():
                if info.state == WorkerState.IDLE:
                    worker_id = wid
                    break

            if not worker_id:
                return None  # No worker available

            info = self._workers[worker_id]
            executor = self._executors[worker_id]

            # Get handler
            handler = self._get_handler(task.task_name)
            if not handler:
                logger.error(f"[WORKER_GROUP] No handler for task {task.task_name}")
                return ExecutionResult(
                    task_id=task.task_id,
                    status=ExecutionStatus.FAILURE,
                    error_message=f"No handler registered for {task.task_name}",
                    error_type="HandlerNotFoundError",
                    worker_id=worker_id,
                )

            # Update worker state
            info.state = WorkerState.BUSY
            info.current_task_id = task.task_id
            info.current_task_started = timezone.now()

        # Execute (outside lock to allow concurrent execution)
        try:
            result = executor.execute(task, handler)
        except Exception as e:
            result = ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILURE,
                error_message=str(e),
                error_type=type(e).__name__,
                worker_id=worker_id,
            )

        # Update state and metrics
        with self._lock:
            if worker_id in self._workers:
                info = self._workers[worker_id]
                info.state = WorkerState.IDLE
                info.current_task_id = None
                info.current_task_started = None
                info.last_heartbeat = timezone.now()

                if result.is_success:
                    info.tasks_completed += 1
                else:
                    info.tasks_failed += 1

                # Track duration
                if result.duration_ms > 0:
                    self._task_durations.append(result.duration_ms)
                    if len(self._task_durations) > 1000:
                        self._task_durations = self._task_durations[-1000:]

                # Check if worker needs recycling
                total_tasks = info.tasks_completed + info.tasks_failed
                if total_tasks >= self.config.recycle_after_tasks:
                    self._recycle_worker(worker_id)

        # Callback
        self._on_result(result)

        return result

    def _recycle_worker(self, worker_id: str) -> None:
        """Recycle a worker (replace with fresh instance)."""
        # Get worker info before removal for event
        info = self._workers.get(worker_id)
        tasks_completed = info.tasks_completed if info else 0
        tasks_failed = info.tasks_failed if info else 0

        logger.info(f"[WORKER_GROUP] Recycling worker {worker_id}")

        # Remove old worker
        self._remove_worker(worker_id)

        # Create replacement
        if self._running and len(self._workers) < self.config.min_workers:
            self._create_worker()

        # Emit recycled event (SCH-003 compliance)
        self._emit_worker_recycled_event(worker_id, tasks_completed, tasks_failed)

    def _emit_worker_recycled_event(
        self, worker_id: str, tasks_completed: int, tasks_failed: int
    ) -> None:
        """Emit scheduler.worker.recycled event per SCH-003."""
        try:
            from syn.synara.cortex import Cortex

            Cortex.emit(
                "scheduler.worker.recycled",
                {
                    "worker_id": worker_id,
                    "resource_class": self.resource_class.value,
                    "reason": "task_threshold",
                    "tasks_completed": tasks_completed,
                    "tasks_failed": tasks_failed,
                    "timestamp": timezone.now().isoformat(),
                },
            )
        except Exception as e:
            logger.debug(f"[WORKER_GROUP] Non-critical event emission failed: {e}")

    def scale(self, target_workers: int) -> int:
        """
        Scale worker count to target.

        Args:
            target_workers: Target worker count

        Returns:
            Number of workers after scaling
        """
        with self._lock:
            target = max(
                self.config.min_workers, min(self.config.max_workers, target_workers)
            )
            current = len(self._workers)

            if target > current:
                # Scale up
                for _ in range(target - current):
                    self._create_worker()
                logger.info(
                    f"[WORKER_GROUP] Scaled up {self.resource_class.value}: {current} → {target}"
                )
                # Emit scaled event (SCH-003 compliance)
                self._emit_worker_scaled_event(current, target, "up")

            elif target < current:
                # Scale down (remove idle workers first)
                to_remove = current - target
                removed = 0
                for worker_id, info in list(self._workers.items()):
                    if info.state == WorkerState.IDLE and removed < to_remove:
                        self._remove_worker(worker_id)
                        removed += 1
                new_count = len(self._workers)
                logger.info(
                    f"[WORKER_GROUP] Scaled down {self.resource_class.value}: {current} → {new_count}"
                )
                # Emit scaled event (SCH-003 compliance)
                self._emit_worker_scaled_event(current, new_count, "down")

            self._last_scale_time = timezone.now()
            return len(self._workers)

    def _emit_worker_scaled_event(
        self, previous_count: int, new_count: int, direction: str
    ) -> None:
        """Emit scheduler.worker.scaled event per SCH-003."""
        try:
            from syn.synara.cortex import Cortex

            Cortex.emit(
                "scheduler.worker.scaled",
                {
                    "resource_class": self.resource_class.value,
                    "previous_count": previous_count,
                    "new_count": new_count,
                    "direction": direction,
                    "timestamp": timezone.now().isoformat(),
                },
            )
        except Exception as e:
            logger.debug(f"[WORKER_GROUP] Non-critical event emission failed: {e}")

    def get_available_capacity(self) -> int:
        """Get number of tasks this group can accept."""
        with self._lock:
            idle_workers = sum(
                1 for w in self._workers.values() if w.state == WorkerState.IDLE
            )
            return idle_workers * self.config.max_concurrent_per_worker

    def get_utilization(self) -> float:
        """Get current utilization (0.0 to 1.0)."""
        with self._lock:
            total = len(self._workers)
            if total == 0:
                return 0.0
            busy = sum(1 for w in self._workers.values() if w.state == WorkerState.BUSY)
            return busy / total

    def get_metrics(self) -> dict[str, Any]:
        """Get group metrics."""
        with self._lock:
            return {
                "resource_class": self.resource_class.value,
                "total_workers": len(self._workers),
                "idle_workers": sum(
                    1 for w in self._workers.values() if w.state == WorkerState.IDLE
                ),
                "busy_workers": sum(
                    1 for w in self._workers.values() if w.state == WorkerState.BUSY
                ),
                "utilization": self.get_utilization(),
                "avg_duration_ms": (
                    sum(self._task_durations) / len(self._task_durations)
                    if self._task_durations
                    else 0
                ),
                "capacity": self.get_available_capacity(),
            }


class WorkerPool:
    """
    Main worker pool orchestrator.

    Standard: SCH-003 §6

    Features:
    - Manages multiple WorkerGroups (one per resource class)
    - Dispatches tasks to appropriate group
    - Auto-scales based on queue depth
    - Monitors health and recycles workers
    - Provides unified metrics

    Usage:
        pool = WorkerPool()
        pool.start()

        # Submit tasks via execution queue
        queue.enqueue(task)

        # Pool automatically dispatches
        pool.dispatch_loop()

        pool.stop()
    """

    def __init__(
        self,
        execution_queue: ExecutionQueue | None = None,
        worker_configs: dict[ResourceClass, WorkerConfig] | None = None,
        auto_scale: bool = True,
        scale_interval_seconds: int = 30,
    ):
        """
        Initialize worker pool.

        Args:
            execution_queue: Queue to dispatch from (or create new)
            worker_configs: Override default worker configs
            auto_scale: Enable automatic scaling
            scale_interval_seconds: Interval between scaling checks
        """
        self._queue = execution_queue or ExecutionQueue()
        self._configs = worker_configs or WORKER_CONFIGS.copy()
        self._auto_scale = auto_scale
        self._scale_interval = scale_interval_seconds

        # Worker groups
        self._groups: dict[ResourceClass, WorkerGroup] = {}

        # State
        self._running = False
        self._dispatch_thread: threading.Thread | None = None
        self._scale_thread: threading.Thread | None = None
        self._fetch_thread: threading.Thread | None = None

        # Metrics
        self._metrics = WorkerPoolMetrics()
        self._result_lock = threading.Lock()

        # Task handler registry reference
        self._handler_registry: Any | None = None

    def set_handler_registry(self, registry: Any) -> None:
        """Set the task handler registry."""
        self._handler_registry = registry

    def _get_handler(self, task_name: str) -> Callable | None:
        """Get handler for a task name."""
        if self._handler_registry:
            return self._handler_registry.get_handler(task_name)

        # Fallback to importing TaskRegistry
        try:
            from syn.sched.core import TaskRegistry

            return TaskRegistry.get_handler(task_name)
        except Exception:
            return None

    def _on_result(self, result: ExecutionResult) -> None:
        """Handle execution result."""
        with self._result_lock:
            if result.is_success:
                self._metrics.tasks_completed += 1
            elif result.status == ExecutionStatus.TIMEOUT:
                self._metrics.tasks_timed_out += 1
            else:
                self._metrics.tasks_failed += 1

        # Update task in database
        self._update_task_state(result)

        # Emit event
        self._emit_result_event(result)

    def _update_task_state(self, result: ExecutionResult) -> None:
        """Update task state in database."""
        try:
            from syn.sched.models import CognitiveTask, DeadLetterEntry, TaskExecution
            from syn.sched.types import TaskState

            task = CognitiveTask.objects.get(id=result.task_id)

            # Create execution record
            TaskExecution.objects.create(
                task=task,
                attempt_number=result.attempt,
                worker_id=result.worker_id,
                started_at=result.started_at,
                completed_at=result.completed_at,
                duration_ms=result.duration_ms,
                is_success=result.is_success,
                result=result.result,
                error_message=result.error_message,
                error_type=result.error_type,
                error_traceback=result.error_traceback,
                memory_peak_mb=result.memory_peak_mb,
                cpu_time_ms=result.cpu_time_ms,
            )

            if result.is_success:
                task.state = TaskState.SUCCESS.value
                task.result = result.result
                task.completed_at = result.completed_at
            elif result.is_retryable and task.attempts < task.max_attempts:
                # Schedule retry
                task.schedule_retry()
            else:
                # Move to DLQ
                task.state = TaskState.DEAD_LETTERED.value
                task.error_message = result.error_message
                task.error_type = result.error_type
                DeadLetterEntry.create_from_task(
                    task, result.error_message or "Unknown error"
                )

            task.save()

        except Exception as e:
            logger.error(f"[WORKER_POOL] Failed to update task state: {e}")

    def _emit_result_event(self, result: ExecutionResult) -> None:
        """Emit execution result event."""
        try:
            from syn.synara.cortex import Cortex

            event_name = (
                "scheduler.task.completed"
                if result.is_success
                else "scheduler.task.failed"
            )
            Cortex.emit(
                event_name,
                {
                    "task_id": str(result.task_id),
                    "status": result.status.value,
                    "duration_ms": result.duration_ms,
                    "worker_id": result.worker_id,
                    "attempt": result.attempt,
                    "error_type": result.error_type,
                },
            )
        except Exception as e:
            # Event emission is non-critical, log at debug level
            logger.debug(f"[WORKER_POOL] Non-critical event emission failed: {e}")

    def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        self._running = True

        # Create worker groups
        for resource_class, config in self._configs.items():
            group = WorkerGroup(
                resource_class=resource_class,
                config=config,
                task_handler_getter=self._get_handler,
                result_callback=self._on_result,
            )
            group.start()
            self._groups[resource_class] = group

        # Start dispatch thread
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="WorkerPool-Dispatch",
            daemon=True,
        )
        self._dispatch_thread.start()

        # Start fetch thread
        self._fetch_thread = threading.Thread(
            target=self._fetch_loop,
            name="WorkerPool-Fetch",
            daemon=True,
        )
        self._fetch_thread.start()

        # Start scale thread if auto-scaling enabled
        if self._auto_scale:
            self._scale_thread = threading.Thread(
                target=self._scale_loop,
                name="WorkerPool-Scale",
                daemon=True,
            )
            self._scale_thread.start()

        logger.info(f"[WORKER_POOL] Started with {len(self._groups)} worker groups")

    def stop(self, graceful: bool = True, timeout: float = 60.0) -> None:
        """Stop the worker pool."""
        self._running = False

        # Stop worker groups
        for group in self._groups.values():
            group.stop(graceful=graceful, timeout=timeout / len(self._groups))

        # Wait for threads
        if graceful:
            if self._dispatch_thread:
                self._dispatch_thread.join(timeout=5)
            if self._fetch_thread:
                self._fetch_thread.join(timeout=5)
            if self._scale_thread:
                self._scale_thread.join(timeout=5)

        self._groups.clear()
        logger.info("[WORKER_POOL] Stopped")

    def _dispatch_loop(self) -> None:
        """Main dispatch loop."""
        logger.info("[WORKER_POOL] Dispatch loop started")

        while self._running:
            try:
                dispatched = False

                # Try each resource class
                for resource_class in ResourceClass:
                    if not self._running:
                        break

                    group = self._groups.get(resource_class)
                    if not group:
                        continue

                    # Check capacity
                    if group.get_available_capacity() <= 0:
                        continue

                    # Get task for this resource class
                    task = self._queue.get_next(resource_class=resource_class)
                    if task:
                        result = group.execute_task(task)
                        if result:
                            dispatched = True

                # Sleep if no work
                if not dispatched:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"[WORKER_POOL] Dispatch error: {e}")
                time.sleep(1)

        logger.info("[WORKER_POOL] Dispatch loop stopped")

    def _fetch_loop(self) -> None:
        """Background loop to fetch tasks from database."""
        logger.info("[WORKER_POOL] Fetch loop started")

        while self._running:
            try:
                # Fetch if queue is low
                if self._queue.size() < 100:
                    self._queue.fetch_batch(batch_size=50)

                time.sleep(1)

            except Exception as e:
                logger.error(f"[WORKER_POOL] Fetch error: {e}")
                time.sleep(5)

        logger.info("[WORKER_POOL] Fetch loop stopped")

    def _scale_loop(self) -> None:
        """Background loop for auto-scaling."""
        logger.info("[WORKER_POOL] Scale loop started")

        while self._running:
            try:
                for resource_class, group in self._groups.items():
                    config = self._configs[resource_class]

                    # Check cooldown
                    if group._last_scale_time:
                        cooldown_end = group._last_scale_time + timedelta(
                            seconds=config.scale_cooldown_seconds
                        )
                        if timezone.now() < cooldown_end:
                            continue

                    utilization = group.get_utilization()
                    current = len(group._workers)

                    if (
                        utilization >= config.scale_up_threshold
                        and current < config.max_workers
                    ):
                        # Scale up
                        target = min(current + 1, config.max_workers)
                        group.scale(target)
                        self._metrics.scale_up_count += 1

                    elif (
                        utilization <= config.scale_down_threshold
                        and current > config.min_workers
                    ):
                        # Scale down
                        target = max(current - 1, config.min_workers)
                        group.scale(target)
                        self._metrics.scale_down_count += 1

                time.sleep(self._scale_interval)

            except Exception as e:
                logger.error(f"[WORKER_POOL] Scale error: {e}")
                time.sleep(10)

        logger.info("[WORKER_POOL] Scale loop stopped")

    def submit(self, task: QueuedTask) -> bool:
        """Submit a task to the queue."""
        return self._queue.enqueue(task)

    def get_metrics(self) -> WorkerPoolMetrics:
        """Get pool metrics."""
        total_workers = 0
        active_workers = 0
        workers_by_class = {}

        for rc, group in self._groups.items():
            metrics = group.get_metrics()
            total_workers += metrics["total_workers"]
            active_workers += metrics["busy_workers"]
            workers_by_class[rc.value] = metrics["total_workers"]

        self._metrics.total_workers = total_workers
        self._metrics.active_workers = active_workers
        self._metrics.idle_workers = total_workers - active_workers
        self._metrics.workers_by_class = workers_by_class
        self._metrics.queue_depth = self._queue.size()
        self._metrics.utilization = (
            active_workers / total_workers if total_workers > 0 else 0
        )

        return self._metrics

    def get_queue(self) -> ExecutionQueue:
        """Get the execution queue."""
        return self._queue

    @property
    def is_running(self) -> bool:
        """Check if pool is running."""
        return self._running
