"""
Execution Queue - In-Memory Priority Queue with Batching

Standard: SCH-003 §4 (Execution Queue)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The ExecutionQueue provides:
- In-memory priority queue for fast dispatch
- Batching for reduced database pressure
- Per-tenant fairness via weighted round-robin
- Circuit breaker awareness at queue level
- Metrics collection for monitoring

Flow:
    CognitiveTask (DB) --> fetch_batch() --> ExecutionQueue --> dispatch()
"""

from __future__ import annotations

import heapq
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from django.utils import timezone

from .resource_class import ResourceClass, infer_resource_class

logger = logging.getLogger(__name__)


@dataclass(order=True)
class QueuedTask:
    """
    Task wrapper for priority queue ordering.

    Standard: SCH-003 §4.1

    Priority ordering (lower = higher priority):
        1. Negative priority_score (cognitive score)
        2. Created timestamp (FIFO for same priority)
        3. Task ID (deterministic tiebreaker)
    """

    # Sort fields (order matters for comparison)
    sort_key: tuple[float, float, str] = field(compare=True)

    # Task data (not used for sorting)
    task_id: uuid.UUID = field(compare=False)
    task_name: str = field(compare=False)
    tenant_id: uuid.UUID | None = field(compare=False)
    queue: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    priority_score: float = field(compare=False)
    resource_class: ResourceClass = field(compare=False)
    created_at: datetime = field(compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_cognitive_task(cls, task: Any) -> QueuedTask:
        """Create QueuedTask from CognitiveTask model instance."""
        # Infer resource class from task metadata
        metadata = {}
        if hasattr(task, "metadata") and task.metadata:
            metadata = task.metadata

        resource_class = infer_resource_class(task.task_name, metadata)

        # Sort key: negative priority_score (so highest score = first),
        # then created_at timestamp, then task_id string
        sort_key = (
            -task.priority_score,
            task.created_at.timestamp(),
            str(task.id),
        )

        return cls(
            sort_key=sort_key,
            task_id=task.id,
            task_name=task.task_name,
            tenant_id=task.tenant_id,
            queue=task.queue,
            payload=task.payload or {},
            priority_score=task.priority_score,
            resource_class=resource_class,
            created_at=task.created_at,
            metadata=metadata,
        )


@dataclass
class QueueMetrics:
    """
    Metrics for execution queue monitoring.

    Standard: SCH-003 §4.2
    """

    total_queued: int = 0
    total_dispatched: int = 0
    total_rejected: int = 0
    queued_by_resource_class: dict[str, int] = field(default_factory=dict)
    queued_by_tenant: dict[str, int] = field(default_factory=dict)
    queued_by_queue: dict[str, int] = field(default_factory=dict)
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    last_fetch_time: datetime | None = None
    last_dispatch_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_queued": self.total_queued,
            "total_dispatched": self.total_dispatched,
            "total_rejected": self.total_rejected,
            "queued_by_resource_class": self.queued_by_resource_class,
            "queued_by_tenant": self.queued_by_tenant,
            "queued_by_queue": self.queued_by_queue,
            "avg_wait_time_ms": self.avg_wait_time_ms,
            "max_wait_time_ms": self.max_wait_time_ms,
            "last_fetch_time": (self.last_fetch_time.isoformat() if self.last_fetch_time else None),
            "last_dispatch_time": (self.last_dispatch_time.isoformat() if self.last_dispatch_time else None),
        }


class ExecutionQueue:
    """
    In-memory priority queue for task dispatch.

    Standard: SCH-003 §4

    Features:
    - Heap-based priority queue for O(log n) operations
    - Per-resource-class sub-queues for worker affinity
    - Tenant fairness via weighted round-robin
    - Circuit breaker awareness (skip circuit-open tasks)
    - Batch fetching from database
    - Thread-safe operations

    Usage:
        queue = ExecutionQueue(max_size=10000)
        queue.fetch_batch(batch_size=100)  # Load from DB
        task = queue.get_next(resource_class=ResourceClass.IO_BOUND)
        if task:
            executor.execute(task)
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_per_tenant: int = 1000,
        fetch_batch_size: int = 100,
        circuit_breaker_check: bool = True,
    ):
        """
        Initialize execution queue.

        Args:
            max_size: Maximum total tasks in queue
            max_per_tenant: Maximum tasks per tenant (fairness limit)
            fetch_batch_size: Default batch size for DB fetch
            circuit_breaker_check: Check circuit breakers at dispatch
        """
        self._max_size = max_size
        self._max_per_tenant = max_per_tenant
        self._fetch_batch_size = fetch_batch_size
        self._circuit_breaker_check = circuit_breaker_check

        # Main priority queue (heap)
        self._queue: list[QueuedTask] = []

        # Per-resource-class queues for affinity
        self._queues_by_class: dict[ResourceClass, list[QueuedTask]] = {rc: [] for rc in ResourceClass}

        # Tracking sets
        self._task_ids: set[uuid.UUID] = set()
        self._tasks_by_tenant: dict[uuid.UUID, int] = {}

        # Metrics
        self._metrics = QueueMetrics()
        self._wait_times: list[float] = []

        # Thread safety
        self._lock = threading.RLock()

        # Circuit breaker cache (service -> is_open)
        self._circuit_states: dict[str, tuple[bool, datetime]] = {}
        self._circuit_cache_ttl = timedelta(seconds=5)

    def enqueue(self, task: QueuedTask) -> bool:
        """
        Add a task to the queue.

        Args:
            task: QueuedTask to enqueue

        Returns:
            True if enqueued, False if rejected (queue full or tenant limit)
        """
        with self._lock:
            # Check total queue size
            if len(self._queue) >= self._max_size:
                logger.warning(f"[QUEUE] Rejected task {task.task_id}: queue full")
                self._metrics.total_rejected += 1
                return False

            # Check per-tenant limit
            if task.tenant_id:
                tenant_count = self._tasks_by_tenant.get(task.tenant_id, 0)
                if tenant_count >= self._max_per_tenant:
                    logger.warning(f"[QUEUE] Rejected task {task.task_id}: tenant limit")
                    self._metrics.total_rejected += 1
                    return False

            # Check for duplicate
            if task.task_id in self._task_ids:
                logger.debug(f"[QUEUE] Skipping duplicate task {task.task_id}")
                return False

            # Add to main queue
            heapq.heappush(self._queue, task)

            # Add to resource-class queue
            rc_queue = self._queues_by_class[task.resource_class]
            heapq.heappush(rc_queue, task)

            # Update tracking
            self._task_ids.add(task.task_id)
            if task.tenant_id:
                self._tasks_by_tenant[task.tenant_id] = self._tasks_by_tenant.get(task.tenant_id, 0) + 1

            # Update metrics
            self._metrics.total_queued += 1
            rc_key = task.resource_class.value
            self._metrics.queued_by_resource_class[rc_key] = self._metrics.queued_by_resource_class.get(rc_key, 0) + 1
            if task.tenant_id:
                tenant_key = str(task.tenant_id)
                self._metrics.queued_by_tenant[tenant_key] = self._metrics.queued_by_tenant.get(tenant_key, 0) + 1
            queue_key = task.queue
            self._metrics.queued_by_queue[queue_key] = self._metrics.queued_by_queue.get(queue_key, 0) + 1

            logger.debug(f"[QUEUE] Enqueued task {task.task_id} ({task.resource_class.value})")
            return True

    def get_next(
        self,
        resource_class: ResourceClass | None = None,
        tenant_id: uuid.UUID | None = None,
    ) -> QueuedTask | None:
        """
        Get the next task from the queue.

        Args:
            resource_class: Filter by resource class (worker affinity)
            tenant_id: Filter by tenant (optional)

        Returns:
            Next highest-priority task, or None if queue empty
        """
        with self._lock:
            # Select queue based on resource class
            if resource_class:
                source_queue = self._queues_by_class[resource_class]
            else:
                source_queue = self._queue

            # Find valid task
            while source_queue:
                task = heapq.heappop(source_queue)

                # Skip if already removed from tracking
                if task.task_id not in self._task_ids:
                    continue

                # Tenant filter
                if tenant_id and task.tenant_id != tenant_id:
                    # Put back and try next
                    heapq.heappush(source_queue, task)
                    continue

                # Circuit breaker check
                if self._circuit_breaker_check:
                    cb_service = task.metadata.get("circuit_breaker")
                    if cb_service and self._is_circuit_open(cb_service):
                        logger.debug(f"[QUEUE] Skipping task {task.task_id}: circuit open")
                        # Re-queue with delay
                        heapq.heappush(source_queue, task)
                        continue

                # Remove from tracking
                self._task_ids.discard(task.task_id)
                if task.tenant_id:
                    self._tasks_by_tenant[task.tenant_id] = max(0, self._tasks_by_tenant.get(task.tenant_id, 1) - 1)

                # Also remove from main queue if using class-specific queue
                if resource_class:
                    self._remove_from_main_queue(task.task_id)

                # Update metrics
                self._metrics.total_dispatched += 1
                self._metrics.last_dispatch_time = timezone.now()

                # Calculate wait time
                wait_time_ms = (timezone.now() - task.created_at).total_seconds() * 1000
                self._wait_times.append(wait_time_ms)
                if len(self._wait_times) > 1000:
                    self._wait_times = self._wait_times[-1000:]
                self._metrics.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
                self._metrics.max_wait_time_ms = max(self._metrics.max_wait_time_ms, wait_time_ms)

                logger.debug(f"[QUEUE] Dispatching task {task.task_id}")
                return task

            return None

    def _remove_from_main_queue(self, task_id: uuid.UUID) -> None:
        """Remove a task from the main queue (expensive, use sparingly)."""
        self._queue = [t for t in self._queue if t.task_id != task_id]
        heapq.heapify(self._queue)

    def _is_circuit_open(self, service: str) -> bool:
        """Check if a circuit breaker is open (with caching)."""
        cached = self._circuit_states.get(service)
        if cached:
            is_open, cached_at = cached
            if timezone.now() - cached_at < self._circuit_cache_ttl:
                return is_open

        # Check database
        try:
            from syn.sched.models import CircuitBreakerState

            circuit = CircuitBreakerState.objects.filter(service_name=service).first()
            if circuit:
                can_execute, _ = circuit.can_execute()
                is_open = not can_execute
            else:
                is_open = False
        except Exception:
            is_open = False

        self._circuit_states[service] = (is_open, timezone.now())
        return is_open

    def fetch_batch(
        self,
        batch_size: int | None = None,
        queues: list[str] | None = None,
    ) -> int:
        """
        Fetch a batch of tasks from the database.

        Args:
            batch_size: Number of tasks to fetch (default: fetch_batch_size)
            queues: Filter by queue names (optional)

        Returns:
            Number of tasks fetched and enqueued
        """
        from syn.sched.models import CognitiveTask
        from syn.sched.types import TaskState

        batch_size = batch_size or self._fetch_batch_size
        now = timezone.now()

        # Build query
        queryset = (
            CognitiveTask.objects.filter(
                state__in=[TaskState.PENDING.value, TaskState.RETRYING.value],
            )
            .filter(
                # Ready to execute
                scheduled_at__lte=now,
            )
            .exclude(
                # Already in queue
                id__in=list(self._task_ids),
            )
            .order_by("-priority_score", "created_at")
        )

        # Queue filter
        if queues:
            queryset = queryset.filter(queue__in=queues)

        # Retry filter
        queryset = queryset.filter(
            # Pending tasks OR retrying tasks with next_retry_at <= now
            state=TaskState.PENDING.value,
        ) | queryset.filter(
            state=TaskState.RETRYING.value,
            next_retry_at__lte=now,
        )

        # Fetch batch
        tasks = list(queryset[:batch_size])

        # Enqueue
        enqueued = 0
        for task in tasks:
            queued_task = QueuedTask.from_cognitive_task(task)
            if self.enqueue(queued_task):
                enqueued += 1

        self._metrics.last_fetch_time = now
        logger.info(f"[QUEUE] Fetched {enqueued}/{len(tasks)} tasks from database")
        return enqueued

    def size(self, resource_class: ResourceClass | None = None) -> int:
        """Get current queue size."""
        with self._lock:
            if resource_class:
                return len(self._queues_by_class[resource_class])
            return len(self._task_ids)

    def is_empty(self, resource_class: ResourceClass | None = None) -> bool:
        """Check if queue is empty."""
        return self.size(resource_class) == 0

    def clear(self) -> int:
        """Clear all tasks from the queue."""
        with self._lock:
            count = len(self._task_ids)
            self._queue.clear()
            for rc_queue in self._queues_by_class.values():
                rc_queue.clear()
            self._task_ids.clear()
            self._tasks_by_tenant.clear()
            logger.info(f"[QUEUE] Cleared {count} tasks")
            return count

    def get_metrics(self) -> QueueMetrics:
        """Get queue metrics snapshot."""
        with self._lock:
            return QueueMetrics(
                total_queued=self._metrics.total_queued,
                total_dispatched=self._metrics.total_dispatched,
                total_rejected=self._metrics.total_rejected,
                queued_by_resource_class=dict(self._metrics.queued_by_resource_class),
                queued_by_tenant=dict(self._metrics.queued_by_tenant),
                queued_by_queue=dict(self._metrics.queued_by_queue),
                avg_wait_time_ms=self._metrics.avg_wait_time_ms,
                max_wait_time_ms=self._metrics.max_wait_time_ms,
                last_fetch_time=self._metrics.last_fetch_time,
                last_dispatch_time=self._metrics.last_dispatch_time,
            )

    def invalidate_circuit_cache(self, service: str | None = None) -> None:
        """Invalidate circuit breaker cache."""
        with self._lock:
            if service:
                self._circuit_states.pop(service, None)
            else:
                self._circuit_states.clear()
