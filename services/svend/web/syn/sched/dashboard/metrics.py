"""
Scheduler Metrics Collector - Aggregates All Observable State

Standard: SCH-005 §2 (Metrics Collection)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The SchedulerMetricsCollector aggregates metrics from:
- CognitiveTask model (queue depths, states)
- TaskExecution model (latencies, success rates)
- Schedule model (last/next runs)
- DeadLetterEntry model (DLQ state)
- CircuitBreakerState model (circuit states)
- BackpressureController (throttle state)
- WorkerPool (worker health)

This is the single source of truth for dashboard data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from django.db.models import Avg, Count, Max, Min, Q, Sum
from django.db.models.functions import TruncHour
from django.utils import timezone

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker health states."""
    ALIVE = "alive"
    BUSY = "busy"
    IDLE = "idle"
    RESTARTING = "restarting"
    QUARANTINED = "quarantined"
    DEAD = "dead"


@dataclass
class QueueMetrics:
    """
    Metrics for a single queue.

    Standard: SCH-005 §2.1
    """
    queue_name: str
    depth: int = 0
    pending: int = 0
    scheduled: int = 0
    running: int = 0
    completed_last_hour: int = 0
    failed_last_hour: int = 0
    avg_wait_time_seconds: float = 0.0
    oldest_task_age_seconds: float = 0.0
    throughput_per_minute: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_name": self.queue_name,
            "depth": self.depth,
            "pending": self.pending,
            "scheduled": self.scheduled,
            "running": self.running,
            "completed_last_hour": self.completed_last_hour,
            "failed_last_hour": self.failed_last_hour,
            "avg_wait_time_seconds": self.avg_wait_time_seconds,
            "oldest_task_age_seconds": self.oldest_task_age_seconds,
            "throughput_per_minute": self.throughput_per_minute,
        }


@dataclass
class WorkerMetrics:
    """
    Metrics for a single worker.

    Standard: SCH-005 §2.2
    """
    worker_id: str
    state: WorkerState = WorkerState.ALIVE
    resource_class: str = "mixed"
    current_task_id: Optional[str] = None
    current_task_name: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: Optional[datetime] = None
    memory_mb: float = 0.0
    cpu_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "state": self.state.value,
            "resource_class": self.resource_class,
            "current_task_id": self.current_task_id,
            "current_task_name": self.current_task_name,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "uptime_seconds": self.uptime_seconds,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
        }


@dataclass
class TaskTypeMetrics:
    """
    Metrics for a specific task type.

    Standard: SCH-005 §2.3
    """
    task_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    success_rate: float = 0.0
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    last_execution: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "p50_duration_ms": self.p50_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "p99_duration_ms": self.p99_duration_ms,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
        }


@dataclass
class ScheduleMetrics:
    """
    Metrics for a schedule.

    Standard: SCH-005 §2.4
    """
    schedule_id: str
    name: str
    task_name: str
    enabled: bool = True
    schedule_type: str = "cron"
    expression: str = ""
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0
    last_status: str = "unknown"
    avg_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "task_name": self.task_name,
            "enabled": self.enabled,
            "schedule_type": self.schedule_type,
            "expression": self.expression,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "run_count": self.run_count,
            "failure_count": self.failure_count,
            "last_status": self.last_status,
            "avg_duration_ms": self.avg_duration_ms,
        }


@dataclass
class CircuitMetrics:
    """
    Metrics for a circuit breaker.

    Standard: SCH-005 §2.5
    """
    service_name: str
    state: str = "closed"
    failure_count: int = 0
    success_count: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    half_open_at: Optional[datetime] = None
    failure_rate: float = 0.0
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "half_open_at": self.half_open_at.isoformat() if self.half_open_at else None,
            "failure_rate": self.failure_rate,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class DLQMetrics:
    """
    Metrics for the Dead Letter Queue.

    Standard: SCH-005 §2.6
    """
    total_entries: int = 0
    pending_entries: int = 0
    resolved_entries: int = 0
    retried_entries: int = 0
    discarded_entries: int = 0
    growth_rate_per_hour: float = 0.0
    oldest_entry_age_hours: float = 0.0
    entries_by_task: Dict[str, int] = field(default_factory=dict)
    entries_by_error_type: Dict[str, int] = field(default_factory=dict)
    avg_resolution_time_hours: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "pending_entries": self.pending_entries,
            "resolved_entries": self.resolved_entries,
            "retried_entries": self.retried_entries,
            "discarded_entries": self.discarded_entries,
            "growth_rate_per_hour": self.growth_rate_per_hour,
            "oldest_entry_age_hours": self.oldest_entry_age_hours,
            "entries_by_task": self.entries_by_task,
            "entries_by_error_type": self.entries_by_error_type,
            "avg_resolution_time_hours": self.avg_resolution_time_hours,
        }


@dataclass
class ResourceClassMetrics:
    """
    Metrics for a resource class (CPU_BOUND, IO_BOUND, etc.).

    Standard: SCH-005 §2.7
    """
    resource_class: str
    active_workers: int = 0
    max_workers: int = 0
    utilization: float = 0.0
    tasks_in_flight: int = 0
    max_concurrent: int = 0
    avg_task_duration_ms: float = 0.0
    memory_used_mb: float = 0.0
    memory_limit_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_class": self.resource_class,
            "active_workers": self.active_workers,
            "max_workers": self.max_workers,
            "utilization": self.utilization,
            "tasks_in_flight": self.tasks_in_flight,
            "max_concurrent": self.max_concurrent,
            "avg_task_duration_ms": self.avg_task_duration_ms,
            "memory_used_mb": self.memory_used_mb,
            "memory_limit_mb": self.memory_limit_mb,
        }


@dataclass
class ThrottleMetrics:
    """
    Metrics for backpressure throttling.

    Standard: SCH-005 §2.8
    """
    current_level: str = "NONE"
    level_value: float = 0.0
    confidence_penalty: float = 0.0
    is_emergency: bool = False
    emergency_reason: Optional[str] = None
    paused_schedules: bool = False
    skip_low_priority: bool = False
    skip_batch_tasks: bool = False
    decisions_total: int = 0
    decisions_allowed: int = 0
    decisions_denied: int = 0
    time_at_level: Dict[str, float] = field(default_factory=dict)
    level_changes_last_hour: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_level": self.current_level,
            "level_value": self.level_value,
            "confidence_penalty": self.confidence_penalty,
            "is_emergency": self.is_emergency,
            "emergency_reason": self.emergency_reason,
            "paused_schedules": self.paused_schedules,
            "skip_low_priority": self.skip_low_priority,
            "skip_batch_tasks": self.skip_batch_tasks,
            "decisions_total": self.decisions_total,
            "decisions_allowed": self.decisions_allowed,
            "decisions_denied": self.decisions_denied,
            "time_at_level": self.time_at_level,
            "level_changes_last_hour": self.level_changes_last_hour,
        }


@dataclass
class DashboardMetrics:
    """
    Complete dashboard metrics snapshot.

    Standard: SCH-005 §2
    """
    collected_at: datetime = field(default_factory=timezone.now)

    # Summary metrics
    total_pending_tasks: int = 0
    total_running_tasks: int = 0
    total_completed_today: int = 0
    total_failed_today: int = 0
    overall_success_rate: float = 0.0

    # Component metrics
    queues: List[QueueMetrics] = field(default_factory=list)
    workers: List[WorkerMetrics] = field(default_factory=list)
    task_types: List[TaskTypeMetrics] = field(default_factory=list)
    schedules: List[ScheduleMetrics] = field(default_factory=list)
    circuits: List[CircuitMetrics] = field(default_factory=list)
    resource_classes: List[ResourceClassMetrics] = field(default_factory=list)
    dlq: DLQMetrics = field(default_factory=DLQMetrics)
    throttle: ThrottleMetrics = field(default_factory=ThrottleMetrics)

    # Health indicators
    scheduler_running: bool = False
    backpressure_running: bool = False
    workers_healthy: int = 0
    workers_unhealthy: int = 0
    circuits_open: int = 0
    circuits_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collected_at": self.collected_at.isoformat(),
            "summary": {
                "total_pending_tasks": self.total_pending_tasks,
                "total_running_tasks": self.total_running_tasks,
                "total_completed_today": self.total_completed_today,
                "total_failed_today": self.total_failed_today,
                "overall_success_rate": self.overall_success_rate,
            },
            "health": {
                "scheduler_running": self.scheduler_running,
                "backpressure_running": self.backpressure_running,
                "workers_healthy": self.workers_healthy,
                "workers_unhealthy": self.workers_unhealthy,
                "circuits_open": self.circuits_open,
                "circuits_total": self.circuits_total,
            },
            "queues": [q.to_dict() for q in self.queues],
            "workers": [w.to_dict() for w in self.workers],
            "task_types": [t.to_dict() for t in self.task_types],
            "schedules": [s.to_dict() for s in self.schedules],
            "circuits": [c.to_dict() for c in self.circuits],
            "resource_classes": [r.to_dict() for r in self.resource_classes],
            "dlq": self.dlq.to_dict(),
            "throttle": self.throttle.to_dict(),
        }


class SchedulerMetricsCollector:
    """
    Collects and aggregates all scheduler metrics.

    Standard: SCH-005 §2

    Features:
    - Queue depth per queue type
    - Per-resource-class utilization
    - Task type latency statistics
    - Schedule status and timing
    - Circuit breaker states
    - DLQ analysis
    - Backpressure state

    Usage:
        collector = SchedulerMetricsCollector(scheduler=scheduler)
        metrics = collector.collect()

        print(f"Queue depth: {metrics.total_pending_tasks}")
        print(f"Throttle level: {metrics.throttle.current_level}")
    """

    def __init__(
        self,
        scheduler: Optional[Any] = None,
        lookback_hours: int = 24,
    ):
        """
        Initialize metrics collector.

        Args:
            scheduler: Optional CognitiveScheduler instance
            lookback_hours: Hours of history for statistics
        """
        self._scheduler = scheduler
        self._lookback_hours = lookback_hours

    def set_scheduler(self, scheduler: Any) -> None:
        """Set the scheduler reference."""
        self._scheduler = scheduler

    def collect(self) -> DashboardMetrics:
        """
        Collect all scheduler metrics.

        Returns:
            DashboardMetrics snapshot
        """
        metrics = DashboardMetrics(collected_at=timezone.now())

        # Collect each category
        self._collect_summary_metrics(metrics)
        self._collect_queue_metrics(metrics)
        self._collect_worker_metrics(metrics)
        self._collect_task_type_metrics(metrics)
        self._collect_schedule_metrics(metrics)
        self._collect_circuit_metrics(metrics)
        self._collect_resource_class_metrics(metrics)
        self._collect_dlq_metrics(metrics)
        self._collect_throttle_metrics(metrics)
        self._collect_health_indicators(metrics)

        return metrics

    def _collect_summary_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect summary metrics."""
        try:
            from syn.sched.models import CognitiveTask
            from syn.sched.types import TaskState

            now = timezone.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # Pending and running counts
            metrics.total_pending_tasks = CognitiveTask.objects.filter(
                state__in=[TaskState.PENDING.value, TaskState.SCHEDULED.value, TaskState.RETRYING.value]
            ).count()

            metrics.total_running_tasks = CognitiveTask.objects.filter(
                state=TaskState.RUNNING.value
            ).count()

            # Today's stats
            metrics.total_completed_today = CognitiveTask.objects.filter(
                state=TaskState.SUCCESS.value,
                completed_at__gte=today_start,
            ).count()

            metrics.total_failed_today = CognitiveTask.objects.filter(
                state__in=[TaskState.FAILURE.value, TaskState.DEAD_LETTERED.value],
                completed_at__gte=today_start,
            ).count()

            # Success rate
            total_today = metrics.total_completed_today + metrics.total_failed_today
            if total_today > 0:
                metrics.overall_success_rate = metrics.total_completed_today / total_today

        except Exception as e:
            logger.debug(f"[DASHBOARD] Summary metrics error: {e}")

    def _collect_queue_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect per-queue metrics."""
        try:
            from syn.sched.models import CognitiveTask
            from syn.sched.types import QueueType, TaskState

            now = timezone.now()
            one_hour_ago = now - timedelta(hours=1)

            for queue_type in QueueType:
                queue_name = queue_type.value
                queue_tasks = CognitiveTask.objects.filter(queue=queue_name)

                qm = QueueMetrics(queue_name=queue_name)

                # Counts by state
                qm.pending = queue_tasks.filter(state=TaskState.PENDING.value).count()
                qm.scheduled = queue_tasks.filter(state=TaskState.SCHEDULED.value).count()
                qm.running = queue_tasks.filter(state=TaskState.RUNNING.value).count()
                qm.depth = qm.pending + qm.scheduled

                # Last hour stats
                qm.completed_last_hour = queue_tasks.filter(
                    state=TaskState.SUCCESS.value,
                    completed_at__gte=one_hour_ago,
                ).count()

                qm.failed_last_hour = queue_tasks.filter(
                    state__in=[TaskState.FAILURE.value, TaskState.DEAD_LETTERED.value],
                    completed_at__gte=one_hour_ago,
                ).count()

                # Throughput (completions per minute)
                qm.throughput_per_minute = qm.completed_last_hour / 60.0

                # Oldest task age
                oldest = queue_tasks.filter(
                    state__in=[TaskState.PENDING.value, TaskState.SCHEDULED.value]
                ).order_by('created_at').first()

                if oldest:
                    qm.oldest_task_age_seconds = (now - oldest.created_at).total_seconds()

                # Average wait time (from created to started for recently completed)
                from syn.sched.models import TaskExecution
                recent_executions = TaskExecution.objects.filter(
                    task__queue=queue_name,
                    started_at__gte=one_hour_ago,
                )
                if recent_executions.exists():
                    wait_times = []
                    for exe in recent_executions[:100]:
                        if exe.started_at and exe.task.created_at:
                            wait = (exe.started_at - exe.task.created_at).total_seconds()
                            wait_times.append(wait)
                    if wait_times:
                        qm.avg_wait_time_seconds = sum(wait_times) / len(wait_times)

                metrics.queues.append(qm)

        except Exception as e:
            logger.debug(f"[DASHBOARD] Queue metrics error: {e}")

    def _collect_worker_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect worker metrics."""
        try:
            if not self._scheduler:
                return

            for worker in getattr(self._scheduler, '_workers', []):
                wm = WorkerMetrics(worker_id=worker.worker_id)

                # State determination
                if not worker._running:
                    wm.state = WorkerState.DEAD
                elif worker._current_task:
                    wm.state = WorkerState.BUSY
                    wm.current_task_id = str(worker._current_task.id)
                    wm.current_task_name = worker._current_task.task_name
                else:
                    wm.state = WorkerState.IDLE

                # Stats
                wm.tasks_completed = worker.completed_tasks
                wm.tasks_failed = worker.failed_tasks

                if worker.started_at:
                    wm.uptime_seconds = (timezone.now() - worker.started_at).total_seconds()
                    wm.last_heartbeat = timezone.now()

                metrics.workers.append(wm)

        except Exception as e:
            logger.debug(f"[DASHBOARD] Worker metrics error: {e}")

    def _collect_task_type_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect per-task-type metrics."""
        try:
            from syn.sched.models import CognitiveTask, TaskExecution
            from syn.sched.types import TaskState

            lookback = timezone.now() - timedelta(hours=self._lookback_hours)

            # Get distinct task names with recent activity
            task_names = CognitiveTask.objects.filter(
                created_at__gte=lookback
            ).values_list('task_name', flat=True).distinct()

            for task_name in task_names[:50]:  # Limit to top 50
                ttm = TaskTypeMetrics(task_name=task_name)

                # Get executions
                executions = TaskExecution.objects.filter(
                    task__task_name=task_name,
                    completed_at__gte=lookback,
                )

                ttm.total_executions = executions.count()
                ttm.successful_executions = executions.filter(is_success=True).count()
                ttm.failed_executions = executions.filter(is_success=False).count()

                if ttm.total_executions > 0:
                    ttm.success_rate = ttm.successful_executions / ttm.total_executions

                # Duration stats
                successful = executions.filter(is_success=True, duration_ms__isnull=False)
                if successful.exists():
                    durations = list(successful.values_list('duration_ms', flat=True))
                    durations.sort()

                    ttm.avg_duration_ms = sum(durations) / len(durations)
                    ttm.p50_duration_ms = durations[len(durations) // 2]
                    ttm.p95_duration_ms = durations[int(len(durations) * 0.95)]
                    ttm.p99_duration_ms = durations[int(len(durations) * 0.99)]

                # Last timestamps
                last_exec = executions.order_by('-completed_at').first()
                if last_exec:
                    ttm.last_execution = last_exec.completed_at

                last_success = executions.filter(is_success=True).order_by('-completed_at').first()
                if last_success:
                    ttm.last_success = last_success.completed_at

                last_failure = executions.filter(is_success=False).order_by('-completed_at').first()
                if last_failure:
                    ttm.last_failure = last_failure.completed_at

                metrics.task_types.append(ttm)

        except Exception as e:
            logger.debug(f"[DASHBOARD] Task type metrics error: {e}")

    def _collect_schedule_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect schedule metrics."""
        try:
            from syn.sched.models import Schedule

            for schedule in Schedule.objects.all()[:100]:  # Limit
                sm = ScheduleMetrics(
                    schedule_id=schedule.schedule_id,
                    name=schedule.name,
                    task_name=schedule.task_name,
                    enabled=schedule.is_enabled,
                    schedule_type=schedule.schedule_type,
                    expression=schedule.cron_expression or str(schedule.interval_seconds) + "s",
                    last_run_at=schedule.last_run_at,
                    next_run_at=schedule.next_run_at,
                    run_count=schedule.run_count,
                    failure_count=getattr(schedule, 'failure_count', 0),
                )

                # Determine last status
                if schedule.last_run_at:
                    from syn.sched.models import CognitiveTask
                    from syn.sched.types import TaskState

                    recent_task = CognitiveTask.objects.filter(
                        task_name=schedule.task_name,
                        created_at__gte=schedule.last_run_at - timedelta(seconds=10),
                        created_at__lte=schedule.last_run_at + timedelta(seconds=10),
                    ).order_by('-created_at').first()

                    if recent_task:
                        sm.last_status = recent_task.state

                metrics.schedules.append(sm)

        except Exception as e:
            logger.debug(f"[DASHBOARD] Schedule metrics error: {e}")

    def _collect_circuit_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect circuit breaker metrics."""
        try:
            from syn.sched.models import CircuitBreakerState

            for circuit in CircuitBreakerState.objects.all():
                cm = CircuitMetrics(
                    service_name=circuit.service_name,
                    state=circuit.state,
                    failure_count=circuit.failure_count,
                    success_count=circuit.success_count,
                    last_failure=circuit.last_failure_at,
                    last_success=circuit.last_success_at,
                    opened_at=circuit.opened_at,
                    consecutive_failures=circuit.consecutive_failures,
                )

                # Calculate failure rate
                total = cm.failure_count + cm.success_count
                if total > 0:
                    cm.failure_rate = cm.failure_count / total

                metrics.circuits.append(cm)

        except Exception as e:
            logger.debug(f"[DASHBOARD] Circuit metrics error: {e}")

    def _collect_resource_class_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect per-resource-class metrics."""
        try:
            from syn.sched.execution import WORKER_CONFIGS, ResourceClass

            for rc in ResourceClass:
                config = WORKER_CONFIGS.get(rc, None)
                if not config:
                    continue

                rcm = ResourceClassMetrics(
                    resource_class=rc.value,
                    max_workers=config.max_workers,
                    max_concurrent=config.max_concurrent_per_worker,
                    memory_limit_mb=config.memory_limit_mb,
                )

                # Get active count from scheduler
                if self._scheduler and hasattr(self._scheduler, 'backpressure'):
                    health = self._scheduler.backpressure.get_health_metrics() if self._scheduler.backpressure else None
                    if health:
                        rcm.utilization = health.worker_utilization

                metrics.resource_classes.append(rcm)

        except Exception as e:
            logger.debug(f"[DASHBOARD] Resource class metrics error: {e}")

    def _collect_dlq_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect Dead Letter Queue metrics."""
        try:
            from syn.sched.models import DeadLetterEntry

            dlq = DLQMetrics()

            dlq.total_entries = DeadLetterEntry.objects.count()
            dlq.pending_entries = DeadLetterEntry.objects.filter(status="pending").count()
            dlq.resolved_entries = DeadLetterEntry.objects.filter(status="resolved").count()
            dlq.retried_entries = DeadLetterEntry.objects.filter(status="retried").count()
            dlq.discarded_entries = DeadLetterEntry.objects.filter(status="discarded").count()

            # Growth rate (entries in last hour)
            one_hour_ago = timezone.now() - timedelta(hours=1)
            recent_entries = DeadLetterEntry.objects.filter(created_at__gte=one_hour_ago).count()
            dlq.growth_rate_per_hour = float(recent_entries)

            # Oldest entry
            oldest = DeadLetterEntry.objects.filter(status="pending").order_by('created_at').first()
            if oldest:
                dlq.oldest_entry_age_hours = (timezone.now() - oldest.created_at).total_seconds() / 3600

            # By task name
            by_task = DeadLetterEntry.objects.values('task_name').annotate(
                count=Count('id')
            ).order_by('-count')[:10]
            dlq.entries_by_task = {item['task_name']: item['count'] for item in by_task}

            # By error type
            by_error = DeadLetterEntry.objects.values('error_type').annotate(
                count=Count('id')
            ).order_by('-count')[:10]
            dlq.entries_by_error_type = {
                item['error_type'] or 'unknown': item['count']
                for item in by_error
            }

            metrics.dlq = dlq

        except Exception as e:
            logger.debug(f"[DASHBOARD] DLQ metrics error: {e}")

    def _collect_throttle_metrics(self, metrics: DashboardMetrics) -> None:
        """Collect backpressure throttle metrics."""
        try:
            if not self._scheduler:
                return

            bp = getattr(self._scheduler, '_backpressure', None)
            if not bp:
                return

            tm = ThrottleMetrics()

            # Current state
            level = bp.get_current_level()
            tm.current_level = level.name
            tm.level_value = level.value
            tm.confidence_penalty = bp.get_confidence_penalty()
            tm.is_emergency = bp.is_emergency()
            tm.emergency_reason = bp.get_emergency_reason()

            # Latest decision state
            decision = bp.should_schedule(bypass_cache=True)
            tm.paused_schedules = decision.pause_schedules
            tm.skip_low_priority = decision.skip_low_priority
            tm.skip_batch_tasks = decision.skip_batch_tasks

            # Decision stats
            bp_metrics = bp.get_backpressure_metrics()
            tm.decisions_total = bp_metrics.decisions_total
            tm.decisions_allowed = bp_metrics.decisions_allowed
            tm.decisions_denied = bp_metrics.decisions_denied
            tm.time_at_level = bp_metrics.time_at_level
            tm.level_changes_last_hour = bp_metrics.level_changes

            metrics.throttle = tm

        except Exception as e:
            logger.debug(f"[DASHBOARD] Throttle metrics error: {e}")

    def _collect_health_indicators(self, metrics: DashboardMetrics) -> None:
        """Collect overall health indicators."""
        try:
            # Scheduler state
            if self._scheduler:
                metrics.scheduler_running = self._scheduler.is_running
                bp = getattr(self._scheduler, '_backpressure', None)
                metrics.backpressure_running = bp.is_running if bp else False

            # Worker health
            for wm in metrics.workers:
                if wm.state in [WorkerState.ALIVE, WorkerState.BUSY, WorkerState.IDLE]:
                    metrics.workers_healthy += 1
                else:
                    metrics.workers_unhealthy += 1

            # Circuit health
            metrics.circuits_total = len(metrics.circuits)
            metrics.circuits_open = sum(1 for c in metrics.circuits if c.state in ['open', 'half_open'])

        except Exception as e:
            logger.debug(f"[DASHBOARD] Health indicators error: {e}")

    # Convenience methods for specific queries

    def get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depths."""
        metrics = self.collect()
        return {q.queue_name: q.depth for q in metrics.queues}

    def get_throttle_level(self) -> str:
        """Get current throttle level."""
        if self._scheduler and hasattr(self._scheduler, '_backpressure'):
            bp = self._scheduler._backpressure
            if bp:
                return bp.get_current_level().name
        return "UNKNOWN"

    def get_dlq_summary(self) -> Dict[str, int]:
        """Get DLQ summary counts."""
        metrics = self.collect()
        return {
            "total": metrics.dlq.total_entries,
            "pending": metrics.dlq.pending_entries,
            "resolved": metrics.dlq.resolved_entries,
        }

    def get_worker_status(self) -> Dict[str, int]:
        """Get worker status counts."""
        metrics = self.collect()
        return {
            "healthy": metrics.workers_healthy,
            "unhealthy": metrics.workers_unhealthy,
            "total": len(metrics.workers),
        }
