"""
System Health Monitor - Collects Metrics for Backpressure Decisions

Standard: SCH-004 §2 (System Health Monitoring)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The SystemHealthMonitor collects real-time metrics about:
- Queue depth and growth rate
- Dead Letter Queue (DLQ) size and growth
- Governance denial rates
- Circuit breaker states
- Worker pool utilization
- Task latency and throughput

These metrics feed into the BackpressureController to make
throttling decisions.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from django.db.models import Avg, Count
from django.utils import timezone

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall system health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STRESSED = "stressed"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """
    Snapshot of system health metrics.

    Standard: SCH-004 §2.1
    """

    # Timestamp
    collected_at: datetime = field(default_factory=timezone.now)

    # Queue Metrics
    queue_depth: int = 0
    queue_growth_rate: float = 0.0  # Tasks/minute
    queue_depth_threshold: int = 1000
    queue_utilization: float = 0.0  # 0.0-1.0

    # DLQ Metrics
    dlq_size: int = 0
    dlq_growth_rate: float = 0.0  # Tasks/minute
    dlq_pending_count: int = 0
    dlq_threshold: int = 100

    # Governance Metrics
    governance_denial_rate: float = 0.0  # 0.0-1.0
    governance_escalation_rate: float = 0.0  # 0.0-1.0
    governance_decisions_last_minute: int = 0
    governance_denial_threshold: float = 0.3  # 30%

    # Circuit Breaker Metrics
    circuits_open: int = 0
    circuits_half_open: int = 0
    circuits_total: int = 0
    circuit_open_ratio: float = 0.0

    # Worker Pool Metrics
    worker_utilization: float = 0.0  # 0.0-1.0
    active_workers: int = 0
    idle_workers: int = 0
    tasks_in_flight: int = 0

    @property
    def running_tasks(self) -> int:
        """Alias for tasks_in_flight (used by temporal controller)."""
        return self.tasks_in_flight

    @property
    def worker_count(self) -> int:
        """Total worker count (active + idle)."""
        return self.active_workers + self.idle_workers

    # Latency Metrics
    avg_task_latency_ms: float = 0.0
    p95_task_latency_ms: float = 0.0
    latency_trend: float = 0.0  # Positive = getting worse

    # Throughput Metrics
    tasks_completed_last_minute: int = 0
    tasks_failed_last_minute: int = 0
    throughput_trend: float = 0.0  # Negative = getting worse

    # Cascade Metrics
    cascade_depth_exceeded_count: int = 0
    max_cascade_depth_seen: int = 0

    # Overall Status
    status: HealthStatus = HealthStatus.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "collected_at": self.collected_at.isoformat(),
            "queue": {
                "depth": self.queue_depth,
                "growth_rate": self.queue_growth_rate,
                "utilization": self.queue_utilization,
            },
            "dlq": {
                "size": self.dlq_size,
                "growth_rate": self.dlq_growth_rate,
                "pending_count": self.dlq_pending_count,
            },
            "governance": {
                "denial_rate": self.governance_denial_rate,
                "escalation_rate": self.governance_escalation_rate,
                "decisions_last_minute": self.governance_decisions_last_minute,
            },
            "circuits": {
                "open": self.circuits_open,
                "half_open": self.circuits_half_open,
                "total": self.circuits_total,
                "open_ratio": self.circuit_open_ratio,
            },
            "workers": {
                "utilization": self.worker_utilization,
                "active": self.active_workers,
                "idle": self.idle_workers,
                "in_flight": self.tasks_in_flight,
            },
            "latency": {
                "avg_ms": self.avg_task_latency_ms,
                "p95_ms": self.p95_task_latency_ms,
                "trend": self.latency_trend,
            },
            "throughput": {
                "completed_last_minute": self.tasks_completed_last_minute,
                "failed_last_minute": self.tasks_failed_last_minute,
                "trend": self.throughput_trend,
            },
            "cascade": {
                "depth_exceeded_count": self.cascade_depth_exceeded_count,
                "max_depth_seen": self.max_cascade_depth_seen,
            },
            "status": self.status.value,
        }


class SystemHealthMonitor:
    """
    Monitors system health for backpressure decisions.

    Standard: SCH-004 §2

    Features:
    - Periodic metric collection
    - Trend analysis (growth rates)
    - Health status computation
    - Historical data for averaging

    Usage:
        monitor = SystemHealthMonitor()
        monitor.start()

        metrics = monitor.get_metrics()
        if metrics.status == HealthStatus.CRITICAL:
            # Apply emergency throttling
            pass

        monitor.stop()
    """

    def __init__(
        self,
        collection_interval_seconds: int = 10,
        history_window_minutes: int = 5,
        queue_depth_threshold: int = 1000,
        dlq_threshold: int = 100,
        governance_denial_threshold: float = 0.3,
    ):
        """
        Initialize health monitor.

        Args:
            collection_interval_seconds: How often to collect metrics
            history_window_minutes: Window for trend calculation
            queue_depth_threshold: Queue depth warning threshold
            dlq_threshold: DLQ size warning threshold
            governance_denial_threshold: Governance denial rate threshold
        """
        self._collection_interval = collection_interval_seconds
        self._history_window = timedelta(minutes=history_window_minutes)
        self._queue_depth_threshold = queue_depth_threshold
        self._dlq_threshold = dlq_threshold
        self._governance_denial_threshold = governance_denial_threshold

        # State
        self._running = False
        self._collection_thread: threading.Thread | None = None
        self._lock = threading.RLock()

        # Metrics history for trend analysis
        self._metrics_history: list[HealthMetrics] = []
        self._max_history_size = int(
            history_window_minutes * 60 / collection_interval_seconds
        )

        # Current metrics
        self._current_metrics: HealthMetrics | None = None

    def start(self) -> None:
        """Start the health monitor."""
        if self._running:
            return

        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            name="HealthMonitor-Collection",
            daemon=True,
        )
        self._collection_thread.start()
        logger.info("[HEALTH] System health monitor started")

    def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("[HEALTH] System health monitor stopped")

    def _collection_loop(self) -> None:
        """Background loop for metric collection."""
        while self._running:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self._current_metrics = metrics
                    self._metrics_history.append(metrics)
                    if len(self._metrics_history) > self._max_history_size:
                        self._metrics_history = self._metrics_history[
                            -self._max_history_size :
                        ]
            except Exception as e:
                logger.error(f"[HEALTH] Collection error: {e}")

            time.sleep(self._collection_interval)

    def _collect_metrics(self) -> HealthMetrics:
        """Collect all system health metrics."""
        metrics = HealthMetrics(collected_at=timezone.now())

        # Collect each category
        self._collect_queue_metrics(metrics)
        self._collect_dlq_metrics(metrics)
        self._collect_governance_metrics(metrics)
        self._collect_circuit_metrics(metrics)
        self._collect_worker_metrics(metrics)
        self._collect_latency_metrics(metrics)
        self._collect_throughput_metrics(metrics)
        self._collect_cascade_metrics(metrics)

        # Calculate trends
        self._calculate_trends(metrics)

        # Determine overall status
        metrics.status = self._compute_health_status(metrics)

        return metrics

    def _collect_queue_metrics(self, metrics: HealthMetrics) -> None:
        """Collect queue depth metrics."""
        try:
            from syn.sched.models import CognitiveTask
            from syn.sched.types import TaskState

            # Count pending tasks
            pending_states = [
                TaskState.PENDING.value,
                TaskState.SCHEDULED.value,
                TaskState.RETRYING.value,
            ]
            metrics.queue_depth = CognitiveTask.objects.filter(
                state__in=pending_states
            ).count()

            # Queue utilization
            metrics.queue_utilization = min(
                1.0, metrics.queue_depth / max(1, self._queue_depth_threshold)
            )

            # Growth rate from history
            if len(self._metrics_history) >= 2:
                old_depth = self._metrics_history[-2].queue_depth
                time_delta = (
                    metrics.collected_at - self._metrics_history[-2].collected_at
                ).total_seconds() / 60
                if time_delta > 0:
                    metrics.queue_growth_rate = (
                        metrics.queue_depth - old_depth
                    ) / time_delta

        except Exception as e:
            logger.debug(f"[HEALTH] Queue metrics error: {e}")

    def _collect_dlq_metrics(self, metrics: HealthMetrics) -> None:
        """Collect Dead Letter Queue metrics."""
        try:
            from syn.sched.models import DeadLetterEntry

            # Total DLQ size
            metrics.dlq_size = DeadLetterEntry.objects.count()

            # Pending (unresolved) entries
            metrics.dlq_pending_count = DeadLetterEntry.objects.filter(
                status="pending"
            ).count()

            # Growth rate from history
            if len(self._metrics_history) >= 2:
                old_size = self._metrics_history[-2].dlq_size
                time_delta = (
                    metrics.collected_at - self._metrics_history[-2].collected_at
                ).total_seconds() / 60
                if time_delta > 0:
                    metrics.dlq_growth_rate = (metrics.dlq_size - old_size) / time_delta

        except Exception as e:
            logger.debug(f"[HEALTH] DLQ metrics error: {e}")

    def _collect_governance_metrics(self, metrics: HealthMetrics) -> None:
        """Collect governance decision metrics."""
        try:
            from syn.synara.models import GovernanceJudgement

            # Last minute window
            one_minute_ago = timezone.now() - timedelta(minutes=1)

            # Count decisions
            recent_decisions = GovernanceJudgement.objects.filter(
                created_at__gte=one_minute_ago
            )

            metrics.governance_decisions_last_minute = recent_decisions.count()

            if metrics.governance_decisions_last_minute > 0:
                # Denial rate
                denied = recent_decisions.filter(result="BLOCKED").count()
                metrics.governance_denial_rate = (
                    denied / metrics.governance_decisions_last_minute
                )

                # Escalation rate
                escalated = recent_decisions.filter(result="ESCALATED").count()
                metrics.governance_escalation_rate = (
                    escalated / metrics.governance_decisions_last_minute
                )

        except Exception as e:
            logger.debug(f"[HEALTH] Governance metrics error: {e}")

    def _collect_circuit_metrics(self, metrics: HealthMetrics) -> None:
        """Collect circuit breaker metrics."""
        try:
            from syn.sched.models import CircuitBreakerState

            circuits = CircuitBreakerState.objects.all()
            metrics.circuits_total = circuits.count()
            metrics.circuits_open = circuits.filter(state="open").count()
            metrics.circuits_half_open = circuits.filter(state="half_open").count()

            if metrics.circuits_total > 0:
                metrics.circuit_open_ratio = (
                    metrics.circuits_open + metrics.circuits_half_open * 0.5
                ) / metrics.circuits_total

        except Exception as e:
            logger.debug(f"[HEALTH] Circuit metrics error: {e}")

    def _collect_worker_metrics(self, metrics: HealthMetrics) -> None:
        """Collect worker pool metrics."""
        try:
            from syn.sched.models import CognitiveTask
            from syn.sched.types import TaskState

            # Tasks currently running
            metrics.tasks_in_flight = CognitiveTask.objects.filter(
                state=TaskState.RUNNING.value
            ).count()

            # Worker utilization estimate (tasks in flight / expected capacity)
            # Assume default 10 workers with mixed config
            expected_capacity = 10 * 3  # workers * concurrent per worker
            metrics.worker_utilization = min(
                1.0, metrics.tasks_in_flight / max(1, expected_capacity)
            )

            # Active/idle split (estimate)
            metrics.active_workers = min(10, metrics.tasks_in_flight)
            metrics.idle_workers = max(0, 10 - metrics.active_workers)

        except Exception as e:
            logger.debug(f"[HEALTH] Worker metrics error: {e}")

    def _collect_latency_metrics(self, metrics: HealthMetrics) -> None:
        """Collect task latency metrics."""
        try:
            from syn.sched.models import TaskExecution

            # Last minute executions
            one_minute_ago = timezone.now() - timedelta(minutes=1)
            recent_executions = TaskExecution.objects.filter(
                completed_at__gte=one_minute_ago,
                is_success=True,
            )

            if recent_executions.exists():
                # Average latency
                avg_result = recent_executions.aggregate(
                    avg_duration=Avg("duration_ms")
                )
                metrics.avg_task_latency_ms = avg_result["avg_duration"] or 0.0

                # P95 approximation (get sorted durations)
                durations = list(
                    recent_executions.values_list("duration_ms", flat=True)
                )
                if durations:
                    durations.sort()
                    p95_index = int(len(durations) * 0.95)
                    metrics.p95_task_latency_ms = durations[
                        min(p95_index, len(durations) - 1)
                    ]

        except Exception as e:
            logger.debug(f"[HEALTH] Latency metrics error: {e}")

    def _collect_throughput_metrics(self, metrics: HealthMetrics) -> None:
        """Collect throughput metrics."""
        try:
            from syn.sched.models import CognitiveTask
            from syn.sched.types import TaskState

            one_minute_ago = timezone.now() - timedelta(minutes=1)

            # Completed in last minute
            metrics.tasks_completed_last_minute = CognitiveTask.objects.filter(
                state=TaskState.SUCCESS.value,
                completed_at__gte=one_minute_ago,
            ).count()

            # Failed in last minute (excluding DLQ)
            metrics.tasks_failed_last_minute = CognitiveTask.objects.filter(
                state=TaskState.FAILURE.value,
                completed_at__gte=one_minute_ago,
            ).count()

        except Exception as e:
            logger.debug(f"[HEALTH] Throughput metrics error: {e}")

    def _collect_cascade_metrics(self, metrics: HealthMetrics) -> None:
        """Collect cascade depth metrics."""
        try:
            from syn.sched.models import CognitiveTask

            # Max cascade depth in recent tasks
            one_minute_ago = timezone.now() - timedelta(minutes=1)
            recent_tasks = CognitiveTask.objects.filter(created_at__gte=one_minute_ago)

            max_depth_result = recent_tasks.aggregate(max_depth=Count("cascade_depth"))
            if max_depth_result["max_depth"]:
                metrics.max_cascade_depth_seen = max_depth_result["max_depth"]

            # Count cascade limit violations (from events or logs)
            # Approximation: tasks at max depth
            metrics.cascade_depth_exceeded_count = recent_tasks.filter(
                cascade_depth__gte=5  # Default cascade limit
            ).count()

        except Exception as e:
            logger.debug(f"[HEALTH] Cascade metrics error: {e}")

    def _calculate_trends(self, metrics: HealthMetrics) -> None:
        """Calculate trend metrics from history."""
        if len(self._metrics_history) < 3:
            return

        # Latency trend (compare avg over windows)
        old_metrics = self._metrics_history[-3]
        if old_metrics.avg_task_latency_ms > 0:
            metrics.latency_trend = (
                metrics.avg_task_latency_ms - old_metrics.avg_task_latency_ms
            ) / old_metrics.avg_task_latency_ms

        # Throughput trend
        old_throughput = old_metrics.tasks_completed_last_minute
        if old_throughput > 0:
            metrics.throughput_trend = (
                metrics.tasks_completed_last_minute - old_throughput
            ) / old_throughput

    def _compute_health_status(self, metrics: HealthMetrics) -> HealthStatus:
        """Compute overall health status from metrics."""
        # Critical indicators
        critical_conditions = [
            metrics.queue_utilization >= 0.95,
            metrics.circuit_open_ratio >= 0.5,
            metrics.governance_denial_rate >= 0.8,
            metrics.dlq_growth_rate > 10,  # 10+ per minute
        ]
        if any(critical_conditions):
            return HealthStatus.CRITICAL

        # Stressed indicators
        stressed_conditions = [
            metrics.queue_utilization >= 0.8,
            metrics.circuit_open_ratio >= 0.3,
            metrics.governance_denial_rate >= 0.5,
            metrics.dlq_size >= self._dlq_threshold * 2,
            metrics.worker_utilization >= 0.9,
        ]
        if sum(stressed_conditions) >= 2:
            return HealthStatus.STRESSED

        # Degraded indicators
        degraded_conditions = [
            metrics.queue_utilization >= 0.6,
            metrics.governance_denial_rate >= self._governance_denial_threshold,
            metrics.dlq_size >= self._dlq_threshold,
            metrics.latency_trend > 0.5,  # 50% latency increase
        ]
        if sum(degraded_conditions) >= 2:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def get_metrics(self) -> HealthMetrics:
        """Get current health metrics."""
        with self._lock:
            if self._current_metrics:
                return self._current_metrics
            # Collect on-demand if not running
            return self._collect_metrics()

    def get_history(self, minutes: int | None = None) -> list[HealthMetrics]:
        """Get historical metrics."""
        with self._lock:
            if minutes is None:
                return list(self._metrics_history)

            cutoff = timezone.now() - timedelta(minutes=minutes)
            return [m for m in self._metrics_history if m.collected_at >= cutoff]

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running
