"""
Dashboard Service - High-Level Dashboard API

Standard: SCH-005 §3 (Dashboard Service)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The DashboardService provides:
- Unified access to scheduler metrics
- Caching for performance
- Historical data aggregation
- Alert threshold evaluation
- Export functionality

This is the primary interface for the admin panel and API.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from django.core.cache import cache
from django.utils import timezone

from .metrics import (
    DashboardMetrics,
    SchedulerMetricsCollector,
)

logger = logging.getLogger(__name__)


class AlertSeverity:
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Dashboard alert."""

    id: str
    severity: str
    title: str
    message: str
    metric: str
    value: Any
    threshold: Any
    timestamp: datetime = field(default_factory=timezone.now)
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class DashboardConfig:
    """
    Configuration for DashboardService.

    Standard: SCH-005 §3.1
    """

    # Caching
    cache_ttl_seconds: int = 30
    cache_key_prefix: str = "synara:dashboard"

    # Alert thresholds
    queue_depth_warning: int = 500
    queue_depth_critical: int = 900
    dlq_warning: int = 50
    dlq_critical: int = 100
    throttle_warning_levels: set[str] = field(default_factory=lambda: {"MODERATE", "HEAVY"})
    throttle_critical_levels: set[str] = field(default_factory=lambda: {"CRITICAL"})
    worker_unhealthy_warning: int = 1
    circuit_open_warning: int = 1

    # History
    history_retention_hours: int = 168  # 1 week
    history_resolution_minutes: int = 5

    # Callbacks
    on_alert: Callable[[Alert], None] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "queue_depth_warning": self.queue_depth_warning,
            "queue_depth_critical": self.queue_depth_critical,
            "dlq_warning": self.dlq_warning,
            "dlq_critical": self.dlq_critical,
            "throttle_warning_levels": list(self.throttle_warning_levels),
            "throttle_critical_levels": list(self.throttle_critical_levels),
            "worker_unhealthy_warning": self.worker_unhealthy_warning,
            "circuit_open_warning": self.circuit_open_warning,
            "history_retention_hours": self.history_retention_hours,
            "history_resolution_minutes": self.history_resolution_minutes,
        }


class DashboardService:
    """
    High-level dashboard service for scheduler observability.

    Standard: SCH-005 §3

    Features:
    - Cached metrics access
    - Alert evaluation
    - Historical data queries
    - Summary views for admin panel
    - Export functionality

    Usage:
        dashboard = DashboardService(scheduler=scheduler)

        # Get overview
        overview = dashboard.get_overview()

        # Get specific sections
        queues = dashboard.get_queue_status()
        workers = dashboard.get_worker_status()
        throttle = dashboard.get_throttle_status()

        # Get alerts
        alerts = dashboard.get_active_alerts()
    """

    def __init__(
        self,
        scheduler: Any | None = None,
        config: DashboardConfig | None = None,
        tenant_id: str | None = None,
    ):
        """
        Initialize dashboard service.

        Args:
            scheduler: CognitiveScheduler instance
            config: Dashboard configuration
            tenant_id: Tenant ID for cache key isolation (SEC-001 §9.7)

        Security:
            SEC-001 §9.7: Cache keys are prefixed with tenant_id to prevent
            cross-tenant cache collisions.
        """
        self._scheduler = scheduler
        self._config = config or DashboardConfig()
        self._tenant_id = tenant_id  # SEC-001 §9.7: Tenant isolation
        self._collector = SchedulerMetricsCollector(scheduler=scheduler)
        self._lock = threading.RLock()

        # Alert state
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []

        # Metrics history
        self._metrics_history: list[DashboardMetrics] = []
        self._max_history_size = int(
            self._config.history_retention_hours * 60 / self._config.history_resolution_minutes
        )

    def set_scheduler(self, scheduler: Any) -> None:
        """Set or update the scheduler reference."""
        self._scheduler = scheduler
        self._collector.set_scheduler(scheduler)

    def set_tenant_id(self, tenant_id: str | None) -> None:
        """
        Set or update the tenant ID for cache key isolation.

        Args:
            tenant_id: Tenant ID for cache key isolation (SEC-001 §9.7)

        Security:
            SEC-001 §9.7: Setting tenant_id ensures cache operations
            are isolated to the specified tenant.
        """
        self._tenant_id = tenant_id

    # =========================================================================
    # Core Data Access
    # =========================================================================

    def _build_cache_key(self, suffix: str) -> str:
        """
        Build tenant-prefixed cache key per SEC-001 §9.7.

        Args:
            suffix: Cache key suffix (e.g., 'metrics', 'alerts')

        Returns:
            Tenant-prefixed cache key

        Security:
            SEC-001 §9.7: All cache keys include tenant_id prefix.
        """
        tenant_prefix = self._tenant_id if self._tenant_id else "global"
        return f"{tenant_prefix}:{self._config.cache_key_prefix}:{suffix}"

    def get_metrics(self, use_cache: bool = True) -> DashboardMetrics:
        """
        Get current dashboard metrics.

        Args:
            use_cache: Whether to use cached data

        Returns:
            Complete DashboardMetrics snapshot

        Security:
            SEC-001 §9.7: Uses tenant-prefixed cache keys.
        """
        # SEC-001 §9.7: Build tenant-prefixed cache key
        cache_key = self._build_cache_key("metrics")

        if use_cache:
            cached = cache.get(cache_key)
            if cached:
                return cached

        metrics = self._collector.collect()

        # Cache result with tenant-prefixed key
        if use_cache:
            cache.set(cache_key, metrics, timeout=self._config.cache_ttl_seconds)

        # Store in history
        with self._lock:
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history_size:
                self._metrics_history = self._metrics_history[-self._max_history_size :]

        # Evaluate alerts
        self._evaluate_alerts(metrics)

        return metrics

    def get_overview(self) -> dict[str, Any]:
        """
        Get dashboard overview for admin panel.

        Returns dict with summary, health, and key metrics.
        """
        metrics = self.get_metrics()
        alerts = self.get_active_alerts()

        return {
            "timestamp": timezone.now().isoformat(),
            "summary": {
                "pending_tasks": metrics.total_pending_tasks,
                "running_tasks": metrics.total_running_tasks,
                "completed_today": metrics.total_completed_today,
                "failed_today": metrics.total_failed_today,
                "success_rate": f"{metrics.overall_success_rate:.1%}",
            },
            "health": {
                "scheduler": "running" if metrics.scheduler_running else "stopped",
                "backpressure": "active" if metrics.backpressure_running else "inactive",
                "workers": f"{metrics.workers_healthy}/{metrics.workers_healthy + metrics.workers_unhealthy} healthy",
                "circuits": f"{metrics.circuits_open}/{metrics.circuits_total} open",
                "throttle_level": metrics.throttle.current_level,
            },
            "alerts": {
                "critical": sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL),
                "warning": sum(1 for a in alerts if a.severity == AlertSeverity.WARNING),
                "total": len(alerts),
            },
            "dlq": {
                "pending": metrics.dlq.pending_entries,
                "growth_rate": f"{metrics.dlq.growth_rate_per_hour:.1f}/hr",
            },
        }

    # =========================================================================
    # Section-Specific Views
    # =========================================================================

    def get_queue_status(self) -> dict[str, Any]:
        """Get queue status for admin panel."""
        metrics = self.get_metrics()

        queues = []
        for q in metrics.queues:
            status = "healthy"
            if q.depth >= self._config.queue_depth_critical:
                status = "critical"
            elif q.depth >= self._config.queue_depth_warning:
                status = "warning"

            queues.append(
                {
                    "name": q.queue_name,
                    "depth": q.depth,
                    "pending": q.pending,
                    "running": q.running,
                    "throughput": f"{q.throughput_per_minute:.1f}/min",
                    "oldest_age": self._format_duration(q.oldest_task_age_seconds),
                    "status": status,
                }
            )

        return {
            "timestamp": timezone.now().isoformat(),
            "queues": queues,
            "total_depth": sum(q.depth for q in metrics.queues),
        }

    def get_worker_status(self) -> dict[str, Any]:
        """Get worker status for admin panel."""
        metrics = self.get_metrics()

        workers = []
        for w in metrics.workers:
            workers.append(
                {
                    "id": w.worker_id,
                    "state": w.state.value,
                    "resource_class": w.resource_class,
                    "current_task": w.current_task_name,
                    "completed": w.tasks_completed,
                    "failed": w.tasks_failed,
                    "uptime": self._format_duration(w.uptime_seconds),
                }
            )

        # Group by state
        by_state = {}
        for w in metrics.workers:
            state = w.state.value
            by_state[state] = by_state.get(state, 0) + 1

        return {
            "timestamp": timezone.now().isoformat(),
            "workers": workers,
            "summary": {
                "total": len(workers),
                "healthy": metrics.workers_healthy,
                "unhealthy": metrics.workers_unhealthy,
                "by_state": by_state,
            },
        }

    def get_throttle_status(self) -> dict[str, Any]:
        """Get backpressure throttle status."""
        metrics = self.get_metrics()
        t = metrics.throttle

        # Determine status
        status = "healthy"
        if t.is_emergency:
            status = "emergency"
        elif t.current_level in self._config.throttle_critical_levels:
            status = "critical"
        elif t.current_level in self._config.throttle_warning_levels:
            status = "warning"

        return {
            "timestamp": timezone.now().isoformat(),
            "level": t.current_level,
            "level_value": f"{t.level_value:.0%}",
            "status": status,
            "restrictions": {
                "paused_schedules": t.paused_schedules,
                "skip_low_priority": t.skip_low_priority,
                "skip_batch_tasks": t.skip_batch_tasks,
            },
            "confidence_penalty": f"{t.confidence_penalty:.2f}",
            "emergency": {
                "active": t.is_emergency,
                "reason": t.emergency_reason,
            },
            "decisions": {
                "total": t.decisions_total,
                "allowed": t.decisions_allowed,
                "denied": t.decisions_denied,
                "deny_rate": f"{t.decisions_denied / max(1, t.decisions_total):.1%}",
            },
            "time_at_level": {level: self._format_duration(seconds) for level, seconds in t.time_at_level.items()},
        }

    def get_schedule_status(self) -> dict[str, Any]:
        """Get schedule status for admin panel."""
        metrics = self.get_metrics()
        now = timezone.now()

        schedules = []
        for s in metrics.schedules:
            # Time until next run
            time_to_next = None
            if s.next_run_at:
                delta = s.next_run_at - now
                time_to_next = self._format_duration(delta.total_seconds())

            # Time since last run
            time_since_last = None
            if s.last_run_at:
                delta = now - s.last_run_at
                time_since_last = self._format_duration(delta.total_seconds())

            schedules.append(
                {
                    "id": s.schedule_id,
                    "name": s.name,
                    "task": s.task_name,
                    "enabled": s.is_enabled,
                    "expression": s.expression,
                    "last_run": time_since_last or "never",
                    "next_run": time_to_next or "not scheduled",
                    "run_count": s.run_count,
                    "last_status": s.last_status,
                }
            )

        return {
            "timestamp": timezone.now().isoformat(),
            "schedules": schedules,
            "summary": {
                "total": len(schedules),
                "enabled": sum(1 for s in metrics.schedules if s.enabled),
                "disabled": sum(1 for s in metrics.schedules if not s.enabled),
            },
        }

    def get_circuit_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        metrics = self.get_metrics()

        circuits = []
        for c in metrics.circuits:
            status = "healthy"
            if c.state == "open":
                status = "critical"
            elif c.state == "half_open":
                status = "warning"

            circuits.append(
                {
                    "service": c.service_name,
                    "state": c.state,
                    "status": status,
                    "failures": c.failure_count,
                    "successes": c.success_count,
                    "failure_rate": f"{c.failure_rate:.1%}",
                    "consecutive_failures": c.consecutive_failures,
                }
            )

        return {
            "timestamp": timezone.now().isoformat(),
            "circuits": circuits,
            "summary": {
                "total": len(circuits),
                "closed": sum(1 for c in metrics.circuits if c.state == "closed"),
                "open": sum(1 for c in metrics.circuits if c.state == "open"),
                "half_open": sum(1 for c in metrics.circuits if c.state == "half_open"),
            },
        }

    def get_dlq_status(self) -> dict[str, Any]:
        """Get Dead Letter Queue status."""
        metrics = self.get_metrics()
        d = metrics.dlq

        status = "healthy"
        if d.pending_entries >= self._config.dlq_critical:
            status = "critical"
        elif d.pending_entries >= self._config.dlq_warning:
            status = "warning"

        return {
            "timestamp": timezone.now().isoformat(),
            "status": status,
            "counts": {
                "total": d.total_entries,
                "pending": d.pending_entries,
                "resolved": d.resolved_entries,
                "retried": d.retried_entries,
                "discarded": d.discarded_entries,
            },
            "growth_rate": f"{d.growth_rate_per_hour:.1f}/hr",
            "oldest_age": f"{d.oldest_entry_age_hours:.1f}h",
            "by_task": d.entries_by_task,
            "by_error_type": d.entries_by_error_type,
        }

    def get_task_types(self) -> dict[str, Any]:
        """Get task type statistics."""
        metrics = self.get_metrics()

        task_types = []
        for t in metrics.task_types:
            task_types.append(
                {
                    "name": t.task_name,
                    "total": t.total_executions,
                    "success": t.successful_executions,
                    "failed": t.failed_executions,
                    "success_rate": f"{t.success_rate:.1%}",
                    "avg_duration": f"{t.avg_duration_ms:.0f}ms",
                    "p95_duration": f"{t.p95_duration_ms:.0f}ms",
                    "last_run": t.last_execution.isoformat() if t.last_execution else None,
                }
            )

        # Sort by total executions
        task_types.sort(key=lambda x: -x["total"])

        return {
            "timestamp": timezone.now().isoformat(),
            "task_types": task_types[:20],  # Top 20
            "total_types": len(metrics.task_types),
        }

    # =========================================================================
    # Alert Management
    # =========================================================================

    def _evaluate_alerts(self, metrics: DashboardMetrics) -> None:
        """Evaluate alert conditions."""
        new_alerts: dict[str, Alert] = {}

        # Queue depth alerts
        total_depth = sum(q.depth for q in metrics.queues)
        if total_depth >= self._config.queue_depth_critical:
            new_alerts["queue_depth_critical"] = Alert(
                id="queue_depth_critical",
                severity=AlertSeverity.CRITICAL,
                title="Queue Depth Critical",
                message=f"Total queue depth ({total_depth}) exceeds critical threshold",
                metric="queue_depth",
                value=total_depth,
                threshold=self._config.queue_depth_critical,
            )
        elif total_depth >= self._config.queue_depth_warning:
            new_alerts["queue_depth_warning"] = Alert(
                id="queue_depth_warning",
                severity=AlertSeverity.WARNING,
                title="Queue Depth Warning",
                message=f"Total queue depth ({total_depth}) exceeds warning threshold",
                metric="queue_depth",
                value=total_depth,
                threshold=self._config.queue_depth_warning,
            )

        # DLQ alerts
        if metrics.dlq.pending_entries >= self._config.dlq_critical:
            new_alerts["dlq_critical"] = Alert(
                id="dlq_critical",
                severity=AlertSeverity.CRITICAL,
                title="DLQ Critical",
                message=f"DLQ pending entries ({metrics.dlq.pending_entries}) exceeds critical threshold",
                metric="dlq_pending",
                value=metrics.dlq.pending_entries,
                threshold=self._config.dlq_critical,
            )
        elif metrics.dlq.pending_entries >= self._config.dlq_warning:
            new_alerts["dlq_warning"] = Alert(
                id="dlq_warning",
                severity=AlertSeverity.WARNING,
                title="DLQ Warning",
                message=f"DLQ pending entries ({metrics.dlq.pending_entries}) exceeds warning threshold",
                metric="dlq_pending",
                value=metrics.dlq.pending_entries,
                threshold=self._config.dlq_warning,
            )

        # Throttle level alerts
        level = metrics.throttle.current_level
        if level in self._config.throttle_critical_levels:
            new_alerts["throttle_critical"] = Alert(
                id="throttle_critical",
                severity=AlertSeverity.CRITICAL,
                title="Throttle Critical",
                message=f"Backpressure at {level} level",
                metric="throttle_level",
                value=level,
                threshold="CRITICAL",
            )
        elif level in self._config.throttle_warning_levels:
            new_alerts["throttle_warning"] = Alert(
                id="throttle_warning",
                severity=AlertSeverity.WARNING,
                title="Throttle Warning",
                message=f"Backpressure at {level} level",
                metric="throttle_level",
                value=level,
                threshold="MODERATE+",
            )

        # Emergency alert
        if metrics.throttle.is_emergency:
            new_alerts["emergency"] = Alert(
                id="emergency",
                severity=AlertSeverity.CRITICAL,
                title="EMERGENCY",
                message=f"Backpressure emergency: {metrics.throttle.emergency_reason}",
                metric="emergency",
                value=True,
                threshold=False,
            )

        # Worker health alerts
        if metrics.workers_unhealthy >= self._config.worker_unhealthy_warning:
            new_alerts["workers_unhealthy"] = Alert(
                id="workers_unhealthy",
                severity=AlertSeverity.WARNING,
                title="Unhealthy Workers",
                message=f"{metrics.workers_unhealthy} worker(s) unhealthy",
                metric="workers_unhealthy",
                value=metrics.workers_unhealthy,
                threshold=self._config.worker_unhealthy_warning,
            )

        # Circuit breaker alerts
        if metrics.circuits_open >= self._config.circuit_open_warning:
            severity = AlertSeverity.CRITICAL if metrics.circuits_open >= 3 else AlertSeverity.WARNING
            new_alerts["circuits_open"] = Alert(
                id="circuits_open",
                severity=severity,
                title="Circuits Open",
                message=f"{metrics.circuits_open}/{metrics.circuits_total} circuit breaker(s) open",
                metric="circuits_open",
                value=metrics.circuits_open,
                threshold=self._config.circuit_open_warning,
            )

        # Update active alerts
        with self._lock:
            # Check for new alerts
            for alert_id, alert in new_alerts.items():
                if alert_id not in self._active_alerts:
                    self._active_alerts[alert_id] = alert
                    self._alert_history.append(alert)
                    if self._config.on_alert:
                        try:
                            self._config.on_alert(alert)
                        except Exception as e:
                            logger.error(f"Alert callback error: {e}")
                    logger.warning(f"[DASHBOARD] Alert triggered: {alert.title}")

            # Remove cleared alerts
            cleared = set(self._active_alerts.keys()) - set(new_alerts.keys())
            for alert_id in cleared:
                logger.info(f"[DASHBOARD] Alert cleared: {self._active_alerts[alert_id].title}")
                del self._active_alerts[alert_id]

    def get_active_alerts(self) -> list[Alert]:
        """Get list of active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].acknowledged = True
                return True
            return False

    def get_alert_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get alert history."""
        with self._lock:
            return [a.to_dict() for a in self._alert_history[-limit:]]

    # =========================================================================
    # Historical Data
    # =========================================================================

    def get_history(
        self,
        metric: str = "queue_depth",
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get historical metric data.

        Args:
            metric: Metric name to retrieve
            hours: Hours of history

        Returns:
            Time series data
        """
        cutoff = timezone.now() - timedelta(hours=hours)

        with self._lock:
            relevant = [m for m in self._metrics_history if m.collected_at >= cutoff]

        data_points = []
        for m in relevant:
            value = None
            if metric == "queue_depth":
                value = m.total_pending_tasks
            elif metric == "running_tasks":
                value = m.total_running_tasks
            elif metric == "throttle_level":
                value = m.throttle.level_value
            elif metric == "dlq_pending":
                value = m.dlq.pending_entries
            elif metric == "success_rate":
                value = m.overall_success_rate

            if value is not None:
                data_points.append(
                    {
                        "timestamp": m.collected_at.isoformat(),
                        "value": value,
                    }
                )

        return {
            "metric": metric,
            "hours": hours,
            "data_points": data_points,
            "count": len(data_points),
        }

    # =========================================================================
    # Export
    # =========================================================================

    def export_metrics(self, format: str = "json") -> Any:
        """
        Export current metrics.

        Args:
            format: Export format ("json" or "prometheus")

        Returns:
            Formatted metrics
        """
        metrics = self.get_metrics(use_cache=False)

        if format == "json":
            return metrics.to_dict()

        elif format == "prometheus":
            lines = []

            # Queue metrics
            for q in metrics.queues:
                lines.append(f'synara_queue_depth{{queue="{q.queue_name}"}} {q.depth}')
                lines.append(f'synara_queue_throughput{{queue="{q.queue_name}"}} {q.throughput_per_minute}')

            # Task summary
            lines.append(f"synara_tasks_pending {metrics.total_pending_tasks}")
            lines.append(f"synara_tasks_running {metrics.total_running_tasks}")
            lines.append(f"synara_tasks_completed_today {metrics.total_completed_today}")
            lines.append(f"synara_tasks_failed_today {metrics.total_failed_today}")

            # Throttle
            lines.append(f"synara_throttle_level {metrics.throttle.level_value}")
            lines.append(f"synara_throttle_emergency {1 if metrics.throttle.is_emergency else 0}")

            # DLQ
            lines.append(f"synara_dlq_pending {metrics.dlq.pending_entries}")
            lines.append(f"synara_dlq_growth_rate {metrics.dlq.growth_rate_per_hour}")

            # Workers
            lines.append(f"synara_workers_healthy {metrics.workers_healthy}")
            lines.append(f"synara_workers_unhealthy {metrics.workers_unhealthy}")

            # Circuits
            lines.append(f"synara_circuits_open {metrics.circuits_open}")
            lines.append(f"synara_circuits_total {metrics.circuits_total}")

            return "\n".join(lines)

        raise ValueError(f"Unknown format: {format}")

    # =========================================================================
    # Utilities
    # =========================================================================

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 0:
            return "0s"
        elif seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        else:
            return f"{seconds / 86400:.1f}d"

    @property
    def config(self) -> DashboardConfig:
        """Get dashboard configuration."""
        return self._config

    @property
    def collector(self) -> SchedulerMetricsCollector:
        """Get metrics collector."""
        return self._collector
