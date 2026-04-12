"""
Synara CognitiveScheduler Dashboard - Observability & Introspection

Standard: SCH-005 (Scheduler Dashboard & Observability)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Dashboard System                             │
    │  ┌──────────────────┐   ┌──────────────────┐                    │
    │  │ MetricsCollector │───│ DashboardService │                    │
    │  └──────────────────┘   └────────┬─────────┘                    │
    │                                  │                               │
    │         ┌────────────────────────┼────────────────────────┐     │
    │         ▼                        ▼                        ▼     │
    │    REST API              Django Admin            WebSocket       │
    │  /api/dashboard/         Custom Views          (real-time)       │
    └─────────────────────────────────────────────────────────────────┘

This package provides complete visibility into the scheduler's internal state:
- Queue depth per queue type
- Per-resource-class utilization
- Throttle level (NONE → CRITICAL)
- DLQ growth trends
- Circuit breaker states
- Average latency per task type
- Schedule last/next run timestamps
- Worker health (alive, restarting, quarantined)

The Dashboard is Synara's "window into the temporal cortex."
"""

from .metrics import (
    CircuitMetrics,
    DashboardMetrics,
    DLQMetrics,
    QueueMetrics,
    ScheduleMetrics,
    SchedulerMetricsCollector,
    TaskTypeMetrics,
    WorkerMetrics,
)
from .service import DashboardConfig, DashboardService
from .views import (
    dashboard_circuits,
    dashboard_dlq,
    dashboard_overview,
    dashboard_queues,
    dashboard_schedules,
    dashboard_workers,
)

__all__ = [
    # Metrics Collection
    "SchedulerMetricsCollector",
    "DashboardMetrics",
    "QueueMetrics",
    "WorkerMetrics",
    "TaskTypeMetrics",
    "ScheduleMetrics",
    "CircuitMetrics",
    "DLQMetrics",
    # Service
    "DashboardService",
    "DashboardConfig",
    # Views
    "dashboard_overview",
    "dashboard_queues",
    "dashboard_workers",
    "dashboard_schedules",
    "dashboard_circuits",
    "dashboard_dlq",
]
