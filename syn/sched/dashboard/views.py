"""
Dashboard Views - REST API and Admin Views

Standard: SCH-005 §4 (Dashboard Views)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

This module provides:
- REST API endpoints for dashboard data
- Django admin custom views
- WebSocket-compatible data structures

All views return JSON for easy consumption by:
- Admin panel JavaScript
- External monitoring tools
- WebSocket real-time updates
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .service import DashboardService

logger = logging.getLogger(__name__)

# Global dashboard service instance
_dashboard_service: DashboardService | None = None


def get_dashboard_service() -> DashboardService:
    """Get or create the dashboard service singleton."""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service


def set_dashboard_scheduler(scheduler: Any) -> None:
    """Set the scheduler for the dashboard service."""
    service = get_dashboard_service()
    service.set_scheduler(scheduler)


def dashboard_api(func: Callable) -> Callable:
    """Decorator for dashboard API endpoints."""

    @wraps(func)
    def wrapper(request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            result = func(request, *args, **kwargs)
            if isinstance(result, JsonResponse):
                return result
            return JsonResponse(result)
        except Exception as e:
            logger.error(f"Dashboard API error: {e}", exc_info=True)
            return JsonResponse(
                {"error": str(e), "status": "error"},
                status=500,
            )

    return wrapper


# =============================================================================
# REST API ENDPOINTS
# =============================================================================


@require_GET
@staff_member_required
@dashboard_api
def dashboard_overview(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/

    Returns complete dashboard overview.

    Response:
    {
        "timestamp": "2025-12-08T10:00:00Z",
        "summary": {...},
        "health": {...},
        "alerts": {...},
        "dlq": {...}
    }
    """
    service = get_dashboard_service()
    return service.get_overview()


@require_GET
@staff_member_required
@dashboard_api
def dashboard_queues(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/queues/

    Returns queue status for all queues.

    Response:
    {
        "timestamp": "...",
        "queues": [
            {"name": "core", "depth": 100, "status": "healthy", ...},
            ...
        ],
        "total_depth": 250
    }
    """
    service = get_dashboard_service()
    return service.get_queue_status()


@require_GET
@staff_member_required
@dashboard_api
def dashboard_workers(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/workers/

    Returns worker status.

    Response:
    {
        "timestamp": "...",
        "workers": [...],
        "summary": {"total": 4, "healthy": 4, "unhealthy": 0, ...}
    }
    """
    service = get_dashboard_service()
    return service.get_worker_status()


@require_GET
@staff_member_required
@dashboard_api
def dashboard_schedules(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/schedules/

    Returns schedule status.

    Response:
    {
        "timestamp": "...",
        "schedules": [...],
        "summary": {"total": 10, "enabled": 8, "disabled": 2}
    }
    """
    service = get_dashboard_service()
    return service.get_schedule_status()


@require_GET
@staff_member_required
@dashboard_api
def dashboard_circuits(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/circuits/

    Returns circuit breaker status.

    Response:
    {
        "timestamp": "...",
        "circuits": [...],
        "summary": {"total": 5, "closed": 4, "open": 1, ...}
    }
    """
    service = get_dashboard_service()
    return service.get_circuit_status()


@require_GET
@staff_member_required
@dashboard_api
def dashboard_dlq(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/dlq/

    Returns Dead Letter Queue status.

    Response:
    {
        "timestamp": "...",
        "status": "warning",
        "counts": {...},
        "by_task": {...},
        "by_error_type": {...}
    }
    """
    service = get_dashboard_service()
    return service.get_dlq_status()


@require_GET
@staff_member_required
@dashboard_api
def dashboard_throttle(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/throttle/

    Returns backpressure throttle status.

    Response:
    {
        "timestamp": "...",
        "level": "MODERATE",
        "status": "warning",
        "restrictions": {...},
        "emergency": {...},
        "decisions": {...}
    }
    """
    service = get_dashboard_service()
    return service.get_throttle_status()


@require_GET
@staff_member_required
@dashboard_api
def dashboard_task_types(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/tasks/

    Returns task type statistics.

    Response:
    {
        "timestamp": "...",
        "task_types": [...],
        "total_types": 25
    }
    """
    service = get_dashboard_service()
    return service.get_task_types()


@require_GET
@staff_member_required
@dashboard_api
def dashboard_alerts(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/alerts/

    Returns active alerts and recent history.

    Response:
    {
        "active": [...],
        "history": [...]
    }
    """
    service = get_dashboard_service()
    return {
        "active": [a.to_dict() for a in service.get_active_alerts()],
        "history": service.get_alert_history(limit=50),
    }


@require_POST
@staff_member_required
@dashboard_api
def dashboard_alert_acknowledge(request: HttpRequest, alert_id: str) -> dict[str, Any]:
    """
    POST /api/scheduler/dashboard/alerts/<alert_id>/acknowledge/

    Acknowledge an alert.

    Response:
    {
        "acknowledged": true,
        "alert_id": "..."
    }
    """
    service = get_dashboard_service()
    success = service.acknowledge_alert(alert_id)
    return {
        "acknowledged": success,
        "alert_id": alert_id,
    }


@require_GET
@staff_member_required
@dashboard_api
def dashboard_history(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/history/?metric=queue_depth&hours=24

    Returns historical metric data.

    Query params:
    - metric: queue_depth, running_tasks, throttle_level, dlq_pending, success_rate
    - hours: Number of hours (default: 24)

    Response:
    {
        "metric": "queue_depth",
        "hours": 24,
        "data_points": [{"timestamp": "...", "value": 100}, ...]
    }
    """
    service = get_dashboard_service()
    metric = request.GET.get("metric", "queue_depth")
    hours = int(request.GET.get("hours", "24"))

    return service.get_history(metric=metric, hours=hours)


@require_GET
@staff_member_required
@dashboard_api
def dashboard_metrics_raw(request: HttpRequest) -> dict[str, Any]:
    """
    GET /api/scheduler/dashboard/metrics/

    Returns complete raw metrics for debugging.

    Response:
    Complete DashboardMetrics as JSON
    """
    service = get_dashboard_service()
    metrics = service.get_metrics(use_cache=False)
    return metrics.to_dict()


@require_GET
@staff_member_required
def dashboard_metrics_prometheus(request: HttpRequest) -> HttpResponse:
    """
    GET /api/scheduler/dashboard/metrics/prometheus

    Returns metrics in Prometheus format.
    """
    service = get_dashboard_service()
    prometheus_data = service.export_metrics(format="prometheus")
    return HttpResponse(prometheus_data, content_type="text/plain")


# =============================================================================
# ADMIN PANEL INTEGRATION
# =============================================================================


def get_admin_dashboard_context() -> dict[str, Any]:
    """
    Get dashboard data for Django admin template context.

    This is called by the custom admin index view.
    """
    service = get_dashboard_service()

    try:
        overview = service.get_overview()
        throttle = service.get_throttle_status()
        queues = service.get_queue_status()
        alerts = [a.to_dict() for a in service.get_active_alerts()]

        return {
            "scheduler_dashboard": {
                "overview": overview,
                "throttle": throttle,
                "queues": queues,
                "alerts": alerts,
                "timestamp": timezone.now().isoformat(),
            }
        }
    except Exception as e:
        logger.error(f"Admin dashboard error: {e}")
        return {
            "scheduler_dashboard": {
                "error": str(e),
                "timestamp": timezone.now().isoformat(),
            }
        }


# =============================================================================
# URL PATTERNS
# =============================================================================


def get_dashboard_urls():
    """
    Get URL patterns for dashboard API.

    Include these in your urls.py:

        from syn.sched.dashboard.views import get_dashboard_urls
        urlpatterns += get_dashboard_urls()
    """
    from django.urls import path

    return [
        path(
            "api/scheduler/dashboard/",
            dashboard_overview,
            name="scheduler_dashboard_overview",
        ),
        path(
            "api/scheduler/dashboard/queues/",
            dashboard_queues,
            name="scheduler_dashboard_queues",
        ),
        path(
            "api/scheduler/dashboard/workers/",
            dashboard_workers,
            name="scheduler_dashboard_workers",
        ),
        path(
            "api/scheduler/dashboard/schedules/",
            dashboard_schedules,
            name="scheduler_dashboard_schedules",
        ),
        path(
            "api/scheduler/dashboard/circuits/",
            dashboard_circuits,
            name="scheduler_dashboard_circuits",
        ),
        path(
            "api/scheduler/dashboard/dlq/",
            dashboard_dlq,
            name="scheduler_dashboard_dlq",
        ),
        path(
            "api/scheduler/dashboard/throttle/",
            dashboard_throttle,
            name="scheduler_dashboard_throttle",
        ),
        path(
            "api/scheduler/dashboard/tasks/",
            dashboard_task_types,
            name="scheduler_dashboard_tasks",
        ),
        path(
            "api/scheduler/dashboard/alerts/",
            dashboard_alerts,
            name="scheduler_dashboard_alerts",
        ),
        path(
            "api/scheduler/dashboard/alerts/<str:alert_id>/acknowledge/",
            dashboard_alert_acknowledge,
            name="scheduler_dashboard_alert_acknowledge",
        ),
        path(
            "api/scheduler/dashboard/history/",
            dashboard_history,
            name="scheduler_dashboard_history",
        ),
        path(
            "api/scheduler/dashboard/metrics/",
            dashboard_metrics_raw,
            name="scheduler_dashboard_metrics",
        ),
        path(
            "api/scheduler/dashboard/metrics/prometheus",
            dashboard_metrics_prometheus,
            name="scheduler_dashboard_prometheus",
        ),
    ]
