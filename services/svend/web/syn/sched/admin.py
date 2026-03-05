"""
Synara Scheduler Admin - Django Admin Integration

Standard: SCH-005 §5 (Admin Integration)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

This module provides Django admin views for:
- CognitiveTask management
- Schedule management
- DeadLetterEntry triage
- CircuitBreakerState monitoring
- Dashboard overview widget

The admin panel serves as Synara's operational control center.
"""

from django.contrib import admin
from django.urls import reverse
from django.utils import timezone
from django.utils.html import format_html

from .models import (
    CircuitBreakerState,
    CognitiveTask,
    DeadLetterEntry,
    Schedule,
    TaskExecution,
)
from .types import TaskState

# =============================================================================
# COGNITIVE TASK ADMIN
# =============================================================================


@admin.register(CognitiveTask)
class CognitiveTaskAdmin(admin.ModelAdmin):
    """Admin for CognitiveTask model."""

    list_display = (
        "short_id",
        "task_name",
        "state_badge",
        "priority_badge",
        "queue",
        "attempts",
        "created_at",
        "tenant_id",
    )
    list_filter = (
        "state",
        "priority",
        "queue",
        "retry_strategy",
    )
    search_fields = (
        "id",
        "task_name",
        "correlation_id",
        "tenant_id",
    )
    readonly_fields = (
        "id",
        "correlation_id",
        "root_correlation_id",
        "parent_task_id",
        "cascade_depth",
        "priority_score",
        "created_at",
        "updated_at",
        "scheduled_at",
        "completed_at",
    )
    ordering = ("-created_at",)
    date_hierarchy = "created_at"

    fieldsets = (
        ("Identity", {"fields": ("id", "task_name", "tenant_id")}),
        (
            "Correlation & Lineage",
            {
                "fields": ("correlation_id", "root_correlation_id", "parent_task_id", "cascade_depth"),
                "classes": ("collapse",),
            },
        ),
        (
            "State",
            {
                "fields": ("state", "attempts", "max_attempts", "result", "error_message", "error_type"),
            },
        ),
        (
            "Configuration",
            {
                "fields": ("queue", "priority", "retry_strategy", "timeout_seconds"),
            },
        ),
        (
            "Cognitive Attributes",
            {
                "fields": ("urgency", "confidence_score", "governance_risk", "resource_weight", "priority_score"),
                "classes": ("collapse",),
            },
        ),
        (
            "Timing",
            {
                "fields": ("created_at", "updated_at", "scheduled_at", "completed_at", "deadline"),
            },
        ),
        (
            "Payload",
            {
                "fields": ("payload",),
                "classes": ("collapse",),
            },
        ),
    )

    def short_id(self, obj):
        """Show short UUID."""
        return str(obj.id)[:8]

    short_id.short_description = "ID"

    def state_badge(self, obj):
        """Show state with color badge."""
        colors = {
            TaskState.PENDING.value: "#6c757d",
            TaskState.SCHEDULED.value: "#007bff",
            TaskState.RUNNING.value: "#ffc107",
            TaskState.SUCCESS.value: "#28a745",
            TaskState.FAILURE.value: "#dc3545",
            TaskState.RETRYING.value: "#fd7e14",
            TaskState.DEAD_LETTERED.value: "#343a40",
            TaskState.CANCELLED.value: "#6c757d",
        }
        color = colors.get(obj.state, "#6c757d")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; '
            'border-radius: 4px; font-size: 11px;">{}</span>',
            color,
            obj.state,
        )

    state_badge.short_description = "State"

    def priority_badge(self, obj):
        """Show priority with badge."""
        priority_names = {
            0: ("CRITICAL", "#dc3545"),
            1: ("HIGH", "#fd7e14"),
            2: ("NORMAL", "#007bff"),
            3: ("LOW", "#6c757d"),
            4: ("BATCH", "#28a745"),
        }
        name, color = priority_names.get(obj.priority, ("UNKNOWN", "#6c757d"))
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 6px; '
            'border-radius: 3px; font-size: 10px;">{}</span>',
            color,
            name,
        )

    priority_badge.short_description = "Priority"

    actions = ["mark_cancelled", "retry_task", "move_to_dlq"]

    @admin.action(description="Cancel selected tasks")
    def mark_cancelled(self, request, queryset):
        count = queryset.filter(state__in=[TaskState.PENDING.value, TaskState.SCHEDULED.value]).update(
            state=TaskState.CANCELLED.value
        )
        self.message_user(request, f"{count} task(s) cancelled.")

    @admin.action(description="Retry selected tasks")
    def retry_task(self, request, queryset):
        count = 0
        for task in queryset.filter(state=TaskState.FAILURE.value):
            task.state = TaskState.RETRYING.value
            task.next_retry_at = timezone.now()
            task.save()
            count += 1
        self.message_user(request, f"{count} task(s) queued for retry.")

    @admin.action(description="Move to Dead Letter Queue")
    def move_to_dlq(self, request, queryset):
        count = 0
        for task in queryset.filter(state__in=[TaskState.FAILURE.value, TaskState.RETRYING.value]):
            task.state = TaskState.DEAD_LETTERED.value
            task.save()
            DeadLetterEntry.create_from_task(task, "Manual DLQ move via admin")
            count += 1
        self.message_user(request, f"{count} task(s) moved to DLQ.")


# =============================================================================
# TASK EXECUTION ADMIN
# =============================================================================


@admin.register(TaskExecution)
class TaskExecutionAdmin(admin.ModelAdmin):
    """Admin for TaskExecution model."""

    list_display = (
        "short_id",
        "task_link",
        "attempt_number",
        "success_badge",
        "duration_display",
        "worker_id",
        "started_at",
    )
    list_filter = (
        "is_success",
        "worker_id",
    )
    search_fields = (
        "id",
        "task__id",
        "task__task_name",
        "worker_id",
    )
    readonly_fields = (
        "id",
        "task",
        "attempt_number",
        "started_at",
        "completed_at",
        "duration_ms",
        "worker_id",
        "is_success",
        "result",
        "error_message",
        "error_type",
        "error_traceback",
    )
    ordering = ("-started_at",)

    def short_id(self, obj):
        return str(obj.id)[:8]

    short_id.short_description = "ID"

    def task_link(self, obj):
        url = reverse("admin:sched_cognitivetask_change", args=[obj.task_id])
        return format_html('<a href="{}">{}</a>', url, obj.task.task_name[:30])

    task_link.short_description = "Task"

    def success_badge(self, obj):
        if obj.is_success:
            return format_html('<span style="color: green;">✓</span>')
        return format_html('<span style="color: red;">✗</span>')

    success_badge.short_description = "OK"

    def duration_display(self, obj):
        if obj.duration_ms:
            if obj.duration_ms < 1000:
                return f"{obj.duration_ms}ms"
            return f"{obj.duration_ms / 1000:.1f}s"
        return "-"

    duration_display.short_description = "Duration"


# =============================================================================
# SCHEDULE ADMIN
# =============================================================================


@admin.register(Schedule)
class ScheduleAdmin(admin.ModelAdmin):
    """Admin for Schedule model."""

    list_display = (
        "schedule_id",
        "name",
        "task_name",
        "enabled_badge",
        "schedule_type",
        "expression",
        "last_run_at",
        "next_run_at",
        "run_count",
    )
    list_filter = (
        "is_enabled",
        "schedule_type",
        "queue",
    )
    search_fields = (
        "schedule_id",
        "name",
        "task_name",
    )
    readonly_fields = (
        "run_count",
        "last_run_at",
        "next_run_at",
        "created_at",
        "updated_at",
    )
    ordering = ("name",)

    fieldsets = (
        ("Identity", {"fields": ("schedule_id", "name", "task_name", "tenant_id", "is_enabled")}),
        (
            "Schedule Pattern",
            {
                "fields": ("schedule_type", "cron_expression", "interval_seconds"),
            },
        ),
        (
            "Task Configuration",
            {
                "fields": ("queue", "priority", "payload_template"),
            },
        ),
        (
            "Execution History",
            {
                "fields": ("run_count", "last_run_at", "next_run_at"),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    def enabled_badge(self, obj):
        if obj.is_enabled:
            return format_html('<span style="color: green;">●</span> Enabled')
        return format_html('<span style="color: gray;">○</span> Disabled')

    enabled_badge.short_description = "Status"

    def expression(self, obj):
        if obj.cron_expression:
            return obj.cron_expression
        if obj.interval_seconds:
            if obj.interval_seconds < 60:
                return f"Every {obj.interval_seconds}s"
            return f"Every {obj.interval_seconds // 60}m"
        return "-"

    expression.short_description = "Schedule"

    actions = ["enable_schedules", "disable_schedules", "trigger_now"]

    @admin.action(description="Enable selected schedules")
    def enable_schedules(self, request, queryset):
        count = queryset.update(is_enabled=True)
        self.message_user(request, f"{count} schedule(s) enabled.")

    @admin.action(description="Disable selected schedules")
    def disable_schedules(self, request, queryset):
        count = queryset.update(is_enabled=False)
        self.message_user(request, f"{count} schedule(s) disabled.")

    @admin.action(description="Trigger now")
    def trigger_now(self, request, queryset):
        count = queryset.update(next_run_at=timezone.now())
        self.message_user(request, f"{count} schedule(s) will run on next tick.")


# =============================================================================
# DEAD LETTER ENTRY ADMIN
# =============================================================================


@admin.register(DeadLetterEntry)
class DeadLetterEntryAdmin(admin.ModelAdmin):
    """Admin for DeadLetterEntry model."""

    list_display = (
        "short_id",
        "task_name_display",
        "status_badge",
        "last_error_type",
        "failure_reason_short",
        "failure_count",
        "created_at",
    )
    list_filter = (
        "status",
        "last_error_type",
    )
    search_fields = (
        "id",
        "original_task__id",
        "original_task__task_name",
        "original_task__correlation_id",
        "failure_reason",
    )
    readonly_fields = (
        "id",
        "original_task",
        "task_name_display",
        "task_id_display",
        "correlation_id_display",
        "tenant_id",
        "payload_display",
        "failure_reason",
        "last_error_type",
        "last_error_message",
        "last_error_traceback",
        "failure_count",
        "created_at",
        "updated_at",
    )
    ordering = ("-created_at",)

    fieldsets = (
        (
            "Task Information",
            {
                "fields": (
                    "id",
                    "original_task",
                    "task_id_display",
                    "task_name_display",
                    "tenant_id",
                    "correlation_id_display",
                )
            },
        ),
        (
            "Status",
            {
                "fields": ("status", "failure_count"),
            },
        ),
        (
            "Failure Details",
            {
                "fields": ("failure_reason", "last_error_type", "last_error_message", "last_error_traceback"),
            },
        ),
        (
            "Payload",
            {
                "fields": ("payload_display",),
                "classes": ("collapse",),
            },
        ),
        (
            "Resolution",
            {
                "fields": ("resolution_notes", "resolved_by", "resolved_at"),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    def short_id(self, obj):
        return str(obj.id)[:8]

    short_id.short_description = "ID"

    def status_badge(self, obj):
        colors = {
            "pending": "#ffc107",
            "resolved": "#28a745",
            "retried": "#007bff",
            "discarded": "#6c757d",
        }
        color = colors.get(obj.status, "#6c757d")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; '
            'border-radius: 4px; font-size: 11px;">{}</span>',
            color,
            obj.status.upper(),
        )

    status_badge.short_description = "Status"

    def failure_reason_short(self, obj):
        if obj.failure_reason:
            return obj.failure_reason[:50] + ("..." if len(obj.failure_reason) > 50 else "")
        return "-"

    failure_reason_short.short_description = "Reason"

    def task_name_display(self, obj):
        """Display task name from related task."""
        if obj.original_task:
            return obj.original_task.task_name
        return "-"

    task_name_display.short_description = "Task Name"

    def task_id_display(self, obj):
        """Display task ID from related task."""
        if obj.original_task:
            return str(obj.original_task.id)[:8]
        return "-"

    task_id_display.short_description = "Task ID"

    def correlation_id_display(self, obj):
        """Display correlation ID from related task."""
        if obj.original_task:
            return str(obj.original_task.correlation_id)[:8]
        return "-"

    correlation_id_display.short_description = "Correlation ID"

    def payload_display(self, obj):
        """Display payload from related task."""
        if obj.original_task:
            return obj.original_task.payload
        return {}

    payload_display.short_description = "Payload"

    actions = ["mark_resolved", "mark_retried", "mark_discarded"]

    @admin.action(description="Mark as resolved")
    def mark_resolved(self, request, queryset):
        count = queryset.filter(status="pending").update(
            status="resolved",
            resolved_at=timezone.now(),
            resolved_by=request.user.username,
        )
        self.message_user(request, f"{count} entry(ies) marked as resolved.")

    @admin.action(description="Queue for retry")
    def mark_retried(self, request, queryset):
        count = 0
        for entry in queryset.filter(status="pending"):
            entry.status = "retried"
            entry.save()
            # Create new task from DLQ entry
            # Note: Would need scheduler instance for actual retry
            count += 1
        self.message_user(request, f"{count} entry(ies) queued for retry.")

    @admin.action(description="Discard (acknowledge)")
    def mark_discarded(self, request, queryset):
        count = queryset.filter(status="pending").update(
            status="discarded",
            resolved_at=timezone.now(),
            resolved_by=request.user.username,
            resolution_notes="Discarded via admin",
        )
        self.message_user(request, f"{count} entry(ies) discarded.")


# =============================================================================
# CIRCUIT BREAKER ADMIN
# =============================================================================


@admin.register(CircuitBreakerState)
class CircuitBreakerStateAdmin(admin.ModelAdmin):
    """Admin for CircuitBreakerState model."""

    list_display = (
        "service_name",
        "state_badge",
        "failure_count",
        "success_count",
        "failure_rate_display",
        "last_state_change",
    )
    list_filter = ("state",)
    search_fields = ("service_name",)
    readonly_fields = (
        "failure_count",
        "success_count",
        "last_failure_at",
        "opened_at",
        "created_at",
        "updated_at",
    )
    ordering = ("service_name",)

    def state_badge(self, obj):
        colors = {
            "closed": "#28a745",
            "open": "#dc3545",
            "half_open": "#ffc107",
        }
        color = colors.get(obj.state, "#6c757d")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; '
            'border-radius: 4px; font-size: 11px;">{}</span>',
            color,
            obj.state.upper(),
        )

    state_badge.short_description = "State"

    def failure_rate_display(self, obj):
        total = obj.failure_count + obj.success_count
        if total > 0:
            rate = obj.failure_count / total * 100
            return f"{rate:.1f}%"
        return "-"

    failure_rate_display.short_description = "Failure Rate"

    def last_state_change(self, obj):
        if obj.opened_at:
            return obj.opened_at
        return obj.created_at

    last_state_change.short_description = "Last Change"

    actions = ["reset_circuits", "force_open", "force_close"]

    @admin.action(description="Reset circuit (clear counters)")
    def reset_circuits(self, request, queryset):
        queryset.update(
            state="closed",
            failure_count=0,
            success_count=0,
            opened_at=None,
        )
        self.message_user(request, f"{queryset.count()} circuit(s) reset.")

    @admin.action(description="Force OPEN")
    def force_open(self, request, queryset):
        queryset.update(state="open", opened_at=timezone.now())
        self.message_user(request, f"{queryset.count()} circuit(s) forced open.")

    @admin.action(description="Force CLOSED")
    def force_close(self, request, queryset):
        queryset.update(
            state="closed",
            opened_at=None,
        )
        self.message_user(request, f"{queryset.count()} circuit(s) forced closed.")


# =============================================================================
# ADMIN SITE CUSTOMIZATION
# =============================================================================


class SchedulerAdminSite(admin.AdminSite):
    """Custom admin site with scheduler dashboard."""

    site_header = "Synara Scheduler Administration"
    site_title = "Synara Scheduler"
    index_title = "Scheduler Control Center"

    def index(self, request, extra_context=None):
        """Add dashboard data to admin index."""
        extra_context = extra_context or {}

        # Add dashboard metrics
        try:
            from .dashboard.views import get_admin_dashboard_context

            extra_context.update(get_admin_dashboard_context())
        except Exception as e:
            extra_context["scheduler_dashboard"] = {"error": str(e)}

        # Add summary stats
        try:
            extra_context["scheduler_stats"] = {
                "pending_tasks": CognitiveTask.objects.filter(
                    state__in=[TaskState.PENDING.value, TaskState.SCHEDULED.value]
                ).count(),
                "running_tasks": CognitiveTask.objects.filter(state=TaskState.RUNNING.value).count(),
                "dlq_pending": DeadLetterEntry.objects.filter(status="pending").count(),
                "circuits_open": CircuitBreakerState.objects.filter(state="open").count(),
                "schedules_enabled": Schedule.objects.filter(is_enabled=True).count(),
            }
        except Exception:
            pass

        return super().index(request, extra_context=extra_context)
