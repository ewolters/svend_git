"""
Synara Cognitive Scheduler Models (SCH-001/002)
================================================

Django models for the cognitive scheduler implementing task management,
execution tracking, scheduling patterns, and dead letter queue handling.

Standard:     SCH-001 §5-8, SCH-002 §5-8
Compliance:   ISO 9001:2015 §9.5, SOC 2 CC7.2, NIST SP 800-53 SC-5
Location:     syn/sched/models.py
Version:      1.0.0

Models:
-------
- CognitiveTask: Unit of work with cognitive metadata (SCH-001 §5)
- TaskExecution: Execution attempt history (SCH-001 §6)
- Schedule: Recurring schedule definitions (SCH-001 §6)
- DeadLetterEntry: Failed tasks for manual review (SCH-002 §13)
- CircuitBreakerState: Per-service circuit breaker state (SCH-002 §4)
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from django.db import models, transaction

from django.utils import timezone

from syn.core.base_models import SynaraEntity
from syn.sched.types import (
    CASCADE_BUDGET,
    CircuitBreakerConfig,
    CircuitState,
    CronSchedule,
    IntervalSchedule,
    QueueType,
    RetryConfig,
    RetryStrategy,
    ScheduleType,
    TaskContext,
    TaskPriority,
    TaskState,
    TenantQuota,
    get_cascade_limit,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COGNITIVE TASK MODEL (SCH-001 §core_concepts.task)
# =============================================================================


class CognitiveTask(SynaraEntity):
    """
    Unit of work in the cognitive scheduler per SCH-001 §5.

    A CognitiveTask represents a callable unit with cognitive metadata
    enabling intelligent scheduling decisions based on priority, urgency,
    confidence, and governance risk scores.

    Standard: SCH-001 §core_concepts.task
    Compliance: ISO 9001:2015 §9.5
    """

    # =========================================================================
    # Identity
    # =========================================================================

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique task identifier",
    )
    tenant_id = models.UUIDField(
        db_index=True,
        help_text="Tenant isolation identifier",
    )
    task_name = models.CharField(
        max_length=255,
        db_index=True,
        help_text="Task type/handler name (e.g., 'reflex.escalation.handler')",
    )

    # =========================================================================
    # Correlation and Lineage (SCH-001 §task_context)
    # =========================================================================

    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        db_index=True,
        help_text="Correlation ID for distributed tracing",
    )
    root_correlation_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Root correlation for cascade chains",
    )
    parent_task_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Parent task ID for cascade lineage",
    )
    cascade_depth = models.PositiveSmallIntegerField(
        default=0,
        help_text="Depth in cascade chain (0 = root task)",
    )

    # =========================================================================
    # Task Configuration
    # =========================================================================

    payload = models.JSONField(
        default=dict,
        help_text="Task arguments/payload as JSON",
    )
    queue = models.CharField(
        max_length=50,
        default=QueueType.CORE.value,
        db_index=True,
        help_text="Target queue for routing",
    )
    priority = models.PositiveSmallIntegerField(
        default=TaskPriority.NORMAL.value,
        db_index=True,
        help_text="Task priority (0=critical, 4=batch)",
    )

    # =========================================================================
    # Cognitive Attributes (SCH-001 §task_context)
    # =========================================================================

    reflex_source = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Originating reflex ID if task was triggered by reflex",
    )
    confidence_score = models.FloatField(
        default=1.0,
        help_text="Confidence in task necessity (0.0-1.0)",
    )
    urgency = models.FloatField(
        default=0.5,
        help_text="Task urgency level (0.0-1.0)",
    )
    governance_risk = models.FloatField(
        default=0.0,
        help_text="Governance risk score (0.0-1.0, higher=riskier)",
    )
    resource_weight = models.FloatField(
        default=1.0,
        help_text="Relative resource consumption weight",
    )
    priority_score = models.FloatField(
        default=0.5,
        db_index=True,
        help_text="Computed priority score for scheduling",
    )

    # =========================================================================
    # State and Execution (SCH-001 §task_states)
    # =========================================================================

    state = models.CharField(
        max_length=20,
        default=TaskState.PENDING.value,
        db_index=True,
        choices=[(s.value, s.value) for s in TaskState],
        help_text="Current task state",
    )
    attempts = models.PositiveSmallIntegerField(
        default=0,
        help_text="Number of execution attempts",
    )
    max_attempts = models.PositiveSmallIntegerField(
        default=3,
        help_text="Maximum execution attempts before DLQ",
    )

    # =========================================================================
    # Retry Configuration (SCH-002 §retry_strategies)
    # =========================================================================

    retry_strategy = models.CharField(
        max_length=30,
        default=RetryStrategy.EXPONENTIAL.value,
        choices=[(s.value, s.value) for s in RetryStrategy],
        help_text="Retry backoff strategy",
    )
    retry_base_delay = models.PositiveIntegerField(
        default=1,
        help_text="Base delay for retry in seconds",
    )
    retry_max_delay = models.PositiveIntegerField(
        default=3600,
        help_text="Maximum retry delay in seconds",
    )
    next_retry_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Scheduled time for next retry attempt",
    )

    # =========================================================================
    # Timing
    # =========================================================================

    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Task creation timestamp",
    )
    scheduled_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="When task was scheduled for execution",
    )
    started_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Execution start time",
    )
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Execution completion time",
    )
    deadline = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Hard deadline for task completion",
    )
    timeout_seconds = models.PositiveIntegerField(
        default=60,
        help_text="Execution timeout in seconds",
    )

    # =========================================================================
    # Results
    # =========================================================================

    result = models.JSONField(
        null=True,
        blank=True,
        help_text="Task execution result",
    )
    error_message = models.TextField(
        null=True,
        blank=True,
        help_text="Error message if failed",
    )
    error_type = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Error type classification",
    )

    # =========================================================================
    # Scheduling Reference
    # =========================================================================

    schedule = models.ForeignKey(
        "Schedule",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="tasks",
        help_text="Associated schedule for recurring tasks",
    )

    # =========================================================================
    # FLD-001 §3.1: Standard timestamp fields
    # =========================================================================

    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last modification timestamp (FLD-001)",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "syn_sched_cognitive_task"
        verbose_name = "Cognitive Task"
        verbose_name_plural = "Cognitive Tasks"
        ordering = ["-priority_score", "created_at"]
        indexes = [
            models.Index(
                fields=["tenant_id", "state", "priority_score"],
                name="syn_idx_task_tnt_state_prio",
            ),
            models.Index(
                fields=["tenant_id", "queue", "state"],
                name="syn_idx_task_tnt_q_state",
            ),
            models.Index(
                fields=["state", "next_retry_at"],
                name="syn_idx_task_retry_pend",
            ),
            models.Index(
                fields=["correlation_id"],
                name="syn_idx_task_correlation",
            ),
            models.Index(
                fields=["root_correlation_id"],
                name="syn_idx_task_root_corr",
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.sched.cognitive_task"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        return f"{self.task_name}:{self.id} [{self.state}]"

    def save(self, *args, **kwargs):
        """Compute priority score before saving."""
        self._compute_priority_score()
        if self.root_correlation_id is None:
            self.root_correlation_id = self.correlation_id
        super().save(*args, **kwargs)

    def _compute_priority_score(self) -> None:
        """
        Compute priority score from cognitive attributes per SCH-001 §task_context.

        Formula: (confidence * 0.3) + (urgency * 0.4) + ((1 - governance_risk) * 0.3)
        """
        self.priority_score = (
            self.confidence_score * 0.3
            + self.urgency * 0.4
            + (1 - self.governance_risk) * 0.3
        )

    # =========================================================================
    # State Transitions (SCH-001 §task_states)
    # =========================================================================

    def can_transition_to(self, new_state: TaskState) -> bool:
        """Check if transition to new state is valid."""
        current = TaskState(self.state)
        valid = TaskState.valid_transitions()
        return new_state in valid.get(current, [])

    def transition_to(self, new_state: TaskState, **kwargs) -> bool:
        """
        Transition task to new state with validation.

        Returns True if transition succeeded.
        """
        if not self.can_transition_to(new_state):
            logger.warning(
                f"Invalid state transition: {self.state} -> {new_state.value} "
                f"for task {self.id}"
            )
            return False

        old_state = self.state
        self.state = new_state.value

        # Update timestamps based on transition
        now = timezone.now()
        if new_state == TaskState.SCHEDULED:
            self.scheduled_at = now
        elif new_state == TaskState.RUNNING:
            self.started_at = now
        elif new_state.is_terminal:
            self.completed_at = now

        # Handle additional kwargs
        if "error_message" in kwargs:
            self.error_message = kwargs["error_message"]
        if "error_type" in kwargs:
            self.error_type = kwargs["error_type"]
        if "result" in kwargs:
            self.result = kwargs["result"]

        self.save()

        # Log transition
        logger.info(
            f"Task {self.id} transitioned: {old_state} -> {new_state.value}",
            extra={
                "task_id": str(self.id),
                "correlation_id": str(self.correlation_id),
                "old_state": old_state,
                "new_state": new_state.value,
            },
        )

        return True

    # =========================================================================
    # Retry Logic (SCH-002 §retry_strategies)
    # =========================================================================

    def schedule_retry(self) -> Optional[datetime]:
        """
        Schedule next retry attempt per SCH-002 §12.

        Returns next retry time or None if max attempts exceeded.
        """
        self.attempts += 1

        if self.attempts >= self.max_attempts:
            self.transition_to(TaskState.DEAD_LETTERED)
            return None

        config = RetryConfig(
            strategy=RetryStrategy(self.retry_strategy),
            max_attempts=self.max_attempts,
            base_delay_seconds=self.retry_base_delay,
            max_delay_seconds=self.retry_max_delay,
        )

        delay = config.get_delay(self.attempts)
        self.next_retry_at = timezone.now() + delay
        self.transition_to(TaskState.RETRYING)

        return self.next_retry_at

    @property
    def has_attempts_remaining(self) -> bool:
        """Check if retry attempts remain."""
        return self.attempts < self.max_attempts

    @property
    def is_past_deadline(self) -> bool:
        """Check if task is past its deadline."""
        if self.deadline is None:
            return False
        return timezone.now() > self.deadline

    # =========================================================================
    # Context Conversion
    # =========================================================================

    def to_context(self) -> TaskContext:
        """Convert to TaskContext for scheduler operations."""
        return TaskContext(
            task_id=str(self.id),
            correlation_id=str(self.correlation_id),
            tenant_id=str(self.tenant_id),
            root_correlation_id=str(self.root_correlation_id) if self.root_correlation_id else None,
            parent_task_id=str(self.parent_task_id) if self.parent_task_id else None,
            cascade_depth=self.cascade_depth,
            reflex_source=self.reflex_source,
            confidence_score=self.confidence_score,
            urgency=self.urgency,
            governance_risk=self.governance_risk,
            resource_weight=self.resource_weight,
            created_at=self.created_at,
            scheduled_at=self.scheduled_at,
            deadline=self.deadline,
            attempts=self.attempts,
            max_attempts=self.max_attempts,
        )

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def create_task(
        cls,
        tenant_id: uuid.UUID,
        task_name: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        queue: QueueType = QueueType.CORE,
        correlation_id: Optional[uuid.UUID] = None,
        parent_task: Optional["CognitiveTask"] = None,
        urgency: float = 0.5,
        confidence_score: float = 1.0,
        governance_risk: float = 0.0,
        deadline: Optional[datetime] = None,
        timeout_seconds: int = 60,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        max_attempts: int = 3,
    ) -> "CognitiveTask":
        """
        Factory method to create a new cognitive task.

        Standard: SCH-001 §core_concepts.task
        """
        cascade_depth = 0
        root_correlation_id = correlation_id
        parent_task_id = None

        if parent_task:
            cascade_depth = parent_task.cascade_depth + 1
            root_correlation_id = parent_task.root_correlation_id or parent_task.correlation_id
            parent_task_id = parent_task.id

            # Check cascade budget
            limit = get_cascade_limit(cascade_depth)
            if limit == 0:
                raise ValueError(
                    f"Cascade depth {cascade_depth} exceeds maximum "
                    f"allowed depth {CASCADE_BUDGET['max_depth']}"
                )

        task = cls(
            tenant_id=tenant_id,
            task_name=task_name,
            payload=payload,
            priority=priority.value,
            queue=queue.value,
            correlation_id=correlation_id or uuid.uuid4(),
            root_correlation_id=root_correlation_id,
            parent_task_id=parent_task_id,
            cascade_depth=cascade_depth,
            urgency=urgency,
            confidence_score=confidence_score,
            governance_risk=governance_risk,
            deadline=deadline,
            timeout_seconds=timeout_seconds,
            retry_strategy=retry_strategy.value,
            max_attempts=max_attempts,
        )
        task.save()
        return task


# =============================================================================
# TASK EXECUTION MODEL (SCH-001 §task_execution)
# =============================================================================


class TaskExecution(SynaraEntity):
    """
    Execution attempt record for a cognitive task per SCH-001 §6.

    Each execution attempt is recorded for audit trail and debugging.

    Standard: SCH-001 §task_execution
    Compliance: SOC 2 CC7.2
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    task = models.ForeignKey(
        CognitiveTask,
        on_delete=models.CASCADE,
        related_name="executions",
        help_text="Parent task",
    )
    attempt_number = models.PositiveSmallIntegerField(
        help_text="Attempt number (1-indexed)",
    )
    worker_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Worker instance that executed this attempt",
    )

    # Timing
    started_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Execution start time",
    )
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Execution completion time",
    )
    duration_ms = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Execution duration in milliseconds",
    )

    # Result
    is_success = models.BooleanField(
        default=False,
        db_column="success",
        help_text="Whether execution succeeded",
    )
    result = models.JSONField(
        null=True,
        blank=True,
        help_text="Execution result",
    )
    error_message = models.TextField(
        null=True,
        blank=True,
        help_text="Error message if failed",
    )
    error_type = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Error classification (transient/permanent/rate_limited)",
    )
    error_traceback = models.TextField(
        null=True,
        blank=True,
        help_text="Full error traceback",
    )

    # Resource metrics
    memory_peak_mb = models.FloatField(
        null=True,
        blank=True,
        help_text="Peak memory usage in MB",
    )
    cpu_time_ms = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="CPU time in milliseconds",
    )

    # FLD-001 §3.1: Standard timestamp field
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last modification timestamp (FLD-001)",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "syn_sched_task_execution"
        verbose_name = "Task Execution"
        verbose_name_plural = "Task Executions"
        ordering = ["-started_at"]
        indexes = [
            models.Index(
                fields=["task", "attempt_number"],
                name="syn_idx_exec_task_attempt",
            ),
        ]
        unique_together = [["task", "attempt_number"]]

    class SynaraMeta:
        event_domain = "syn.sched.task_execution"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        status = "✓" if self.is_success else "✗"
        return f"{self.task.task_name} attempt #{self.attempt_number} [{status}]"

    def complete(
        self,
        success: bool,
        result: Any = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> None:
        """Mark execution as complete."""
        self.completed_at = timezone.now()
        self.duration_ms = int(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )
        self.is_success = success
        self.result = result
        self.error_message = error_message
        self.error_type = error_type
        self.error_traceback = error_traceback
        self.save()


# =============================================================================
# SCHEDULE MODEL (SCH-001 §temporal_patterns)
# =============================================================================


class Schedule(SynaraEntity):
    """
    Recurring schedule definition per SCH-001 §6.

    Supports cron expressions, fixed intervals, and one-time delayed execution.

    Standard: SCH-001 §temporal_patterns
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    tenant_id = models.UUIDField(
        db_index=True,
        help_text="Tenant isolation identifier",
    )
    schedule_id = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Human-readable schedule identifier",
    )
    name = models.CharField(
        max_length=255,
        help_text="Schedule name/description",
    )

    # Task template
    task_name = models.CharField(
        max_length=255,
        help_text="Task handler name to invoke",
    )
    payload_template = models.JSONField(
        default=dict,
        help_text="Template for task payload",
    )
    queue = models.CharField(
        max_length=50,
        default=QueueType.CORE.value,
        help_text="Target queue for scheduled tasks",
    )
    priority = models.PositiveSmallIntegerField(
        default=TaskPriority.NORMAL.value,
        help_text="Priority for scheduled tasks",
    )

    # Schedule type and configuration
    schedule_type = models.CharField(
        max_length=20,
        choices=[(s.value, s.value) for s in ScheduleType],
        help_text="Schedule type (cron/interval/once)",
    )

    # Cron fields (for CRON type)
    cron_minute = models.CharField(max_length=50, default="*")
    cron_hour = models.CharField(max_length=50, default="*")
    cron_day_of_month = models.CharField(max_length=50, default="*")
    cron_month = models.CharField(max_length=50, default="*")
    cron_day_of_week = models.CharField(max_length=50, default="*")

    # Interval fields (for INTERVAL type)
    interval_seconds = models.PositiveIntegerField(
        default=0,
        help_text="Interval in seconds",
    )

    # One-time execution (for ONCE type)
    run_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="One-time execution timestamp",
    )

    # State
    is_enabled = models.BooleanField(
        default=True,
        db_index=True,
        db_column="enabled",
        help_text="Whether schedule is active",
    )
    last_run_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last successful execution time",
    )
    next_run_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Next scheduled execution time",
    )
    run_count = models.PositiveIntegerField(
        default=0,
        help_text="Total execution count",
    )

    # Limits
    max_runs = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum number of runs (null = unlimited)",
    )
    expires_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Schedule expiration time",
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Creator identifier",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "syn_sched_schedule"
        verbose_name = "Schedule"
        verbose_name_plural = "Schedules"
        ordering = ["next_run_at"]
        indexes = [
            models.Index(
                fields=["tenant_id", "is_enabled", "next_run_at"],
                name="syn_idx_sched_tnt_enabled",
            ),
        ]
        unique_together = [["tenant_id", "schedule_id"]]

    class SynaraMeta:
        event_domain = "syn.sched.schedule"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        status = "enabled" if self.is_enabled else "disabled"
        return f"{self.name} ({self.schedule_type}) [{status}]"

    @property
    def cron_expression(self) -> str:
        """Get full cron expression."""
        return (
            f"{self.cron_minute} {self.cron_hour} {self.cron_day_of_month} "
            f"{self.cron_month} {self.cron_day_of_week}"
        )

    def calculate_next_run(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate next run time based on schedule type."""
        from_time = from_time or timezone.now()

        if self.schedule_type == ScheduleType.ONCE.value:
            if self.run_at and self.run_at > from_time:
                return self.run_at
            return None

        if self.schedule_type == ScheduleType.INTERVAL.value:
            if self.last_run_at:
                return self.last_run_at + timedelta(seconds=self.interval_seconds)
            return from_time + timedelta(seconds=self.interval_seconds)

        if self.schedule_type == ScheduleType.CRON.value:
            try:
                from croniter import croniter
                cron_expr = self.cron_expression
                cron = croniter(cron_expr, from_time)
                return cron.get_next(datetime)
            except Exception:
                # Fallback: next hour
                next_hour = from_time.replace(minute=0, second=0, microsecond=0)
                if next_hour <= from_time:
                    next_hour += timedelta(hours=1)
                return next_hour

        return None

    def update_next_run(self) -> None:
        """Update next_run_at based on current time."""
        # Check limits
        if self.max_runs and self.run_count >= self.max_runs:
            self.is_enabled = False
            self.next_run_at = None
        elif self.expires_at and timezone.now() >= self.expires_at:
            self.is_enabled = False
            self.next_run_at = None
        else:
            self.next_run_at = self.calculate_next_run()
        self.save()

    def record_run(self) -> None:
        """Record a successful run."""
        self.last_run_at = timezone.now()
        self.run_count += 1
        self.update_next_run()

    @classmethod
    def create_cron_schedule(
        cls,
        tenant_id: uuid.UUID,
        schedule_id: str,
        name: str,
        task_name: str,
        cron: CronSchedule,
        **kwargs,
    ) -> "Schedule":
        """Create a cron-based schedule."""
        schedule = cls(
            tenant_id=tenant_id,
            schedule_id=schedule_id,
            name=name,
            task_name=task_name,
            schedule_type=ScheduleType.CRON.value,
            cron_minute=cron.minute,
            cron_hour=cron.hour,
            cron_day_of_month=cron.day_of_month,
            cron_month=cron.month,
            cron_day_of_week=cron.day_of_week,
            **kwargs,
        )
        schedule.next_run_at = schedule.calculate_next_run()
        schedule.save()
        return schedule

    @classmethod
    def create_interval_schedule(
        cls,
        tenant_id: uuid.UUID,
        schedule_id: str,
        name: str,
        task_name: str,
        interval: IntervalSchedule,
        **kwargs,
    ) -> "Schedule":
        """Create an interval-based schedule."""
        schedule = cls(
            tenant_id=tenant_id,
            schedule_id=schedule_id,
            name=name,
            task_name=task_name,
            schedule_type=ScheduleType.INTERVAL.value,
            interval_seconds=interval.total_seconds,
            **kwargs,
        )
        schedule.next_run_at = schedule.calculate_next_run()
        schedule.save()
        return schedule


# =============================================================================
# DEAD LETTER QUEUE MODEL (SCH-002 §dlq_handling)
# =============================================================================


class DeadLetterEntry(SynaraEntity):
    """
    Dead Letter Queue entry for tasks that exhausted retries per SCH-002 §13.

    Failed tasks are moved here for manual review and potential reprocessing.

    Standard: SCH-002 §dlq_handling
    Compliance: SOC 2 CC7.2
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    tenant_id = models.UUIDField(
        db_index=True,
        help_text="Tenant isolation identifier",
    )
    original_task = models.OneToOneField(
        CognitiveTask,
        on_delete=models.CASCADE,
        related_name="dlq_entry",
        help_text="Original failed task",
    )

    # Failure details
    failure_reason = models.TextField(
        help_text="Reason for DLQ placement",
    )
    failure_count = models.PositiveIntegerField(
        help_text="Total failure count before DLQ",
    )
    last_error_message = models.TextField(
        null=True,
        blank=True,
        help_text="Last error message",
    )
    last_error_type = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Last error classification",
    )
    last_error_traceback = models.TextField(
        null=True,
        blank=True,
        help_text="Last error traceback",
    )

    # State
    status = models.CharField(
        max_length=20,
        default="pending",
        db_index=True,
        choices=[
            ("pending", "Pending Review"),
            ("reviewing", "Under Review"),
            ("reprocessing", "Reprocessing"),
            ("resolved", "Resolved"),
            ("discarded", "Discarded"),
        ],
        help_text="DLQ entry status",
    )
    resolution_notes = models.TextField(
        null=True,
        blank=True,
        help_text="Notes on resolution",
    )
    resolved_by = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Who resolved this entry",
    )
    resolved_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Resolution timestamp",
    )

    # Reprocessing
    reprocess_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of reprocess attempts",
    )
    reprocessed_task_id = models.UUIDField(
        null=True,
        blank=True,
        help_text="ID of reprocessed task if any",
    )

    # FLD-001 §3.1: Standard timestamp fields
    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Entry creation timestamp (FLD-001)",
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last modification timestamp (FLD-001)",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "syn_sched_dead_letter"
        verbose_name = "Dead Letter Entry"
        verbose_name_plural = "Dead Letter Entries"
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["tenant_id", "status"],
                name="syn_idx_dlq_tnt_status",
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.sched.dead_letter_entry"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        return f"DLQ: {self.original_task.task_name} [{self.status}]"

    @classmethod
    def create_from_task(
        cls,
        task: CognitiveTask,
        failure_reason: str,
    ) -> "DeadLetterEntry":
        """Create DLQ entry from a failed task."""
        # Get last execution for error details
        last_execution = task.executions.order_by("-attempt_number").first()

        entry = cls(
            tenant_id=task.tenant_id,
            original_task=task,
            failure_reason=failure_reason,
            failure_count=task.attempts,
            last_error_message=last_execution.error_message if last_execution else task.error_message,
            last_error_type=last_execution.error_type if last_execution else task.error_type,
            last_error_traceback=last_execution.error_traceback if last_execution else None,
        )
        entry.save()

        logger.warning(
            f"Task {task.id} moved to DLQ: {failure_reason}",
            extra={
                "task_id": str(task.id),
                "correlation_id": str(task.correlation_id),
                "failure_reason": failure_reason,
                "failure_count": task.attempts,
            },
        )

        return entry

    def reprocess(self, modified_payload: Optional[Dict[str, Any]] = None) -> CognitiveTask:
        """
        Create a new task from this DLQ entry for reprocessing.

        Returns the newly created task.
        """
        original = self.original_task
        payload = modified_payload or original.payload

        new_task = CognitiveTask.create_task(
            tenant_id=original.tenant_id,
            task_name=original.task_name,
            payload=payload,
            priority=TaskPriority(original.priority),
            queue=QueueType(original.queue),
            correlation_id=original.correlation_id,
            urgency=original.urgency,
            confidence_score=original.confidence_score,
            governance_risk=original.governance_risk,
            timeout_seconds=original.timeout_seconds,
            retry_strategy=RetryStrategy(original.retry_strategy),
            max_attempts=original.max_attempts,
        )

        self.status = "reprocessing"
        self.reprocess_count += 1
        self.reprocessed_task_id = new_task.id
        self.save()

        logger.info(
            f"DLQ entry {self.id} reprocessing as task {new_task.id}",
            extra={
                "dlq_id": str(self.id),
                "new_task_id": str(new_task.id),
                "reprocess_count": self.reprocess_count,
            },
        )

        return new_task

    def resolve(self, status: str, notes: str, resolved_by: str) -> None:
        """Mark DLQ entry as resolved."""
        self.status = status
        self.resolution_notes = notes
        self.resolved_by = resolved_by
        self.resolved_at = timezone.now()
        self.save()


# =============================================================================
# CIRCUIT BREAKER STATE MODEL (SCH-002 §circuit_breaker)
# =============================================================================


class CircuitBreakerState(SynaraEntity):
    """
    Per-service circuit breaker state per SCH-002 §4.

    Tracks circuit breaker state for external services to prevent
    cascade failures.

    Standard: SCH-002 §circuit_breaker
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    service_name = models.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        help_text="Service identifier (e.g., 'external_api', 'database')",
    )

    # State
    state = models.CharField(
        max_length=20,
        default=CircuitState.CLOSED.value,
        choices=[(s.value, s.value) for s in CircuitState],
        help_text="Current circuit state",
    )

    # Counters
    failure_count = models.PositiveIntegerField(
        default=0,
        help_text="Consecutive failure count",
    )
    success_count = models.PositiveIntegerField(
        default=0,
        help_text="Success count in half-open state",
    )
    half_open_calls = models.PositiveIntegerField(
        default=0,
        help_text="Calls made in half-open state",
    )

    # Configuration
    failure_threshold = models.PositiveIntegerField(
        default=5,
        help_text="Failures before opening circuit",
    )
    recovery_timeout_seconds = models.PositiveIntegerField(
        default=30,
        help_text="Seconds before testing recovery",
    )
    success_threshold = models.PositiveIntegerField(
        default=3,
        help_text="Successes needed to close circuit",
    )
    half_open_max_calls = models.PositiveIntegerField(
        default=3,
        help_text="Max calls in half-open state",
    )

    # Timing
    last_failure_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last failure timestamp",
    )
    opened_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When circuit was opened",
    )

    # FLD-001 §3.1: Standard timestamp fields
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Circuit breaker creation timestamp (FLD-001)",
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last modification timestamp (FLD-001)",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "syn_sched_circuit_breaker"
        verbose_name = "Circuit Breaker State"
        verbose_name_plural = "Circuit Breaker States"

    class SynaraMeta:
        event_domain = "syn.sched.circuit_breaker_state"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        return f"{self.service_name}: {self.state}"

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN.value:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED.value:
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = 0
                self.save()

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_at = timezone.now()

        if self.state == CircuitState.HALF_OPEN.value:
            # Any failure in half-open reopens circuit
            self._open_circuit()
        elif self.state == CircuitState.CLOSED.value:
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()
            else:
                self.save()

    def can_execute(self) -> Tuple[bool, str]:
        """
        Check if execution is allowed through this circuit.

        Returns (allowed, reason).
        """
        if self.state == CircuitState.CLOSED.value:
            return True, "Circuit closed"

        if self.state == CircuitState.OPEN.value:
            # Check if recovery timeout has passed
            if self.opened_at:
                recovery_time = self.opened_at + timedelta(
                    seconds=self.recovery_timeout_seconds
                )
                if timezone.now() >= recovery_time:
                    self._half_open_circuit()
                    return True, "Circuit half-open (testing)"
            return False, "Circuit open"

        if self.state == CircuitState.HALF_OPEN.value:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                self.save()
                return True, "Circuit half-open (testing)"
            return False, "Circuit half-open (max test calls reached)"

        return False, "Unknown circuit state"

    def _open_circuit(self) -> None:
        """Open the circuit."""
        self.state = CircuitState.OPEN.value
        self.opened_at = timezone.now()
        self.success_count = 0
        self.half_open_calls = 0
        self.save()

        logger.warning(
            f"Circuit breaker OPENED for {self.service_name}",
            extra={
                "service": self.service_name,
                "failure_count": self.failure_count,
            },
        )

    def _half_open_circuit(self) -> None:
        """Transition to half-open state."""
        self.state = CircuitState.HALF_OPEN.value
        self.success_count = 0
        self.half_open_calls = 0
        self.save()

        logger.info(
            f"Circuit breaker HALF-OPEN for {self.service_name}",
            extra={"service": self.service_name},
        )

    def _close_circuit(self) -> None:
        """Close the circuit."""
        self.state = CircuitState.CLOSED.value
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.opened_at = None
        self.save()

        logger.info(
            f"Circuit breaker CLOSED for {self.service_name}",
            extra={"service": self.service_name},
        )

    @classmethod
    def get_or_create_for_service(
        cls,
        service_name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> "CircuitBreakerState":
        """Get or create circuit breaker state for a service."""
        config = config or CircuitBreakerConfig()

        state, created = cls.objects.get_or_create(
            service_name=service_name,
            defaults={
                "failure_threshold": config.failure_threshold,
                "recovery_timeout_seconds": config.recovery_timeout_seconds,
                "success_threshold": config.success_threshold,
                "half_open_max_calls": config.half_open_max_calls,
            },
        )

        return state
