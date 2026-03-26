"""
Synara Cognitive Scheduler Primitives (SCH-001/002)
====================================================

Type definitions and enumerations for the cognitive scheduler.

Standard:     SCH-001 §3-4, SCH-002 §3-4
Compliance:   ISO 9001:2015 §9.5, SOC 2 CC7.2, NIST SP 800-53 SC-5
Version:      1.0.0
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# =============================================================================
# TASK STATE (SCH-001 §core_concepts)
# =============================================================================


class TaskState(str, Enum):
    """
    Task execution states per SCH-001.

    Lifecycle: PENDING -> SCHEDULED -> RUNNING -> (SUCCESS|FAILURE|CANCELLED)
    """

    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"
    DEAD_LETTERED = "DEAD_LETTERED"

    @property
    def is_terminal(self) -> bool:
        """Whether this state is terminal (no further transitions)."""
        return self in (
            TaskState.SUCCESS,
            TaskState.FAILURE,
            TaskState.CANCELLED,
            TaskState.DEAD_LETTERED,
        )

    @property
    def is_active(self) -> bool:
        """Whether the task is actively being processed."""
        return self in (TaskState.RUNNING, TaskState.RETRYING)

    @classmethod
    def valid_transitions(cls) -> dict[TaskState, list[TaskState]]:
        """Valid state transitions."""
        return {
            cls.PENDING: [cls.SCHEDULED, cls.CANCELLED],
            cls.SCHEDULED: [cls.RUNNING, cls.CANCELLED],
            cls.RUNNING: [cls.SUCCESS, cls.FAILURE, cls.RETRYING, cls.CANCELLED],
            cls.RETRYING: [cls.RUNNING, cls.FAILURE, cls.CANCELLED, cls.DEAD_LETTERED],
            cls.SUCCESS: [],
            cls.FAILURE: [cls.RETRYING, cls.DEAD_LETTERED],
            cls.CANCELLED: [],
            cls.DEAD_LETTERED: [],
        }


# =============================================================================
# PRIORITY LEVELS (SCH-001 §core_concepts)
# =============================================================================


class TaskPriority(int, Enum):
    """
    Task priority levels per SCH-001.

    Lower numbers = higher priority (like Unix nice levels).
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4

    @property
    def weight(self) -> float:
        """Priority weight for scoring (higher = more important)."""
        weights = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.8,
            TaskPriority.NORMAL: 0.5,
            TaskPriority.LOW: 0.3,
            TaskPriority.BATCH: 0.1,
        }
        return weights[self]


# =============================================================================
# RETRY STRATEGIES (SCH-002 §retry_strategies)
# =============================================================================


class RetryStrategy(str, Enum):
    """
    Retry backoff strategies per SCH-002 §12.

    - EXPONENTIAL: Exponential backoff with jitter
    - LINEAR: Linear backoff with fixed increment
    - FIXED: Fixed delay between retries (no backoff)
    - IMMEDIATE: Immediate retry (no delay)
    - NONE: No retry, send to DLQ on failure
    """

    EXPONENTIAL = "exponential_backoff"
    LINEAR = "linear_backoff"
    FIXED = "fixed_delay"
    IMMEDIATE = "immediate_retry"
    NONE = "none"


@dataclass
class RetryConfig:
    """
    Retry configuration per SCH-002 §retry_strategies.

    Standard: SCH-002 §12
    """

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_attempts: int = 5
    base_delay_seconds: int = 1
    max_delay_seconds: int = 3600
    multiplier: float = 2.0
    jitter: bool = True
    delay_increment_seconds: int = 60  # For LINEAR strategy

    def get_delay(self, attempt: int) -> timedelta:
        """Calculate delay for given attempt number."""
        if self.strategy == RetryStrategy.NONE:
            return timedelta(seconds=0)

        if self.strategy == RetryStrategy.IMMEDIATE:
            return timedelta(seconds=0)

        if self.strategy == RetryStrategy.FIXED:
            # Fixed delay - always use base_delay_seconds
            return timedelta(seconds=self.base_delay_seconds)

        if self.strategy == RetryStrategy.LINEAR:
            delay = self.delay_increment_seconds * attempt
            delay = min(delay, self.max_delay_seconds)
            return timedelta(seconds=delay)

        # EXPONENTIAL
        delay = self.base_delay_seconds * (self.multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay_seconds)

        if self.jitter:
            import random

            jitter_factor = random.uniform(0.5, 1.5)
            delay = delay * jitter_factor

        return timedelta(seconds=int(delay))


# Default retry configurations per error type
RETRY_CONFIGS = {
    "transient": RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL,
        max_attempts=5,
        base_delay_seconds=1,
        max_delay_seconds=300,
    ),
    "permanent": RetryConfig(
        strategy=RetryStrategy.NONE,
        max_attempts=1,
    ),
    "rate_limited": RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL,
        max_attempts=10,
        base_delay_seconds=60,
        max_delay_seconds=3600,
    ),
    "default": RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL,
        max_attempts=3,
        base_delay_seconds=5,
    ),
}


# =============================================================================
# CIRCUIT BREAKER STATES (SCH-002 §circuit_breaker)
# =============================================================================


class CircuitState(str, Enum):
    """
    Circuit breaker states per SCH-002 §4.

    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration per SCH-002 §circuit_breaker.

    Standard: SCH-002 §4
    """

    failure_threshold: int = 5
    recovery_timeout_seconds: int = 30
    success_threshold: int = 3
    half_open_max_calls: int = 3


# Per-service circuit breaker configs
CIRCUIT_CONFIGS = {
    "default": CircuitBreakerConfig(),
    "external_api": CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60,
    ),
    "database": CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout_seconds=15,
    ),
    "redis": CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=10,
    ),
}


# =============================================================================
# SCHEDULE TYPES (SCH-001 §temporal_patterns)
# =============================================================================


class ScheduleType(str, Enum):
    """
    Schedule types per SCH-001 §6.

    - CRON: Standard cron expression
    - INTERVAL: Fixed interval execution
    - ONCE: One-time delayed execution
    """

    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"


@dataclass
class CronSchedule:
    """
    Cron schedule configuration per SCH-001 §temporal_patterns.cron.

    Standard cron format: minute hour day month weekday
    """

    minute: str = "*"
    hour: str = "*"
    day_of_month: str = "*"
    month: str = "*"
    day_of_week: str = "*"

    @property
    def expression(self) -> str:
        """Return cron expression string."""
        return f"{self.minute} {self.hour} {self.day_of_month} {self.month} {self.day_of_week}"

    @classmethod
    def from_expression(cls, expr: str) -> CronSchedule:
        """Parse cron expression string."""
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expr}")
        return cls(
            minute=parts[0],
            hour=parts[1],
            day_of_month=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )

    @classmethod
    def every_hour(cls) -> CronSchedule:
        """Run every hour at minute 0."""
        return cls(minute="0")

    @classmethod
    def every_minute(cls) -> CronSchedule:
        """Run every minute."""
        return cls()

    @classmethod
    def daily_at(cls, hour: int, minute: int = 0) -> CronSchedule:
        """Run daily at specified time."""
        return cls(minute=str(minute), hour=str(hour))

    @classmethod
    def weekdays_at(cls, hour: int, minute: int = 0) -> CronSchedule:
        """Run on weekdays (Mon-Fri) at specified time."""
        return cls(minute=str(minute), hour=str(hour), day_of_week="1-5")


@dataclass
class IntervalSchedule:
    """
    Interval schedule configuration per SCH-001 §temporal_patterns.periodic.

    Fixed interval between executions.
    """

    seconds: int = 0
    minutes: int = 0
    hours: int = 0
    days: int = 0

    @property
    def total_seconds(self) -> int:
        """Total interval in seconds."""
        return self.seconds + self.minutes * 60 + self.hours * 3600 + self.days * 86400

    @property
    def as_timedelta(self) -> timedelta:
        """Return as timedelta."""
        return timedelta(seconds=self.total_seconds)


# =============================================================================
# TASK CONTEXT (SCH-001 §core_concepts.task_context)
# =============================================================================


@dataclass
class TaskContext:
    """
    Cognitive metadata attached to every task per SCH-001 §task_context.

    This context enables cognitive scheduling decisions including
    priority scoring, governance checks, and cascade tracking.

    Standard: SCH-001 §core_concepts.task_context
    """

    # Identity
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str | None = None

    # Lineage
    root_correlation_id: str | None = None
    parent_task_id: str | None = None
    cascade_depth: int = 0

    # Cognitive attributes
    reflex_source: str | None = None
    confidence_score: float = 1.0
    urgency: float = 0.5
    governance_risk: float = 0.0
    resource_weight: float = 1.0

    # Computed
    priority_score: float = field(default=0.5, init=False)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: datetime | None = None
    deadline: datetime | None = None

    # Execution tracking
    attempts: int = 0
    max_attempts: int = 3

    def __post_init__(self):
        """Compute derived values."""
        if self.root_correlation_id is None:
            self.root_correlation_id = self.correlation_id
        self._compute_priority_score()

    def _compute_priority_score(self) -> None:
        """
        Compute priority score from cognitive attributes.

        Score formula: (confidence * 0.3) + (urgency * 0.4) + ((1 - governance_risk) * 0.3)
        Higher score = higher priority.
        """
        self.priority_score = (
            self.confidence_score * 0.3
            + self.urgency * 0.4
            + (1 - self.governance_risk) * 0.3
        )

    def increment_attempt(self) -> None:
        """Increment attempt counter."""
        self.attempts += 1

    @property
    def has_attempts_remaining(self) -> bool:
        """Check if retry attempts remain."""
        return self.attempts < self.max_attempts

    @property
    def is_past_deadline(self) -> bool:
        """Check if task is past its deadline."""
        if self.deadline is None:
            return False
        return datetime.utcnow() > self.deadline

    def as_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
            "tenant_id": self.tenant_id,
            "root_correlation_id": self.root_correlation_id,
            "parent_task_id": self.parent_task_id,
            "cascade_depth": self.cascade_depth,
            "reflex_source": self.reflex_source,
            "confidence_score": self.confidence_score,
            "urgency": self.urgency,
            "governance_risk": self.governance_risk,
            "resource_weight": self.resource_weight,
            "priority_score": self.priority_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "scheduled_at": (
                self.scheduled_at.isoformat() if self.scheduled_at else None
            ),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskContext:
        """Deserialize from dictionary."""
        ctx = cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            correlation_id=data.get("correlation_id", str(uuid.uuid4())),
            tenant_id=data.get("tenant_id"),
            root_correlation_id=data.get("root_correlation_id"),
            parent_task_id=data.get("parent_task_id"),
            cascade_depth=data.get("cascade_depth", 0),
            reflex_source=data.get("reflex_source"),
            confidence_score=data.get("confidence_score", 1.0),
            urgency=data.get("urgency", 0.5),
            governance_risk=data.get("governance_risk", 0.0),
            resource_weight=data.get("resource_weight", 1.0),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
        )

        if data.get("created_at"):
            ctx.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("scheduled_at"):
            ctx.scheduled_at = datetime.fromisoformat(data["scheduled_at"])
        if data.get("deadline"):
            ctx.deadline = datetime.fromisoformat(data["deadline"])

        return ctx


# =============================================================================
# RESOURCE QUOTAS (SCH-002 §resource_quotas)
# =============================================================================


@dataclass
class TenantQuota:
    """
    Tenant resource quota per SCH-002 §7.

    Standard: SCH-002 §resource_quotas
    """

    max_concurrent_tasks: int = 100
    max_queue_depth: int = 10000
    max_cascade_depth: int = 5
    rate_limit_per_minute: int = 1000


TENANT_QUOTAS = {
    "default": TenantQuota(),
    "premium": TenantQuota(
        max_concurrent_tasks=500,
        max_queue_depth=50000,
        max_cascade_depth=10,
        rate_limit_per_minute=5000,
    ),
    "enterprise": TenantQuota(
        max_concurrent_tasks=1000,
        max_queue_depth=100000,
        max_cascade_depth=15,
        rate_limit_per_minute=10000,
    ),
}


# =============================================================================
# CASCADE THROTTLING (SCH-002 §reflex_throttling)
# =============================================================================


CASCADE_BUDGET = {
    "max_depth": 5,
    "max_tasks_per_depth": {
        1: 100,
        2: 50,
        3: 25,
        4: 10,
        5: 5,
    },
}


def get_cascade_limit(depth: int) -> int:
    """Get maximum tasks allowed at given cascade depth."""
    max_depth = CASCADE_BUDGET["max_depth"]
    if depth > max_depth:
        return 0
    return CASCADE_BUDGET["max_tasks_per_depth"].get(depth, 0)


# =============================================================================
# QUEUE CONFIGURATION
# =============================================================================


class QueueType(str, Enum):
    """Queue types for task routing."""

    CORE = "core"
    TELEMETRY = "telemetry"
    BATCH = "batch"
    CRITICAL = "critical"
    GOVERNANCE = "governance"


QUEUE_CONFIG = {
    QueueType.CORE: {
        "priority": TaskPriority.NORMAL,
        "max_workers": 10,
        "timeout_seconds": 60,
    },
    QueueType.TELEMETRY: {
        "priority": TaskPriority.LOW,
        "max_workers": 2,
        "timeout_seconds": 30,
    },
    QueueType.BATCH: {
        "priority": TaskPriority.BATCH,
        "max_workers": 4,
        "timeout_seconds": 300,
    },
    QueueType.CRITICAL: {
        "priority": TaskPriority.CRITICAL,
        "max_workers": 5,
        "timeout_seconds": 30,
    },
    QueueType.GOVERNANCE: {
        "priority": TaskPriority.HIGH,
        "max_workers": 3,
        "timeout_seconds": 60,
    },
}
