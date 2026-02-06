"""
Tempora Type Definitions
========================

Core type definitions, enums, and configuration dataclasses for the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Callable
import uuid


# =============================================================================
# ENUMS
# =============================================================================


class TaskState(str, Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRYING = "retrying"
    DEAD_LETTERED = "dead_lettered"
    CANCELLED = "cancelled"


class TaskPriority(IntEnum):
    """Task priority levels (higher = more urgent)."""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class QueueType(str, Enum):
    """Queue types for task routing."""
    CORE = "core"
    BATCH = "batch"
    REALTIME = "realtime"
    GOVERNANCE = "governance"


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""
    IMMEDIATE = "immediate"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class ScheduleType(str, Enum):
    """Schedule types."""
    CRON = "cron"
    INTERVAL = "interval"
    ONE_TIME = "one_time"


class ThrottleLevel(IntEnum):
    """Backpressure throttle levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class TaskContext:
    """Context passed to task handlers during execution."""
    task_id: uuid.UUID
    correlation_id: uuid.UUID
    tenant_id: uuid.UUID
    task_name: str
    attempt: int
    max_attempts: int
    created_at: datetime
    deadline: Optional[datetime] = None
    parent_task_id: Optional[uuid.UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Retry configuration for tasks."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay_seconds: int = 1
    max_delay_seconds: int = 3600
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    half_open_max_calls: int = 3


@dataclass
class TenantQuota:
    """Resource quotas per tenant."""
    max_queue_depth: int = 10000
    max_concurrent_tasks: int = 100
    max_cascade_depth: int = 5


@dataclass
class CronSchedule:
    """Cron schedule definition."""
    minute: str = "*"
    hour: str = "*"
    day_of_month: str = "*"
    month: str = "*"
    day_of_week: str = "*"

    def __str__(self) -> str:
        return f"{self.minute} {self.hour} {self.day_of_month} {self.month} {self.day_of_week}"


@dataclass
class IntervalSchedule:
    """Interval schedule definition."""
    seconds: int = 0
    minutes: int = 0
    hours: int = 0
    days: int = 0

    def to_timedelta(self) -> timedelta:
        return timedelta(
            days=self.days,
            hours=self.hours,
            minutes=self.minutes,
            seconds=self.seconds,
        )


# =============================================================================
# BUDGET AND LIMITS
# =============================================================================


CASCADE_BUDGET = {
    "max_depth": 10,
    "max_children_per_level": [100, 50, 25, 10, 5, 3, 2, 1, 1, 1],
}


def get_cascade_limit(depth: int) -> int:
    """Get max children allowed at a cascade depth."""
    limits = CASCADE_BUDGET["max_children_per_level"]
    if depth >= len(limits):
        return 0
    return limits[depth]


QUEUE_CONFIG = {
    QueueType.CORE: {"workers": 4, "priority_weight": 1.0},
    QueueType.BATCH: {"workers": 2, "priority_weight": 0.5},
    QueueType.REALTIME: {"workers": 8, "priority_weight": 2.0},
    QueueType.GOVERNANCE: {"workers": 2, "priority_weight": 1.5},
}


CIRCUIT_CONFIGS: Dict[str, CircuitBreakerConfig] = {}


TENANT_QUOTAS: Dict[str, TenantQuota] = {
    "default": TenantQuota(),
}


__all__ = [
    # Enums
    "TaskState",
    "TaskPriority",
    "QueueType",
    "RetryStrategy",
    "CircuitState",
    "ScheduleType",
    "ThrottleLevel",
    # Dataclasses
    "TaskContext",
    "RetryConfig",
    "CircuitBreakerConfig",
    "TenantQuota",
    "CronSchedule",
    "IntervalSchedule",
    # Config
    "CASCADE_BUDGET",
    "get_cascade_limit",
    "QUEUE_CONFIG",
    "CIRCUIT_CONFIGS",
    "TENANT_QUOTAS",
]
