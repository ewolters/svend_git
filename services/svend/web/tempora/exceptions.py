"""
Tempora Exceptions
==================

Custom exceptions for the scheduler system.
"""

from typing import Optional
from tempora.types import ThrottleLevel


class TemporaError(Exception):
    """Base exception for Tempora scheduler."""
    pass


class QuotaExceededError(TemporaError):
    """Raised when tenant quota is exceeded."""

    def __init__(self, message: str, quota_type: str = "", current: int = 0, limit: int = 0):
        super().__init__(message)
        self.quota_type = quota_type
        self.current = current
        self.limit = limit


class ThrottledError(TemporaError):
    """Raised when task is throttled by backpressure."""

    def __init__(
        self,
        message: str,
        throttle_level: ThrottleLevel = ThrottleLevel.NONE,
        retry_after_seconds: int = 0,
    ):
        super().__init__(message)
        self.throttle_level = throttle_level
        self.retry_after_seconds = retry_after_seconds


class CascadeLimitError(TemporaError):
    """Raised when cascade depth or child limit is exceeded."""

    def __init__(self, message: str, depth: int = 0, limit: int = 0):
        super().__init__(message)
        self.depth = depth
        self.limit = limit


class CircuitOpenError(TemporaError):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, service_name: str = "", retry_after_seconds: int = 0):
        super().__init__(message)
        self.service_name = service_name
        self.retry_after_seconds = retry_after_seconds


class TaskNotFoundError(TemporaError):
    """Raised when a task is not found."""
    pass


class ScheduleNotFoundError(TemporaError):
    """Raised when a schedule is not found."""
    pass


class NotLeaderError(TemporaError):
    """Raised when operation requires leader but node is not leader."""

    def __init__(self, message: str = "Not the cluster leader", leader_hint: Optional[str] = None):
        super().__init__(message)
        self.leader_hint = leader_hint


class ReplicationError(TemporaError):
    """Raised when log replication fails."""
    pass


class ElectionError(TemporaError):
    """Raised when leader election fails."""
    pass


__all__ = [
    "TemporaError",
    "QuotaExceededError",
    "ThrottledError",
    "CascadeLimitError",
    "CircuitOpenError",
    "TaskNotFoundError",
    "ScheduleNotFoundError",
    "NotLeaderError",
    "ReplicationError",
    "ElectionError",
]
