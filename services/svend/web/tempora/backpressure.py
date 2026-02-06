"""
Tempora Backpressure Controller
===============================

Adaptive load management and throttling for the scheduler.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

from tempora.types import ThrottleLevel

logger = logging.getLogger(__name__)


@dataclass
class BackpressureConfig:
    """Configuration for backpressure controller."""
    # Thresholds for each level (queue depth)
    low_threshold: int = 1000
    medium_threshold: int = 5000
    high_threshold: int = 10000
    critical_threshold: int = 20000

    # Memory thresholds (percent)
    memory_warning_percent: float = 70.0
    memory_critical_percent: float = 90.0

    # Callbacks
    on_level_change: Optional[Callable[[ThrottleLevel, ThrottleLevel], None]] = None
    on_emergency: Optional[Callable[[str], None]] = None

    # Monitoring
    check_interval_seconds: float = 1.0


@dataclass
class SchedulingDecision:
    """Result of backpressure scheduling check."""
    allow: bool
    level: ThrottleLevel
    reasons: List[str] = field(default_factory=list)
    schedule_delay_seconds: int = 0
    confidence_penalty: float = 0.0
    pause_schedules: bool = False


class HealthStatus:
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class BackpressureController:
    """
    Monitors system load and applies backpressure to prevent overload.

    Features:
    - Queue depth monitoring
    - Memory pressure tracking
    - Adaptive throttling levels
    - Schedule pausing under load
    """

    def __init__(self, config: Optional[BackpressureConfig] = None):
        self.config = config or BackpressureConfig()
        self._level = ThrottleLevel.NONE
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Metrics
        self._queue_depth = 0
        self._memory_percent = 0.0
        self._last_check = datetime.now()

    def start(self) -> None:
        """Start the backpressure monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="backpressure-monitor",
        )
        self._monitor_thread.start()
        logger.info("Backpressure controller started")

    def stop(self) -> None:
        """Stop the backpressure monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Backpressure controller stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._update_metrics()
                self._evaluate_level()
            except Exception as e:
                logger.error(f"Backpressure monitor error: {e}")

            time.sleep(self.config.check_interval_seconds)

    def _update_metrics(self) -> None:
        """Update system metrics."""
        # Queue depth from Django ORM
        try:
            from tempora.models import CognitiveTask
            from tempora.types import TaskState
            self._queue_depth = CognitiveTask.objects.filter(
                state__in=[TaskState.PENDING.value, TaskState.SCHEDULED.value]
            ).count()
        except Exception:
            pass

        # Memory usage
        try:
            import psutil
            self._memory_percent = psutil.virtual_memory().percent
        except ImportError:
            pass

        self._last_check = datetime.now()

    def _evaluate_level(self) -> None:
        """Evaluate and update throttle level."""
        new_level = ThrottleLevel.NONE

        # Check queue depth
        if self._queue_depth >= self.config.critical_threshold:
            new_level = ThrottleLevel.CRITICAL
        elif self._queue_depth >= self.config.high_threshold:
            new_level = max(new_level, ThrottleLevel.HIGH)
        elif self._queue_depth >= self.config.medium_threshold:
            new_level = max(new_level, ThrottleLevel.MEDIUM)
        elif self._queue_depth >= self.config.low_threshold:
            new_level = max(new_level, ThrottleLevel.LOW)

        # Check memory
        if self._memory_percent >= self.config.memory_critical_percent:
            new_level = ThrottleLevel.CRITICAL
        elif self._memory_percent >= self.config.memory_warning_percent:
            new_level = max(new_level, ThrottleLevel.HIGH)

        # Update level if changed
        if new_level != self._level:
            old_level = self._level
            with self._lock:
                self._level = new_level

            if self.config.on_level_change:
                self.config.on_level_change(old_level, new_level)

            # Emergency callback
            if new_level == ThrottleLevel.CRITICAL and self.config.on_emergency:
                self.config.on_emergency(
                    f"Critical backpressure: queue={self._queue_depth}, memory={self._memory_percent}%"
                )

    @property
    def level(self) -> ThrottleLevel:
        """Current throttle level."""
        return self._level

    def should_schedule(
        self,
        task_name: str,
        priority: int,
        is_batch: bool = False,
    ) -> SchedulingDecision:
        """
        Check if a task should be scheduled given current backpressure.

        Args:
            task_name: Task type name
            priority: Task priority (0-4)
            is_batch: Whether this is a batch task

        Returns:
            SchedulingDecision with allow/deny and reasons
        """
        level = self._level
        reasons = []

        # Critical - only allow critical priority
        if level == ThrottleLevel.CRITICAL:
            if priority < 4:  # TaskPriority.CRITICAL
                return SchedulingDecision(
                    allow=False,
                    level=level,
                    reasons=["System at critical load, only critical tasks allowed"],
                    schedule_delay_seconds=60,
                )

        # High - block batch, delay low priority
        if level >= ThrottleLevel.HIGH:
            if is_batch:
                return SchedulingDecision(
                    allow=False,
                    level=level,
                    reasons=["Batch tasks paused during high load"],
                    schedule_delay_seconds=30,
                    pause_schedules=True,
                )
            if priority < 2:  # Below NORMAL
                return SchedulingDecision(
                    allow=False,
                    level=level,
                    reasons=["Low priority tasks delayed during high load"],
                    schedule_delay_seconds=15,
                )

        # Medium - add delay for batch
        if level >= ThrottleLevel.MEDIUM:
            if is_batch:
                return SchedulingDecision(
                    allow=True,
                    level=level,
                    reasons=["Batch task delayed"],
                    schedule_delay_seconds=5,
                    confidence_penalty=0.1,
                )

        return SchedulingDecision(allow=True, level=level, reasons=reasons)

    def should_trigger_schedule(self, schedule_type: str) -> SchedulingDecision:
        """Check if schedules should trigger."""
        if self._level >= ThrottleLevel.HIGH:
            return SchedulingDecision(
                allow=False,
                level=self._level,
                reasons=["Schedules paused during high load"],
                pause_schedules=True,
            )
        return SchedulingDecision(allow=True, level=self._level)

    def adjust_governance_confidence(self, confidence: float) -> float:
        """Apply confidence penalty based on load."""
        penalties = {
            ThrottleLevel.NONE: 0.0,
            ThrottleLevel.LOW: 0.05,
            ThrottleLevel.MEDIUM: 0.10,
            ThrottleLevel.HIGH: 0.20,
            ThrottleLevel.CRITICAL: 0.50,
        }
        penalty = penalties.get(self._level, 0.0)
        return max(0.0, confidence - penalty)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current backpressure metrics."""
        return {
            "level": self._level.name,
            "queue_depth": self._queue_depth,
            "memory_percent": self._memory_percent,
            "last_check": self._last_check.isoformat(),
        }


__all__ = [
    "BackpressureController",
    "BackpressureConfig",
    "SchedulingDecision",
    "HealthStatus",
    "ThrottleLevel",
]
