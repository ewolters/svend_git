"""
Backpressure Controller - Orchestrates System Load Regulation

Standard: SCH-004 §4 (Backpressure Controller)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The BackpressureController orchestrates:
- Health monitoring via SystemHealthMonitor
- Throttle decisions via ThrottlePolicy
- Confidence degradation for governance
- Schedule rate limiting
- Emergency shutdown triggers

This is the main integration point for the CognitiveScheduler.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   BackpressureController                     │
    │  ┌──────────────────┐      ┌──────────────────┐             │
    │  │ SystemHealth     │ ──── │ ThrottlePolicy   │             │
    │  │ Monitor          │      │                  │             │
    │  └────────┬─────────┘      └────────┬─────────┘             │
    │           │                          │                       │
    │           └─────────┬────────────────┘                       │
    │                     ▼                                        │
    │             ThrottleDecision                                 │
    │                     │                                        │
    │    ┌────────────────┼───────────────────┐                   │
    │    ▼                ▼                   ▼                   │
    │  Scheduler      Governance        CircuitBreakers           │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from django.utils import timezone

from .health import HealthMetrics, HealthStatus, SystemHealthMonitor
from .throttle import ThrottleDecision, ThrottleLevel, ThrottlePolicy, ThrottleRule

logger = logging.getLogger(__name__)


@dataclass
class BackpressureConfig:
    """
    Configuration for BackpressureController.

    Standard: SCH-004 §4.1
    """

    # Health monitoring
    health_collection_interval_seconds: int = 10
    health_history_window_minutes: int = 5

    # Thresholds
    queue_depth_threshold: int = 1000
    dlq_threshold: int = 100
    governance_denial_threshold: float = 0.3

    # Confidence degradation
    enable_confidence_degradation: bool = True
    max_confidence_penalty: float = 0.5  # Maximum penalty to apply
    confidence_recovery_rate: float = 0.1  # Recovery per minute

    # Emergency controls
    enable_emergency_shutdown: bool = True
    emergency_queue_threshold: float = 0.99
    emergency_dlq_threshold: int = 1000
    emergency_circuit_threshold: float = 0.8

    # Callbacks
    on_level_change: Callable[[ThrottleLevel, ThrottleLevel], None] | None = None
    on_emergency: Callable[[str], None] | None = None

    # Decision caching
    decision_cache_seconds: int = 5  # Cache decisions for this long

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "health_collection_interval_seconds": self.health_collection_interval_seconds,
            "health_history_window_minutes": self.health_history_window_minutes,
            "queue_depth_threshold": self.queue_depth_threshold,
            "dlq_threshold": self.dlq_threshold,
            "governance_denial_threshold": self.governance_denial_threshold,
            "enable_confidence_degradation": self.enable_confidence_degradation,
            "max_confidence_penalty": self.max_confidence_penalty,
            "confidence_recovery_rate": self.confidence_recovery_rate,
            "enable_emergency_shutdown": self.enable_emergency_shutdown,
            "emergency_queue_threshold": self.emergency_queue_threshold,
            "emergency_dlq_threshold": self.emergency_dlq_threshold,
            "emergency_circuit_threshold": self.emergency_circuit_threshold,
            "decision_cache_seconds": self.decision_cache_seconds,
        }


@dataclass
class BackpressureMetrics:
    """
    Metrics for backpressure monitoring.

    Standard: SCH-004 §4.2
    """

    # Current state
    current_level: ThrottleLevel = ThrottleLevel.NONE
    current_confidence_penalty: float = 0.0
    health_status: HealthStatus = HealthStatus.UNKNOWN

    # Decision counts
    decisions_total: int = 0
    decisions_allowed: int = 0
    decisions_denied: int = 0

    # Level time tracking
    time_at_level: dict[str, float] = field(default_factory=dict)  # seconds at each level
    level_changes: int = 0
    last_level_change: datetime | None = None

    # Emergency counts
    emergency_triggers: int = 0
    last_emergency: datetime | None = None

    # Paused tasks
    paused_task_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_level": self.current_level.name,
            "current_confidence_penalty": self.current_confidence_penalty,
            "health_status": self.health_status.value,
            "decisions_total": self.decisions_total,
            "decisions_allowed": self.decisions_allowed,
            "decisions_denied": self.decisions_denied,
            "time_at_level": self.time_at_level,
            "level_changes": self.level_changes,
            "last_level_change": self.last_level_change.isoformat() if self.last_level_change else None,
            "emergency_triggers": self.emergency_triggers,
            "last_emergency": self.last_emergency.isoformat() if self.last_emergency else None,
            "paused_task_count": self.paused_task_count,
        }


class BackpressureController:
    """
    Orchestrates backpressure decisions for the CognitiveScheduler.

    Standard: SCH-004 §4

    Features:
    - Unified throttle decision API
    - Health monitoring integration
    - Confidence degradation for governance
    - Emergency shutdown triggers
    - Decision caching for performance
    - Metrics and history for debugging

    Usage:
        controller = BackpressureController()
        controller.start()

        # Before scheduling a task
        decision = controller.should_schedule(task_name="myapp.process", priority=2)
        if not decision.allow:
            # Apply backpressure
            pass

        # Get confidence penalty for governance
        penalty = controller.get_confidence_penalty()

        controller.stop()

    Integration with CognitiveScheduler:
        scheduler = CognitiveScheduler()
        scheduler.set_backpressure_controller(controller)
    """

    def __init__(
        self,
        config: BackpressureConfig | None = None,
        health_monitor: SystemHealthMonitor | None = None,
        throttle_policy: ThrottlePolicy | None = None,
    ):
        """
        Initialize backpressure controller.

        Args:
            config: Configuration options
            health_monitor: Custom health monitor (or create default)
            throttle_policy: Custom throttle policy (or create default)
        """
        self._config = config or BackpressureConfig()

        # Components
        self._health_monitor = health_monitor or SystemHealthMonitor(
            collection_interval_seconds=self._config.health_collection_interval_seconds,
            history_window_minutes=self._config.health_history_window_minutes,
            queue_depth_threshold=self._config.queue_depth_threshold,
            dlq_threshold=self._config.dlq_threshold,
            governance_denial_threshold=self._config.governance_denial_threshold,
        )
        self._throttle_policy = throttle_policy or ThrottlePolicy()

        # State
        self._running = False
        self._lock = threading.RLock()

        # Current throttle level
        self._current_level = ThrottleLevel.NONE
        self._level_changed_at = timezone.now()

        # Confidence penalty tracking
        self._current_confidence_penalty = 0.0
        self._penalty_updated_at = timezone.now()

        # Decision cache
        self._cached_decision: ThrottleDecision | None = None
        self._cache_expires_at: datetime | None = None

        # Metrics
        self._metrics = BackpressureMetrics()
        self._level_time_tracker: dict[ThrottleLevel, datetime] = {}

        # Decision history for debugging
        self._decision_history: list[ThrottleDecision] = []
        self._max_history_size = 100

        # Emergency state
        self._emergency_active = False
        self._emergency_reason: str | None = None

    def start(self) -> None:
        """Start the backpressure controller."""
        if self._running:
            return

        self._running = True
        self._health_monitor.start()

        # Start level tracking
        self._level_time_tracker[self._current_level] = timezone.now()

        logger.info("[BACKPRESSURE] Controller started")

    def stop(self) -> None:
        """Stop the backpressure controller."""
        self._running = False
        self._health_monitor.stop()

        # Update final level time
        self._update_level_time()

        logger.info("[BACKPRESSURE] Controller stopped")

    def should_schedule(
        self,
        task_name: str | None = None,
        priority: int | None = None,
        is_batch: bool = False,
        bypass_cache: bool = False,
    ) -> ThrottleDecision:
        """
        Determine if a task should be scheduled.

        Args:
            task_name: Name of the task to schedule
            priority: Task priority (0=critical, 4=batch)
            is_batch: Whether this is a batch task
            bypass_cache: Force fresh decision (ignore cache)

        Returns:
            ThrottleDecision with allow/deny and restrictions
        """
        with self._lock:
            # Check cache
            if not bypass_cache and self._is_cache_valid():
                cached = self._cached_decision
                # Re-evaluate task-specific restrictions
                if task_name or priority is not None or is_batch:
                    return self._apply_task_restrictions(cached, task_name, priority, is_batch)
                return cached

            # Get health metrics
            metrics = self._health_monitor.get_metrics()

            # Check for emergency conditions
            if self._config.enable_emergency_shutdown:
                self._check_emergency_conditions(metrics)

            # Evaluate throttle policy
            decision = self._throttle_policy.evaluate(
                metrics=metrics,
                task_name=task_name,
                priority=priority,
                is_batch=is_batch,
            )

            # Apply confidence penalty from decision
            if decision.confidence_penalty > 0:
                self._apply_confidence_penalty(decision.confidence_penalty)

            # Add current confidence penalty to decision
            decision.confidence_penalty = self._current_confidence_penalty

            # Track level changes
            if decision.level != self._current_level:
                self._handle_level_change(self._current_level, decision.level)

            # Update cache
            self._cached_decision = decision
            self._cache_expires_at = timezone.now() + timedelta(seconds=self._config.decision_cache_seconds)

            # Update metrics
            self._update_metrics(decision)

            # Store in history
            self._decision_history.append(decision)
            if len(self._decision_history) > self._max_history_size:
                self._decision_history = self._decision_history[-self._max_history_size :]

            return decision

    def should_trigger_schedule(self, schedule_name: str) -> ThrottleDecision:
        """
        Determine if a schedule trigger should proceed.

        Args:
            schedule_name: Name of the schedule

        Returns:
            ThrottleDecision specific to schedule triggers
        """
        decision = self.should_schedule()

        # Additional schedule-specific logic
        if decision.pause_schedules:
            decision.allow = False
            decision.reasons.append(f"Schedule {schedule_name} paused due to backpressure")

        # Apply schedule delay
        if decision.schedule_delay_seconds > 0:
            decision.reasons.append(f"Schedule delayed by {decision.schedule_delay_seconds:.1f}s")

        return decision

    def get_confidence_penalty(self) -> float:
        """
        Get current confidence penalty for governance decisions.

        Returns:
            Penalty value (0.0-1.0) to subtract from governance confidence
        """
        with self._lock:
            # Apply recovery over time
            self._apply_confidence_recovery()
            return min(self._current_confidence_penalty, self._config.max_confidence_penalty)

    def adjust_governance_confidence(self, base_confidence: float) -> float:
        """
        Adjust governance confidence score with backpressure penalty.

        Args:
            base_confidence: Original confidence score (0.0-1.0)

        Returns:
            Adjusted confidence score after penalty
        """
        penalty = self.get_confidence_penalty()
        adjusted = base_confidence - penalty
        return max(0.0, adjusted)

    def _apply_confidence_penalty(self, penalty: float) -> None:
        """Apply a confidence penalty."""
        self._current_confidence_penalty = min(
            self._config.max_confidence_penalty,
            self._current_confidence_penalty + penalty,
        )
        self._penalty_updated_at = timezone.now()
        logger.debug(f"[BACKPRESSURE] Applied confidence penalty: {penalty}, total: {self._current_confidence_penalty}")

    def _apply_confidence_recovery(self) -> None:
        """Apply gradual recovery to confidence penalty."""
        if self._current_confidence_penalty <= 0:
            return

        elapsed_minutes = (timezone.now() - self._penalty_updated_at).total_seconds() / 60
        recovery = elapsed_minutes * self._config.confidence_recovery_rate

        self._current_confidence_penalty = max(0.0, self._current_confidence_penalty - recovery)
        self._penalty_updated_at = timezone.now()

    def _check_emergency_conditions(self, metrics: HealthMetrics) -> None:
        """Check for emergency shutdown conditions."""
        emergency_reasons = []

        if metrics.queue_utilization >= self._config.emergency_queue_threshold:
            emergency_reasons.append(f"Queue at {metrics.queue_utilization:.0%}")

        if metrics.dlq_pending_count >= self._config.emergency_dlq_threshold:
            emergency_reasons.append(f"DLQ at {metrics.dlq_pending_count}")

        if metrics.circuit_open_ratio >= self._config.emergency_circuit_threshold:
            emergency_reasons.append(f"Circuits {metrics.circuit_open_ratio:.0%} open")

        if emergency_reasons:
            if not self._emergency_active:
                self._trigger_emergency(", ".join(emergency_reasons))
        else:
            if self._emergency_active:
                self._clear_emergency()

    def _trigger_emergency(self, reason: str) -> None:
        """Trigger emergency state."""
        self._emergency_active = True
        self._emergency_reason = reason
        self._metrics.emergency_triggers += 1
        self._metrics.last_emergency = timezone.now()

        logger.warning(f"[BACKPRESSURE] EMERGENCY triggered: {reason}")

        if self._config.on_emergency:
            try:
                self._config.on_emergency(reason)
            except Exception as e:
                logger.error(f"[BACKPRESSURE] Emergency callback failed: {e}")

    def _clear_emergency(self) -> None:
        """Clear emergency state."""
        self._emergency_active = False
        self._emergency_reason = None
        logger.info("[BACKPRESSURE] Emergency cleared")

    def _handle_level_change(self, old_level: ThrottleLevel, new_level: ThrottleLevel) -> None:
        """Handle throttle level change."""
        # Update time tracking
        self._update_level_time()

        self._current_level = new_level
        self._level_changed_at = timezone.now()
        self._level_time_tracker[new_level] = timezone.now()

        self._metrics.level_changes += 1
        self._metrics.last_level_change = timezone.now()

        logger.info(f"[BACKPRESSURE] Level changed: {old_level.name} → {new_level.name}")

        if self._config.on_level_change:
            try:
                self._config.on_level_change(old_level, new_level)
            except Exception as e:
                logger.error(f"[BACKPRESSURE] Level change callback failed: {e}")

    def _update_level_time(self) -> None:
        """Update time spent at current level."""
        if self._current_level in self._level_time_tracker:
            elapsed = (timezone.now() - self._level_time_tracker[self._current_level]).total_seconds()
            level_name = self._current_level.name
            self._metrics.time_at_level[level_name] = self._metrics.time_at_level.get(level_name, 0.0) + elapsed

    def _is_cache_valid(self) -> bool:
        """Check if decision cache is valid."""
        if not self._cached_decision or not self._cache_expires_at:
            return False
        return timezone.now() < self._cache_expires_at

    def _apply_task_restrictions(
        self,
        base_decision: ThrottleDecision,
        task_name: str | None,
        priority: int | None,
        is_batch: bool,
    ) -> ThrottleDecision:
        """Apply task-specific restrictions to cached decision."""
        # Clone decision
        ThrottleDecision(
            level=base_decision.level,
            allow=base_decision.allow,
            reasons=list(base_decision.reasons),
            skip_low_priority=base_decision.skip_low_priority,
            skip_batch_tasks=base_decision.skip_batch_tasks,
            pause_schedules=base_decision.pause_schedules,
            pause_specific_tasks=set(base_decision.pause_specific_tasks),
            schedule_delay_seconds=base_decision.schedule_delay_seconds,
            confidence_penalty=base_decision.confidence_penalty,
            contributing_metrics=dict(base_decision.contributing_metrics),
        )

        # Re-evaluate task-specific restrictions
        return self._throttle_policy.evaluate(
            metrics=self._health_monitor.get_metrics(),
            task_name=task_name,
            priority=priority,
            is_batch=is_batch,
        )

    def _update_metrics(self, decision: ThrottleDecision) -> None:
        """Update metrics from decision."""
        self._metrics.decisions_total += 1
        if decision.allow:
            self._metrics.decisions_allowed += 1
        else:
            self._metrics.decisions_denied += 1

        self._metrics.current_level = decision.level
        self._metrics.current_confidence_penalty = self._current_confidence_penalty
        self._metrics.health_status = self._health_monitor.get_metrics().status
        self._metrics.paused_task_count = len(self._throttle_policy.get_paused_tasks())

    # Task management methods

    def pause_task(self, task_name: str, duration_minutes: int = 30) -> None:
        """Pause a specific task from being scheduled."""
        self._throttle_policy.pause_task(task_name, duration_minutes)

    def unpause_task(self, task_name: str) -> None:
        """Unpause a specific task."""
        self._throttle_policy.unpause_task(task_name)

    def get_paused_tasks(self) -> set[str]:
        """Get set of currently paused tasks."""
        return self._throttle_policy.get_paused_tasks()

    # Diagnostic methods

    def get_health_metrics(self) -> HealthMetrics:
        """Get current health metrics."""
        return self._health_monitor.get_metrics()

    def get_backpressure_metrics(self) -> BackpressureMetrics:
        """Get backpressure controller metrics."""
        with self._lock:
            self._update_level_time()
            return BackpressureMetrics(
                current_level=self._metrics.current_level,
                current_confidence_penalty=self._metrics.current_confidence_penalty,
                health_status=self._metrics.health_status,
                decisions_total=self._metrics.decisions_total,
                decisions_allowed=self._metrics.decisions_allowed,
                decisions_denied=self._metrics.decisions_denied,
                time_at_level=dict(self._metrics.time_at_level),
                level_changes=self._metrics.level_changes,
                last_level_change=self._metrics.last_level_change,
                emergency_triggers=self._metrics.emergency_triggers,
                last_emergency=self._metrics.last_emergency,
                paused_task_count=self._metrics.paused_task_count,
            )

    def get_decision_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent throttle decisions."""
        with self._lock:
            recent = self._decision_history[-limit:]
            return [d.to_dict() for d in recent]

    def get_current_level(self) -> ThrottleLevel:
        """Get current throttle level."""
        return self._current_level

    def is_emergency(self) -> bool:
        """Check if emergency state is active."""
        return self._emergency_active

    def get_emergency_reason(self) -> str | None:
        """Get reason for current emergency."""
        return self._emergency_reason

    @property
    def config(self) -> BackpressureConfig:
        """Get controller configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    # Rule management (delegated to policy)

    def add_throttle_rule(self, rule: ThrottleRule) -> None:
        """Add a custom throttle rule."""
        self._throttle_policy.add_rule(rule)

    def remove_throttle_rule(self, rule_name: str) -> bool:
        """Remove a throttle rule by name."""
        return self._throttle_policy.remove_rule(rule_name)

    def set_rule_enabled(self, rule_name: str, enabled: bool) -> bool:
        """Enable or disable a throttle rule."""
        return self._throttle_policy.set_rule_enabled(rule_name, enabled)

    def get_throttle_rules(self) -> list[ThrottleRule]:
        """Get all throttle rules."""
        return self._throttle_policy.get_rules()
