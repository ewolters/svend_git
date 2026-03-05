"""
Throttle Policy - Dynamic Throttling Rules and Decisions

Standard: SCH-004 §3 (Throttle Policy)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

ThrottlePolicy defines rules for dynamic throttling based on:
- Queue depth thresholds
- DLQ size thresholds
- Governance denial rates
- Circuit breaker states
- Worker utilization

Throttle Levels:
    NONE (0.0)     - Normal operation, no restrictions
    LIGHT (0.25)   - 25% reduction in throughput
    MODERATE (0.5) - 50% reduction, skip low-priority
    HEAVY (0.75)   - 75% reduction, only critical tasks
    CRITICAL (1.0) - Full stop, circuit open
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from django.utils import timezone

from .health import HealthMetrics

logger = logging.getLogger(__name__)


class ThrottleLevel(Enum):
    """
    Throttle intensity levels.

    Standard: SCH-004 §3.1

    Each level has an associated reduction factor:
    - NONE: 0% reduction (normal operation)
    - LIGHT: 25% reduction
    - MODERATE: 50% reduction
    - HEAVY: 75% reduction
    - CRITICAL: 100% reduction (full stop)
    """

    NONE = 0.0
    LIGHT = 0.25
    MODERATE = 0.5
    HEAVY = 0.75
    CRITICAL = 1.0

    @property
    def throughput_multiplier(self) -> float:
        """Get throughput multiplier (1.0 - reduction)."""
        return 1.0 - self.value

    @property
    def delay_multiplier(self) -> float:
        """Get delay multiplier for schedule triggers."""
        if self == ThrottleLevel.NONE:
            return 1.0
        elif self == ThrottleLevel.LIGHT:
            return 1.5
        elif self == ThrottleLevel.MODERATE:
            return 2.0
        elif self == ThrottleLevel.HEAVY:
            return 4.0
        else:  # CRITICAL
            return float("inf")  # Infinite delay = stop

    @classmethod
    def from_value(cls, value: float) -> ThrottleLevel:
        """Get throttle level from numeric value."""
        if value >= 1.0:
            return cls.CRITICAL
        elif value >= 0.75:
            return cls.HEAVY
        elif value >= 0.5:
            return cls.MODERATE
        elif value >= 0.25:
            return cls.LIGHT
        else:
            return cls.NONE


@dataclass
class ThrottleDecision:
    """
    Result of throttle policy evaluation.

    Standard: SCH-004 §3.2
    """

    # Overall throttle level
    level: ThrottleLevel

    # Whether to allow the action
    allow: bool

    # Reasons for throttling
    reasons: list[str] = field(default_factory=list)

    # Specific restrictions
    skip_low_priority: bool = False
    skip_batch_tasks: bool = False
    pause_schedules: bool = False
    pause_specific_tasks: set[str] = field(default_factory=set)

    # Suggested delay for schedules (seconds)
    schedule_delay_seconds: float = 0.0

    # Confidence degradation for governance
    confidence_penalty: float = 0.0

    # Timestamp
    decided_at: datetime = field(default_factory=timezone.now)

    # Contributing metrics
    contributing_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.name,
            "level_value": self.level.value,
            "allow": self.allow,
            "reasons": self.reasons,
            "skip_low_priority": self.skip_low_priority,
            "skip_batch_tasks": self.skip_batch_tasks,
            "pause_schedules": self.pause_schedules,
            "pause_specific_tasks": list(self.pause_specific_tasks),
            "schedule_delay_seconds": self.schedule_delay_seconds,
            "confidence_penalty": self.confidence_penalty,
            "decided_at": self.decided_at.isoformat(),
            "contributing_metrics": self.contributing_metrics,
        }


@dataclass
class ThrottleRule:
    """
    A single throttle rule.

    Standard: SCH-004 §3.3
    """

    name: str
    description: str
    metric: str  # Metric name to check
    threshold: float  # Threshold value
    comparison: str  # "gt", "gte", "lt", "lte", "eq"
    throttle_level: ThrottleLevel
    actions: list[str] = field(default_factory=list)  # Additional actions
    enabled: bool = True

    def evaluate(self, metrics: HealthMetrics) -> tuple[bool, str] | None:
        """Evaluate rule against metrics. Returns (triggered, reason)."""
        if not self.enabled:
            return None

        # Get metric value
        value = self._get_metric_value(metrics)
        if value is None:
            return None

        # Compare
        triggered = False
        if self.comparison == "gt":
            triggered = value > self.threshold
        elif self.comparison == "gte":
            triggered = value >= self.threshold
        elif self.comparison == "lt":
            triggered = value < self.threshold
        elif self.comparison == "lte":
            triggered = value <= self.threshold
        elif self.comparison == "eq":
            triggered = value == self.threshold

        if triggered:
            return (True, f"{self.name}: {self.metric}={value:.2f} {self.comparison} {self.threshold}")

        return (False, "")

    def _get_metric_value(self, metrics: HealthMetrics) -> float | None:
        """Extract metric value from HealthMetrics."""
        metric_map = {
            "queue_depth": metrics.queue_depth,
            "queue_utilization": metrics.queue_utilization,
            "queue_growth_rate": metrics.queue_growth_rate,
            "dlq_size": metrics.dlq_size,
            "dlq_growth_rate": metrics.dlq_growth_rate,
            "dlq_pending_count": metrics.dlq_pending_count,
            "governance_denial_rate": metrics.governance_denial_rate,
            "governance_escalation_rate": metrics.governance_escalation_rate,
            "circuit_open_ratio": metrics.circuit_open_ratio,
            "circuits_open": metrics.circuits_open,
            "worker_utilization": metrics.worker_utilization,
            "tasks_in_flight": metrics.tasks_in_flight,
            "avg_task_latency_ms": metrics.avg_task_latency_ms,
            "latency_trend": metrics.latency_trend,
            "throughput_trend": metrics.throughput_trend,
        }
        return metric_map.get(self.metric)


# Default throttle rules (SCH-004 §3.4)
DEFAULT_THROTTLE_RULES: list[ThrottleRule] = [
    # Queue depth rules
    ThrottleRule(
        name="queue_light",
        description="Light throttle when queue is 60% full",
        metric="queue_utilization",
        threshold=0.6,
        comparison="gte",
        throttle_level=ThrottleLevel.LIGHT,
        actions=["skip_batch"],
    ),
    ThrottleRule(
        name="queue_moderate",
        description="Moderate throttle when queue is 80% full",
        metric="queue_utilization",
        threshold=0.8,
        comparison="gte",
        throttle_level=ThrottleLevel.MODERATE,
        actions=["skip_batch", "skip_low_priority"],
    ),
    ThrottleRule(
        name="queue_heavy",
        description="Heavy throttle when queue is 95% full",
        metric="queue_utilization",
        threshold=0.95,
        comparison="gte",
        throttle_level=ThrottleLevel.HEAVY,
        actions=["skip_batch", "skip_low_priority", "pause_schedules"],
    ),
    # DLQ rules
    ThrottleRule(
        name="dlq_growing",
        description="Throttle when DLQ is growing quickly",
        metric="dlq_growth_rate",
        threshold=5.0,  # 5 per minute
        comparison="gt",
        throttle_level=ThrottleLevel.LIGHT,
        actions=["confidence_penalty:0.1"],
    ),
    ThrottleRule(
        name="dlq_large",
        description="Moderate throttle when DLQ is large",
        metric="dlq_pending_count",
        threshold=100,
        comparison="gte",
        throttle_level=ThrottleLevel.MODERATE,
        actions=["confidence_penalty:0.2", "pause_failing_tasks"],
    ),
    ThrottleRule(
        name="dlq_critical",
        description="Heavy throttle when DLQ is very large",
        metric="dlq_pending_count",
        threshold=500,
        comparison="gte",
        throttle_level=ThrottleLevel.HEAVY,
        actions=["confidence_penalty:0.3", "pause_failing_tasks", "pause_schedules"],
    ),
    # Governance rules
    ThrottleRule(
        name="governance_denying",
        description="Throttle when governance denial rate is high",
        metric="governance_denial_rate",
        threshold=0.3,
        comparison="gte",
        throttle_level=ThrottleLevel.LIGHT,
        actions=["confidence_penalty:0.15"],
    ),
    ThrottleRule(
        name="governance_blocking",
        description="Heavy throttle when governance is blocking most requests",
        metric="governance_denial_rate",
        threshold=0.6,
        comparison="gte",
        throttle_level=ThrottleLevel.HEAVY,
        actions=["confidence_penalty:0.4", "pause_schedules"],
    ),
    # Circuit breaker rules
    ThrottleRule(
        name="circuits_opening",
        description="Throttle when circuits are opening",
        metric="circuit_open_ratio",
        threshold=0.2,
        comparison="gte",
        throttle_level=ThrottleLevel.LIGHT,
        actions=["skip_circuit_open_tasks"],
    ),
    ThrottleRule(
        name="circuits_many_open",
        description="Heavy throttle when many circuits are open",
        metric="circuit_open_ratio",
        threshold=0.5,
        comparison="gte",
        throttle_level=ThrottleLevel.HEAVY,
        actions=["skip_circuit_open_tasks", "pause_schedules"],
    ),
    # Worker utilization rules
    ThrottleRule(
        name="workers_saturated",
        description="Throttle when workers are near capacity",
        metric="worker_utilization",
        threshold=0.9,
        comparison="gte",
        throttle_level=ThrottleLevel.MODERATE,
        actions=["skip_batch", "delay_schedules:2.0"],
    ),
    # Latency rules
    ThrottleRule(
        name="latency_increasing",
        description="Throttle when latency is trending up",
        metric="latency_trend",
        threshold=0.5,  # 50% increase
        comparison="gt",
        throttle_level=ThrottleLevel.LIGHT,
        actions=["skip_batch"],
    ),
]


class ThrottlePolicy:
    """
    Evaluates throttle rules and produces throttle decisions.

    Standard: SCH-004 §3

    Features:
    - Rule-based throttling
    - Composable rules with priorities
    - Action-based restrictions
    - Confidence degradation for governance

    Usage:
        policy = ThrottlePolicy()
        decision = policy.evaluate(health_metrics)

        if not decision.allow:
            # Reject or delay the action
            pass

        if decision.skip_batch_tasks:
            # Don't process batch queue
            pass
    """

    def __init__(
        self,
        rules: list[ThrottleRule] | None = None,
        default_level: ThrottleLevel = ThrottleLevel.NONE,
    ):
        """
        Initialize throttle policy.

        Args:
            rules: Custom throttle rules (or use defaults)
            default_level: Default throttle level when no rules trigger
        """
        self._rules = rules or DEFAULT_THROTTLE_RULES.copy()
        self._default_level = default_level

        # Task-specific pause tracking
        self._paused_tasks: set[str] = set()
        self._pause_until: dict[str, datetime] = {}

    def evaluate(
        self,
        metrics: HealthMetrics,
        task_name: str | None = None,
        priority: int | None = None,
        is_batch: bool = False,
    ) -> ThrottleDecision:
        """
        Evaluate throttle policy against current metrics.

        Args:
            metrics: Current health metrics
            task_name: Optional task name to check (for task-specific pauses)
            priority: Optional task priority (0=critical, 4=batch)
            is_batch: Whether this is a batch task

        Returns:
            ThrottleDecision with throttle level and restrictions
        """
        decision = ThrottleDecision(
            level=self._default_level,
            allow=True,
        )

        # Evaluate all rules
        triggered_rules: list[tuple[ThrottleRule, str]] = []
        for rule in self._rules:
            result = rule.evaluate(metrics)
            if result and result[0]:
                triggered_rules.append((rule, result[1]))

        # Determine highest throttle level
        max_level = self._default_level
        for rule, reason in triggered_rules:
            if rule.throttle_level.value > max_level.value:
                max_level = rule.throttle_level
            decision.reasons.append(reason)

        decision.level = max_level

        # Apply actions from triggered rules
        for rule, _ in triggered_rules:
            self._apply_actions(decision, rule.actions)

        # Check task-specific restrictions
        if task_name:
            if self._is_task_paused(task_name):
                decision.allow = False
                decision.reasons.append(f"Task {task_name} is paused")

        # Priority-based filtering
        if priority is not None:
            if decision.skip_low_priority and priority >= 3:  # LOW or BATCH
                decision.allow = False
                decision.reasons.append("Low priority tasks skipped")
            if decision.skip_batch_tasks and priority >= 4:  # BATCH only
                decision.allow = False
                decision.reasons.append("Batch tasks skipped")

        # Batch task filtering
        if is_batch and decision.skip_batch_tasks:
            decision.allow = False
            decision.reasons.append("Batch tasks skipped during throttle")

        # Critical level = full stop
        if decision.level == ThrottleLevel.CRITICAL:
            decision.allow = False
            decision.pause_schedules = True

        # Calculate schedule delay
        decision.schedule_delay_seconds = self._calculate_schedule_delay(decision.level)

        # Add contributing metrics
        decision.contributing_metrics = {
            "queue_utilization": metrics.queue_utilization,
            "dlq_pending_count": metrics.dlq_pending_count,
            "governance_denial_rate": metrics.governance_denial_rate,
            "circuit_open_ratio": metrics.circuit_open_ratio,
            "worker_utilization": metrics.worker_utilization,
            "health_status": metrics.status.value,
        }

        return decision

    def _apply_actions(self, decision: ThrottleDecision, actions: list[str]) -> None:
        """Apply rule actions to decision."""
        for action in actions:
            if action == "skip_batch":
                decision.skip_batch_tasks = True
            elif action == "skip_low_priority":
                decision.skip_low_priority = True
            elif action == "pause_schedules":
                decision.pause_schedules = True
            elif action.startswith("confidence_penalty:"):
                penalty = float(action.split(":")[1])
                decision.confidence_penalty = max(decision.confidence_penalty, penalty)
            elif action.startswith("delay_schedules:"):
                delay = float(action.split(":")[1])
                decision.schedule_delay_seconds = max(decision.schedule_delay_seconds, delay)
            elif action == "pause_failing_tasks":
                decision.pause_specific_tasks.update(self._paused_tasks)

    def _calculate_schedule_delay(self, level: ThrottleLevel) -> float:
        """Calculate schedule delay based on throttle level."""
        base_interval = 60.0  # 1 minute base
        return base_interval * (level.delay_multiplier - 1)

    def _is_task_paused(self, task_name: str) -> bool:
        """Check if a task is currently paused."""
        if task_name not in self._paused_tasks:
            return False

        # Check if pause has expired
        if task_name in self._pause_until:
            if timezone.now() >= self._pause_until[task_name]:
                self._paused_tasks.discard(task_name)
                del self._pause_until[task_name]
                return False

        return True

    def pause_task(self, task_name: str, duration_minutes: int = 30) -> None:
        """Pause a specific task for a duration."""
        self._paused_tasks.add(task_name)
        self._pause_until[task_name] = timezone.now() + timedelta(minutes=duration_minutes)
        logger.info(f"[THROTTLE] Paused task {task_name} for {duration_minutes} minutes")

    def unpause_task(self, task_name: str) -> None:
        """Unpause a specific task."""
        self._paused_tasks.discard(task_name)
        self._pause_until.pop(task_name, None)
        logger.info(f"[THROTTLE] Unpaused task {task_name}")

    def get_paused_tasks(self) -> set[str]:
        """Get set of currently paused tasks."""
        # Clean expired pauses
        now = timezone.now()
        expired = [t for t, until in self._pause_until.items() if now >= until]
        for task_name in expired:
            self._paused_tasks.discard(task_name)
            del self._pause_until[task_name]

        return self._paused_tasks.copy()

    def add_rule(self, rule: ThrottleRule) -> None:
        """Add a custom throttle rule."""
        self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a throttle rule by name."""
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        return len(self._rules) < original_len

    def set_rule_enabled(self, rule_name: str, enabled: bool) -> bool:
        """Enable or disable a rule."""
        for rule in self._rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                return True
        return False

    def get_rules(self) -> list[ThrottleRule]:
        """Get all throttle rules."""
        return self._rules.copy()
