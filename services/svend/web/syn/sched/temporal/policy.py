"""
Temporal Policy - Governance-to-Timing Rules

Standard: SCH-006 §2 (Temporal Policy)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

TemporalPolicy defines how governance conditions translate to timing actions:
- Triggers: What conditions activate the policy
- Actions: What temporal adjustments to make
- Scope: Which schedules/tasks are affected

Examples:
    - risk > 0.7 → pause non-critical schedules
    - incident_active → accelerate health checks 4x
    - tenant_unstable → degrade priority by 2 levels
    - cascade_depth > 3 → schedule compensating cleanup
    - load > 0.9 → extend task TTLs by 50%
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from django.utils import timezone

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of conditions that can trigger temporal actions."""

    # Governance triggers
    RISK_THRESHOLD = "risk_threshold"
    GOVERNANCE_BLOCKED = "governance_blocked"
    GOVERNANCE_ESCALATED = "governance_escalated"
    CONFIDENCE_LOW = "confidence_low"

    # Incident triggers
    INCIDENT_ACTIVE = "incident_active"
    INCIDENT_SEVERITY = "incident_severity"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_OPEN_RATIO = "circuit_open_ratio"

    # Tenant triggers
    TENANT_UNSTABLE = "tenant_unstable"
    TENANT_QUOTA_HIGH = "tenant_quota_high"
    TENANT_FAILURE_RATE = "tenant_failure_rate"

    # Load triggers
    QUEUE_DEPTH = "queue_depth"
    QUEUE_UTILIZATION = "queue_utilization"
    WORKER_UTILIZATION = "worker_utilization"
    THROTTLE_LEVEL = "throttle_level"

    # Cascade triggers
    CASCADE_DEPTH = "cascade_depth"
    CASCADE_BUDGET_LOW = "cascade_budget_low"

    # DLQ triggers
    DLQ_SIZE = "dlq_size"
    DLQ_GROWTH_RATE = "dlq_growth_rate"

    # Custom
    CUSTOM = "custom"


class ActionType(Enum):
    """Types of temporal actions."""

    # Schedule actions
    PAUSE_SCHEDULE = "pause_schedule"
    RESUME_SCHEDULE = "resume_schedule"
    DELAY_SCHEDULE = "delay_schedule"
    ACCELERATE_SCHEDULE = "accelerate_schedule"

    # Priority actions
    DEGRADE_PRIORITY = "degrade_priority"
    BOOST_PRIORITY = "boost_priority"
    SET_PRIORITY = "set_priority"

    # Task actions
    SCHEDULE_COMPENSATING = "schedule_compensating"
    CANCEL_TASKS = "cancel_tasks"
    EXTEND_TTL = "extend_ttl"
    REDUCE_TTL = "reduce_ttl"

    # Tenant actions
    TENANT_THROTTLE = "tenant_throttle"
    TENANT_PAUSE = "tenant_pause"

    # Health actions
    ACCELERATE_HEALTH_CHECK = "accelerate_health_check"
    TRIGGER_HEALTH_CHECK = "trigger_health_check"

    # Custom
    CUSTOM = "custom"


@dataclass
class TemporalTrigger:
    """
    Defines a condition that activates a temporal policy.

    Standard: SCH-006 §2.1
    """

    trigger_type: TriggerType
    threshold: float | None = None
    comparison: str = "gte"  # gt, gte, lt, lte, eq, ne
    metric_name: str | None = None
    tenant_id: str | None = None  # None = all tenants
    schedule_pattern: str | None = None  # Regex for schedule matching
    task_pattern: str | None = None  # Regex for task matching
    custom_evaluator: Callable[[dict[str, Any]], bool] | None = None

    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate if trigger condition is met.

        Args:
            context: Current system state

        Returns:
            True if trigger condition is met
        """
        if self.trigger_type == TriggerType.CUSTOM:
            if self.custom_evaluator:
                return self.custom_evaluator(context)
            return False

        # Get metric value
        value = self._get_metric_value(context)
        if value is None:
            return False

        # Compare against threshold
        if self.threshold is None:
            return bool(value)

        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        elif self.comparison == "eq":
            return value == self.threshold
        elif self.comparison == "ne":
            return value != self.threshold

        return False

    def _get_metric_value(self, context: dict[str, Any]) -> float | None:
        """Extract metric value from context."""
        metric_map = {
            TriggerType.RISK_THRESHOLD: "governance_risk",
            TriggerType.GOVERNANCE_BLOCKED: "governance_blocked_rate",
            TriggerType.GOVERNANCE_ESCALATED: "governance_escalation_rate",
            TriggerType.CONFIDENCE_LOW: "confidence_score",
            TriggerType.INCIDENT_ACTIVE: "incident_active",
            TriggerType.INCIDENT_SEVERITY: "incident_severity",
            TriggerType.CIRCUIT_OPEN: "circuits_open",
            TriggerType.CIRCUIT_OPEN_RATIO: "circuit_open_ratio",
            TriggerType.TENANT_UNSTABLE: "tenant_stability",
            TriggerType.TENANT_QUOTA_HIGH: "tenant_quota_utilization",
            TriggerType.TENANT_FAILURE_RATE: "tenant_failure_rate",
            TriggerType.QUEUE_DEPTH: "queue_depth",
            TriggerType.QUEUE_UTILIZATION: "queue_utilization",
            TriggerType.WORKER_UTILIZATION: "worker_utilization",
            TriggerType.THROTTLE_LEVEL: "throttle_level_value",
            TriggerType.CASCADE_DEPTH: "cascade_depth",
            TriggerType.CASCADE_BUDGET_LOW: "cascade_budget_remaining",
            TriggerType.DLQ_SIZE: "dlq_size",
            TriggerType.DLQ_GROWTH_RATE: "dlq_growth_rate",
        }

        metric_name = self.metric_name or metric_map.get(self.trigger_type)
        if metric_name:
            return context.get(metric_name)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_type": self.trigger_type.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "metric_name": self.metric_name,
            "tenant_id": self.tenant_id,
            "schedule_pattern": self.schedule_pattern,
            "task_pattern": self.task_pattern,
        }


@dataclass
class TemporalAction:
    """
    Defines what temporal adjustment to make when triggered.

    Standard: SCH-006 §2.2
    """

    action_type: ActionType
    value: float | None = None  # Multiplier, priority level, etc.
    duration_minutes: int | None = None  # How long the action lasts
    schedule_pattern: str | None = None  # Which schedules to affect
    task_pattern: str | None = None  # Which tasks to affect
    task_name: str | None = None  # For compensating tasks
    task_payload: dict[str, Any] | None = None
    priority_delta: int = 0  # For priority adjustments
    ttl_multiplier: float = 1.0  # For TTL adjustments
    interval_multiplier: float = 1.0  # For schedule interval adjustments
    custom_executor: Callable[[dict[str, Any]], None] | None = None

    def matches_schedule(self, schedule_name: str) -> bool:
        """Check if schedule matches the pattern."""
        if not self.schedule_pattern:
            return True
        try:
            return bool(re.match(self.schedule_pattern, schedule_name))
        except re.error:
            return schedule_name == self.schedule_pattern

    def matches_task(self, task_name: str) -> bool:
        """Check if task matches the pattern."""
        if not self.task_pattern:
            return True
        try:
            return bool(re.match(self.task_pattern, task_name))
        except re.error:
            return task_name == self.task_pattern

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "value": self.value,
            "duration_minutes": self.duration_minutes,
            "schedule_pattern": self.schedule_pattern,
            "task_pattern": self.task_pattern,
            "task_name": self.task_name,
            "priority_delta": self.priority_delta,
            "ttl_multiplier": self.ttl_multiplier,
            "interval_multiplier": self.interval_multiplier,
        }


@dataclass
class TemporalPolicyRule:
    """
    A single temporal policy rule: trigger → actions.

    Standard: SCH-006 §2.3
    """

    rule_id: str
    name: str
    description: str
    trigger: TemporalTrigger
    actions: list[TemporalAction]
    enabled: bool = True
    priority: int = 100  # Lower = higher priority
    cooldown_minutes: int = 5  # Minimum time between activations
    max_activations: int | None = None  # Max activations per hour

    # State
    last_activated: datetime | None = None
    activation_count: int = 0
    activation_count_hour: datetime | None = None

    def can_activate(self) -> bool:
        """Check if rule can activate (cooldown, max activations)."""
        if not self.enabled:
            return False

        now = timezone.now()

        # Check cooldown
        if self.last_activated:
            cooldown_end = self.last_activated + timedelta(
                minutes=self.cooldown_minutes
            )
            if now < cooldown_end:
                return False

        # Check max activations per hour
        if self.max_activations:
            if self.activation_count_hour:
                if now - self.activation_count_hour > timedelta(hours=1):
                    # Reset hourly counter
                    self.activation_count = 0
                    self.activation_count_hour = now
            else:
                self.activation_count_hour = now

            if self.activation_count >= self.max_activations:
                return False

        return True

    def record_activation(self) -> None:
        """Record that the rule was activated."""
        self.last_activated = timezone.now()
        self.activation_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "priority": self.priority,
            "cooldown_minutes": self.cooldown_minutes,
            "max_activations": self.max_activations,
            "trigger": self.trigger.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "last_activated": (
                self.last_activated.isoformat() if self.last_activated else None
            ),
            "activation_count": self.activation_count,
        }


# Default temporal policy rules
DEFAULT_TEMPORAL_RULES: list[TemporalPolicyRule] = [
    # High risk → pause non-critical schedules
    TemporalPolicyRule(
        rule_id="risk_pause_schedules",
        name="High Risk Schedule Pause",
        description="Pause non-critical schedules when governance risk exceeds threshold",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.RISK_THRESHOLD,
            threshold=0.7,
            comparison="gte",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.PAUSE_SCHEDULE,
                schedule_pattern=r"^(?!health_|critical_).*",  # Non-health, non-critical
                duration_minutes=30,
            ),
        ],
        priority=10,
        cooldown_minutes=15,
    ),
    # Incident active → accelerate health checks
    TemporalPolicyRule(
        rule_id="incident_accelerate_health",
        name="Incident Health Acceleration",
        description="Run health checks more frequently during incidents",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.INCIDENT_ACTIVE,
            threshold=1,
            comparison="gte",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.ACCELERATE_SCHEDULE,
                schedule_pattern=r"^health_.*",
                interval_multiplier=0.25,  # 4x faster
                duration_minutes=60,
            ),
        ],
        priority=5,
        cooldown_minutes=5,
    ),
    # High tenant failure rate → degrade priority
    TemporalPolicyRule(
        rule_id="tenant_failure_degrade",
        name="Tenant Instability Priority Degradation",
        description="Lower priority for tasks from unstable tenants",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.TENANT_FAILURE_RATE,
            threshold=0.3,
            comparison="gte",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.DEGRADE_PRIORITY,
                priority_delta=2,  # Lower by 2 levels
                duration_minutes=60,
            ),
        ],
        priority=20,
        cooldown_minutes=30,
    ),
    # High cascade depth → schedule cleanup
    TemporalPolicyRule(
        rule_id="cascade_compensate",
        name="Cascade Depth Compensation",
        description="Schedule cleanup task when cascade depth is high",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.CASCADE_DEPTH,
            threshold=4,
            comparison="gte",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.SCHEDULE_COMPENSATING,
                task_name="synara.cascade.cleanup",
                task_payload={"reason": "cascade_depth_exceeded"},
            ),
        ],
        priority=15,
        cooldown_minutes=10,
        max_activations=6,
    ),
    # High load → extend TTLs
    TemporalPolicyRule(
        rule_id="load_extend_ttl",
        name="High Load TTL Extension",
        description="Extend task TTLs when system is under heavy load",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.QUEUE_UTILIZATION,
            threshold=0.8,
            comparison="gte",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.EXTEND_TTL,
                ttl_multiplier=1.5,  # 50% longer
                duration_minutes=30,
            ),
        ],
        priority=30,
        cooldown_minutes=10,
    ),
    # DLQ growing → pause failing task types
    TemporalPolicyRule(
        rule_id="dlq_pause_failing",
        name="DLQ Growth Pause",
        description="Pause task types that are filling the DLQ",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.DLQ_GROWTH_RATE,
            threshold=10,  # 10/min
            comparison="gt",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.PAUSE_SCHEDULE,
                duration_minutes=15,
            ),
            TemporalAction(
                action_type=ActionType.SCHEDULE_COMPENSATING,
                task_name="synara.dlq.analyze",
                task_payload={"reason": "dlq_growth_rate_exceeded"},
            ),
        ],
        priority=10,
        cooldown_minutes=20,
    ),
    # Circuits opening → reduce load
    TemporalPolicyRule(
        rule_id="circuit_reduce_load",
        name="Circuit Breaker Load Reduction",
        description="Reduce load when circuits are opening",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.CIRCUIT_OPEN_RATIO,
            threshold=0.3,
            comparison="gte",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.DELAY_SCHEDULE,
                schedule_pattern=r"^(?!health_|critical_).*",
                interval_multiplier=2.0,  # 2x slower
                duration_minutes=30,
            ),
            TemporalAction(
                action_type=ActionType.TRIGGER_HEALTH_CHECK,
            ),
        ],
        priority=8,
        cooldown_minutes=10,
    ),
    # Low confidence → reduce priority
    TemporalPolicyRule(
        rule_id="low_confidence_degrade",
        name="Low Confidence Priority Reduction",
        description="Reduce priority when governance confidence is low",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.CONFIDENCE_LOW,
            threshold=0.5,
            comparison="lt",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.DEGRADE_PRIORITY,
                priority_delta=1,
                duration_minutes=30,
            ),
        ],
        priority=25,
        cooldown_minutes=15,
    ),
    # Governance blocking frequently → slow down
    TemporalPolicyRule(
        rule_id="governance_blocking_slow",
        name="Governance Block Slowdown",
        description="Slow schedule triggers when governance is blocking frequently",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.GOVERNANCE_BLOCKED,
            threshold=0.4,
            comparison="gte",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.DELAY_SCHEDULE,
                interval_multiplier=1.5,
                duration_minutes=20,
            ),
        ],
        priority=20,
        cooldown_minutes=10,
    ),
    # Critical throttle → emergency response
    TemporalPolicyRule(
        rule_id="throttle_critical_response",
        name="Critical Throttle Emergency",
        description="Emergency response when throttle is critical",
        trigger=TemporalTrigger(
            trigger_type=TriggerType.THROTTLE_LEVEL,
            threshold=1.0,  # CRITICAL level
            comparison="gte",
        ),
        actions=[
            TemporalAction(
                action_type=ActionType.PAUSE_SCHEDULE,
                schedule_pattern=r".*",  # All schedules
                duration_minutes=10,
            ),
            TemporalAction(
                action_type=ActionType.ACCELERATE_HEALTH_CHECK,
                interval_multiplier=0.1,  # 10x faster
                duration_minutes=10,
            ),
            TemporalAction(
                action_type=ActionType.SCHEDULE_COMPENSATING,
                task_name="synara.emergency.response",
                task_payload={"reason": "critical_throttle"},
            ),
        ],
        priority=1,  # Highest priority
        cooldown_minutes=5,
        max_activations=12,
    ),
]


class TemporalPolicy:
    """
    Manages temporal policy rules and evaluates triggers.

    Standard: SCH-006 §2

    Features:
    - Rule-based temporal governance
    - Trigger evaluation with context
    - Action recommendation
    - Priority-based rule ordering
    - Cooldown and rate limiting

    Usage:
        policy = TemporalPolicy()
        context = {"governance_risk": 0.8, "queue_depth": 500}

        activated_rules = policy.evaluate(context)
        for rule in activated_rules:
            for action in rule.actions:
                # Apply temporal action
                pass
    """

    def __init__(
        self,
        rules: list[TemporalPolicyRule] | None = None,
    ):
        """
        Initialize temporal policy.

        Args:
            rules: Custom rules (or use defaults)
        """
        self._rules = rules if rules is not None else DEFAULT_TEMPORAL_RULES.copy()
        self._rules.sort(key=lambda r: r.priority)

    def evaluate(self, context: dict[str, Any]) -> list[TemporalPolicyRule]:
        """
        Evaluate all rules against current context.

        Args:
            context: Current system state

        Returns:
            List of activated rules (in priority order)
        """
        activated = []

        for rule in self._rules:
            if not rule.can_activate():
                continue

            if rule.trigger.evaluate(context):
                activated.append(rule)
                logger.info(
                    f"[TEMPORAL] Rule activated: {rule.name}",
                    extra={
                        "rule_id": rule.rule_id,
                        "trigger_type": rule.trigger.trigger_type.value,
                    },
                )

        return activated

    def add_rule(self, rule: TemporalPolicyRule) -> None:
        """Add a custom rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return len(self._rules) < original_len

    def set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        """Enable or disable a rule."""
        for rule in self._rules:
            if rule.rule_id == rule_id:
                rule.enabled = enabled
                return True
        return False

    def get_rule(self, rule_id: str) -> TemporalPolicyRule | None:
        """Get a rule by ID."""
        for rule in self._rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def get_rules(self) -> list[TemporalPolicyRule]:
        """Get all rules."""
        return self._rules.copy()

    def get_active_rules(self) -> list[TemporalPolicyRule]:
        """Get enabled rules that can activate."""
        return [r for r in self._rules if r.enabled and r.can_activate()]
