"""
Tempora Temporal Reflexes
=========================

Governance-linked temporal control for adaptive scheduling.
Allows governance rules to affect timing: pause schedules,
adjust priorities, modify TTLs based on system state.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Types of temporal triggers."""
    RISK_THRESHOLD = "risk_threshold"
    HEALTH_DEGRADED = "health_degraded"
    LOAD_SPIKE = "load_spike"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"


class ActionType(str, Enum):
    """Types of temporal actions."""
    PAUSE_SCHEDULE = "pause_schedule"
    ADJUST_PRIORITY = "adjust_priority"
    MODIFY_TTL = "modify_ttl"
    SPAWN_COMPENSATING = "spawn_compensating"
    PAUSE_TENANT = "pause_tenant"


@dataclass
class TemporalTrigger:
    """Condition that activates a temporal policy."""
    trigger_type: TriggerType
    threshold: float = 0.0
    duration_seconds: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalAction:
    """Action to take when trigger activates."""
    action_type: ActionType
    target: str = ""  # schedule_id, task_name, tenant_id
    value: Any = None  # adjustment amount, new TTL, etc.
    duration_seconds: int = 300  # How long action persists


@dataclass
class TemporalPolicyRule:
    """A single rule in a temporal policy."""
    name: str
    trigger: TemporalTrigger
    actions: List[TemporalAction]
    enabled: bool = True
    cooldown_seconds: int = 60


@dataclass
class TemporalPolicy:
    """Collection of temporal rules."""
    policy_id: str
    name: str
    rules: List[TemporalPolicyRule]
    enabled: bool = True


@dataclass
class CompensatingTask:
    """Task to spawn as compensation/self-healing."""
    task_name: str
    payload: Dict[str, Any]
    priority: int = 2  # NORMAL
    queue: str = "core"
    delay_seconds: int = 0


@dataclass
class ReflexOutcome:
    """Result of evaluating temporal reflexes."""
    actions_taken: List[str]
    schedules_paused: Set[str]
    tenants_paused: Set[str]
    priority_adjustments: Dict[str, int]
    ttl_multipliers: Dict[str, float]
    compensating_tasks: List[CompensatingTask]


class ReflexState(str, Enum):
    """State of a temporal reflex."""
    IDLE = "idle"
    TRIGGERED = "triggered"
    ACTIVE = "active"
    COOLING_DOWN = "cooling_down"


@dataclass
class TemporalControllerConfig:
    """Configuration for temporal controller."""
    evaluation_interval_seconds: float = 5.0
    task_submitter: Optional[Callable[[CompensatingTask], Any]] = None
    context_provider: Optional[Callable[[], Dict[str, Any]]] = None


class TemporalReflex:
    """
    Single temporal reflex instance tracking state and activation.
    """

    def __init__(self, rule: TemporalPolicyRule):
        self.rule = rule
        self.state = ReflexState.IDLE
        self.last_triggered: Optional[datetime] = None
        self.last_action: Optional[datetime] = None
        self.activation_count = 0

    def can_trigger(self) -> bool:
        """Check if reflex can trigger (not in cooldown)."""
        if not self.rule.enabled:
            return False

        if self.last_triggered is None:
            return True

        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.rule.cooldown_seconds

    def trigger(self) -> None:
        """Mark reflex as triggered."""
        self.state = ReflexState.TRIGGERED
        self.last_triggered = datetime.now()
        self.activation_count += 1

    def activate(self) -> None:
        """Mark reflex as actively applying actions."""
        self.state = ReflexState.ACTIVE
        self.last_action = datetime.now()

    def deactivate(self) -> None:
        """Mark reflex as cooling down."""
        self.state = ReflexState.COOLING_DOWN


class TemporalController:
    """
    Controller for governance-linked temporal reflexes.

    Evaluates policies against system state and applies
    temporal adjustments (pausing, priority changes, TTL mods).
    """

    def __init__(self, config: Optional[TemporalControllerConfig] = None):
        self.config = config or TemporalControllerConfig()
        self._policies: Dict[str, TemporalPolicy] = {}
        self._reflexes: Dict[str, TemporalReflex] = {}
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Active effects
        self._paused_schedules: Set[str] = set()
        self._paused_tenants: Set[str] = set()
        self._priority_adjustments: Dict[str, int] = {}
        self._ttl_multipliers: Dict[str, float] = {}
        self._schedule_interval_multipliers: Dict[str, float] = {}

        # Backpressure integration
        self._backpressure = None

    def set_backpressure(self, controller) -> None:
        """Wire up backpressure controller for load-based reflexes."""
        self._backpressure = controller

    def register_policy(self, policy: TemporalPolicy) -> None:
        """Register a temporal policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy
            for rule in policy.rules:
                reflex_id = f"{policy.policy_id}.{rule.name}"
                self._reflexes[reflex_id] = TemporalReflex(rule)

        logger.info(f"Registered temporal policy: {policy.name}")

    def unregister_policy(self, policy_id: str) -> None:
        """Unregister a temporal policy."""
        with self._lock:
            if policy_id in self._policies:
                policy = self._policies.pop(policy_id)
                for rule in policy.rules:
                    reflex_id = f"{policy_id}.{rule.name}"
                    self._reflexes.pop(reflex_id, None)

    def start(self) -> None:
        """Start the temporal controller."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._evaluation_loop,
            daemon=True,
            name="temporal-controller",
        )
        self._monitor_thread.start()
        logger.info("Temporal controller started")

    def stop(self) -> None:
        """Stop the temporal controller."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Temporal controller stopped")

    def _evaluation_loop(self) -> None:
        """Background loop evaluating policies."""
        while self._running:
            try:
                self._evaluate_policies()
            except Exception as e:
                logger.error(f"Temporal evaluation error: {e}")

            time.sleep(self.config.evaluation_interval_seconds)

    def _evaluate_policies(self) -> None:
        """Evaluate all registered policies."""
        context = self._get_context()

        for policy_id, policy in self._policies.items():
            if not policy.enabled:
                continue

            for rule in policy.rules:
                reflex_id = f"{policy_id}.{rule.name}"
                reflex = self._reflexes.get(reflex_id)
                if not reflex or not reflex.can_trigger():
                    continue

                if self._check_trigger(rule.trigger, context):
                    self._apply_actions(reflex, rule.actions)

    def _get_context(self) -> Dict[str, Any]:
        """Get current system context for policy evaluation."""
        context = {
            "timestamp": datetime.now(),
            "risk_score": 0.0,
            "health_status": "healthy",
            "error_rate": 0.0,
            "load_percent": 0.0,
        }

        # Add backpressure info
        if self._backpressure:
            metrics = self._backpressure.get_metrics()
            context["load_percent"] = metrics.get("queue_depth", 0) / 100
            context["memory_percent"] = metrics.get("memory_percent", 0)

        # Add custom context
        if self.config.context_provider:
            try:
                custom = self.config.context_provider()
                context.update(custom)
            except Exception as e:
                logger.error(f"Context provider error: {e}")

        return context

    def _check_trigger(self, trigger: TemporalTrigger, context: Dict[str, Any]) -> bool:
        """Check if a trigger condition is met."""
        if trigger.trigger_type == TriggerType.RISK_THRESHOLD:
            return context.get("risk_score", 0) >= trigger.threshold

        elif trigger.trigger_type == TriggerType.HEALTH_DEGRADED:
            return context.get("health_status") != "healthy"

        elif trigger.trigger_type == TriggerType.LOAD_SPIKE:
            return context.get("load_percent", 0) >= trigger.threshold

        elif trigger.trigger_type == TriggerType.ERROR_RATE:
            return context.get("error_rate", 0) >= trigger.threshold

        elif trigger.trigger_type == TriggerType.MANUAL:
            return trigger.metadata.get("activated", False)

        return False

    def _apply_actions(self, reflex: TemporalReflex, actions: List[TemporalAction]) -> None:
        """Apply temporal actions."""
        reflex.trigger()
        reflex.activate()

        for action in actions:
            try:
                self._apply_action(action)
            except Exception as e:
                logger.error(f"Action error: {e}")

    def _apply_action(self, action: TemporalAction) -> None:
        """Apply a single temporal action."""
        if action.action_type == ActionType.PAUSE_SCHEDULE:
            self._paused_schedules.add(action.target)
            logger.info(f"[TEMPORAL] Paused schedule: {action.target}")

        elif action.action_type == ActionType.PAUSE_TENANT:
            self._paused_tenants.add(action.target)
            logger.info(f"[TEMPORAL] Paused tenant: {action.target}")

        elif action.action_type == ActionType.ADJUST_PRIORITY:
            self._priority_adjustments[action.target] = action.value
            logger.info(f"[TEMPORAL] Priority adjustment for {action.target}: {action.value}")

        elif action.action_type == ActionType.MODIFY_TTL:
            self._ttl_multipliers[action.target] = action.value
            logger.info(f"[TEMPORAL] TTL multiplier for {action.target}: {action.value}")

        elif action.action_type == ActionType.SPAWN_COMPENSATING:
            if self.config.task_submitter and isinstance(action.value, CompensatingTask):
                self.config.task_submitter(action.value)
                logger.info(f"[TEMPORAL] Spawned compensating task: {action.value.task_name}")

    # =========================================================================
    # Query Methods (called by scheduler)
    # =========================================================================

    def is_schedule_paused(self, schedule_id: str) -> bool:
        """Check if a schedule is paused by temporal policy."""
        return schedule_id in self._paused_schedules

    def is_tenant_paused(self, tenant_id: str) -> bool:
        """Check if a tenant is paused by temporal policy."""
        return tenant_id in self._paused_tenants

    def get_priority_adjustment(self, task_name: str) -> int:
        """Get priority adjustment for a task type."""
        return self._priority_adjustments.get(task_name, 0)

    def get_ttl_multiplier(self, task_name: str) -> float:
        """Get TTL multiplier for a task type."""
        return self._ttl_multipliers.get(task_name, 1.0)

    def get_schedule_interval_multiplier(self, schedule_id: str) -> float:
        """Get interval multiplier for a schedule."""
        return self._schedule_interval_multipliers.get(schedule_id, 1.0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get temporal controller metrics."""
        return {
            "policies_count": len(self._policies),
            "reflexes_count": len(self._reflexes),
            "paused_schedules": list(self._paused_schedules),
            "paused_tenants": list(self._paused_tenants),
            "priority_adjustments": dict(self._priority_adjustments),
            "ttl_multipliers": dict(self._ttl_multipliers),
        }


class TemporalMetrics:
    """Metrics collection for temporal controller."""

    def __init__(self, controller: TemporalController):
        self.controller = controller

    def get_reflex_stats(self) -> Dict[str, Any]:
        """Get statistics on reflex activations."""
        stats = {}
        for reflex_id, reflex in self.controller._reflexes.items():
            stats[reflex_id] = {
                "state": reflex.state.value,
                "activation_count": reflex.activation_count,
                "last_triggered": reflex.last_triggered.isoformat() if reflex.last_triggered else None,
            }
        return stats


__all__ = [
    # Enums
    "TriggerType",
    "ActionType",
    "ReflexState",
    # Dataclasses
    "TemporalTrigger",
    "TemporalAction",
    "TemporalPolicyRule",
    "TemporalPolicy",
    "CompensatingTask",
    "ReflexOutcome",
    "TemporalControllerConfig",
    # Classes
    "TemporalReflex",
    "TemporalController",
    "TemporalMetrics",
]
