"""
Temporal Reflex Engine - Executes Temporal Actions

Standard: SCH-006 §3 (Temporal Reflex Engine)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The TemporalReflex engine executes temporal actions defined by policies:
- Schedule pausing/resuming
- Priority adjustments
- TTL modifications
- Compensating task scheduling
- Health check acceleration

This is the "muscle" of the temporal system - it turns decisions into action.

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                     Temporal Reflex Engine                        │
    │  ┌─────────────────┐   ┌────────────────┐   ┌───────────────┐   │
    │  │ TemporalReflex  │──▶│ ReflexOutcome  │──▶│ ReflexState   │   │
    │  │ (executor)      │   │ (result)       │   │ (active)      │   │
    │  └────────┬────────┘   └────────────────┘   └───────────────┘   │
    │           │                                                       │
    │    ┌──────┴───────┐                                              │
    │    ▼              ▼                                              │
    │  Schedules    Priorities    TTLs    CompensatingTasks            │
    └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from django.utils import timezone

from .policy import (
    ActionType,
    TemporalAction,
    TemporalPolicyRule,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ReflexStatus(Enum):
    """Status of a reflex action execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERTED = "reverted"
    EXPIRED = "expired"


@dataclass
class CompensatingTask:
    """
    Represents a compensating task to be scheduled.

    Standard: SCH-006 §3.2
    """

    task_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 2  # NORMAL
    queue: str = "core"
    delay_seconds: int = 0
    triggered_by_rule: str | None = None
    triggered_at: datetime = field(default_factory=timezone.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "payload": self.payload,
            "priority": self.priority,
            "queue": self.queue,
            "delay_seconds": self.delay_seconds,
            "triggered_by_rule": self.triggered_by_rule,
            "triggered_at": self.triggered_at.isoformat(),
        }


@dataclass
class ReflexOutcome:
    """
    Records the outcome of a reflex action execution.

    Standard: SCH-006 §3.3
    """

    outcome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    action_type: ActionType = ActionType.CUSTOM
    status: ReflexStatus = ReflexStatus.PENDING

    # What was affected
    affected_schedules: list[str] = field(default_factory=list)
    affected_tasks: list[str] = field(default_factory=list)
    compensating_tasks: list[CompensatingTask] = field(default_factory=list)

    # Modifications made
    priority_adjustments: dict[str, int] = field(
        default_factory=dict
    )  # task/schedule -> delta
    ttl_adjustments: dict[str, float] = field(
        default_factory=dict
    )  # task -> multiplier
    interval_adjustments: dict[str, float] = field(
        default_factory=dict
    )  # schedule -> multiplier

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None

    # Error tracking
    error_message: str | None = None

    # Original values for reverting
    original_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "rule_id": self.rule_id,
            "action_type": self.action_type.value,
            "status": self.status.value,
            "affected_schedules": self.affected_schedules,
            "affected_tasks": self.affected_tasks,
            "compensating_tasks": [ct.to_dict() for ct in self.compensating_tasks],
            "priority_adjustments": self.priority_adjustments,
            "ttl_adjustments": self.ttl_adjustments,
            "interval_adjustments": self.interval_adjustments,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "error_message": self.error_message,
        }


@dataclass
class ReflexState:
    """
    Tracks active temporal reflexes and their state.

    Standard: SCH-006 §3.4
    """

    # Active modifications
    paused_schedules: set[str] = field(default_factory=set)
    accelerated_schedules: dict[str, float] = field(
        default_factory=dict
    )  # schedule -> multiplier
    delayed_schedules: dict[str, float] = field(
        default_factory=dict
    )  # schedule -> multiplier

    # Priority modifications
    priority_degradations: dict[str, int] = field(
        default_factory=dict
    )  # pattern -> delta
    priority_boosts: dict[str, int] = field(default_factory=dict)  # pattern -> delta

    # TTL modifications
    ttl_extensions: dict[str, float] = field(
        default_factory=dict
    )  # pattern -> multiplier
    ttl_reductions: dict[str, float] = field(
        default_factory=dict
    )  # pattern -> multiplier

    # Tenant modifications
    paused_tenants: set[str] = field(default_factory=set)
    throttled_tenants: dict[str, float] = field(
        default_factory=dict
    )  # tenant -> throttle level

    # Health check modifications
    accelerated_health_checks: dict[str, float] = field(
        default_factory=dict
    )  # pattern -> multiplier

    # Active outcomes (with expiration tracking)
    active_outcomes: dict[str, ReflexOutcome] = field(
        default_factory=dict
    )  # outcome_id -> outcome

    # History
    outcome_history: list[ReflexOutcome] = field(default_factory=list)
    max_history_size: int = 500

    def add_outcome(self, outcome: ReflexOutcome) -> None:
        """Add an outcome to tracking."""
        if outcome.expires_at:
            self.active_outcomes[outcome.outcome_id] = outcome
        self.outcome_history.append(outcome)

        # Trim history if needed
        if len(self.outcome_history) > self.max_history_size:
            self.outcome_history = self.outcome_history[-self.max_history_size :]

    def remove_outcome(self, outcome_id: str) -> ReflexOutcome | None:
        """Remove an outcome from active tracking."""
        return self.active_outcomes.pop(outcome_id, None)

    def get_expired_outcomes(self) -> list[ReflexOutcome]:
        """Get list of expired outcomes that need reverting."""
        now = timezone.now()
        expired = []
        for outcome in self.active_outcomes.values():
            if outcome.expires_at and outcome.expires_at <= now:
                expired.append(outcome)
        return expired

    def is_schedule_paused(self, schedule_name: str) -> bool:
        """Check if a schedule is currently paused."""
        return schedule_name in self.paused_schedules

    def get_schedule_interval_multiplier(self, schedule_name: str) -> float:
        """Get the current interval multiplier for a schedule."""
        # Accelerated = smaller interval
        if schedule_name in self.accelerated_schedules:
            return self.accelerated_schedules[schedule_name]
        # Delayed = larger interval
        if schedule_name in self.delayed_schedules:
            return self.delayed_schedules[schedule_name]
        return 1.0

    def get_priority_adjustment(self, task_name: str) -> int:
        """Get net priority adjustment for a task type."""
        adjustment = 0
        for pattern, delta in self.priority_degradations.items():
            if self._matches_pattern(task_name, pattern):
                adjustment += delta
        for pattern, delta in self.priority_boosts.items():
            if self._matches_pattern(task_name, pattern):
                adjustment -= delta  # Boost = lower priority number = higher priority
        return adjustment

    def get_ttl_multiplier(self, task_name: str) -> float:
        """Get TTL multiplier for a task type."""
        multiplier = 1.0
        for pattern, mult in self.ttl_extensions.items():
            if self._matches_pattern(task_name, pattern):
                multiplier = max(multiplier, mult)
        for pattern, mult in self.ttl_reductions.items():
            if self._matches_pattern(task_name, pattern):
                multiplier = min(multiplier, mult)
        return multiplier

    def is_tenant_paused(self, tenant_id: str) -> bool:
        """Check if a tenant is paused."""
        return tenant_id in self.paused_tenants

    def get_tenant_throttle(self, tenant_id: str) -> float:
        """Get throttle level for a tenant (0.0-1.0)."""
        return self.throttled_tenants.get(tenant_id, 0.0)

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches pattern (supports * wildcard)."""
        import fnmatch

        return fnmatch.fnmatch(name, pattern)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paused_schedules": list(self.paused_schedules),
            "accelerated_schedules": self.accelerated_schedules,
            "delayed_schedules": self.delayed_schedules,
            "priority_degradations": self.priority_degradations,
            "priority_boosts": self.priority_boosts,
            "ttl_extensions": self.ttl_extensions,
            "ttl_reductions": self.ttl_reductions,
            "paused_tenants": list(self.paused_tenants),
            "throttled_tenants": self.throttled_tenants,
            "accelerated_health_checks": self.accelerated_health_checks,
            "active_outcomes_count": len(self.active_outcomes),
            "history_size": len(self.outcome_history),
        }


class TemporalReflex:
    """
    Executes temporal actions based on policy rules.

    Standard: SCH-006 §3

    The TemporalReflex engine is responsible for:
    - Executing temporal actions (pause, accelerate, degrade, etc.)
    - Tracking active modifications
    - Reverting expired modifications
    - Scheduling compensating tasks
    - Providing state queries for the scheduler

    Usage:
        reflex = TemporalReflex()

        # Execute an action
        outcome = reflex.execute_action(rule, action, context)

        # Check if schedule is affected
        if reflex.is_schedule_paused("maintenance_cleanup"):
            # Skip this schedule
            pass

        # Get priority adjustment
        adjustment = reflex.get_priority_adjustment("myapp.process")
    """

    def __init__(
        self,
        task_submitter: Callable[[CompensatingTask], Any] | None = None,
        schedule_modifier: Callable[[str, str, Any], bool] | None = None,
    ):
        """
        Initialize the temporal reflex engine.

        Args:
            task_submitter: Callback to submit compensating tasks
            schedule_modifier: Callback to modify schedules (name, field, value)
        """
        self._state = ReflexState()
        self._task_submitter = task_submitter
        self._schedule_modifier = schedule_modifier

        # Action executors
        self._executors: dict[ActionType, Callable] = {
            ActionType.PAUSE_SCHEDULE: self._execute_pause_schedule,
            ActionType.RESUME_SCHEDULE: self._execute_resume_schedule,
            ActionType.DELAY_SCHEDULE: self._execute_delay_schedule,
            ActionType.ACCELERATE_SCHEDULE: self._execute_accelerate_schedule,
            ActionType.DEGRADE_PRIORITY: self._execute_degrade_priority,
            ActionType.BOOST_PRIORITY: self._execute_boost_priority,
            ActionType.SET_PRIORITY: self._execute_set_priority,
            ActionType.SCHEDULE_COMPENSATING: self._execute_schedule_compensating,
            ActionType.CANCEL_TASKS: self._execute_cancel_tasks,
            ActionType.EXTEND_TTL: self._execute_extend_ttl,
            ActionType.REDUCE_TTL: self._execute_reduce_ttl,
            ActionType.TENANT_THROTTLE: self._execute_tenant_throttle,
            ActionType.TENANT_PAUSE: self._execute_tenant_pause,
            ActionType.ACCELERATE_HEALTH_CHECK: self._execute_accelerate_health_check,
            ActionType.TRIGGER_HEALTH_CHECK: self._execute_trigger_health_check,
            ActionType.CUSTOM: self._execute_custom,
        }

    def execute_rule(
        self,
        rule: TemporalPolicyRule,
        context: dict[str, Any],
    ) -> list[ReflexOutcome]:
        """
        Execute all actions for a triggered rule.

        Args:
            rule: The triggered rule
            context: Current system context

        Returns:
            List of ReflexOutcome for each action
        """
        outcomes = []

        for action in rule.actions:
            try:
                outcome = self.execute_action(rule, action, context)
                outcomes.append(outcome)
            except Exception as e:
                logger.error(
                    f"[TEMPORAL REFLEX] Action execution failed: {e}",
                    extra={
                        "rule_id": rule.rule_id,
                        "action_type": action.action_type.value,
                    },
                    exc_info=True,
                )
                # Create failed outcome
                failed_outcome = ReflexOutcome(
                    rule_id=rule.rule_id,
                    action_type=action.action_type,
                    status=ReflexStatus.FAILED,
                    error_message=str(e),
                    started_at=timezone.now(),
                    completed_at=timezone.now(),
                )
                outcomes.append(failed_outcome)

        # Record rule activation
        rule.record_activation()

        return outcomes

    def execute_action(
        self,
        rule: TemporalPolicyRule,
        action: TemporalAction,
        context: dict[str, Any],
    ) -> ReflexOutcome:
        """
        Execute a single temporal action.

        Args:
            rule: The parent rule
            action: The action to execute
            context: Current system context

        Returns:
            ReflexOutcome describing what happened
        """
        outcome = ReflexOutcome(
            rule_id=rule.rule_id,
            action_type=action.action_type,
            status=ReflexStatus.EXECUTING,
            started_at=timezone.now(),
        )

        # Calculate expiration
        if action.duration_minutes:
            outcome.expires_at = timezone.now() + timedelta(
                minutes=action.duration_minutes
            )

        try:
            # Get executor for action type
            executor = self._executors.get(action.action_type)
            if not executor:
                raise ValueError(f"No executor for action type: {action.action_type}")

            # Execute
            executor(action, outcome, context)

            outcome.status = ReflexStatus.COMPLETED
            outcome.completed_at = timezone.now()

            logger.info(
                f"[TEMPORAL REFLEX] Action executed: {action.action_type.value}",
                extra={
                    "rule_id": rule.rule_id,
                    "outcome_id": outcome.outcome_id,
                    "affected_schedules": len(outcome.affected_schedules),
                    "affected_tasks": len(outcome.affected_tasks),
                    "expires_at": (
                        outcome.expires_at.isoformat() if outcome.expires_at else None
                    ),
                },
            )

        except Exception as e:
            outcome.status = ReflexStatus.FAILED
            outcome.error_message = str(e)
            outcome.completed_at = timezone.now()
            logger.error(f"[TEMPORAL REFLEX] Action failed: {e}")

        # Track outcome
        self._state.add_outcome(outcome)

        return outcome

    def process_expirations(self) -> list[ReflexOutcome]:
        """
        Process expired outcomes and revert their modifications.

        Returns:
            List of reverted outcomes
        """
        expired = self._state.get_expired_outcomes()
        reverted = []

        for outcome in expired:
            try:
                self._revert_outcome(outcome)
                outcome.status = ReflexStatus.REVERTED
                reverted.append(outcome)
                self._state.remove_outcome(outcome.outcome_id)

                logger.info(
                    f"[TEMPORAL REFLEX] Outcome expired and reverted: {outcome.outcome_id}",
                    extra={
                        "rule_id": outcome.rule_id,
                        "action_type": outcome.action_type.value,
                    },
                )
            except Exception as e:
                logger.error(f"[TEMPORAL REFLEX] Failed to revert outcome: {e}")

        return reverted

    def _revert_outcome(self, outcome: ReflexOutcome) -> None:
        """Revert the modifications made by an outcome."""
        action_type = outcome.action_type

        if action_type == ActionType.PAUSE_SCHEDULE:
            for schedule_name in outcome.affected_schedules:
                self._state.paused_schedules.discard(schedule_name)

        elif action_type == ActionType.ACCELERATE_SCHEDULE:
            for schedule_name in outcome.affected_schedules:
                self._state.accelerated_schedules.pop(schedule_name, None)

        elif action_type == ActionType.DELAY_SCHEDULE:
            for schedule_name in outcome.affected_schedules:
                self._state.delayed_schedules.pop(schedule_name, None)

        elif action_type == ActionType.DEGRADE_PRIORITY:
            for pattern in outcome.priority_adjustments.keys():
                self._state.priority_degradations.pop(pattern, None)

        elif action_type == ActionType.BOOST_PRIORITY:
            for pattern in outcome.priority_adjustments.keys():
                self._state.priority_boosts.pop(pattern, None)

        elif action_type == ActionType.EXTEND_TTL:
            for pattern in outcome.ttl_adjustments.keys():
                self._state.ttl_extensions.pop(pattern, None)

        elif action_type == ActionType.REDUCE_TTL:
            for pattern in outcome.ttl_adjustments.keys():
                self._state.ttl_reductions.pop(pattern, None)

        elif action_type == ActionType.TENANT_PAUSE:
            for (
                tenant_id
            ) in outcome.affected_tasks:  # Reusing affected_tasks for tenant IDs
                self._state.paused_tenants.discard(tenant_id)

        elif action_type == ActionType.TENANT_THROTTLE:
            for tenant_id in outcome.affected_tasks:
                self._state.throttled_tenants.pop(tenant_id, None)

        elif action_type == ActionType.ACCELERATE_HEALTH_CHECK:
            for pattern in outcome.interval_adjustments.keys():
                self._state.accelerated_health_checks.pop(pattern, None)

    # =========================================================================
    # Action Executors
    # =========================================================================

    def _execute_pause_schedule(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Pause matching schedules."""
        schedules = self._get_matching_schedules(action.schedule_pattern, context)

        for schedule_name in schedules:
            self._state.paused_schedules.add(schedule_name)
            outcome.affected_schedules.append(schedule_name)

            # Store original state for reverting
            outcome.original_values[f"schedule:{schedule_name}:paused"] = False

    def _execute_resume_schedule(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Resume matching schedules."""
        schedules = self._get_matching_schedules(action.schedule_pattern, context)

        for schedule_name in schedules:
            was_paused = schedule_name in self._state.paused_schedules
            self._state.paused_schedules.discard(schedule_name)
            outcome.affected_schedules.append(schedule_name)
            outcome.original_values[f"schedule:{schedule_name}:paused"] = was_paused

    def _execute_delay_schedule(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Delay matching schedules (increase interval)."""
        schedules = self._get_matching_schedules(action.schedule_pattern, context)

        for schedule_name in schedules:
            multiplier = action.interval_multiplier
            self._state.delayed_schedules[schedule_name] = multiplier
            outcome.affected_schedules.append(schedule_name)
            outcome.interval_adjustments[schedule_name] = multiplier

    def _execute_accelerate_schedule(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Accelerate matching schedules (decrease interval)."""
        schedules = self._get_matching_schedules(action.schedule_pattern, context)

        for schedule_name in schedules:
            multiplier = action.interval_multiplier
            self._state.accelerated_schedules[schedule_name] = multiplier
            outcome.affected_schedules.append(schedule_name)
            outcome.interval_adjustments[schedule_name] = multiplier

    def _execute_degrade_priority(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Degrade priority for matching tasks."""
        pattern = action.task_pattern or "*"
        delta = action.priority_delta

        self._state.priority_degradations[pattern] = delta
        outcome.priority_adjustments[pattern] = delta

    def _execute_boost_priority(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Boost priority for matching tasks."""
        pattern = action.task_pattern or "*"
        delta = action.priority_delta

        self._state.priority_boosts[pattern] = delta
        outcome.priority_adjustments[pattern] = -delta  # Negative = higher priority

    def _execute_set_priority(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Set absolute priority for matching tasks."""
        # This requires tracking absolute priorities differently
        # For now, treat as a boost/degrade from normal (2)
        target_priority = int(action.value or 2)
        delta = 2 - target_priority  # 2 is NORMAL

        if delta > 0:
            self._execute_boost_priority(action, outcome, context)
        elif delta < 0:
            action.priority_delta = abs(delta)
            self._execute_degrade_priority(action, outcome, context)

    def _execute_schedule_compensating(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Schedule a compensating task."""
        if not action.task_name:
            raise ValueError("Compensating task action requires task_name")

        # Build compensating task
        comp_task = CompensatingTask(
            task_name=action.task_name,
            payload=action.task_payload or {},
            priority=int(action.value) if action.value else 2,
            triggered_by_rule=outcome.rule_id,
        )

        # Add context to payload
        comp_task.payload["_triggered_by"] = outcome.rule_id
        comp_task.payload["_triggered_at"] = timezone.now().isoformat()
        comp_task.payload["_trigger_context"] = {
            k: v
            for k, v in context.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }

        outcome.compensating_tasks.append(comp_task)

        # Submit if callback is set
        if self._task_submitter:
            try:
                self._task_submitter(comp_task)
                logger.info(
                    f"[TEMPORAL REFLEX] Compensating task scheduled: {action.task_name}",
                    extra={"task_name": action.task_name, "rule_id": outcome.rule_id},
                )
            except Exception as e:
                logger.error(
                    f"[TEMPORAL REFLEX] Failed to submit compensating task: {e}"
                )

    def _execute_cancel_tasks(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Cancel matching pending tasks."""
        # This would need integration with the scheduler to cancel tasks
        # For now, just log the intent
        pattern = action.task_pattern or "*"
        logger.info(
            f"[TEMPORAL REFLEX] Cancel tasks requested for pattern: {pattern}",
            extra={"pattern": pattern},
        )
        outcome.affected_tasks.append(f"pattern:{pattern}")

    def _execute_extend_ttl(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Extend TTL for matching tasks."""
        pattern = action.task_pattern or "*"
        multiplier = action.ttl_multiplier

        self._state.ttl_extensions[pattern] = multiplier
        outcome.ttl_adjustments[pattern] = multiplier

    def _execute_reduce_ttl(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Reduce TTL for matching tasks."""
        pattern = action.task_pattern or "*"
        multiplier = action.ttl_multiplier

        self._state.ttl_reductions[pattern] = multiplier
        outcome.ttl_adjustments[pattern] = multiplier

    def _execute_tenant_throttle(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Apply throttling to a tenant."""
        tenant_id = context.get("tenant_id")
        if not tenant_id:
            logger.warning(
                "[TEMPORAL REFLEX] Tenant throttle requested but no tenant_id in context"
            )
            return

        throttle_level = action.value or 0.5
        self._state.throttled_tenants[str(tenant_id)] = throttle_level
        outcome.affected_tasks.append(str(tenant_id))

    def _execute_tenant_pause(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Pause all tasks for a tenant."""
        tenant_id = context.get("tenant_id")
        if not tenant_id:
            logger.warning(
                "[TEMPORAL REFLEX] Tenant pause requested but no tenant_id in context"
            )
            return

        self._state.paused_tenants.add(str(tenant_id))
        outcome.affected_tasks.append(str(tenant_id))

    def _execute_accelerate_health_check(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Accelerate health check schedules."""
        pattern = action.schedule_pattern or "health_*"
        multiplier = action.interval_multiplier

        self._state.accelerated_health_checks[pattern] = multiplier
        outcome.interval_adjustments[pattern] = multiplier

        # Also apply to schedules
        schedules = self._get_matching_schedules(pattern, context)
        for schedule_name in schedules:
            self._state.accelerated_schedules[schedule_name] = multiplier
            outcome.affected_schedules.append(schedule_name)

    def _execute_trigger_health_check(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Trigger an immediate health check."""
        # Schedule a health check task
        health_task = CompensatingTask(
            task_name="synara.health.check",
            payload={"triggered_by": "temporal_reflex"},
            priority=0,  # CRITICAL
            triggered_by_rule=outcome.rule_id,
        )

        outcome.compensating_tasks.append(health_task)

        if self._task_submitter:
            try:
                self._task_submitter(health_task)
                logger.info("[TEMPORAL REFLEX] Health check triggered")
            except Exception as e:
                logger.error(f"[TEMPORAL REFLEX] Failed to trigger health check: {e}")

    def _execute_custom(
        self,
        action: TemporalAction,
        outcome: ReflexOutcome,
        context: dict[str, Any],
    ) -> None:
        """Execute custom action via executor callback."""
        if action.custom_executor:
            action.custom_executor(context)
            logger.info("[TEMPORAL REFLEX] Custom action executed")
        else:
            logger.warning(
                "[TEMPORAL REFLEX] Custom action requested but no executor provided"
            )

    # =========================================================================
    # Query Methods (for scheduler integration)
    # =========================================================================

    def is_schedule_paused(self, schedule_name: str) -> bool:
        """Check if a schedule is currently paused by temporal policy."""
        return self._state.is_schedule_paused(schedule_name)

    def get_schedule_interval_multiplier(self, schedule_name: str) -> float:
        """Get interval multiplier for a schedule (< 1 = faster, > 1 = slower)."""
        return self._state.get_schedule_interval_multiplier(schedule_name)

    def get_priority_adjustment(self, task_name: str) -> int:
        """Get priority adjustment for a task type (positive = lower priority)."""
        return self._state.get_priority_adjustment(task_name)

    def get_ttl_multiplier(self, task_name: str) -> float:
        """Get TTL multiplier for a task type."""
        return self._state.get_ttl_multiplier(task_name)

    def is_tenant_paused(self, tenant_id: str) -> bool:
        """Check if a tenant is paused by temporal policy."""
        return self._state.is_tenant_paused(tenant_id)

    def get_tenant_throttle(self, tenant_id: str) -> float:
        """Get throttle level for a tenant."""
        return self._state.get_tenant_throttle(tenant_id)

    def get_state(self) -> ReflexState:
        """Get current reflex state."""
        return self._state

    def get_active_outcomes(self) -> list[ReflexOutcome]:
        """Get list of active (non-expired) outcomes."""
        return list(self._state.active_outcomes.values())

    def get_outcome_history(self, limit: int = 50) -> list[ReflexOutcome]:
        """Get recent outcome history."""
        return self._state.outcome_history[-limit:]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_matching_schedules(
        self,
        pattern: str | None,
        context: dict[str, Any],
    ) -> list[str]:
        """Get schedule names matching a pattern."""
        import fnmatch

        # Get available schedules from context or fetch
        available = context.get("available_schedules", [])

        if not pattern:
            return list(available)

        return [s for s in available if fnmatch.fnmatch(s, pattern)]

    def set_task_submitter(self, submitter: Callable[[CompensatingTask], Any]) -> None:
        """Set the callback for submitting compensating tasks."""
        self._task_submitter = submitter

    def set_schedule_modifier(self, modifier: Callable[[str, str, Any], bool]) -> None:
        """Set the callback for modifying schedules."""
        self._schedule_modifier = modifier
