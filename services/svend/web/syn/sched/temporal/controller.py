"""
Temporal Controller - Adaptive Governance-to-Timing Loop

Standard: SCH-006 §4 (Temporal Controller)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

The TemporalController runs the adaptive loop that connects:
- Governance Layer: Risk assessment, confidence scoring
- Scheduler Layer: Schedules, task execution, backpressure
- Cognition Layer: Pattern recognition, anomaly detection

This is the "brain" of Synara's temporal cortex - continuously evaluating
conditions and triggering reflexes.

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                    ADAPTIVE CONTROL LOOP                            │
    │                                                                     │
    │   ┌───────────────┐         ┌──────────────────┐                   │
    │   │   Context     │────────▶│ TemporalPolicy   │                   │
    │   │   Collector   │         │ Evaluator        │                   │
    │   └───────────────┘         └────────┬─────────┘                   │
    │          ▲                           │                             │
    │          │                           ▼                             │
    │   ┌──────┴────────┐         ┌──────────────────┐                   │
    │   │   Feedback    │◀────────│ TemporalReflex   │                   │
    │   │   Loop        │         │ Executor         │                   │
    │   └───────────────┘         └──────────────────┘                   │
    │                                                                     │
    │   Sources:                  Outputs:                               │
    │   - BackpressureController  - Schedule pauses                      │
    │   - SystemHealthMonitor     - Priority adjustments                 │
    │   - GovernanceEngine        - Compensating tasks                   │
    │   - DashboardService        - TTL modifications                    │
    │   - CircuitBreakerState     - Health check acceleration            │
    │                                                                     │
    └────────────────────────────────────────────────────────────────────┘

NO OTHER SYSTEM DOES THIS.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from django.utils import timezone

from .policy import TemporalPolicy, TemporalPolicyRule
from .reflex import CompensatingTask, ReflexOutcome, TemporalReflex

if TYPE_CHECKING:
    from syn.sched.backpressure import BackpressureController

logger = logging.getLogger(__name__)


@dataclass
class TemporalControllerConfig:
    """
    Configuration for TemporalController.

    Standard: SCH-006 §4.1
    """

    # Evaluation frequency
    evaluation_interval_seconds: float = 5.0
    expiration_check_interval_seconds: float = 30.0

    # Context collection
    enable_governance_context: bool = True
    enable_scheduler_context: bool = True
    enable_backpressure_context: bool = True
    enable_circuit_context: bool = True
    enable_dlq_context: bool = True

    # Limits
    max_rules_per_cycle: int = 10
    max_compensating_tasks_per_cycle: int = 5

    # Callbacks
    on_rule_activated: Callable[[TemporalPolicyRule, dict], None] | None = None
    on_outcome_completed: Callable[[ReflexOutcome], None] | None = None
    on_error: Callable[[Exception], None] | None = None

    # Integration callbacks
    task_submitter: Callable[[CompensatingTask], Any] | None = None
    context_provider: Callable[[], dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_interval_seconds": self.evaluation_interval_seconds,
            "expiration_check_interval_seconds": self.expiration_check_interval_seconds,
            "enable_governance_context": self.enable_governance_context,
            "enable_scheduler_context": self.enable_scheduler_context,
            "enable_backpressure_context": self.enable_backpressure_context,
            "enable_circuit_context": self.enable_circuit_context,
            "enable_dlq_context": self.enable_dlq_context,
            "max_rules_per_cycle": self.max_rules_per_cycle,
            "max_compensating_tasks_per_cycle": self.max_compensating_tasks_per_cycle,
        }


@dataclass
class TemporalMetrics:
    """
    Metrics for temporal controller monitoring.

    Standard: SCH-006 §4.2
    """

    # Cycle counts
    evaluation_cycles: int = 0
    last_evaluation: datetime | None = None

    # Rule metrics
    rules_evaluated: int = 0
    rules_activated: int = 0
    rules_by_trigger: dict[str, int] = field(default_factory=dict)

    # Outcome metrics
    outcomes_completed: int = 0
    outcomes_failed: int = 0
    outcomes_reverted: int = 0

    # Compensating task metrics
    compensating_tasks_scheduled: int = 0
    compensating_tasks_by_type: dict[str, int] = field(default_factory=dict)

    # Active state
    active_outcomes: int = 0
    paused_schedules: int = 0
    priority_adjustments: int = 0
    ttl_adjustments: int = 0

    # Timing
    avg_evaluation_ms: float = 0.0
    max_evaluation_ms: float = 0.0
    total_evaluation_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_cycles": self.evaluation_cycles,
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
            "rules_evaluated": self.rules_evaluated,
            "rules_activated": self.rules_activated,
            "rules_by_trigger": self.rules_by_trigger,
            "outcomes_completed": self.outcomes_completed,
            "outcomes_failed": self.outcomes_failed,
            "outcomes_reverted": self.outcomes_reverted,
            "compensating_tasks_scheduled": self.compensating_tasks_scheduled,
            "compensating_tasks_by_type": self.compensating_tasks_by_type,
            "active_outcomes": self.active_outcomes,
            "paused_schedules": self.paused_schedules,
            "priority_adjustments": self.priority_adjustments,
            "ttl_adjustments": self.ttl_adjustments,
            "avg_evaluation_ms": round(self.avg_evaluation_ms, 2),
            "max_evaluation_ms": round(self.max_evaluation_ms, 2),
        }


class TemporalController:
    """
    Runs the adaptive governance-to-timing control loop.

    Standard: SCH-006 §4

    The TemporalController continuously:
    1. Collects context from governance, scheduler, backpressure
    2. Evaluates temporal policies against current context
    3. Executes triggered actions via TemporalReflex
    4. Tracks metrics and provides state queries
    5. Processes expired outcomes for reversion

    This creates a feedback loop where system state drives timing decisions,
    and timing decisions affect system state.

    Usage:
        policy = TemporalPolicy()
        reflex = TemporalReflex()
        config = TemporalControllerConfig(
            task_submitter=scheduler.submit_compensating,
        )

        controller = TemporalController(
            policy=policy,
            reflex=reflex,
            config=config,
        )

        # Set context sources
        controller.set_backpressure(backpressure_controller)
        controller.set_governance(governance_engine)

        # Start the loop
        controller.start()

        # Query state
        if controller.is_schedule_paused("maintenance"):
            pass

        # Stop
        controller.stop()

    Integration with CognitiveScheduler:
        scheduler.set_temporal_controller(controller)
    """

    def __init__(
        self,
        policy: TemporalPolicy | None = None,
        reflex: TemporalReflex | None = None,
        config: TemporalControllerConfig | None = None,
    ):
        """
        Initialize the temporal controller.

        Args:
            policy: Temporal policy for rule evaluation
            reflex: Temporal reflex for action execution
            config: Controller configuration
        """
        self._config = config or TemporalControllerConfig()
        self._policy = policy or TemporalPolicy()
        self._reflex = reflex or TemporalReflex()

        # Set up reflex task submitter
        if self._config.task_submitter:
            self._reflex.set_task_submitter(self._config.task_submitter)

        # Context sources
        self._backpressure: BackpressureController | None = None
        self._governance_provider: Callable[[], dict[str, Any]] | None = None
        self._scheduler_provider: Callable[[], dict[str, Any]] | None = None
        self._custom_context_provider: Callable[[], dict[str, Any]] | None = None

        # State
        self._running = False
        self._lock = threading.RLock()
        self._eval_thread: threading.Thread | None = None
        self._expiration_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Metrics
        self._metrics = TemporalMetrics()

        # Last context for debugging
        self._last_context: dict[str, Any] = {}
        self._last_activated_rules: list[TemporalPolicyRule] = []

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the temporal controller loop."""
        if self._running:
            logger.warning("[TEMPORAL CONTROLLER] Already running")
            return

        self._running = True
        self._shutdown_event.clear()

        # Start evaluation loop
        self._eval_thread = threading.Thread(
            target=self._evaluation_loop,
            daemon=True,
            name="temporal-eval",
        )
        self._eval_thread.start()

        # Start expiration checker
        self._expiration_thread = threading.Thread(
            target=self._expiration_loop,
            daemon=True,
            name="temporal-expiry",
        )
        self._expiration_thread.start()

        logger.info("[TEMPORAL CONTROLLER] Started adaptive control loop")

    def stop(self) -> None:
        """Stop the temporal controller loop."""
        if not self._running:
            return

        logger.info("[TEMPORAL CONTROLLER] Stopping...")
        self._running = False
        self._shutdown_event.set()

        # Wait for threads
        if self._eval_thread:
            self._eval_thread.join(timeout=5.0)
        if self._expiration_thread:
            self._expiration_thread.join(timeout=5.0)

        logger.info("[TEMPORAL CONTROLLER] Stopped")

    @property
    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    # =========================================================================
    # Context Source Configuration
    # =========================================================================

    def set_backpressure(self, controller: BackpressureController) -> None:
        """Set the backpressure controller for context."""
        self._backpressure = controller

    def set_governance_provider(self, provider: Callable[[], dict[str, Any]]) -> None:
        """Set callback to get governance context."""
        self._governance_provider = provider

    def set_scheduler_provider(self, provider: Callable[[], dict[str, Any]]) -> None:
        """Set callback to get scheduler context."""
        self._scheduler_provider = provider

    def set_custom_context_provider(self, provider: Callable[[], dict[str, Any]]) -> None:
        """Set callback to provide additional context."""
        self._custom_context_provider = provider

    def set_task_submitter(self, submitter: Callable[[CompensatingTask], Any]) -> None:
        """Set the callback for submitting compensating tasks."""
        self._config.task_submitter = submitter
        self._reflex.set_task_submitter(submitter)

    # =========================================================================
    # Control Loop
    # =========================================================================

    def _evaluation_loop(self) -> None:
        """Main evaluation loop."""
        while self._running:
            try:
                start_time = time.time()

                # Collect context
                context = self._collect_context()

                # Evaluate and execute
                with self._lock:
                    activated_rules = self._policy.evaluate(context)
                    self._last_context = context
                    self._last_activated_rules = activated_rules

                    # Execute actions for activated rules (limited)
                    rules_executed = 0
                    compensating_count = 0

                    for rule in activated_rules:
                        if rules_executed >= self._config.max_rules_per_cycle:
                            logger.warning(
                                f"[TEMPORAL CONTROLLER] Rule limit reached ({rules_executed}), "
                                f"skipping remaining {len(activated_rules) - rules_executed} rules"
                            )
                            break

                        outcomes = self._reflex.execute_rule(rule, context)
                        rules_executed += 1

                        # Track metrics
                        self._metrics.rules_activated += 1
                        trigger_type = rule.trigger.trigger_type.value
                        self._metrics.rules_by_trigger[trigger_type] = (
                            self._metrics.rules_by_trigger.get(trigger_type, 0) + 1
                        )

                        # Process outcomes
                        for outcome in outcomes:
                            if outcome.status.value == "completed":
                                self._metrics.outcomes_completed += 1
                            else:
                                self._metrics.outcomes_failed += 1

                            # Count compensating tasks
                            for comp_task in outcome.compensating_tasks:
                                compensating_count += 1
                                task_type = comp_task.task_name
                                self._metrics.compensating_tasks_by_type[task_type] = (
                                    self._metrics.compensating_tasks_by_type.get(task_type, 0) + 1
                                )

                            # Callback
                            if self._config.on_outcome_completed:
                                try:
                                    self._config.on_outcome_completed(outcome)
                                except Exception as e:
                                    logger.error(f"[TEMPORAL CONTROLLER] Outcome callback failed: {e}")

                        # Rule callback
                        if self._config.on_rule_activated:
                            try:
                                self._config.on_rule_activated(rule, context)
                            except Exception as e:
                                logger.error(f"[TEMPORAL CONTROLLER] Rule callback failed: {e}")

                        # Check compensating task limit
                        if compensating_count >= self._config.max_compensating_tasks_per_cycle:
                            break

                    self._metrics.compensating_tasks_scheduled += compensating_count

                # Update metrics
                elapsed_ms = (time.time() - start_time) * 1000
                self._update_timing_metrics(elapsed_ms)
                self._update_state_metrics()

            except Exception as e:
                logger.error(f"[TEMPORAL CONTROLLER] Evaluation error: {e}", exc_info=True)
                if self._config.on_error:
                    try:
                        self._config.on_error(e)
                    except Exception:
                        pass

            # Wait for next cycle
            self._shutdown_event.wait(timeout=self._config.evaluation_interval_seconds)

    def _expiration_loop(self) -> None:
        """Loop to process expired outcomes."""
        while self._running:
            try:
                with self._lock:
                    reverted = self._reflex.process_expirations()
                    self._metrics.outcomes_reverted += len(reverted)

                    for outcome in reverted:
                        logger.debug(
                            f"[TEMPORAL CONTROLLER] Outcome reverted: {outcome.outcome_id}",
                            extra={"rule_id": outcome.rule_id},
                        )
            except Exception as e:
                logger.error(f"[TEMPORAL CONTROLLER] Expiration processing error: {e}")

            self._shutdown_event.wait(timeout=self._config.expiration_check_interval_seconds)

    # =========================================================================
    # Context Collection
    # =========================================================================

    def _collect_context(self) -> dict[str, Any]:
        """Collect context from all sources."""
        context: dict[str, Any] = {
            "timestamp": timezone.now().isoformat(),
        }

        # Backpressure context
        if self._config.enable_backpressure_context and self._backpressure:
            try:
                health = self._backpressure.get_health_metrics()
                bp_metrics = self._backpressure.get_backpressure_metrics()

                context.update(
                    {
                        # Queue metrics
                        "queue_depth": health.queue_depth,
                        "queue_utilization": health.queue_utilization,
                        # Throttle metrics
                        "throttle_level": bp_metrics.current_level.name,
                        "throttle_level_value": bp_metrics.current_level.value,
                        "confidence_penalty": bp_metrics.current_confidence_penalty,
                        "is_emergency": self._backpressure.is_emergency(),
                        # Health status
                        "health_status": health.status.value,
                        "running_tasks": health.running_tasks,
                        "worker_count": health.worker_count,
                        "worker_utilization": health.worker_utilization,
                        # DLQ metrics
                        "dlq_size": health.dlq_pending_count,
                        "dlq_growth_rate": health.dlq_growth_rate,
                        # Circuit metrics
                        "circuits_open": health.circuit_breakers_open,
                        "circuits_total": health.circuit_breakers_total,
                        "circuit_open_ratio": health.circuit_open_ratio,
                    }
                )
            except Exception as e:
                logger.warning(f"[TEMPORAL CONTROLLER] Failed to collect backpressure context: {e}")

        # Governance context
        if self._config.enable_governance_context and self._governance_provider:
            try:
                gov_context = self._governance_provider()
                context.update(
                    {
                        "governance_risk": gov_context.get("risk_score", 0.0),
                        "governance_blocked_rate": gov_context.get("blocked_rate", 0.0),
                        "governance_escalation_rate": gov_context.get("escalation_rate", 0.0),
                        "confidence_score": gov_context.get("confidence_score", 1.0),
                        "incident_active": 1 if gov_context.get("incident_active", False) else 0,
                        "incident_severity": gov_context.get("incident_severity", 0),
                    }
                )
            except Exception as e:
                logger.warning(f"[TEMPORAL CONTROLLER] Failed to collect governance context: {e}")

        # Scheduler context
        if self._config.enable_scheduler_context and self._scheduler_provider:
            try:
                sched_context = self._scheduler_provider()
                context.update(
                    {
                        "available_schedules": sched_context.get("schedules", []),
                        "pending_tasks": sched_context.get("pending_tasks", 0),
                        "running_tasks": sched_context.get("running_tasks", context.get("running_tasks", 0)),
                        "cascade_depth": sched_context.get("max_cascade_depth", 0),
                        "cascade_budget_remaining": sched_context.get("cascade_budget_remaining", 1.0),
                    }
                )
            except Exception as e:
                logger.warning(f"[TEMPORAL CONTROLLER] Failed to collect scheduler context: {e}")

        # Custom context
        if self._custom_context_provider:
            try:
                custom = self._custom_context_provider()
                context.update(custom)
            except Exception as e:
                logger.warning(f"[TEMPORAL CONTROLLER] Failed to collect custom context: {e}")

        # Config context provider
        if self._config.context_provider:
            try:
                config_context = self._config.context_provider()
                context.update(config_context)
            except Exception as e:
                logger.warning(f"[TEMPORAL CONTROLLER] Failed to collect config context: {e}")

        return context

    # =========================================================================
    # Query Methods (for scheduler integration)
    # =========================================================================

    def is_schedule_paused(self, schedule_name: str) -> bool:
        """Check if a schedule is paused by temporal policy."""
        return self._reflex.is_schedule_paused(schedule_name)

    def get_schedule_interval_multiplier(self, schedule_name: str) -> float:
        """Get interval multiplier for a schedule."""
        return self._reflex.get_schedule_interval_multiplier(schedule_name)

    def get_priority_adjustment(self, task_name: str) -> int:
        """Get priority adjustment for a task."""
        return self._reflex.get_priority_adjustment(task_name)

    def get_ttl_multiplier(self, task_name: str) -> float:
        """Get TTL multiplier for a task."""
        return self._reflex.get_ttl_multiplier(task_name)

    def is_tenant_paused(self, tenant_id: str) -> bool:
        """Check if a tenant is paused."""
        return self._reflex.is_tenant_paused(tenant_id)

    def get_tenant_throttle(self, tenant_id: str) -> float:
        """Get tenant throttle level."""
        return self._reflex.get_tenant_throttle(tenant_id)

    # =========================================================================
    # Manual Evaluation (for testing/debugging)
    # =========================================================================

    def evaluate_now(self, context: dict[str, Any] | None = None) -> list[ReflexOutcome]:
        """
        Force immediate policy evaluation.

        Args:
            context: Optional context override

        Returns:
            List of outcomes from triggered rules
        """
        if context is None:
            context = self._collect_context()

        outcomes = []

        with self._lock:
            activated_rules = self._policy.evaluate(context)

            for rule in activated_rules:
                rule_outcomes = self._reflex.execute_rule(rule, context)
                outcomes.extend(rule_outcomes)

        return outcomes

    def simulate(self, context: dict[str, Any]) -> list[TemporalPolicyRule]:
        """
        Simulate policy evaluation without executing actions.

        Args:
            context: Context to evaluate against

        Returns:
            List of rules that would be activated
        """
        return self._policy.evaluate(context)

    # =========================================================================
    # Metrics and State
    # =========================================================================

    def _update_timing_metrics(self, elapsed_ms: float) -> None:
        """Update timing metrics."""
        self._metrics.evaluation_cycles += 1
        self._metrics.last_evaluation = timezone.now()
        self._metrics.total_evaluation_ms += elapsed_ms
        self._metrics.max_evaluation_ms = max(self._metrics.max_evaluation_ms, elapsed_ms)
        self._metrics.avg_evaluation_ms = self._metrics.total_evaluation_ms / self._metrics.evaluation_cycles

    def _update_state_metrics(self) -> None:
        """Update state metrics."""
        state = self._reflex.get_state()
        self._metrics.active_outcomes = len(state.active_outcomes)
        self._metrics.paused_schedules = len(state.paused_schedules)
        self._metrics.priority_adjustments = len(state.priority_degradations) + len(state.priority_boosts)
        self._metrics.ttl_adjustments = len(state.ttl_extensions) + len(state.ttl_reductions)
        self._metrics.rules_evaluated = len(self._policy.get_rules())

    def get_metrics(self) -> TemporalMetrics:
        """Get controller metrics."""
        self._update_state_metrics()
        return TemporalMetrics(
            evaluation_cycles=self._metrics.evaluation_cycles,
            last_evaluation=self._metrics.last_evaluation,
            rules_evaluated=self._metrics.rules_evaluated,
            rules_activated=self._metrics.rules_activated,
            rules_by_trigger=dict(self._metrics.rules_by_trigger),
            outcomes_completed=self._metrics.outcomes_completed,
            outcomes_failed=self._metrics.outcomes_failed,
            outcomes_reverted=self._metrics.outcomes_reverted,
            compensating_tasks_scheduled=self._metrics.compensating_tasks_scheduled,
            compensating_tasks_by_type=dict(self._metrics.compensating_tasks_by_type),
            active_outcomes=self._metrics.active_outcomes,
            paused_schedules=self._metrics.paused_schedules,
            priority_adjustments=self._metrics.priority_adjustments,
            ttl_adjustments=self._metrics.ttl_adjustments,
            avg_evaluation_ms=self._metrics.avg_evaluation_ms,
            max_evaluation_ms=self._metrics.max_evaluation_ms,
            total_evaluation_ms=self._metrics.total_evaluation_ms,
        )

    def get_reflex_state(self) -> dict[str, Any]:
        """Get current reflex state."""
        return self._reflex.get_state().to_dict()

    def get_last_context(self) -> dict[str, Any]:
        """Get the last collected context (for debugging)."""
        return dict(self._last_context)

    def get_last_activated_rules(self) -> list[dict[str, Any]]:
        """Get the last activated rules (for debugging)."""
        return [r.to_dict() for r in self._last_activated_rules]

    def get_active_outcomes(self) -> list[dict[str, Any]]:
        """Get currently active outcomes."""
        return [o.to_dict() for o in self._reflex.get_active_outcomes()]

    def get_outcome_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent outcome history."""
        return [o.to_dict() for o in self._reflex.get_outcome_history(limit)]

    # =========================================================================
    # Policy Management
    # =========================================================================

    def add_rule(self, rule: TemporalPolicyRule) -> None:
        """Add a rule to the policy."""
        self._policy.add_rule(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the policy."""
        return self._policy.remove_rule(rule_id)

    def set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        """Enable or disable a rule."""
        return self._policy.set_rule_enabled(rule_id, enabled)

    def get_rules(self) -> list[dict[str, Any]]:
        """Get all policy rules."""
        return [r.to_dict() for r in self._policy.get_rules()]

    def get_rule(self, rule_id: str) -> dict[str, Any] | None:
        """Get a specific rule by ID."""
        rule = self._policy.get_rule(rule_id)
        return rule.to_dict() if rule else None

    @property
    def policy(self) -> TemporalPolicy:
        """Get the temporal policy."""
        return self._policy

    @property
    def reflex(self) -> TemporalReflex:
        """Get the temporal reflex engine."""
        return self._reflex

    @property
    def config(self) -> TemporalControllerConfig:
        """Get controller configuration."""
        return self._config
