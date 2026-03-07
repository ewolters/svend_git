"""
Tests for syn.sched.types — pure-Python type definitions.

Standard: SCH-001 §3-4, SCH-002 §3-4
No database access required (SimpleTestCase).
"""

from datetime import datetime, timedelta

from django.test import SimpleTestCase

from syn.sched.types import (
    CASCADE_BUDGET,
    CronSchedule,
    RetryConfig,
    RetryStrategy,
    TaskContext,
    TaskState,
    get_cascade_limit,
)

# =============================================================================
# TaskState
# =============================================================================


class TaskStateTest(SimpleTestCase):
    """Tests for TaskState enum and its properties."""

    def test_terminal_states_success(self):
        self.assertTrue(TaskState.SUCCESS.is_terminal)

    def test_terminal_states_failure(self):
        self.assertTrue(TaskState.FAILURE.is_terminal)

    def test_terminal_states_cancelled(self):
        self.assertTrue(TaskState.CANCELLED.is_terminal)

    def test_terminal_states_dead_lettered(self):
        self.assertTrue(TaskState.DEAD_LETTERED.is_terminal)

    def test_non_terminal_states(self):
        non_terminal = [
            TaskState.PENDING,
            TaskState.SCHEDULED,
            TaskState.RUNNING,
            TaskState.RETRYING,
        ]
        for state in non_terminal:
            with self.subTest(state=state):
                self.assertFalse(state.is_terminal)

    def test_is_active_running(self):
        self.assertTrue(TaskState.RUNNING.is_active)

    def test_is_active_retrying(self):
        self.assertTrue(TaskState.RETRYING.is_active)

    def test_is_active_false_for_non_active(self):
        inactive = [
            TaskState.PENDING,
            TaskState.SCHEDULED,
            TaskState.SUCCESS,
            TaskState.FAILURE,
            TaskState.CANCELLED,
            TaskState.DEAD_LETTERED,
        ]
        for state in inactive:
            with self.subTest(state=state):
                self.assertFalse(state.is_active)

    def test_valid_transition_pending_to_scheduled(self):
        transitions = TaskState.valid_transitions()
        self.assertIn(TaskState.SCHEDULED, transitions[TaskState.PENDING])

    def test_valid_transition_pending_to_cancelled(self):
        transitions = TaskState.valid_transitions()
        self.assertIn(TaskState.CANCELLED, transitions[TaskState.PENDING])

    def test_valid_transition_running_to_success(self):
        transitions = TaskState.valid_transitions()
        self.assertIn(TaskState.SUCCESS, transitions[TaskState.RUNNING])

    def test_valid_transition_running_to_failure(self):
        transitions = TaskState.valid_transitions()
        self.assertIn(TaskState.FAILURE, transitions[TaskState.RUNNING])

    def test_valid_transition_running_to_retrying(self):
        transitions = TaskState.valid_transitions()
        self.assertIn(TaskState.RETRYING, transitions[TaskState.RUNNING])

    def test_valid_transition_retrying_to_dead_lettered(self):
        transitions = TaskState.valid_transitions()
        self.assertIn(TaskState.DEAD_LETTERED, transitions[TaskState.RETRYING])

    def test_invalid_transition_success_has_no_targets(self):
        transitions = TaskState.valid_transitions()
        self.assertEqual(transitions[TaskState.SUCCESS], [])

    def test_invalid_transition_dead_lettered_has_no_targets(self):
        transitions = TaskState.valid_transitions()
        self.assertEqual(transitions[TaskState.DEAD_LETTERED], [])

    def test_invalid_transition_pending_cannot_go_to_running(self):
        transitions = TaskState.valid_transitions()
        self.assertNotIn(TaskState.RUNNING, transitions[TaskState.PENDING])

    def test_all_states_present_in_transitions(self):
        transitions = TaskState.valid_transitions()
        for state in TaskState:
            with self.subTest(state=state):
                self.assertIn(state, transitions)

    def test_str_enum_values(self):
        self.assertEqual(TaskState.PENDING, "PENDING")
        self.assertEqual(TaskState.RUNNING, "RUNNING")


# =============================================================================
# RetryConfig
# =============================================================================


class RetryConfigTest(SimpleTestCase):
    """Tests for RetryConfig backoff calculations."""

    def test_default_values(self):
        cfg = RetryConfig()
        self.assertEqual(cfg.strategy, RetryStrategy.EXPONENTIAL)
        self.assertEqual(cfg.max_attempts, 5)
        self.assertEqual(cfg.base_delay_seconds, 1)
        self.assertEqual(cfg.max_delay_seconds, 3600)
        self.assertEqual(cfg.multiplier, 2.0)
        self.assertTrue(cfg.jitter)

    def test_exponential_backoff_doubles_without_jitter(self):
        cfg = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_seconds=1,
            multiplier=2.0,
            jitter=False,
        )
        # attempt 1: 1 * 2^0 = 1
        self.assertEqual(cfg.get_delay(1), timedelta(seconds=1))
        # attempt 2: 1 * 2^1 = 2
        self.assertEqual(cfg.get_delay(2), timedelta(seconds=2))
        # attempt 3: 1 * 2^2 = 4
        self.assertEqual(cfg.get_delay(3), timedelta(seconds=4))

    def test_exponential_caps_at_max_delay(self):
        cfg = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_seconds=100,
            max_delay_seconds=500,
            multiplier=2.0,
            jitter=False,
        )
        # attempt 3: 100 * 2^2 = 400 (under cap)
        self.assertEqual(cfg.get_delay(3), timedelta(seconds=400))
        # attempt 4: 100 * 2^3 = 800 -> capped to 500
        self.assertEqual(cfg.get_delay(4), timedelta(seconds=500))

    def test_exponential_jitter_varies_results(self):
        cfg = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_seconds=100,
            multiplier=2.0,
            jitter=True,
        )
        results = {cfg.get_delay(3) for _ in range(20)}
        # With jitter factor 0.5-1.5, we should get variation
        self.assertGreater(len(results), 1, "Jitter should produce varying delays")

    def test_fixed_delay_constant(self):
        cfg = RetryConfig(
            strategy=RetryStrategy.FIXED,
            base_delay_seconds=30,
        )
        self.assertEqual(cfg.get_delay(1), timedelta(seconds=30))
        self.assertEqual(cfg.get_delay(5), timedelta(seconds=30))
        self.assertEqual(cfg.get_delay(10), timedelta(seconds=30))

    def test_linear_backoff(self):
        cfg = RetryConfig(
            strategy=RetryStrategy.LINEAR,
            delay_increment_seconds=60,
            max_delay_seconds=300,
        )
        self.assertEqual(cfg.get_delay(1), timedelta(seconds=60))
        self.assertEqual(cfg.get_delay(3), timedelta(seconds=180))
        # attempt 6: 360 -> capped to 300
        self.assertEqual(cfg.get_delay(6), timedelta(seconds=300))

    def test_immediate_retry_zero_delay(self):
        cfg = RetryConfig(strategy=RetryStrategy.IMMEDIATE)
        self.assertEqual(cfg.get_delay(1), timedelta(seconds=0))
        self.assertEqual(cfg.get_delay(5), timedelta(seconds=0))

    def test_none_strategy_zero_delay(self):
        cfg = RetryConfig(strategy=RetryStrategy.NONE)
        self.assertEqual(cfg.get_delay(1), timedelta(seconds=0))


# =============================================================================
# CronSchedule
# =============================================================================


class CronScheduleTest(SimpleTestCase):
    """Tests for CronSchedule parsing and factory methods."""

    def test_from_expression_valid(self):
        sched = CronSchedule.from_expression("0 * * * *")
        self.assertEqual(sched.minute, "0")
        self.assertEqual(sched.hour, "*")
        self.assertEqual(sched.day_of_month, "*")
        self.assertEqual(sched.month, "*")
        self.assertEqual(sched.day_of_week, "*")

    def test_from_expression_all_fields(self):
        sched = CronSchedule.from_expression("30 14 1 6 3")
        self.assertEqual(sched.minute, "30")
        self.assertEqual(sched.hour, "14")
        self.assertEqual(sched.day_of_month, "1")
        self.assertEqual(sched.month, "6")
        self.assertEqual(sched.day_of_week, "3")

    def test_from_expression_invalid_too_few_parts(self):
        with self.assertRaises(ValueError):
            CronSchedule.from_expression("0 * *")

    def test_from_expression_invalid_too_many_parts(self):
        with self.assertRaises(ValueError):
            CronSchedule.from_expression("0 * * * * *")

    def test_from_expression_invalid_empty(self):
        with self.assertRaises(ValueError):
            CronSchedule.from_expression("")

    def test_expression_property_roundtrip(self):
        expr = "15 3 * * 1-5"
        sched = CronSchedule.from_expression(expr)
        self.assertEqual(sched.expression, expr)

    def test_every_hour(self):
        sched = CronSchedule.every_hour()
        self.assertEqual(sched.expression, "0 * * * *")

    def test_every_minute(self):
        sched = CronSchedule.every_minute()
        self.assertEqual(sched.expression, "* * * * *")

    def test_daily_at(self):
        sched = CronSchedule.daily_at(14, 30)
        self.assertEqual(sched.expression, "30 14 * * *")

    def test_daily_at_default_minute(self):
        sched = CronSchedule.daily_at(8)
        self.assertEqual(sched.expression, "0 8 * * *")

    def test_weekdays_at(self):
        sched = CronSchedule.weekdays_at(9, 0)
        self.assertEqual(sched.expression, "0 9 * * 1-5")

    def test_default_all_stars(self):
        sched = CronSchedule()
        self.assertEqual(sched.expression, "* * * * *")


# =============================================================================
# TaskContext
# =============================================================================


class TaskContextTest(SimpleTestCase):
    """Tests for TaskContext cognitive metadata."""

    def test_default_values(self):
        ctx = TaskContext()
        self.assertEqual(ctx.cascade_depth, 0)
        self.assertEqual(ctx.confidence_score, 1.0)
        self.assertEqual(ctx.urgency, 0.5)
        self.assertEqual(ctx.governance_risk, 0.0)
        self.assertEqual(ctx.resource_weight, 1.0)
        self.assertEqual(ctx.attempts, 0)
        self.assertEqual(ctx.max_attempts, 3)
        self.assertIsNone(ctx.tenant_id)
        self.assertIsNone(ctx.parent_task_id)

    def test_priority_score_formula_defaults(self):
        # (1.0 * 0.3) + (0.5 * 0.4) + ((1 - 0.0) * 0.3) = 0.3 + 0.2 + 0.3 = 0.8
        ctx = TaskContext()
        self.assertAlmostEqual(ctx.priority_score, 0.8)

    def test_priority_score_formula_custom(self):
        ctx = TaskContext(confidence_score=0.5, urgency=1.0, governance_risk=0.5)
        # (0.5 * 0.3) + (1.0 * 0.4) + ((1 - 0.5) * 0.3) = 0.15 + 0.4 + 0.15 = 0.7
        self.assertAlmostEqual(ctx.priority_score, 0.7)

    def test_priority_score_all_zeros(self):
        ctx = TaskContext(confidence_score=0.0, urgency=0.0, governance_risk=1.0)
        # (0.0 * 0.3) + (0.0 * 0.4) + ((1 - 1.0) * 0.3) = 0.0
        self.assertAlmostEqual(ctx.priority_score, 0.0)

    def test_priority_score_all_max(self):
        ctx = TaskContext(confidence_score=1.0, urgency=1.0, governance_risk=0.0)
        # (1.0 * 0.3) + (1.0 * 0.4) + ((1 - 0.0) * 0.3) = 1.0
        self.assertAlmostEqual(ctx.priority_score, 1.0)

    def test_root_correlation_id_defaults_to_correlation_id(self):
        ctx = TaskContext()
        self.assertEqual(ctx.root_correlation_id, ctx.correlation_id)

    def test_root_correlation_id_explicit(self):
        ctx = TaskContext(root_correlation_id="explicit-root")
        self.assertEqual(ctx.root_correlation_id, "explicit-root")

    def test_increment_attempt(self):
        ctx = TaskContext()
        self.assertEqual(ctx.attempts, 0)
        ctx.increment_attempt()
        self.assertEqual(ctx.attempts, 1)
        ctx.increment_attempt()
        self.assertEqual(ctx.attempts, 2)

    def test_has_attempts_remaining(self):
        ctx = TaskContext(max_attempts=2)
        self.assertTrue(ctx.has_attempts_remaining)
        ctx.increment_attempt()
        self.assertTrue(ctx.has_attempts_remaining)
        ctx.increment_attempt()
        self.assertFalse(ctx.has_attempts_remaining)

    def test_as_dict_contains_all_keys(self):
        ctx = TaskContext()
        d = ctx.as_dict()
        expected_keys = {
            "task_id",
            "correlation_id",
            "tenant_id",
            "root_correlation_id",
            "parent_task_id",
            "cascade_depth",
            "reflex_source",
            "confidence_score",
            "urgency",
            "governance_risk",
            "resource_weight",
            "priority_score",
            "created_at",
            "scheduled_at",
            "deadline",
            "attempts",
            "max_attempts",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_as_dict_from_dict_roundtrip(self):
        ctx = TaskContext(
            task_id="test-id-123",
            correlation_id="corr-456",
            tenant_id="tenant-789",
            cascade_depth=2,
            confidence_score=0.7,
            urgency=0.9,
            governance_risk=0.1,
            resource_weight=2.0,
            attempts=1,
            max_attempts=5,
        )
        d = ctx.as_dict()
        restored = TaskContext.from_dict(d)
        self.assertEqual(restored.task_id, ctx.task_id)
        self.assertEqual(restored.correlation_id, ctx.correlation_id)
        self.assertEqual(restored.tenant_id, ctx.tenant_id)
        self.assertEqual(restored.cascade_depth, ctx.cascade_depth)
        self.assertAlmostEqual(restored.confidence_score, ctx.confidence_score)
        self.assertAlmostEqual(restored.urgency, ctx.urgency)
        self.assertAlmostEqual(restored.governance_risk, ctx.governance_risk)
        self.assertAlmostEqual(restored.resource_weight, ctx.resource_weight)
        self.assertEqual(restored.attempts, ctx.attempts)
        self.assertEqual(restored.max_attempts, ctx.max_attempts)
        # Priority score is recomputed from cognitive attributes
        self.assertAlmostEqual(restored.priority_score, ctx.priority_score)

    def test_from_dict_with_minimal_data(self):
        ctx = TaskContext.from_dict({})
        # Should get defaults and auto-generated IDs
        self.assertIsNotNone(ctx.task_id)
        self.assertIsNotNone(ctx.correlation_id)
        self.assertEqual(ctx.cascade_depth, 0)

    def test_unique_ids_generated(self):
        ctx1 = TaskContext()
        ctx2 = TaskContext()
        self.assertNotEqual(ctx1.task_id, ctx2.task_id)
        self.assertNotEqual(ctx1.correlation_id, ctx2.correlation_id)

    def test_is_past_deadline_none(self):
        ctx = TaskContext()
        self.assertFalse(ctx.is_past_deadline)

    def test_is_past_deadline_future(self):
        ctx = TaskContext()
        ctx.deadline = datetime.utcnow() + timedelta(hours=1)
        self.assertFalse(ctx.is_past_deadline)

    def test_is_past_deadline_past(self):
        ctx = TaskContext()
        ctx.deadline = datetime.utcnow() - timedelta(hours=1)
        self.assertTrue(ctx.is_past_deadline)


# =============================================================================
# CascadeBudget
# =============================================================================


class CascadeBudgetTest(SimpleTestCase):
    """Tests for cascade budget limits."""

    def test_depth_1_limit(self):
        self.assertEqual(get_cascade_limit(1), 100)

    def test_depth_2_limit(self):
        self.assertEqual(get_cascade_limit(2), 50)

    def test_depth_3_limit(self):
        self.assertEqual(get_cascade_limit(3), 25)

    def test_depth_4_limit(self):
        self.assertEqual(get_cascade_limit(4), 10)

    def test_depth_5_limit(self):
        self.assertEqual(get_cascade_limit(5), 5)

    def test_beyond_max_depth_returns_zero(self):
        self.assertEqual(get_cascade_limit(6), 0)
        self.assertEqual(get_cascade_limit(100), 0)

    def test_depth_zero_not_in_budget(self):
        # Depth 0 is not in the per-depth map, should return 0 via .get default
        self.assertEqual(get_cascade_limit(0), 0)

    def test_max_depth_config(self):
        self.assertEqual(CASCADE_BUDGET["max_depth"], 5)

    def test_limits_decrease_with_depth(self):
        limits = [get_cascade_limit(d) for d in range(1, 6)]
        for i in range(len(limits) - 1):
            self.assertGreater(limits[i], limits[i + 1])
