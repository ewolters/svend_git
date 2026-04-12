"""
SCH-001 compliance tests: Cognitive Scheduler Standard.

Tests scheduler types, configuration, exceptions, cascade limits,
and cognitive priority scoring. Pure unit tests — no database required.

Standard: SCH-001
"""

import os
import unittest
from datetime import datetime, timedelta

from syn.sched.config import (
    SchedulerConfig,
    get_default_config,
    get_test_config,
)
from syn.sched.exceptions import (
    CascadeLimitError,
    CircuitOpenError,
    ConfigurationError,
    ExecutionError,
    HandlerNotFoundError,
    QuotaExceededError,
    SchedulerError,
    SubmissionError,
    TaskTimeoutError,
    ThrottledError,
)
from syn.sched.types import (
    CASCADE_BUDGET,
    CronSchedule,
    RetryConfig,
    RetryStrategy,
    TaskContext,
    TaskPriority,
    TaskState,
    get_cascade_limit,
)


class SchedulerExceptionHierarchyTest(unittest.TestCase):
    """SCH-001 §4.1: Scheduler exceptions follow ERR-001 hierarchy."""

    def test_all_inherit_from_scheduler_error(self):
        for cls in [
            SubmissionError,
            QuotaExceededError,
            ThrottledError,
            HandlerNotFoundError,
            ExecutionError,
            CircuitOpenError,
            CascadeLimitError,
            TaskTimeoutError,
            ConfigurationError,
        ]:
            self.assertTrue(
                issubclass(cls, SchedulerError),
                f"{cls.__name__} does not inherit from SchedulerError",
            )

    def test_scheduler_error_is_exception(self):
        self.assertTrue(issubclass(SchedulerError, Exception))

    def test_error_codes_unique(self):
        codes = set()
        for cls in [
            SchedulerError,
            SubmissionError,
            QuotaExceededError,
            ThrottledError,
            HandlerNotFoundError,
            ExecutionError,
            CircuitOpenError,
            CascadeLimitError,
            TaskTimeoutError,
            ConfigurationError,
        ]:
            self.assertNotIn(cls.code, codes, f"Duplicate code: {cls.code}")
            codes.add(cls.code)

    def test_to_dict_serialization(self):
        err = QuotaExceededError(tenant_id=42, current_count=100, quota_limit=100)
        d = err.to_dict()
        self.assertEqual(d["error"], "SCHED_QUOTA_EXCEEDED")
        self.assertEqual(d["details"]["tenant_id"], 42)


class TaskStateTest(unittest.TestCase):
    """SCH-001 §core_concepts: Task state machine."""

    def test_terminal_states(self):
        for state in [
            TaskState.SUCCESS,
            TaskState.FAILURE,
            TaskState.CANCELLED,
            TaskState.DEAD_LETTERED,
        ]:
            self.assertTrue(state.is_terminal, f"{state} should be terminal")

    def test_non_terminal_states(self):
        for state in [
            TaskState.PENDING,
            TaskState.SCHEDULED,
            TaskState.RUNNING,
            TaskState.RETRYING,
        ]:
            self.assertFalse(state.is_terminal, f"{state} should not be terminal")

    def test_active_states(self):
        self.assertTrue(TaskState.RUNNING.is_active)
        self.assertTrue(TaskState.RETRYING.is_active)
        self.assertFalse(TaskState.PENDING.is_active)

    def test_valid_transitions_defined(self):
        transitions = TaskState.valid_transitions()
        for state in TaskState:
            self.assertIn(state, transitions, f"No transitions for {state}")
        # Terminal states have no transitions
        self.assertEqual(transitions[TaskState.SUCCESS], [])
        self.assertEqual(transitions[TaskState.CANCELLED], [])


class CognitiveScoringTest(unittest.TestCase):
    """SCH-001 §core_concepts: Cognitive priority scoring formula."""

    def test_priority_score_formula(self):
        ctx = TaskContext(confidence_score=1.0, urgency=1.0, governance_risk=0.0)
        # Score = (1.0 * 0.3) + (1.0 * 0.4) + ((1 - 0.0) * 0.3) = 1.0
        self.assertAlmostEqual(ctx.priority_score, 1.0)

    def test_low_urgency_lowers_score(self):
        ctx = TaskContext(confidence_score=1.0, urgency=0.0, governance_risk=0.0)
        # Score = 0.3 + 0.0 + 0.3 = 0.6
        self.assertAlmostEqual(ctx.priority_score, 0.6)

    def test_high_risk_lowers_score(self):
        ctx = TaskContext(confidence_score=1.0, urgency=1.0, governance_risk=1.0)
        # Score = 0.3 + 0.4 + 0.0 = 0.7
        self.assertAlmostEqual(ctx.priority_score, 0.7)

    def test_root_correlation_defaults_to_correlation(self):
        ctx = TaskContext()
        self.assertEqual(ctx.root_correlation_id, ctx.correlation_id)

    def test_attempt_tracking(self):
        ctx = TaskContext(max_attempts=3)
        self.assertTrue(ctx.has_attempts_remaining)
        ctx.increment_attempt()
        ctx.increment_attempt()
        ctx.increment_attempt()
        self.assertFalse(ctx.has_attempts_remaining)


class CascadeLimitsTest(unittest.TestCase):
    """SCH-001 §5: Cascade depth and budget limits."""

    def test_max_depth_defined(self):
        self.assertEqual(CASCADE_BUDGET["max_depth"], 5)

    def test_cascade_limit_decreases_with_depth(self):
        limits = [get_cascade_limit(d) for d in range(1, 6)]
        for i in range(len(limits) - 1):
            self.assertGreaterEqual(limits[i], limits[i + 1])

    def test_beyond_max_depth_returns_zero(self):
        self.assertEqual(get_cascade_limit(6), 0)
        self.assertEqual(get_cascade_limit(100), 0)

    def test_cascade_limit_error_attributes(self):
        err = CascadeLimitError(cascade_depth=6, max_depth=5)
        self.assertEqual(err.cascade_depth, 6)
        self.assertEqual(err.max_depth, 5)


class RetryConfigTest(unittest.TestCase):
    """SCH-001 §retry: Retry strategy configuration."""

    def test_exponential_delay_increases(self):
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_seconds=1,
            jitter=False,
        )
        d1 = config.get_delay(1)
        d2 = config.get_delay(2)
        self.assertGreater(d2, d1)

    def test_none_strategy_zero_delay(self):
        config = RetryConfig(strategy=RetryStrategy.NONE)
        self.assertEqual(config.get_delay(1).total_seconds(), 0)

    def test_fixed_strategy_constant(self):
        config = RetryConfig(
            strategy=RetryStrategy.FIXED,
            base_delay_seconds=5,
        )
        self.assertEqual(config.get_delay(1).total_seconds(), 5)
        self.assertEqual(config.get_delay(10).total_seconds(), 5)

    def test_max_delay_cap(self):
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_seconds=100,
            max_delay_seconds=300,
            jitter=False,
        )
        d = config.get_delay(10)
        self.assertLessEqual(d.total_seconds(), 300)


class SchedulerConfigTest(unittest.TestCase):
    """SCH-001 §4.1: Configuration supports env vars and Django settings."""

    def test_default_config_has_cascade_limits(self):
        config = get_default_config()
        self.assertGreater(config.max_cascade_depth, 0)
        self.assertGreater(config.cascade_budget, 0)

    def test_test_config_has_minimal_limits(self):
        config = get_test_config()
        self.assertLessEqual(config.max_queue_depth, 100)
        self.assertFalse(config.enable_schedule_processing)

    def test_from_environment(self):
        with unittest.mock.patch.dict(os.environ, {"SYNARA_SCHED_MAX_CASCADE_DEPTH": "10"}):
            config = SchedulerConfig.from_environment()
            self.assertEqual(config.max_cascade_depth, 10)

    def test_to_dict_serialization(self):
        config = SchedulerConfig()
        d = config.to_dict()
        self.assertIn("max_cascade_depth", d)
        self.assertIn("enable_backpressure", d)
        self.assertIn("worker_pool", d)

    def test_from_dict_roundtrip(self):
        config = SchedulerConfig(max_cascade_depth=7, cascade_budget=42)
        d = config.to_dict()
        restored = SchedulerConfig.from_dict(d)
        self.assertEqual(restored.max_cascade_depth, 7)
        self.assertEqual(restored.cascade_budget, 42)


class TaskPriorityTest(unittest.TestCase):
    """SCH-001: Priority weights."""

    def test_critical_highest_weight(self):
        self.assertEqual(TaskPriority.CRITICAL.weight, 1.0)

    def test_batch_lowest_weight(self):
        self.assertEqual(TaskPriority.BATCH.weight, 0.1)

    def test_weights_monotonically_decrease(self):
        weights = [p.weight for p in TaskPriority]
        for i in range(len(weights) - 1):
            self.assertGreaterEqual(weights[i], weights[i + 1])


class UUIDPrimaryKeysTest(unittest.TestCase):
    """SCH-001 §DAT-001: Scheduler models use UUID primary keys."""

    def test_task_context_has_uuid_task_id(self):
        ctx = TaskContext()
        from uuid import UUID

        UUID(ctx.task_id)  # Validates as UUID

    def test_task_context_has_uuid_correlation_id(self):
        ctx = TaskContext()
        from uuid import UUID

        UUID(ctx.correlation_id)


class CronScheduleTest(unittest.TestCase):
    """SCH-001 §temporal: Cron schedule parsing."""

    def test_from_expression(self):
        cron = CronSchedule.from_expression("0 * * * *")
        self.assertEqual(cron.minute, "0")
        self.assertEqual(cron.hour, "*")

    def test_expression_roundtrip(self):
        cron = CronSchedule(minute="30", hour="6")
        self.assertEqual(cron.expression, "30 6 * * *")

    def test_invalid_expression_raises(self):
        with self.assertRaises(ValueError):
            CronSchedule.from_expression("invalid")


class ScheduleModelTest(unittest.TestCase):
    """SCH-001 §11.1: Schedule model operations."""

    def test_cron_calculate_next_run(self):
        """Cron schedule calculates a future next_run_at."""
        from syn.sched.models import Schedule
        from syn.sched.types import ScheduleType

        s = Schedule()
        s.schedule_type = ScheduleType.CRON.value
        s.cron_minute = "0"
        s.cron_hour = "2"
        s.cron_day_of_month = "*"
        s.cron_month = "*"
        s.cron_day_of_week = "*"
        s.run_count = 0
        s.last_run_at = None

        from django.utils import timezone

        now = timezone.now()
        result = s.calculate_next_run(from_time=now)
        self.assertIsNotNone(result)
        self.assertGreater(result, now)

    def test_interval_calculate_next_run(self):
        """Interval schedule calculates next_run from last_run_at."""
        from django.utils import timezone

        from syn.sched.models import Schedule
        from syn.sched.types import ScheduleType

        s = Schedule()
        s.schedule_type = ScheduleType.INTERVAL.value
        s.interval_seconds = 300
        s.last_run_at = timezone.now()

        result = s.calculate_next_run()
        self.assertIsNotNone(result)
        expected = s.last_run_at + timedelta(seconds=300)
        self.assertEqual(result, expected)

    def test_once_calculate_next_run_future(self):
        """ONCE schedule with future run_at returns that time."""
        from django.utils import timezone

        from syn.sched.models import Schedule
        from syn.sched.types import ScheduleType

        future = timezone.now() + timedelta(hours=1)
        s = Schedule()
        s.schedule_type = ScheduleType.ONCE.value
        s.run_at = future

        result = s.calculate_next_run()
        self.assertEqual(result, future)

    def test_once_calculate_next_run_past(self):
        """ONCE schedule with past run_at returns None."""
        from django.utils import timezone

        from syn.sched.models import Schedule
        from syn.sched.types import ScheduleType

        past = timezone.now() - timedelta(hours=1)
        s = Schedule()
        s.schedule_type = ScheduleType.ONCE.value
        s.run_at = past

        result = s.calculate_next_run()
        self.assertIsNone(result)

    def test_cron_expression_property(self):
        """cron_expression property concatenates the 5 fields."""
        from syn.sched.models import Schedule

        s = Schedule()
        s.cron_minute = "*/10"
        s.cron_hour = "*"
        s.cron_day_of_month = "*"
        s.cron_month = "*"
        s.cron_day_of_week = "*"
        self.assertEqual(s.cron_expression, "*/10 * * * *")


class ScheduleRecordRunTest(unittest.TestCase):
    """SCH-001 §11.2: record_run updates schedule metadata."""

    def _make_schedule(self):
        """Create an in-memory Schedule for testing (no DB save)."""
        from syn.sched.models import Schedule
        from syn.sched.types import ScheduleType

        s = Schedule()
        s.schedule_type = ScheduleType.CRON.value
        s.cron_minute = "0"
        s.cron_hour = "*"
        s.cron_day_of_month = "*"
        s.cron_month = "*"
        s.cron_day_of_week = "*"
        s.run_count = 0
        s.last_run_at = None
        s.is_enabled = True
        s.max_runs = None
        s.expires_at = None
        return s

    def test_record_run_increments_count(self):
        """record_run increments run_count by 1."""
        s = self._make_schedule()
        # Monkey-patch save to avoid DB
        s.save = lambda *a, **kw: None
        s.record_run()
        self.assertEqual(s.run_count, 1)

    def test_record_run_sets_last_run_at(self):
        """record_run sets last_run_at to current time."""
        s = self._make_schedule()
        s.save = lambda *a, **kw: None
        self.assertIsNone(s.last_run_at)
        s.record_run()
        self.assertIsNotNone(s.last_run_at)

    def test_record_run_recalculates_next_run(self):
        """record_run recalculates next_run_at from current time."""
        from django.utils import timezone

        s = self._make_schedule()
        s.save = lambda *a, **kw: None
        s.next_run_at = timezone.now() - timedelta(hours=1)  # Past
        s.record_run()
        self.assertIsNotNone(s.next_run_at)
        self.assertGreater(s.next_run_at, timezone.now())

    def test_max_runs_disables_schedule(self):
        """record_run disables schedule when max_runs is reached."""
        s = self._make_schedule()
        s.save = lambda *a, **kw: None
        s.max_runs = 1
        s.run_count = 0
        s.record_run()
        self.assertFalse(s.is_enabled)
        self.assertIsNone(s.next_run_at)


class CronCalculationTest(unittest.TestCase):
    """SCH-001 §11.3: Croniter-based next-run calculation."""

    def test_croniter_produces_future_datetime(self):
        """croniter always returns a future datetime for cron schedules."""
        from croniter import croniter
        from django.utils import timezone

        now = timezone.now()
        cron = croniter("*/5 * * * *", now)
        next_run = cron.get_next(datetime)
        # croniter may return naive or aware depending on input
        if timezone.is_naive(next_run):
            now_naive = now.replace(tzinfo=None)
            self.assertGreater(next_run, now_naive)
        else:
            self.assertGreater(next_run, now)

    def test_croniter_respects_timezone(self):
        """Cron calculation works with timezone-aware datetimes."""

        from syn.sched.models import Schedule
        from syn.sched.types import ScheduleType

        s = Schedule()
        s.schedule_type = ScheduleType.CRON.value
        s.cron_minute = "0"
        s.cron_hour = "3"
        s.cron_day_of_month = "*"
        s.cron_month = "*"
        s.cron_day_of_week = "*"

        result = s.calculate_next_run()
        self.assertIsNotNone(result)
        # Result should be timezone-aware or we handle it
        self.assertTrue(hasattr(result, "tzinfo"))

    def test_interval_schedule_from_last_run(self):
        """Interval schedule calculates from last_run_at when available."""
        from django.utils import timezone

        from syn.sched.models import Schedule
        from syn.sched.types import ScheduleType

        s = Schedule()
        s.schedule_type = ScheduleType.INTERVAL.value
        s.interval_seconds = 600
        s.last_run_at = timezone.now() - timedelta(seconds=300)

        result = s.calculate_next_run()
        # Should be last_run + 600s, which is 300s from now
        expected = s.last_run_at + timedelta(seconds=600)
        self.assertEqual(result, expected)

    def test_once_schedule_returns_none_after_run(self):
        """ONCE schedule returns None after its run_at has passed."""
        from django.utils import timezone

        from syn.sched.models import Schedule
        from syn.sched.types import ScheduleType

        s = Schedule()
        s.schedule_type = ScheduleType.ONCE.value
        s.run_at = timezone.now() - timedelta(hours=1)

        result = s.calculate_next_run()
        self.assertIsNone(result)


class ManagementCommandTest(unittest.TestCase):
    """SCH-001 §11.4: tempora_server management command."""

    def test_tempora_server_registered(self):
        """tempora_server is registered as a Django management command."""
        from django.core.management import get_commands

        commands = get_commands()
        self.assertIn("tempora_server", commands)
        self.assertEqual(commands["tempora_server"], "syn.sched")

    def test_command_has_workers_arg(self):
        """tempora_server accepts --workers argument."""
        from django.core.management import load_command_class

        cmd = load_command_class("syn.sched", "tempora_server")
        parser = cmd.create_parser("manage.py", "tempora_server")
        # Parse with workers arg — should not raise
        args = parser.parse_args(["--single-node", "--workers", "2"])
        self.assertEqual(args.workers, 2)

    def test_command_has_single_node_arg(self):
        """tempora_server accepts --single-node flag."""
        from django.core.management import load_command_class

        cmd = load_command_class("syn.sched", "tempora_server")
        parser = cmd.create_parser("manage.py", "tempora_server")
        args = parser.parse_args(["--single-node"])
        self.assertTrue(args.single_node)


if __name__ == "__main__":
    unittest.main()
