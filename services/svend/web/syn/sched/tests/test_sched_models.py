"""
DB behavior tests for syn.sched models.

Tests: CognitiveTask, TaskExecution, Schedule, DeadLetterEntry
Standard: SCH-001 §5-8, SCH-002 §5-8
"""

import uuid
from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from syn.sched.models import (
    CognitiveTask,
    DeadLetterEntry,
    Schedule,
    TaskExecution,
)
from syn.sched.types import (
    CASCADE_BUDGET,
    CronSchedule,
    IntervalSchedule,
    QueueType,
    ScheduleType,
    TaskPriority,
    TaskState,
)


class CognitiveTaskTest(TestCase):
    """DB tests for CognitiveTask model."""

    def setUp(self):
        self.tenant_id = uuid.uuid4()

    def _create_task(self, **kwargs):
        """Helper: create a task with sensible defaults."""
        defaults = {
            "tenant_id": self.tenant_id,
            "task_name": "test.handler",
            "payload": {"key": "value"},
        }
        defaults.update(kwargs)
        return CognitiveTask.create_task(**defaults)

    # --------------------------------------------------------------------- #
    # create_task factory
    # --------------------------------------------------------------------- #

    def test_create_task_defaults(self):
        """create_task() persists with correct defaults."""
        task = self._create_task()
        task.refresh_from_db()

        self.assertEqual(task.task_name, "test.handler")
        self.assertEqual(task.state, TaskState.PENDING.value)
        self.assertEqual(task.priority, TaskPriority.NORMAL.value)
        self.assertEqual(task.queue, QueueType.CORE.value)
        self.assertEqual(task.attempts, 0)
        self.assertEqual(task.max_attempts, 3)
        self.assertEqual(task.cascade_depth, 0)
        self.assertEqual(task.payload, {"key": "value"})
        self.assertIsNotNone(task.correlation_id)
        self.assertFalse(task.is_deleted)

    def test_create_task_computes_priority_score(self):
        """create_task() computes priority_score = (conf*0.3)+(urg*0.4)+((1-risk)*0.3)."""
        task = self._create_task(
            confidence_score=0.8,
            urgency=0.6,
            governance_risk=0.2,
        )
        task.refresh_from_db()

        expected = (0.8 * 0.3) + (0.6 * 0.4) + ((1 - 0.2) * 0.3)
        self.assertAlmostEqual(task.priority_score, expected, places=6)

    def test_create_task_root_correlation_no_parent(self):
        """root_correlation_id equals correlation_id when no parent."""
        task = self._create_task()
        task.refresh_from_db()
        self.assertEqual(task.root_correlation_id, task.correlation_id)

    def test_create_task_parent_increments_cascade_depth(self):
        """create_task() with parent_task increments cascade_depth."""
        parent = self._create_task(task_name="parent.handler")
        child = self._create_task(task_name="child.handler", parent_task=parent)
        child.refresh_from_db()

        self.assertEqual(child.cascade_depth, parent.cascade_depth + 1)
        self.assertEqual(child.parent_task_id, parent.id)

    def test_create_task_parent_sets_root_correlation(self):
        """create_task() with parent inherits root_correlation_id."""
        parent = self._create_task(task_name="parent.handler")
        child = self._create_task(task_name="child.handler", parent_task=parent)
        child.refresh_from_db()

        self.assertEqual(child.root_correlation_id, parent.root_correlation_id)

    def test_create_task_excessive_cascade_depth_raises(self):
        """create_task() raises ValueError when cascade_depth exceeds budget max."""
        max_depth = CASCADE_BUDGET["max_depth"]
        # Build chain up to max depth
        task = self._create_task(task_name="depth.0")
        for i in range(1, max_depth):
            task = self._create_task(task_name=f"depth.{i}", parent_task=task)

        # Next one should be at depth == max_depth, which has limit 5, still ok
        task = self._create_task(task_name=f"depth.{max_depth}", parent_task=task)

        # Now at max_depth, one more pushes past the limit (get_cascade_limit returns 0)
        with self.assertRaises(ValueError):
            self._create_task(task_name="depth.too.deep", parent_task=task)

    # --------------------------------------------------------------------- #
    # transition_to
    # --------------------------------------------------------------------- #

    def test_transition_pending_to_scheduled(self):
        """PENDING -> SCHEDULED sets scheduled_at."""
        task = self._create_task()
        self.assertTrue(task.transition_to(TaskState.SCHEDULED))
        task.refresh_from_db()
        self.assertEqual(task.state, TaskState.SCHEDULED.value)
        self.assertIsNotNone(task.scheduled_at)

    def test_transition_scheduled_to_running(self):
        """SCHEDULED -> RUNNING sets started_at."""
        task = self._create_task()
        task.transition_to(TaskState.SCHEDULED)
        self.assertTrue(task.transition_to(TaskState.RUNNING))
        task.refresh_from_db()
        self.assertEqual(task.state, TaskState.RUNNING.value)
        self.assertIsNotNone(task.started_at)

    def test_transition_running_to_success(self):
        """RUNNING -> SUCCESS sets completed_at."""
        task = self._create_task()
        task.transition_to(TaskState.SCHEDULED)
        task.transition_to(TaskState.RUNNING)
        self.assertTrue(task.transition_to(TaskState.SUCCESS))
        task.refresh_from_db()
        self.assertEqual(task.state, TaskState.SUCCESS.value)
        self.assertIsNotNone(task.completed_at)

    def test_transition_invalid_returns_false(self):
        """Invalid transition returns False and does not change state."""
        task = self._create_task()
        original_state = task.state
        result = task.transition_to(TaskState.SUCCESS)
        self.assertFalse(result)
        task.refresh_from_db()
        self.assertEqual(task.state, original_state)

    def test_transition_from_terminal_returns_false(self):
        """Transition from terminal state (SUCCESS) returns False."""
        task = self._create_task()
        task.transition_to(TaskState.SCHEDULED)
        task.transition_to(TaskState.RUNNING)
        task.transition_to(TaskState.SUCCESS)
        result = task.transition_to(TaskState.RUNNING)
        self.assertFalse(result)
        task.refresh_from_db()
        self.assertEqual(task.state, TaskState.SUCCESS.value)

    # --------------------------------------------------------------------- #
    # schedule_retry
    # --------------------------------------------------------------------- #

    def test_schedule_retry_increments_attempts(self):
        """schedule_retry() increments attempts and sets next_retry_at."""
        task = self._create_task(max_attempts=5)
        task.transition_to(TaskState.SCHEDULED)
        task.transition_to(TaskState.RUNNING)
        # Move to FAILURE so RETRYING is reachable
        task.transition_to(TaskState.FAILURE)

        result = task.schedule_retry()
        task.refresh_from_db()

        self.assertIsNotNone(result)
        self.assertEqual(task.attempts, 1)
        self.assertEqual(task.state, TaskState.RETRYING.value)
        self.assertIsNotNone(task.next_retry_at)

    def test_schedule_retry_max_exceeded_dead_letters(self):
        """schedule_retry() transitions to DEAD_LETTERED when max attempts exceeded."""
        task = self._create_task(max_attempts=1)
        task.transition_to(TaskState.SCHEDULED)
        task.transition_to(TaskState.RUNNING)
        task.transition_to(TaskState.FAILURE)

        result = task.schedule_retry()
        task.refresh_from_db()

        self.assertIsNone(result)
        self.assertEqual(task.state, TaskState.DEAD_LETTERED.value)

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    def test_has_attempts_remaining(self):
        """has_attempts_remaining is True when attempts < max_attempts."""
        task = self._create_task(max_attempts=3)
        self.assertTrue(task.has_attempts_remaining)
        task.attempts = 3
        self.assertFalse(task.has_attempts_remaining)

    def test_is_past_deadline_no_deadline(self):
        """is_past_deadline is False when deadline is None."""
        task = self._create_task()
        self.assertFalse(task.is_past_deadline)

    def test_is_past_deadline_true(self):
        """is_past_deadline is True when deadline is in the past."""
        task = self._create_task(deadline=timezone.now() - timedelta(hours=1))
        self.assertTrue(task.is_past_deadline)


class ScheduleModelTest(TestCase):
    """DB tests for Schedule model."""

    def setUp(self):
        self.tenant_id = uuid.uuid4()

    def test_create_interval_schedule(self):
        """create_interval_schedule() persists with correct type and next_run."""
        interval = IntervalSchedule(minutes=10)
        sched = Schedule.create_interval_schedule(
            tenant_id=self.tenant_id,
            schedule_id="test-interval",
            name="Every 10 min",
            task_name="periodic.handler",
            interval=interval,
        )
        sched.refresh_from_db()

        self.assertEqual(sched.schedule_type, ScheduleType.INTERVAL.value)
        self.assertEqual(sched.interval_seconds, 600)
        self.assertTrue(sched.is_enabled)
        self.assertIsNotNone(sched.next_run_at)

    def test_create_cron_schedule(self):
        """create_cron_schedule() persists cron fields correctly."""
        cron = CronSchedule(minute="30", hour="6", day_of_week="1-5")
        sched = Schedule.create_cron_schedule(
            tenant_id=self.tenant_id,
            schedule_id="test-cron",
            name="Weekday 6:30",
            task_name="cron.handler",
            cron=cron,
        )
        sched.refresh_from_db()

        self.assertEqual(sched.schedule_type, ScheduleType.CRON.value)
        self.assertEqual(sched.cron_minute, "30")
        self.assertEqual(sched.cron_hour, "6")
        self.assertEqual(sched.cron_day_of_week, "1-5")

    def test_calculate_next_run_interval(self):
        """calculate_next_run() for interval type returns from_time + interval."""
        interval = IntervalSchedule(minutes=5)
        sched = Schedule.create_interval_schedule(
            tenant_id=self.tenant_id,
            schedule_id="calc-interval",
            name="Every 5 min",
            task_name="calc.handler",
            interval=interval,
        )
        now = timezone.now()
        next_run = sched.calculate_next_run(from_time=now)
        # Without last_run_at, should be from_time + interval
        self.assertAlmostEqual(
            next_run.timestamp(),
            (now + timedelta(seconds=300)).timestamp(),
            delta=1,
        )

    def test_record_run_increments_count(self):
        """record_run() increments run_count and updates last_run_at."""
        interval = IntervalSchedule(minutes=10)
        sched = Schedule.create_interval_schedule(
            tenant_id=self.tenant_id,
            schedule_id="rec-run",
            name="Run counter",
            task_name="counter.handler",
            interval=interval,
        )
        self.assertEqual(sched.run_count, 0)
        self.assertIsNone(sched.last_run_at)

        sched.record_run()
        sched.refresh_from_db()

        self.assertEqual(sched.run_count, 1)
        self.assertIsNotNone(sched.last_run_at)

    def test_record_run_max_runs_disables(self):
        """record_run() disables schedule when max_runs reached."""
        interval = IntervalSchedule(minutes=10)
        sched = Schedule.create_interval_schedule(
            tenant_id=self.tenant_id,
            schedule_id="max-run",
            name="Limited",
            task_name="limited.handler",
            interval=interval,
            max_runs=1,
        )
        sched.record_run()
        sched.refresh_from_db()

        self.assertFalse(sched.is_enabled)
        self.assertIsNone(sched.next_run_at)

    def test_cron_expression_property(self):
        """cron_expression assembles fields into standard format."""
        cron = CronSchedule(
            minute="0", hour="12", day_of_month="1", month="*", day_of_week="*"
        )
        sched = Schedule.create_cron_schedule(
            tenant_id=self.tenant_id,
            schedule_id="cron-expr",
            name="Monthly noon",
            task_name="cron.handler",
            cron=cron,
        )
        self.assertEqual(sched.cron_expression, "0 12 1 * *")


class DeadLetterEntryTest(TestCase):
    """DB tests for DeadLetterEntry model."""

    def setUp(self):
        self.tenant_id = uuid.uuid4()

    def _create_failed_task(self):
        """Helper: create a task in DEAD_LETTERED state."""
        task = CognitiveTask.create_task(
            tenant_id=self.tenant_id,
            task_name="failing.handler",
            payload={"data": "test"},
        )
        task.transition_to(TaskState.SCHEDULED)
        task.transition_to(TaskState.RUNNING)
        task.transition_to(
            TaskState.FAILURE, error_message="boom", error_type="transient"
        )
        return task

    def test_create_dead_letter_entry(self):
        """DeadLetterEntry stores required fields correctly."""
        task = self._create_failed_task()
        entry = DeadLetterEntry(
            tenant_id=self.tenant_id,
            original_task=task,
            failure_reason="Max retries exceeded",
            failure_count=3,
            last_error_message="boom",
            last_error_type="transient",
        )
        entry.save()
        entry.refresh_from_db()

        self.assertEqual(entry.tenant_id, self.tenant_id)
        self.assertEqual(entry.original_task_id, task.id)
        self.assertEqual(entry.failure_reason, "Max retries exceeded")
        self.assertEqual(entry.failure_count, 3)
        self.assertEqual(entry.status, "pending")

    def test_dead_letter_fields_stored(self):
        """Optional DLQ fields (error details) persist correctly."""
        task = self._create_failed_task()
        entry = DeadLetterEntry(
            tenant_id=self.tenant_id,
            original_task=task,
            failure_reason="Permanent error",
            failure_count=1,
            last_error_message="connection refused",
            last_error_type="permanent",
            last_error_traceback="Traceback...",
        )
        entry.save()
        entry.refresh_from_db()

        self.assertEqual(entry.last_error_message, "connection refused")
        self.assertEqual(entry.last_error_type, "permanent")
        self.assertEqual(entry.last_error_traceback, "Traceback...")


class TaskExecutionTest(TestCase):
    """DB tests for TaskExecution model."""

    def setUp(self):
        self.tenant_id = uuid.uuid4()

    def _create_task(self):
        return CognitiveTask.create_task(
            tenant_id=self.tenant_id,
            task_name="exec.handler",
            payload={},
        )

    def test_create_execution_with_task(self):
        """TaskExecution persists with FK to task and attempt_number."""
        task = self._create_task()
        exe = TaskExecution(task=task, attempt_number=1, worker_id="worker-1")
        exe.save()
        exe.refresh_from_db()

        self.assertEqual(exe.task_id, task.id)
        self.assertEqual(exe.attempt_number, 1)
        self.assertFalse(exe.is_success)
        self.assertIsNotNone(exe.started_at)

    def test_complete_sets_duration_and_success(self):
        """complete() sets duration_ms and is_success."""
        task = self._create_task()
        exe = TaskExecution(task=task, attempt_number=1)
        exe.save()

        exe.complete(success=True, result={"output": "done"})
        exe.refresh_from_db()

        self.assertTrue(exe.is_success)
        self.assertIsNotNone(exe.completed_at)
        self.assertIsNotNone(exe.duration_ms)
        self.assertEqual(exe.result, {"output": "done"})
