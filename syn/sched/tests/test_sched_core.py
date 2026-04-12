"""
Tests for syn.sched.core — TaskRegistry, CognitiveScheduler, and @task decorator.

Standard: SCH-001 §5-8, SCH-002 §3-8
"""

import uuid

from django.test import SimpleTestCase, TestCase

from syn.sched.core import CognitiveScheduler, TaskRegistry, task
from syn.sched.models import CognitiveTask
from syn.sched.types import (
    QueueType,
    RetryStrategy,
    TaskPriority,
    TaskState,
)

# ============================================================================
# TaskRegistry (no DB needed)
# ============================================================================


class TaskRegistryTest(SimpleTestCase):
    """Tests for TaskRegistry class-level handler management."""

    def setUp(self):
        self._original_handlers = TaskRegistry._handlers.copy()
        self._original_metadata = TaskRegistry._handler_metadata.copy()

    def tearDown(self):
        TaskRegistry._handlers = self._original_handlers
        TaskRegistry._handler_metadata = self._original_metadata

    # ------------------------------------------------------------------ #

    def test_register_and_get_handler(self):
        """register() adds handler that get_handler() retrieves."""

        def my_handler(**kw):
            return "ok"

        TaskRegistry.register("test.my_handler", my_handler)
        retrieved = TaskRegistry.get_handler("test.my_handler")
        self.assertIs(retrieved, my_handler)

    def test_get_handler_returns_none_for_unregistered(self):
        """get_handler() returns None for unregistered task name."""
        self.assertIsNone(TaskRegistry.get_handler("nonexistent.task"))

    def test_list_handlers_returns_registered_names(self):
        """list_handlers() includes registered task names."""
        TaskRegistry.register("test.alpha", lambda **kw: None)
        TaskRegistry.register("test.beta", lambda **kw: None)
        names = TaskRegistry.list_handlers()
        self.assertIn("test.alpha", names)
        self.assertIn("test.beta", names)

    def test_get_metadata_returns_dict(self):
        """get_metadata() returns metadata dict for registered task."""
        TaskRegistry.register(
            "test.meta",
            lambda **kw: None,
            queue=QueueType.BATCH,
            priority=TaskPriority.HIGH,
            timeout_seconds=120,
            retry_strategy=RetryStrategy.LINEAR,
            max_attempts=7,
        )
        meta = TaskRegistry.get_metadata("test.meta")
        self.assertEqual(meta["queue"], QueueType.BATCH)
        self.assertEqual(meta["priority"], TaskPriority.HIGH)
        self.assertEqual(meta["timeout_seconds"], 120)
        self.assertEqual(meta["retry_strategy"], RetryStrategy.LINEAR)
        self.assertEqual(meta["max_attempts"], 7)

    def test_get_metadata_returns_empty_for_unregistered(self):
        """get_metadata() returns empty dict for unknown task."""
        self.assertEqual(TaskRegistry.get_metadata("nope.missing"), {})

    def test_task_decorator_registers_handler(self):
        """The @task decorator registers the function in TaskRegistry."""

        @task("test.decorated", queue=QueueType.TELEMETRY, priority=TaskPriority.LOW)
        def decorated_handler(**kw):
            return "decorated"

        self.assertIs(TaskRegistry.get_handler("test.decorated"), decorated_handler)
        meta = TaskRegistry.get_metadata("test.decorated")
        self.assertEqual(meta["queue"], QueueType.TELEMETRY)
        self.assertEqual(meta["priority"], TaskPriority.LOW)


# ============================================================================
# CognitiveScheduler (needs DB)
# ============================================================================


class CognitiveSchedulerTest(TestCase):
    """Tests for CognitiveScheduler submit/fetch/batch operations."""

    def setUp(self):
        self._original_handlers = TaskRegistry._handlers.copy()
        self._original_metadata = TaskRegistry._handler_metadata.copy()
        TaskRegistry.register("test.handler", lambda **kw: None)
        TaskRegistry.register(
            "test.high_priority",
            lambda **kw: None,
            priority=TaskPriority.HIGH,
            queue=QueueType.CRITICAL,
        )
        self.tenant_id = uuid.uuid4()
        self.scheduler = CognitiveScheduler(
            enable_backpressure=False,
            enable_temporal=False,
        )

    def tearDown(self):
        TaskRegistry._handlers = self._original_handlers
        TaskRegistry._handler_metadata = self._original_metadata

    # ------------------------------------------------------------------ #

    def test_submit_creates_cognitive_task(self):
        """submit() creates a CognitiveTask record in the database."""
        t = self.scheduler.submit(
            task_name="test.handler",
            payload={"foo": "bar"},
            tenant_id=self.tenant_id,
        )
        self.assertIsInstance(t, CognitiveTask)
        self.assertEqual(t.task_name, "test.handler")
        self.assertEqual(t.payload, {"foo": "bar"})
        self.assertEqual(t.tenant_id, self.tenant_id)
        self.assertEqual(t.state, TaskState.PENDING.value)
        # Verify it persisted
        self.assertTrue(CognitiveTask.objects.filter(id=t.id).exists())

    def test_submit_respects_handler_defaults(self):
        """submit() uses handler-registered priority/queue when caller omits them."""
        t = self.scheduler.submit(
            task_name="test.high_priority",
            payload={},
            tenant_id=self.tenant_id,
        )
        self.assertEqual(t.priority, TaskPriority.HIGH.value)
        self.assertEqual(t.queue, QueueType.CRITICAL.value)

    def test_submit_caller_overrides_handler_defaults(self):
        """submit() lets caller override handler defaults for priority/queue."""
        t = self.scheduler.submit(
            task_name="test.high_priority",
            payload={},
            tenant_id=self.tenant_id,
            priority=TaskPriority.LOW,
            queue=QueueType.BATCH,
        )
        self.assertEqual(t.priority, TaskPriority.LOW.value)
        self.assertEqual(t.queue, QueueType.BATCH.value)

    def test_submit_enforces_cascade_budget(self):
        """submit() raises ValueError when cascade depth exceeds budget."""
        # Build a chain up to max depth (max_depth=5, so depths 0..5 are valid)
        parent = self.scheduler.submit(
            task_name="test.handler",
            payload={},
            tenant_id=self.tenant_id,
        )
        for _ in range(5):
            parent = self.scheduler.submit(
                task_name="test.handler",
                payload={},
                tenant_id=self.tenant_id,
                parent_task=parent,
            )
        # parent is now at depth 5, next child would be depth 6 which exceeds max
        with self.assertRaises(ValueError) as ctx:
            self.scheduler.submit(
                task_name="test.handler",
                payload={},
                tenant_id=self.tenant_id,
                parent_task=parent,
            )
        self.assertIn("Cascade depth", str(ctx.exception))

    def test_fetch_next_task_returns_highest_priority(self):
        """fetch_next_task() returns the task with the highest priority score."""
        self.scheduler.submit(
            task_name="test.handler",
            payload={"label": "low"},
            tenant_id=self.tenant_id,
            urgency=0.1,
        )
        high = self.scheduler.submit(
            task_name="test.handler",
            payload={"label": "high"},
            tenant_id=self.tenant_id,
            urgency=1.0,
        )
        fetched = self.scheduler.fetch_next_task(
            queues=[QueueType.CORE],
            worker_id="w-1",
        )
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, high.id)

    def test_fetch_next_task_returns_none_when_empty(self):
        """fetch_next_task() returns None when no tasks are available."""
        fetched = self.scheduler.fetch_next_task(
            queues=[QueueType.CORE],
            worker_id="w-1",
        )
        self.assertIsNone(fetched)

    def test_submit_batch_creates_multiple_tasks(self):
        """submit_batch() creates multiple tasks atomically."""
        specs = [
            {"task_name": "test.handler", "payload": {"i": 1}},
            {"task_name": "test.handler", "payload": {"i": 2}},
            {"task_name": "test.handler", "payload": {"i": 3}},
        ]
        tasks = self.scheduler.submit_batch(
            tasks=specs,
            tenant_id=self.tenant_id,
        )
        self.assertEqual(len(tasks), 3)
        # All share a correlation_id
        cids = {t.correlation_id for t in tasks}
        self.assertEqual(len(cids), 1)
        # All persisted
        self.assertEqual(
            CognitiveTask.objects.filter(tenant_id=self.tenant_id).count(),
            3,
        )

    def test_submit_batch_shares_provided_correlation_id(self):
        """submit_batch() uses caller-provided correlation_id for all tasks."""
        cid = uuid.uuid4()
        specs = [
            {"task_name": "test.handler", "payload": {}},
            {"task_name": "test.handler", "payload": {}},
        ]
        tasks = self.scheduler.submit_batch(
            tasks=specs,
            tenant_id=self.tenant_id,
            correlation_id=cid,
        )
        for t in tasks:
            self.assertEqual(t.correlation_id, cid)
