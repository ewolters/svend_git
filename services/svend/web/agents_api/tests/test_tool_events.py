"""
ToolEventBus behavioral tests.

Standard:     ARCH-001 §4 (Event-Driven Integration)
Compliance:   TST-001 §4

<!-- test: agents_api.tests.test_tool_events.ToolEventEmitTests -->
<!-- test: agents_api.tests.test_tool_events.ToolEventWildcardTests -->
<!-- test: agents_api.tests.test_tool_events.ToolEventIsolationTests -->
<!-- test: agents_api.tests.test_tool_events.ToolEventSubscribeTests -->
<!-- test: agents_api.tests.test_tool_events.ToolEventDataclassTests -->

CR: 8efac44d
"""

from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from agents_api.tool_events import ToolEvent, ToolEventBus

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(calls_list, label=None):
    """Return a handler that appends to *calls_list* when invoked."""

    def handler(event):
        calls_list.append((label or handler.__name__, event))

    handler.__qualname__ = label or "test_handler"
    return handler


def _boom_handler(event):
    """Handler that always raises."""
    raise ValueError("boom")


_boom_handler.__qualname__ = "_boom_handler"


# ---------------------------------------------------------------------------
# ToolEventEmitTests
# ---------------------------------------------------------------------------


@SECURE_OFF
class ToolEventEmitTests(SimpleTestCase):
    def setUp(self):
        self.bus = ToolEventBus()

    def tearDown(self):
        self.bus.clear()

    def test_emit_calls_registered_handler(self):
        calls = []
        self.bus.on("fmea.row_updated")(_make_handler(calls, "h1"))
        self.bus.emit("fmea.row_updated", record="rec")
        self.assertEqual(len(calls), 1)
        event = calls[0][1]
        self.assertIsInstance(event, ToolEvent)
        self.assertEqual(event.name, "fmea.row_updated")

    def test_emit_passes_record_and_user(self):
        calls = []
        self.bus.on("x.y")(_make_handler(calls))
        sentinel_record = object()
        sentinel_user = object()
        self.bus.emit("x.y", sentinel_record, user=sentinel_user)
        event = calls[0][1]
        self.assertIs(event.record, sentinel_record)
        self.assertIs(event.user, sentinel_user)

    def test_emit_passes_extra_kwargs(self):
        calls = []
        self.bus.on("x.y")(_make_handler(calls))
        self.bus.emit("x.y", "rec", user=None, rpn=120, severity=8)
        event = calls[0][1]
        self.assertEqual(event.extra["rpn"], 120)
        self.assertEqual(event.extra["severity"], 8)

    def test_emit_with_no_handlers_does_not_raise(self):
        # No handler registered — should be a no-op.
        self.bus.emit("nonexistent.event", record=None)

    def test_multiple_handlers_all_called(self):
        calls = []
        self.bus.on("e")(_make_handler(calls, "h1"))
        self.bus.on("e")(_make_handler(calls, "h2"))
        self.bus.on("e")(_make_handler(calls, "h3"))
        self.bus.emit("e", "rec")
        labels = [c[0] for c in calls]
        self.assertEqual(labels, ["h1", "h2", "h3"])


# ---------------------------------------------------------------------------
# ToolEventWildcardTests
# ---------------------------------------------------------------------------


@SECURE_OFF
class ToolEventWildcardTests(SimpleTestCase):
    def setUp(self):
        self.bus = ToolEventBus()

    def tearDown(self):
        self.bus.clear()

    def test_star_dot_pattern_matches(self):
        calls = []
        self.bus.on("*.completed")(_make_handler(calls))
        self.bus.emit("fmea.completed", "rec")
        self.assertEqual(len(calls), 1)

    def test_prefix_star_pattern_matches(self):
        calls = []
        self.bus.on("fmea.*")(_make_handler(calls))
        self.bus.emit("fmea.row_updated", "rec")
        self.assertEqual(len(calls), 1)

    def test_wildcard_does_not_match_wrong_event(self):
        calls = []
        self.bus.on("*.completed")(_make_handler(calls))
        self.bus.emit("fmea.created", "rec")
        self.assertEqual(len(calls), 0)

    def test_exact_and_wildcard_both_called(self):
        calls = []
        self.bus.on("fmea.created")(_make_handler(calls, "exact"))
        self.bus.on("fmea.*")(_make_handler(calls, "wild"))
        self.bus.emit("fmea.created", "rec")
        labels = [c[0] for c in calls]
        self.assertIn("exact", labels)
        self.assertIn("wild", labels)
        self.assertEqual(len(labels), 2)


# ---------------------------------------------------------------------------
# ToolEventIsolationTests
# ---------------------------------------------------------------------------


@SECURE_OFF
class ToolEventIsolationTests(SimpleTestCase):
    def setUp(self):
        self.bus = ToolEventBus()

    def tearDown(self):
        self.bus.clear()

    def test_handler_exception_does_not_block_others(self):
        calls = []
        self.bus.on("e")(_make_handler(calls, "h1"))
        self.bus.on("e")(_boom_handler)
        self.bus.on("e")(_make_handler(calls, "h3"))
        self.bus.emit("e", "rec")
        labels = [c[0] for c in calls]
        self.assertEqual(labels, ["h1", "h3"])

    def test_handler_exception_is_logged(self):
        self.bus.on("e")(_boom_handler)
        with patch("agents_api.tool_events.logger") as mock_logger:
            self.bus.emit("e", "rec")
            mock_logger.exception.assert_called_once()
            args = mock_logger.exception.call_args[0]
            self.assertIn("_boom_handler", args[1])
            self.assertIn("e", args[2])


# ---------------------------------------------------------------------------
# ToolEventSubscribeTests
# ---------------------------------------------------------------------------


@SECURE_OFF
class ToolEventSubscribeTests(SimpleTestCase):
    def setUp(self):
        self.bus = ToolEventBus()

    def tearDown(self):
        self.bus.clear()

    def test_subscribe_programmatic(self):
        calls = []
        handler = _make_handler(calls, "prog")
        self.bus.subscribe("fmea.done", handler)
        self.bus.emit("fmea.done", "rec")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "prog")

    def test_clear_removes_all_handlers(self):
        calls = []
        self.bus.on("a")(_make_handler(calls, "exact"))
        self.bus.on("b.*")(_make_handler(calls, "wild"))
        self.bus.clear()
        self.bus.emit("a", "rec")
        self.bus.emit("b.x", "rec")
        self.assertEqual(len(calls), 0)


# ---------------------------------------------------------------------------
# ToolEventDataclassTests
# ---------------------------------------------------------------------------


@SECURE_OFF
class ToolEventDataclassTests(SimpleTestCase):
    def test_tool_event_fields(self):
        event = ToolEvent(name="fmea.row_updated", record="rec", user="usr", extra={"k": "v"})
        self.assertEqual(event.name, "fmea.row_updated")
        self.assertEqual(event.record, "rec")
        self.assertEqual(event.user, "usr")
        self.assertEqual(event.extra, {"k": "v"})

    def test_tool_event_extra_defaults_to_empty_dict(self):
        event = ToolEvent(name="x", record=None, user=None)
        self.assertEqual(event.extra, {})
