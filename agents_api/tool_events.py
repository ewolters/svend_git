"""
Domain event bus for QMS tool lifecycle integration.

Tools emit events, handlers perform cross-cutting concerns (evidence
creation, project logging, investigation bridging) without coupling.

Standard:     ARCH-001 §4 (Event-Driven Integration)
Compliance:   AUD-001 §3 (Audit Trail)

Usage:
    from agents_api.tool_events import tool_events

    # Subscribe (at module import time)
    @tool_events.on("fmea.row_updated")
    def on_fmea_row_updated(record, user, **kwargs):
        create_tool_evidence(...)

    # Subscribe to wildcards
    @tool_events.on("*.completed")
    def on_any_tool_completed(record, user, **kwargs):
        record.project.log_event("tool_completed", ...)

    # Emit (in view functions)
    tool_events.emit("fmea.row_updated", record, user=request.user,
                     extra={"rpn": row.rpn})
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolEvent:
    """Emitted event payload."""

    name: str  # e.g. "fmea.row_updated"
    record: Any  # The model instance
    user: Any  # request.user or None
    extra: dict = field(default_factory=dict)  # Additional context


class ToolEventBus:
    """Lightweight pub/sub for QMS tool domain events."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}
        self._wildcard_handlers: list[tuple[str, Callable]] = []

    def on(self, event_pattern: str):
        """Decorator to subscribe a handler to an event pattern.

        Supports exact match ("fmea.row_updated") and wildcards
        ("*.completed", "fmea.*").
        """

        def decorator(fn):
            if "*" in event_pattern:
                self._wildcard_handlers.append((event_pattern, fn))
            else:
                self._handlers.setdefault(event_pattern, []).append(fn)
            return fn

        return decorator

    def subscribe(self, event_pattern: str, handler: Callable):
        """Programmatic subscription (non-decorator)."""
        if "*" in event_pattern:
            self._wildcard_handlers.append((event_pattern, handler))
        else:
            self._handlers.setdefault(event_pattern, []).append(handler)

    def emit(self, event_name: str, record, *, user=None, **extra):
        """Emit an event. All matching handlers are called synchronously.

        Handlers are called in registration order. If a handler raises,
        the error is logged but does NOT propagate -- other handlers
        still execute (fault isolation).
        """
        event = ToolEvent(name=event_name, record=record, user=user, extra=extra)

        handlers = list(self._handlers.get(event_name, []))
        for pattern, handler in self._wildcard_handlers:
            if fnmatch.fnmatch(event_name, pattern):
                handlers.append(handler)

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("Event handler %s failed for %s", handler.__qualname__, event_name)

    def clear(self):
        """Remove all handlers. Used in tests."""
        self._handlers.clear()
        self._wildcard_handlers.clear()


# Module-level singleton
tool_events = ToolEventBus()
