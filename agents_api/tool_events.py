"""Shim — re-exports from tools.events (CR-0.9).

All internal agents_api code uses ``from .tool_events import tool_events``.
This shim preserves that import path during extraction.
"""

from tools.events import ToolEvent, ToolEventBus, tool_events

__all__ = ["ToolEvent", "ToolEventBus", "tool_events"]
