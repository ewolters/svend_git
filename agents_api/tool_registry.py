"""Shim — re-exports from tools.registry (CR-0.9)."""

from tools.registry import register_tool, register_tools

__all__ = ["register_tools", "register_tool"]
