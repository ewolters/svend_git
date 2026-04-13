"""Shim — handlers now live in tools.handlers (CR-0.9).

This module is imported by agents_api.apps.ready() for backward compat.
The actual handler registration happens in tools.apps.ready() via
tools.handlers import.
"""
