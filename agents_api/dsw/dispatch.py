"""DSW Analysis dispatcher — thin redirect to analysis/dispatch.py.

The canonical dispatcher now lives in agents_api.analysis.dispatch.
This module preserves backward compatibility for any code that imports
from agents_api.dsw.dispatch directly.

Migration: CR d9c36a0b (2026-03-26)
"""

# Re-export the canonical dispatcher so existing imports keep working.
from agents_api.analysis.dispatch import (  # noqa: F401
    _dispatch_analysis,
    _log_rejection,
    _read_csv_safe,
    _resolve_data,
    run_analysis,
)
