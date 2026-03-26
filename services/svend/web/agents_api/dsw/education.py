"""DSW Education module — thin redirect to analysis/education.

The canonical education content now lives in agents_api.analysis.education.
This module preserves backward compatibility.

Migration: CR 07425b56 (2026-03-26)
"""

# Re-export canonical education content
from agents_api.analysis.education import (  # noqa: F401
    EDUCATION_CONTENT,
    _extend,
    get_education,
)
