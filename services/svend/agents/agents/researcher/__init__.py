"""Research Agent - Multi-source research with structured output."""
from .agent import ResearchAgent, ResearchQuery
from .validator import (
    ResearchValidator,
    ValidationResult,
    ClaimValidation,
    SourceConsistency,
    ConfidenceJustification,
    quick_validate,
)

__all__ = [
    "ResearchAgent",
    "ResearchQuery",
    "ResearchValidator",
    "ValidationResult",
    "ClaimValidation",
    "SourceConsistency",
    "ConfidenceJustification",
    "quick_validate",
]
