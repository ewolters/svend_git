"""
Deterministic Tools

Standalone utilities that can be used directly or called by agents.
No LLM required - pure algorithmic analysis.

Tools:
- readability: Text readability scoring (Flesch-Kincaid, Gunning Fog, SMOG, ARI)
- complexity: Code complexity metrics (cyclomatic, cognitive, maintainability)
- schema: JSON Schema validation (type checking, constraints)
- stats: Text/code statistics (word count, functions, imports)
- grammar: Grammar and style checking (spelling, grammar rules, style)
"""

from .readability import ReadabilityScorer, ReadabilityResult, analyze_readability
from .complexity import CodeComplexity, ComplexityResult, analyze_complexity
from .schema import SchemaValidator, ValidationResult, validate_json
from .stats import TextStats, CodeStats, analyze_text, analyze_code
from .grammar import GrammarChecker, GrammarResult, check_grammar

__all__ = [
    # Readability
    "ReadabilityScorer",
    "ReadabilityResult",
    "analyze_readability",
    # Complexity
    "CodeComplexity",
    "ComplexityResult",
    "analyze_complexity",
    # Schema
    "SchemaValidator",
    "ValidationResult",
    "validate_json",
    # Stats
    "TextStats",
    "CodeStats",
    "analyze_text",
    "analyze_code",
    # Grammar
    "GrammarChecker",
    "GrammarResult",
    "check_grammar",
]
