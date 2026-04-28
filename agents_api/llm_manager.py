"""Shim — re-exports from llm.manager.

Canonical home is now llm/manager.py.
This shim keeps internal agents_api imports working during extraction.
"""

from llm.manager import CLAUDE_MODELS, TIER_MODEL_MAP, LLMManager

__all__ = ["CLAUDE_MODELS", "TIER_MODEL_MAP", "LLMManager"]
