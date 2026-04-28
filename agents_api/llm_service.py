"""Shim — re-exports from llm.service.

Canonical home is now llm/service.py.
This shim keeps internal agents_api imports working during extraction.
"""

from llm.service import LLMResult, LLMService, llm_service

__all__ = ["LLMResult", "LLMService", "llm_service"]
