"""
Centralized LLM call interface for all QMS tool modules.

Wraps LLMManager with a simpler API, consistent error handling,
and context-based temperature tuning.

Standard:     LLM-001 §3 (LLM Integration)
Compliance:   BILL-001 §4 (Tier-Based Model Selection)

Usage:
    from llm.service import llm_service

    result = llm_service.chat(request.user, "Analyze this defect pattern",
        system="You are a quality engineer.",
        context="analysis",
        max_tokens=500)

    if result.success:
        logger.info(result.content)
    elif result.rate_limited:
        return JsonResponse({"error": "Rate limit exceeded"}, status=429)
"""

import logging
from dataclasses import dataclass

from .manager import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Immutable result from an LLM call.

    Always returned by LLMService.chat() — never None, never raises.
    Check ``success`` before using ``content``.
    """

    content: str
    model: str
    success: bool
    rate_limited: bool
    error: str
    input_tokens: int
    output_tokens: int


class LLMService:
    """Simplified LLM interface wrapping LLMManager.

    Provides context-based temperature defaults, automatic message
    wrapping, and uniform LLMResult return type.
    """

    # Context-based temperature defaults
    CONTEXT_TEMPERATURES = {
        "analysis": 0.3,  # Structured reasoning
        "critique": 0.5,  # Evaluative
        "chat": 0.7,  # Conversational
        "generation": 0.7,  # Content creation
    }

    DEFAULT_TEMPERATURE = 0.7

    def chat(
        self,
        user,
        prompt: str = "",
        *,
        system: str | None = None,
        context: str = "chat",
        max_tokens: int = 4096,
        temperature: float | None = None,
        messages: list[dict] | None = None,
        model_override: str | None = None,
        skip_rate_limit: bool = False,
    ) -> LLMResult:
        """Single LLM call. Returns LLMResult (never None).

        If ``messages`` is provided, uses it directly (multi-turn).
        Otherwise, wraps ``prompt`` as a single user message.
        Temperature defaults based on context if not explicitly set.

        Args:
            model_override: Force a specific model ID (bypasses tier selection).
                Use for internal/staff endpoints that need a specific model.
            skip_rate_limit: Skip rate limit check. Use for system/scheduled
                tasks and internal endpoints only.
        """
        # Resolve temperature
        if temperature is None:
            temperature = self.CONTEXT_TEMPERATURES.get(context, self.DEFAULT_TEMPERATURE)

        # Build messages
        if messages is not None:
            msg_list = messages
        else:
            msg_list = [{"role": "user", "content": prompt}]

        # Determine model (for result reporting on failure)
        model = model_override or self.get_model_for_user(user)

        try:
            raw = LLMManager.chat(
                user,
                msg_list,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                skip_rate_limit=skip_rate_limit,
                model=model_override,
            )
        except Exception as exc:
            logger.exception("LLMManager.chat() raised unexpectedly")
            return LLMResult(
                content="",
                model=model,
                success=False,
                rate_limited=False,
                error=str(exc),
                input_tokens=0,
                output_tokens=0,
            )

        # Total failure — None
        if raw is None:
            return LLMResult(
                content="",
                model=model,
                success=False,
                rate_limited=False,
                error="LLM request failed",
                input_tokens=0,
                output_tokens=0,
            )

        # Rate-limited response
        if raw.get("rate_limited"):
            return LLMResult(
                content="",
                model=model,
                success=False,
                rate_limited=True,
                error=raw.get("error", "Rate limit exceeded"),
                input_tokens=0,
                output_tokens=0,
            )

        # Success
        usage = raw.get("usage", {})
        return LLMResult(
            content=raw.get("content", ""),
            model=raw.get("model", model),
            success=True,
            rate_limited=False,
            error="",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    def get_model_for_user(self, user) -> str:
        """Return model name for user's tier."""
        return LLMManager.get_model_for_user(user)


# Module-level singleton
llm_service = LLMService()
