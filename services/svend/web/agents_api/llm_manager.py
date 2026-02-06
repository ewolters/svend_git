"""Centralized LLM management for Svend.

All LLM instantiation goes through this manager. This provides:
- Single source of truth for model loading
- Thread-safe singleton pattern
- Configurable model selection via environment
- Tier-based Claude model selection

Usage:
    from agents_api.llm_manager import LLMManager

    # Get Claude model based on user tier
    response = LLMManager.chat(user, messages)

    # Get specific Claude model
    client = LLMManager.get_anthropic()
    response = client.messages.create(model="claude-3-5-haiku-20241022", ...)

Model Selection by Tier:
    - Free/Founder: claude-3-5-haiku-20241022
    - Pro/Team: claude-sonnet-4-20250514
    - Enterprise: claude-opus-4-20250514

Note: Custom Qwen/DeepSeek models temporarily disabled while testing Synara.
"""

import logging
import os
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)


# Model mapping by tier
CLAUDE_MODELS = {
    "haiku": "claude-3-5-haiku-20241022",
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}

TIER_MODEL_MAP = {
    "FREE": "haiku",
    "FOUNDER": "haiku",
    "PRO": "sonnet",
    "TEAM": "sonnet",
    "ENTERPRISE": "opus",
}


class LLMManager:
    """Singleton manager for Claude API access.

    Thread-safe lazy loading. Configure via environment:
    - ANTHROPIC_API_KEY: Required for all LLM access
    - SVEND_DEFAULT_MODEL: Override default model (haiku/sonnet/opus)

    Custom Qwen/DeepSeek models are temporarily disabled.
    """

    _instance = None
    _lock = threading.Lock()

    # Anthropic client
    _anthropic_client = None
    _anthropic_loaded = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_anthropic(cls) -> Optional[Any]:
        """Get Anthropic client.

        Returns None if ANTHROPIC_API_KEY not set.
        """
        with cls._lock:
            if cls._anthropic_loaded:
                return cls._anthropic_client

            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set - LLM features unavailable")
                cls._anthropic_loaded = True
                return None

            try:
                import anthropic
                cls._anthropic_client = anthropic.Anthropic(api_key=api_key)
                cls._anthropic_loaded = True
                logger.info("Anthropic client initialized")
                return cls._anthropic_client

            except ImportError:
                logger.error("anthropic package not installed - run: pip install anthropic")
                cls._anthropic_loaded = True
                return None
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                cls._anthropic_loaded = True
                return None

    @classmethod
    def get_model_for_tier(cls, tier: str) -> str:
        """Get the Claude model ID for a given user tier."""
        model_key = TIER_MODEL_MAP.get(tier.upper(), "haiku")
        return CLAUDE_MODELS[model_key]

    @classmethod
    def get_model_for_user(cls, user) -> str:
        """Get the Claude model ID for a user based on their subscription."""
        try:
            tier = user.subscription_tier if hasattr(user, 'subscription_tier') else "FREE"
            return cls.get_model_for_tier(tier)
        except Exception:
            return CLAUDE_MODELS["haiku"]

    @classmethod
    def chat(
        cls,
        user,
        messages: list[dict],
        system: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> Optional[dict]:
        """Send a chat request using the appropriate Claude model for the user's tier.

        Args:
            user: Django user object (used to determine tier)
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to Anthropic API

        Returns:
            dict with 'content' (str), 'model' (str), 'usage' (dict)
            or None if request fails
        """
        client = cls.get_anthropic()
        if not client:
            return None

        model = cls.get_model_for_user(user)

        try:
            create_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
                **kwargs,
            }
            if system:
                create_kwargs["system"] = system

            response = client.messages.create(**create_kwargs)

            return {
                "content": response.content[0].text if response.content else "",
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            }

        except Exception as e:
            logger.error(f"Claude API request failed: {e}")
            return None

    @classmethod
    def anthropic_available(cls) -> bool:
        """Check if Anthropic API is available."""
        return bool(os.environ.get('ANTHROPIC_API_KEY'))

    @classmethod
    def reset(cls):
        """Reset cached client (for testing)."""
        with cls._lock:
            cls._anthropic_client = None
            cls._anthropic_loaded = False

    @classmethod
    def status(cls) -> dict:
        """Get status of LLM configuration."""
        return {
            "anthropic": {
                "loaded": cls._anthropic_loaded,
                "available": cls._anthropic_client is not None,
                "api_key_set": cls.anthropic_available(),
            },
            "models": CLAUDE_MODELS,
            "tier_mapping": TIER_MODEL_MAP,
            "custom_llm": "disabled",  # Qwen/DeepSeek temporarily disabled
        }

    # =========================================================================
    # Deprecated methods - kept for backwards compatibility during transition
    # =========================================================================

    @classmethod
    def get_shared(cls) -> Optional[Any]:
        """DEPRECATED: Custom LLMs disabled. Use get_anthropic() instead."""
        logger.warning("get_shared() is deprecated - custom LLMs temporarily disabled")
        return None

    @classmethod
    def get_coder(cls) -> Optional[Any]:
        """DEPRECATED: Custom LLMs disabled. Use get_anthropic() instead."""
        logger.warning("get_coder() is deprecated - custom LLMs temporarily disabled")
        return None


# Convenience functions - deprecated but kept for compatibility
def get_shared_llm():
    """DEPRECATED: Use LLMManager.chat() or LLMManager.get_anthropic() instead."""
    logger.warning("get_shared_llm() is deprecated - custom LLMs temporarily disabled")
    return None


def get_coder_llm():
    """DEPRECATED: Use LLMManager.chat() or LLMManager.get_anthropic() instead."""
    logger.warning("get_coder_llm() is deprecated - custom LLMs temporarily disabled")
    return None
