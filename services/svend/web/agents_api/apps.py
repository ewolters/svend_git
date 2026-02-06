"""Agents API Django App Configuration.

Preloads LLMs at startup for fast inference.
"""

import logging
import os
import threading

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AgentsApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "agents_api"
    verbose_name = "Agents API"

    def ready(self):
        """Called when Django starts - preload LLMs."""
        import sys

        # Skip during migrations, shell, or other management commands
        if any(cmd in sys.argv for cmd in ['migrate', 'makemigrations', 'collectstatic', 'shell', 'dbshell']):
            return

        # Preload in background thread to not block startup
        thread = threading.Thread(target=self._preload_llms, daemon=True)
        thread.start()
        logger.info("LLM preload thread started")

    def _preload_llms(self):
        """Preload both LLMs on GPU."""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available - skipping LLM preload")
                return

            logger.info("=" * 60)
            logger.info("PRELOADING LLMs AT STARTUP")
            logger.info("=" * 60)

            from .views import get_shared_llm, get_coder_llm

            # Load shared LLM (Qwen 7B)
            logger.info("Loading shared LLM (Qwen)...")
            shared_llm = get_shared_llm()
            if shared_llm:
                logger.info(f"Shared LLM loaded: {type(shared_llm).__name__}")
            else:
                logger.error("Failed to load shared LLM")

            # Load coder LLM (Qwen Coder)
            logger.info("Loading coder LLM (Qwen Coder)...")
            coder_llm = get_coder_llm()
            if coder_llm:
                logger.info(f"Coder LLM loaded: {type(coder_llm).__name__}")
            else:
                logger.warning("Coder LLM not loaded (may share with base)")

            # Log GPU memory usage
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU Memory: {mem_used:.1f}GB / {mem_total:.1f}GB")

            logger.info("=" * 60)
            logger.info("LLM PRELOAD COMPLETE")
            logger.info("=" * 60)

        except Exception as e:
            logger.exception(f"LLM preload failed: {e}")
