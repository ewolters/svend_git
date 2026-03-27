"""Agents API Django App Configuration."""

import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AgentsApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "agents_api"
    verbose_name = "Agents API"

    def ready(self):
        """Called when Django starts - register event handlers."""
        # Register ToolEventBus handlers (ARCH-001 §10.2)
        import agents_api.tool_event_handlers  # noqa: F401
