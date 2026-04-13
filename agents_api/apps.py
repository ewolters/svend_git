"""Agents API Django App Configuration."""

import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AgentsApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "agents_api"
    verbose_name = "Agents API"

    def ready(self):
        """Called when Django starts."""
        # ToolEventBus handlers moved to tools/apps.py (CR-0.9)
        pass
