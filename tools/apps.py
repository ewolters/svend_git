"""Tools Django App Configuration.

Formalized tool router, event bus, and registration system for QMS modules.
Extracted from agents_api in CR-0.9.
"""

import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ToolsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "tools"
    verbose_name = "QMS Tools"

    def ready(self):
        """Register cross-cutting event handlers at startup."""
        import tools.handlers  # noqa: F401
