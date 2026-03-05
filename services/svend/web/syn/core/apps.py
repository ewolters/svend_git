"""Django app configuration for syn.core."""

from django.apps import AppConfig


class CoreConfig(AppConfig):
    """Configuration for the core utilities app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "syn.core"
    label = "syn_core"
    verbose_name = "Synara Core Utilities"
