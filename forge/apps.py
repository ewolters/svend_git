"""Forge app config."""

from django.apps import AppConfig


class ForgeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "forge"
    verbose_name = "Forge - Synthetic Data Generation"

    def ready(self):
        # Import task handlers to register them with Tempora
        from forge import tasks  # noqa
