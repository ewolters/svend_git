"""Safety app — HIRARC program management."""

from django.apps import AppConfig


class SafetyConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "safety"
    verbose_name = "HIRARC Safety Management"
