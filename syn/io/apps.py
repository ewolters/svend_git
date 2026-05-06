"""
IO App Configuration
====================

Django app configuration for the Input/Output Contract module.

Standard: IO-001/002
"""

from django.apps import AppConfig


class IoConfig(AppConfig):
    """Django app config for syn.io module."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "syn.io"
    label = "io"
    verbose_name = "Input/Output Contracts"
