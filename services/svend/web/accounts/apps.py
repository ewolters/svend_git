"""Accounts app configuration."""

from django.apps import AppConfig


class AccountsConfig(AppConfig):
    """Django app configuration for the Accounts module."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "accounts"
