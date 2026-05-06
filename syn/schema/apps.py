"""
Django app configuration for schema registry.
"""

from django.apps import AppConfig


class SchemaConfig(AppConfig):
    """
    Configuration for the schema registry app.

    Provides event schema management and validation.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "syn.schema"
    verbose_name = "Event Schema Registry"
