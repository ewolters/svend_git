"""
⟡  S Y N A R A
   Event Schema Registry App Configuration
"""

from django.apps import AppConfig


class EventsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "syn.events"
    verbose_name = "Event Schema Registry"
