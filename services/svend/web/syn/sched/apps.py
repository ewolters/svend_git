"""
Synara Cognitive Scheduler Django App Configuration
====================================================

Standard: SCH-001/002
"""

from django.apps import AppConfig


class SchedConfig(AppConfig):
    """
    Django app configuration for syn.sched.

    Implements cognitive scheduling per SCH-001/002:
    - Priority-based task scheduling with cognitive scoring
    - Circuit breaker pattern for external services
    - Dead Letter Queue (DLQ) for failed tasks
    - Cascade throttling and budget enforcement
    - Tenant isolation and quota enforcement

    This module is designed to replace Celery with a cognitive-aware
    scheduler that integrates with Synara's governance and reflex layers.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "syn.sched"
    verbose_name = "Synara Cognitive Scheduler"
    label = "sched"

    def ready(self):
        """
        Initialize scheduler module on Django startup.

        Registers task handlers and event listeners.
        """
        try:
            from syn.sched.svend_tasks import register_svend_tasks
            register_svend_tasks()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to register task handlers: {e}")
