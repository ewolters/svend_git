"""Loop — Closed-loop operating model (LOOP-001).

Three mechanisms: Signals, Mode Transitions, Commitments.
QMS Policy service: org-defined rules that inform system behavior.
PolicyEvaluator: real-time + aggregate policy evaluation.
"""

import logging

from django.apps import AppConfig

logger = logging.getLogger("svend.loop")


class LoopConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "loop"
    verbose_name = "Loop — Closed-Loop Operating Model"

    def ready(self):
        """Register PolicyEvaluator handlers on startup."""
        from .evaluator import register_policy_handlers

        register_policy_handlers()
        logger.info("Loop: PolicyEvaluator handlers registered")
