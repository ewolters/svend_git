"""Loop — Closed-loop operating model (LOOP-001).

Three mechanisms: Signals, Mode Transitions, Commitments.
QMS Policy service: org-defined rules that inform system behavior.
"""

from django.apps import AppConfig


class LoopConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "loop"
    verbose_name = "Loop — Closed-Loop Operating Model"
