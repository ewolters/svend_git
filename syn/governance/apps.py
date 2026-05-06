"""
Synara Governance Django App Configuration (GOV-002)
"""

from django.apps import AppConfig


class GovernanceConfig(AppConfig):
    """
    Django app configuration for syn.governance.

    Governance rules, judgments, approval workflows.
    Standard: GOV-001/002, POL-001/002
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "syn.governance"
    verbose_name = "Synara Governance"
    label = "governance"
