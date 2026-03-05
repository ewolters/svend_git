"""
Synara Audit App Configuration (AUD-001)
=========================================

Django app configuration for audit system.

Standard:     AUD-001 (Audit Logging)
Compliance:   SOC 2 CC7.2, ISO 27001 A.12.7
Architecture: SBL-001 (Event-driven)
"""

import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AuditConfig(AppConfig):
    """
    Configuration for the audit app.

    Provides tamper-proof audit logging with hash chain integrity.

    SBL-001 Compliance:
    - Registers signal handlers for automatic event emission
    - All model changes emit Cortex events
    - Events are forwarded to SIEM for compliance monitoring

    Compliance:
    - SOC 2 CC7.2: System activity monitoring and logging
    - ISO 27001 A.12.7: Audit log protection
    - NIST SP 800-53 AU-2: Audit event selection
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "syn.audit"
    label = "audit"
    verbose_name = "Synara Audit System"

    def ready(self):
        """
        Initialize audit subsystem.

        SBL-001: Register signal handlers for event emission.
        """
        # Import signals immediately to register handlers
        import syn.audit.signals  # noqa: F401

        logger.info("[AUDIT] Audit signals registered (SBL-001 compliant)")
