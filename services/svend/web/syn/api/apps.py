"""
Django app configuration for the API surface module.

Standard: API-001, API-002 §4
Compliance: NIST SP 800-53 SC-8 / ISO 27001 A.13.1 / SOC 2 CC6.1
"""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    """
    Configuration for the API surface app.

    Provides API-001/002 compliant surface conventions with:
    - Required headers middleware (API-002 §8)
    - Cursor pagination (API-002 §7)
    - Idempotency handling (API-002 §9)
    - Standard error envelope (API-002 §10)
    - Runtime operations endpoints (API-002 §17)

    Compliance:
    - NIST SP 800-53 SC-8: Transmission Confidentiality
    - ISO 27001 A.13.1: Network Security Management
    - SOC 2 CC6.1: Logical Access Controls
    """

    default_auto_field = "django.db.models.UUIDField"
    name = "syn.api"
    label = "syn_api"
    verbose_name = "Synara API Surface"

    def ready(self):
        """Initialize API subsystem."""
        # Import signal handlers if needed
        pass
