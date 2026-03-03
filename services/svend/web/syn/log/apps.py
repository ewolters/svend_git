"""
Django app configuration for the logging system.

Standard: LOG-002 §4
Compliance: NIST SP 800-53 AU-2, AU-3, AU-9 / ISO 27001 A.12.4.1-4
"""

from django.apps import AppConfig


class LogConfig(AppConfig):
    """
    Configuration for the log app.

    Provides LOG-001/002 compliant structured logging with:
    - LogEntry: Core log entry model (LOG-002 §4.2)
    - LogStream: Logical grouping with retention (LOG-002 §4.3)
    - LogAlert: Alert configuration (LOG-002 §4.4)
    - LogMetric: Aggregated metrics (LOG-002 §4.5)

    Compliance:
    - NIST SP 800-53 AU-2, AU-3, AU-9: Audit controls
    - ISO 27001 A.12.4.1-4: Event logging controls
    """

    default_auto_field = "django.db.models.UUIDField"
    name = "syn.log"
    label = "syn_log"
    verbose_name = "Synara Logging System"

    def ready(self):
        """Initialize logging subsystem."""
        # Import signal handlers if needed
        pass
