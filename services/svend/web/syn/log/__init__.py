"""
Synara Logging Module (LOG-001/002)
===================================

LOG-001/002 compliant structured logging with event tracking,
stream management, alerting, and metrics aggregation.

Standard:     LOG-001 (Conceptual), LOG-002 (Implementation)
Compliance:   NIST SP 800-53 AU-2/AU-3/AU-9, ISO 27001 A.12.4.1-4
Location:     syn/log/
Version:      1.0.0

Features:
---------
- LogEntry: Core structured log entries (LOG-002 §4.2)
- LogStream: Logical grouping with retention policies (LOG-002 §4.3)
- LogAlert: Alert configuration for monitoring (LOG-002 §4.4)
- LogMetric: Aggregated metrics for dashboards (LOG-002 §4.5)
- Events: Logging lifecycle and alert events (LOG-002 §4)

Constants:
- LOG_LEVEL_CHOICES: Valid log levels
- LOG_LEVEL_NUMERIC: Numeric level mapping
- LOG_SOURCE_CHOICES: Valid log sources
- AGGREGATION_SINK_CHOICES: Valid aggregation sinks

Usage:
------
    from syn.log import (
        LogEntry,
        LogStream,
        LogAlert,
        LogMetric,
        emit_log_event,
        LOG_EVENTS,
        LOG_LEVEL_CHOICES,
    )

    # Create a log stream
    stream = LogStream.objects.create(
        name="application",
        retention_days=90,
        min_level="INFO",
    )

    # Create a log entry
    entry = LogEntry.objects.create(
        stream=stream,
        level="INFO",
        logger="syn.cortex.publisher",
        message="Event published successfully",
        correlation_id=uuid.uuid4(),
    )

    # Configure an alert
    alert = LogAlert.objects.create(
        name="High Error Rate",
        stream=stream,
        level="ERROR",
        threshold_count=10,
        threshold_window_minutes=5,
    )
"""

__version__ = "1.0.0"
__standard__ = "LOG-002"


# Import models and utilities directly from submodules:
#   from syn.log.models import LogEntry, LogStream, LogAlert, LogMetric
#   from syn.log.models import LOG_LEVEL_CHOICES, LOG_SOURCE_CHOICES
#   from syn.log.events import emit_log_event, LOG_EVENTS
#   from syn.log.handlers import SynaraLogHandler, get_synara_logger
#   from syn.log.middleware import CorrelationMiddleware
