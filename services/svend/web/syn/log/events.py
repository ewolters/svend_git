"""
Log events catalog for LOG-001/002 compliant logging.

Standard: LOG-002 §4, EVT-001 §5
Compliance: NIST SP 800-53 AU-2, AU-3 / ISO 27001 A.12.4.1

Events:
- Entry lifecycle: created, archived, deleted
- Stream lifecycle: created, updated, deactivated
- Alert lifecycle: configured, triggered, resolved
- Metric lifecycle: computed, threshold_exceeded
- Retention: cleanup, archive
- Error handling: write_failed, query_failed
"""

import uuid
from datetime import datetime
from typing import Any

# =============================================================================
# Event Catalog (EVT-001 §5)
# =============================================================================

LOG_EVENTS = {
    # -------------------------------------------------------------------------
    # Entry Events (LOG-002 §4.2)
    # -------------------------------------------------------------------------
    "log.entry.created": {
        "description": "A new log entry was persisted",
        "category": "entry",
        "severity": "info",
        "payload_schema": {
            "log_id": "UUID",
            "stream_name": "str",
            "level": "str",
            "logger": "str",
            "correlation_id": "UUID",
            "tenant_id": "UUID | None",
        },
    },
    "log.entry.archived": {
        "description": "A log entry was archived to cold storage",
        "category": "entry",
        "severity": "info",
        "payload_schema": {
            "log_id": "UUID",
            "stream_name": "str",
            "archived_at": "datetime",
        },
    },
    "log.entry.deleted": {
        "description": "A log entry was deleted (retention policy)",
        "category": "entry",
        "severity": "info",
        "payload_schema": {
            "log_id": "UUID",
            "stream_name": "str",
            "reason": "str",
        },
    },
    # -------------------------------------------------------------------------
    # Stream Events (LOG-002 §4.3)
    # -------------------------------------------------------------------------
    "log.stream.created": {
        "description": "A new log stream was created",
        "category": "stream",
        "severity": "info",
        "payload_schema": {
            "stream_id": "UUID",
            "name": "str",
            "retention_days": "int",
            "min_level": "str",
        },
    },
    "log.stream.updated": {
        "description": "A log stream configuration was updated",
        "category": "stream",
        "severity": "info",
        "payload_schema": {
            "stream_id": "UUID",
            "name": "str",
            "changes": "dict",
        },
    },
    "log.stream.deactivated": {
        "description": "A log stream was deactivated",
        "category": "stream",
        "severity": "warning",
        "payload_schema": {
            "stream_id": "UUID",
            "name": "str",
            "reason": "str",
        },
    },
    # -------------------------------------------------------------------------
    # Alert Events (LOG-002 §4.4)
    # -------------------------------------------------------------------------
    "log.alert.configured": {
        "description": "A log alert was configured",
        "category": "alert",
        "severity": "info",
        "payload_schema": {
            "alert_id": "UUID",
            "name": "str",
            "stream_name": "str",
            "level": "str",
            "threshold_count": "int",
            "threshold_window_minutes": "int",
        },
    },
    "log.alert.triggered": {
        "description": "A log alert threshold was exceeded",
        "category": "alert",
        "severity": "warning",
        "payload_schema": {
            "alert_id": "UUID",
            "alert_name": "str",
            "stream_name": "str",
            "level": "str",
            "count": "int",
            "threshold": "int",
            "window_minutes": "int",
        },
    },
    "log.alert.resolved": {
        "description": "A log alert condition was resolved",
        "category": "alert",
        "severity": "info",
        "payload_schema": {
            "alert_id": "UUID",
            "alert_name": "str",
            "resolved_at": "datetime",
        },
    },
    "log.alert.cooldown_started": {
        "description": "Alert entered cooldown period",
        "category": "alert",
        "severity": "info",
        "payload_schema": {
            "alert_id": "UUID",
            "alert_name": "str",
            "cooldown_minutes": "int",
            "cooldown_ends_at": "datetime",
        },
    },
    # -------------------------------------------------------------------------
    # Metric Events (LOG-002 §4.5)
    # -------------------------------------------------------------------------
    "log.metrics.computed": {
        "description": "Log metrics were computed for a time bucket",
        "category": "metric",
        "severity": "info",
        "payload_schema": {
            "metric_id": "UUID",
            "stream_name": "str",
            "bucket_start": "datetime",
            "bucket_duration_minutes": "int",
            "count": "int",
            "error_count": "int",
        },
    },
    "log.metrics.threshold_exceeded": {
        "description": "Log metrics exceeded a configured threshold",
        "category": "metric",
        "severity": "warning",
        "payload_schema": {
            "stream_name": "str",
            "metric_type": "str",
            "current_value": "float",
            "threshold": "float",
            "bucket_start": "datetime",
        },
    },
    "log.metrics.error_rate_anomaly": {
        "description": "Anomalous error rate detected",
        "category": "metric",
        "severity": "warning",
        "payload_schema": {
            "stream_name": "str",
            "error_rate": "float",
            "baseline_rate": "float",
            "deviation_sigma": "float",
        },
    },
    # -------------------------------------------------------------------------
    # Retention Events (LOG-002 §5.5)
    # -------------------------------------------------------------------------
    "log.retention.cleanup_started": {
        "description": "Retention cleanup job started",
        "category": "retention",
        "severity": "info",
        "payload_schema": {
            "stream_name": "str",
            "retention_days": "int",
            "cutoff_date": "datetime",
        },
    },
    "log.retention.cleanup_completed": {
        "description": "Retention cleanup job completed",
        "category": "retention",
        "severity": "info",
        "payload_schema": {
            "stream_name": "str",
            "entries_deleted": "int",
            "entries_archived": "int",
            "duration_ms": "float",
        },
    },
    "log.retention.archive_started": {
        "description": "Archive job started",
        "category": "retention",
        "severity": "info",
        "payload_schema": {
            "stream_name": "str",
            "archive_days": "int",
            "cutoff_date": "datetime",
        },
    },
    "log.retention.archive_completed": {
        "description": "Archive job completed",
        "category": "retention",
        "severity": "info",
        "payload_schema": {
            "stream_name": "str",
            "entries_archived": "int",
            "archive_location": "str",
            "duration_ms": "float",
        },
    },
    # -------------------------------------------------------------------------
    # SIEM Events (LOG-002 §5.4, SEC-002 §8)
    # -------------------------------------------------------------------------
    "log.siem.forward_started": {
        "description": "SIEM forwarding started for security logs",
        "category": "siem",
        "severity": "info",
        "payload_schema": {
            "stream_name": "str",
            "siem_target": "str",
            "batch_size": "int",
        },
    },
    "log.siem.forward_completed": {
        "description": "SIEM forwarding completed",
        "category": "siem",
        "severity": "info",
        "payload_schema": {
            "stream_name": "str",
            "entries_forwarded": "int",
            "duration_ms": "float",
        },
    },
    "log.siem.forward_failed": {
        "description": "SIEM forwarding failed",
        "category": "siem",
        "severity": "error",
        "payload_schema": {
            "stream_name": "str",
            "error": "str",
            "entries_pending": "int",
        },
    },
    # -------------------------------------------------------------------------
    # Error Events (LOG-002 §9)
    # -------------------------------------------------------------------------
    "log.write.failed": {
        "description": "Failed to write a log entry",
        "category": "error",
        "severity": "error",
        "payload_schema": {
            "logger": "str",
            "level": "str",
            "error": "str",
            "correlation_id": "UUID",
        },
    },
    "log.query.failed": {
        "description": "Failed to execute a log query",
        "category": "error",
        "severity": "error",
        "payload_schema": {
            "query_params": "dict",
            "error": "str",
            "correlation_id": "UUID",
        },
    },
    "log.aggregate.failed": {
        "description": "Failed to compute log aggregation",
        "category": "error",
        "severity": "error",
        "payload_schema": {
            "stream_name": "str",
            "bucket_start": "datetime",
            "error": "str",
        },
    },
    # -------------------------------------------------------------------------
    # Governance Events (GOV-001 §5)
    # -------------------------------------------------------------------------
    "log.governance.alert": {
        "description": "Governance alert for logging subsystem",
        "category": "governance",
        "severity": "warning",
        "payload_schema": {
            "alert_type": "str",
            "message": "str",
            "severity": "str",
            "recommended_action": "str",
        },
    },
}


# =============================================================================
# Event Emission (EVT-001 §6)
# =============================================================================


def emit_log_event(
    event_name: str,
    payload: dict[str, Any],
    correlation_id: uuid.UUID | None = None,
    tenant_id: uuid.UUID | str | None = None,
) -> None:
    """
    Emit a log event to the Cortex event bus.

    Standard: EVT-001 §6
    Compliance: NIST SP 800-53 AU-2

    Args:
        event_name: Event name from LOG_EVENTS catalog
        payload: Event payload matching schema
        correlation_id: Correlation ID for tracing
        tenant_id: Tenant identifier for isolation

    Raises:
        ValueError: If event_name not in catalog
    """
    if event_name not in LOG_EVENTS:
        raise ValueError(
            f"Unknown log event: {event_name}. Must be one of: {list(LOG_EVENTS.keys())}"
        )

    # Build full event payload
    event_payload = {
        "event_name": event_name,
        "correlation_id": str(correlation_id) if correlation_id else str(uuid.uuid4()),
        "tenant_id": str(tenant_id) if tenant_id else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": "syn.log",
        "payload": payload,
    }

    # Import Cortex publisher if available
    try:
        from syn.cortex import publish

        publish(event_name, event_payload)
    except ImportError:
        # Cortex not available, log locally
        import logging

        logger = logging.getLogger("syn.log.events")
        logger.info(f"Log event: {event_name}", extra={"payload": event_payload})


# =============================================================================
# Payload Builders (EVT-001 §6.2)
# =============================================================================


def build_entry_created_payload(
    log_id: uuid.UUID,
    stream_name: str,
    level: str,
    logger: str,
    correlation_id: uuid.UUID,
    tenant_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Build payload for log.entry.created event."""
    return {
        "log_id": str(log_id),
        "stream_name": stream_name,
        "level": level,
        "logger": logger,
        "correlation_id": str(correlation_id),
        "tenant_id": str(tenant_id) if tenant_id else None,
    }


def build_entry_archived_payload(
    log_id: uuid.UUID,
    stream_name: str,
    archived_at: datetime,
) -> dict[str, Any]:
    """Build payload for log.entry.archived event."""
    return {
        "log_id": str(log_id),
        "stream_name": stream_name,
        "archived_at": archived_at.isoformat() + "Z",
    }


def build_entry_deleted_payload(
    log_id: uuid.UUID,
    stream_name: str,
    reason: str,
) -> dict[str, Any]:
    """Build payload for log.entry.deleted event."""
    return {
        "log_id": str(log_id),
        "stream_name": stream_name,
        "reason": reason,
    }


def build_stream_created_payload(
    stream_id: uuid.UUID,
    name: str,
    retention_days: int,
    min_level: str,
) -> dict[str, Any]:
    """Build payload for log.stream.created event."""
    return {
        "stream_id": str(stream_id),
        "name": name,
        "retention_days": retention_days,
        "min_level": min_level,
    }


def build_stream_updated_payload(
    stream_id: uuid.UUID,
    name: str,
    changes: dict[str, Any],
) -> dict[str, Any]:
    """Build payload for log.stream.updated event."""
    return {
        "stream_id": str(stream_id),
        "name": name,
        "changes": changes,
    }


def build_stream_deactivated_payload(
    stream_id: uuid.UUID,
    name: str,
    reason: str,
) -> dict[str, Any]:
    """Build payload for log.stream.deactivated event."""
    return {
        "stream_id": str(stream_id),
        "name": name,
        "reason": reason,
    }


def build_alert_configured_payload(
    alert_id: uuid.UUID,
    name: str,
    stream_name: str,
    level: str,
    threshold_count: int,
    threshold_window_minutes: int,
) -> dict[str, Any]:
    """Build payload for log.alert.configured event."""
    return {
        "alert_id": str(alert_id),
        "name": name,
        "stream_name": stream_name,
        "level": level,
        "threshold_count": threshold_count,
        "threshold_window_minutes": threshold_window_minutes,
    }


def build_alert_triggered_payload(
    alert_id: uuid.UUID,
    alert_name: str,
    stream_name: str,
    level: str,
    count: int,
    threshold: int,
    window_minutes: int,
) -> dict[str, Any]:
    """Build payload for log.alert.triggered event."""
    return {
        "alert_id": str(alert_id),
        "alert_name": alert_name,
        "stream_name": stream_name,
        "level": level,
        "count": count,
        "threshold": threshold,
        "window_minutes": window_minutes,
    }


def build_alert_resolved_payload(
    alert_id: uuid.UUID,
    alert_name: str,
    resolved_at: datetime,
) -> dict[str, Any]:
    """Build payload for log.alert.resolved event."""
    return {
        "alert_id": str(alert_id),
        "alert_name": alert_name,
        "resolved_at": resolved_at.isoformat() + "Z",
    }


def build_alert_cooldown_started_payload(
    alert_id: uuid.UUID,
    alert_name: str,
    cooldown_minutes: int,
    cooldown_ends_at: datetime,
) -> dict[str, Any]:
    """Build payload for log.alert.cooldown_started event."""
    return {
        "alert_id": str(alert_id),
        "alert_name": alert_name,
        "cooldown_minutes": cooldown_minutes,
        "cooldown_ends_at": cooldown_ends_at.isoformat() + "Z",
    }


def build_metrics_computed_payload(
    metric_id: uuid.UUID,
    stream_name: str,
    bucket_start: datetime,
    bucket_duration_minutes: int,
    count: int,
    error_count: int,
) -> dict[str, Any]:
    """Build payload for log.metrics.computed event."""
    return {
        "metric_id": str(metric_id),
        "stream_name": stream_name,
        "bucket_start": bucket_start.isoformat() + "Z",
        "bucket_duration_minutes": bucket_duration_minutes,
        "count": count,
        "error_count": error_count,
    }


def build_metrics_threshold_exceeded_payload(
    stream_name: str,
    metric_type: str,
    current_value: float,
    threshold: float,
    bucket_start: datetime,
) -> dict[str, Any]:
    """Build payload for log.metrics.threshold_exceeded event."""
    return {
        "stream_name": stream_name,
        "metric_type": metric_type,
        "current_value": current_value,
        "threshold": threshold,
        "bucket_start": bucket_start.isoformat() + "Z",
    }


def build_error_rate_anomaly_payload(
    stream_name: str,
    error_rate: float,
    baseline_rate: float,
    deviation_sigma: float,
) -> dict[str, Any]:
    """Build payload for log.metrics.error_rate_anomaly event."""
    return {
        "stream_name": stream_name,
        "error_rate": error_rate,
        "baseline_rate": baseline_rate,
        "deviation_sigma": deviation_sigma,
    }


def build_retention_cleanup_started_payload(
    stream_name: str,
    retention_days: int,
    cutoff_date: datetime,
) -> dict[str, Any]:
    """Build payload for log.retention.cleanup_started event."""
    return {
        "stream_name": stream_name,
        "retention_days": retention_days,
        "cutoff_date": cutoff_date.isoformat() + "Z",
    }


def build_retention_cleanup_completed_payload(
    stream_name: str,
    entries_deleted: int,
    entries_archived: int,
    duration_ms: float,
) -> dict[str, Any]:
    """Build payload for log.retention.cleanup_completed event."""
    return {
        "stream_name": stream_name,
        "entries_deleted": entries_deleted,
        "entries_archived": entries_archived,
        "duration_ms": duration_ms,
    }


def build_retention_archive_started_payload(
    stream_name: str,
    archive_days: int,
    cutoff_date: datetime,
) -> dict[str, Any]:
    """Build payload for log.retention.archive_started event."""
    return {
        "stream_name": stream_name,
        "archive_days": archive_days,
        "cutoff_date": cutoff_date.isoformat() + "Z",
    }


def build_retention_archive_completed_payload(
    stream_name: str,
    entries_archived: int,
    archive_location: str,
    duration_ms: float,
) -> dict[str, Any]:
    """Build payload for log.retention.archive_completed event."""
    return {
        "stream_name": stream_name,
        "entries_archived": entries_archived,
        "archive_location": archive_location,
        "duration_ms": duration_ms,
    }


def build_siem_forward_started_payload(
    stream_name: str,
    siem_target: str,
    batch_size: int,
) -> dict[str, Any]:
    """Build payload for log.siem.forward_started event."""
    return {
        "stream_name": stream_name,
        "siem_target": siem_target,
        "batch_size": batch_size,
    }


def build_siem_forward_completed_payload(
    stream_name: str,
    entries_forwarded: int,
    duration_ms: float,
) -> dict[str, Any]:
    """Build payload for log.siem.forward_completed event."""
    return {
        "stream_name": stream_name,
        "entries_forwarded": entries_forwarded,
        "duration_ms": duration_ms,
    }


def build_siem_forward_failed_payload(
    stream_name: str,
    error: str,
    entries_pending: int,
) -> dict[str, Any]:
    """Build payload for log.siem.forward_failed event."""
    return {
        "stream_name": stream_name,
        "error": error,
        "entries_pending": entries_pending,
    }


def build_write_failed_payload(
    logger: str,
    level: str,
    error: str,
    correlation_id: uuid.UUID,
) -> dict[str, Any]:
    """Build payload for log.write.failed event."""
    return {
        "logger": logger,
        "level": level,
        "error": error,
        "correlation_id": str(correlation_id),
    }


def build_query_failed_payload(
    query_params: dict[str, Any],
    error: str,
    correlation_id: uuid.UUID,
) -> dict[str, Any]:
    """Build payload for log.query.failed event."""
    return {
        "query_params": query_params,
        "error": error,
        "correlation_id": str(correlation_id),
    }


def build_aggregate_failed_payload(
    stream_name: str,
    bucket_start: datetime,
    error: str,
) -> dict[str, Any]:
    """Build payload for log.aggregate.failed event."""
    return {
        "stream_name": stream_name,
        "bucket_start": bucket_start.isoformat() + "Z",
        "error": error,
    }


def build_governance_log_alert_payload(
    alert_type: str,
    message: str,
    severity: str,
    recommended_action: str,
) -> dict[str, Any]:
    """Build payload for log.governance.alert event."""
    return {
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "recommended_action": recommended_action,
    }
